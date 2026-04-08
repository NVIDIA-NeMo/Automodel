# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import wandb

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Returns (num_correct, batch_size) for top-1 accuracy."""
    num_correct = output.argmax(dim=1).eq(target).sum()
    return num_correct, target.size(0)


@torch.no_grad()
def batch_mrr(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Returns sum of reciprocal ranks for the batch. Stays on GPU."""
    _, sorted_indices = torch.sort(output, dim=-1, descending=True)
    _, rank = torch.nonzero(sorted_indices.eq(target.unsqueeze(-1)).long(), as_tuple=True)
    rank = rank + 1
    return (1.0 / rank.float()).sum()


class TrainCrossEncoderRecipe(TrainBiEncoderRecipe):
    def _run_train_optim_step(self, batches, max_grad_norm=None):
        self._acc_buffer = []
        result = super()._run_train_optim_step(batches, max_grad_norm)

        # Sum (correct, total) across micro-batches
        total_correct = torch.stack([c for c, _ in self._acc_buffer]).sum()
        total_samples = sum(t for _, t in self._acc_buffer)

        # Allreduce across DP ranks for global accuracy
        if torch.distributed.is_initialized():
            counts = torch.stack(
                [total_correct, torch.tensor(total_samples, device=total_correct.device, dtype=total_correct.dtype)]
            )
            counts = self._dp_allreduce(counts)
            total_correct, total_samples = counts[0], counts[1]

        train_accuracy = total_correct.item() / total_samples.item() if total_samples > 0 else 0.0
        result.metrics["train_accuracy"] = train_accuracy
        return result

    def log_train_metrics(self, log_data: MetricsSample):
        if not self.dist_env.is_main:
            return

        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=self.step_scheduler.step)

        self.metric_logger_train.log(log_data)

        msg = "step {} | epoch {} | loss {:.4f} | train_acc {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | time {:.2f}s".format(
            log_data.step,
            log_data.epoch,
            log_data.metrics["loss"],
            log_data.metrics["train_accuracy"],
            log_data.metrics["grad_norm"],
            log_data.metrics["lr"],
            log_data.metrics["mem"],
            log_data.metrics["time_per_step"],
        )
        logging.info(msg)

        torch.cuda.reset_peak_memory_stats()

    def _forward_backward_step(self, idx, batch, *, loss_buffer, num_batches, is_train: bool = True):
        """Forward and backward pass for a single micro-batch."""
        batch = {
            k: v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        labels = batch.pop("labels")

        model = self.model_parts[0]
        train_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        with train_ctx, sync_ctx:
            outputs = model(**batch, return_dict=True)

            outputs.logits = outputs.logits.view(-1, self.train_n_passages)
            loss = F.cross_entropy(outputs.logits, labels)

            loss_buffer.append(loss.clone().detach())
            self._acc_buffer.append(accuracy(outputs.logits, labels))

            if is_train:
                # Scale loss by number of gradient accumulation steps to get correct average gradients
                # FSDP/DDP will handle averaging across DP ranks automatically
                scaled_loss = loss / num_batches
                scaled_loss.backward()

    def _run_validation_epoch(self, val_dataloader):
        """Run validation for one epoch and compute loss, accuracy@1, and MRR."""
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()
            loss_buffer = []
            all_logits = []
            all_labels = []

            model = self.model_parts[0]
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
            )

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {
                        k: v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    labels = batch.pop("labels")

                    with autocast_ctx:
                        outputs = model(**batch, return_dict=True)
                        logits = outputs.logits.view(-1, self.val_n_passages)
                        loss = F.cross_entropy(logits, labels)

                    loss_buffer.append(loss.clone().detach())
                    all_logits.append(logits.detach())
                    all_labels.append(labels.detach())

            loss_sum = torch.stack(loss_buffer).sum()
            loss_count = torch.tensor(len(loss_buffer), device=self.dist_env.device, dtype=loss_sum.dtype)
            if torch.distributed.is_initialized():
                loss_sum = self._dp_allreduce(loss_sum)
                loss_count = self._dp_allreduce(loss_count)
            avg_loss = loss_sum / loss_count

            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)
            n_samples = labels.size(0)

            num_correct, _ = accuracy(logits, labels)
            num_correct = num_correct.float()
            rr_sum = batch_mrr(logits, labels)

            # Allreduce counts across DP ranks for global metrics
            if torch.distributed.is_initialized():
                counts = torch.tensor([num_correct, rr_sum, n_samples], device=self.dist_env.device, dtype=torch.float)
                counts = self._dp_allreduce(counts)
                num_correct, rr_sum, n_samples = counts[0].item(), counts[1].item(), counts[2].item()

            acc1 = num_correct / n_samples if n_samples > 0 else 0.0
            mrr = rr_sum / n_samples if n_samples > 0 else 0.0

            metrics = {
                "val_loss": avg_loss.item(),
                "val_acc1": acc1,
                "val_mrr": mrr,
            }

            return MetricsSample(
                step=self.step_scheduler.step,
                epoch=self.step_scheduler.epoch,
                metrics=metrics,
            )


def main(default_config_path="examples/retrieval/cross_encoder/llama3_2_1b.yaml"):
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainCrossEncoderRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
