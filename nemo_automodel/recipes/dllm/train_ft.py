# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Diffusion LLM (dLLM) SFT recipe for Automodel.

Extends ``TrainFinetuneRecipeForNextTokenPrediction`` to support diffusion LLM
training. Instead of next-token prediction, the model is trained as a denoiser:
tokens are randomly corrupted and the model predicts the clean token at each
position.  Loss is weighted by the inverse corruption probability.

Model-specific behaviour (loss function, corruption strategy, batch preparation)
is encapsulated in :mod:`~nemo_automodel.recipes.dllm.strategy` so that new
dLLM variants can be added without modifying this recipe.  Current modes:

- **mdlm**: Pure masked denoising.  Uses ``MDLMCrossEntropyLoss``.

Usage::

    python -m torch.distributed.run --nproc-per-node=8 \\
        nemo_automodel/recipes/dllm/train_ft.py \\
        -c examples/dllm_sft/mdlm_sft.yaml
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Optional

import torch
import wandb
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.dllm.collate import DLLMCollator
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.training.utils import (
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.flops_utils import calculate_mfu
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from nemo_automodel.recipes.dllm.strategy import get_dllm_strategy
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

logger = logging.getLogger(__name__)


class DiffusionLMSFTRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Recipe for dLLM (diffusion LLM) supervised fine-tuning.

    Extends the standard fine-tuning recipe by:

    1. Wrapping the dataloader collate function to produce unshifted batches
    2. Applying token corruption before each forward pass
    3. Using dLLM-specific loss functions via a pluggable strategy
    """

    def setup(self):
        """Build all training components, then apply dLLM-specific overrides."""
        # Let parent build model, optimizer, dataloader, scheduler, etc.
        super().setup()

        # --- dLLM config ---
        dllm_cfg = self.cfg.get("dllm", None)
        if dllm_cfg is None:
            raise ValueError("Config must contain a 'dllm' section for DiffusionLMSFTRecipe")

        self.dllm_mode = dllm_cfg.get("mode", "mdlm")
        self.dllm_strategy = get_dllm_strategy(self.dllm_mode)
        if self.dllm_strategy.normalization_mode not in ("supervised", "noise"):
            raise ValueError(
                f"Invalid normalization_mode {self.dllm_strategy.normalization_mode!r} "
                f"from strategy {type(self.dllm_strategy).__name__}. "
                f"Must be 'supervised' or 'noise'."
            )

        self.dllm_eps = float(dllm_cfg.get("eps", 1e-3))
        self.dllm_block_size = dllm_cfg.get("block_size", None)
        if self.dllm_block_size is not None:
            self.dllm_block_size = int(self.dllm_block_size)
        hlr = dllm_cfg.get("half_life_ratio", 0.25)
        self.dllm_half_life_ratio = float(hlr) if hlr is not None else None

        # Padding config (two-stage block-aligned padding)
        pbs = dllm_cfg.get("pad_block_size", None)
        self.dllm_pad_block_size = int(pbs) if pbs is not None else None
        psld = dllm_cfg.get("pad_seq_len_divisible", None)
        self.dllm_pad_seq_len_divisible = int(psld) if psld is not None else None

        # Resolve mask_token_id
        self.mask_token_id = dllm_cfg.get("mask_token_id", None)
        if self.mask_token_id is None:
            # Try to get from tokenizer
            if (
                self.tokenizer is not None
                and hasattr(self.tokenizer, "mask_token_id")
                and self.tokenizer.mask_token_id is not None
            ):
                self.mask_token_id = self.tokenizer.mask_token_id
            else:
                raise ValueError("dllm.mask_token_id must be set in config, or the tokenizer must have a mask_token_id")
        self.mask_token_id = int(self.mask_token_id)

        # --- Build dLLM loss function via strategy ---
        self.dllm_loss_fn = self.dllm_strategy.create_loss_fn(dllm_cfg)

        logger.info(
            f"dLLM SFT setup: mode={self.dllm_mode}, mask_token_id={self.mask_token_id}, "
            f"eps={self.dllm_eps}, block_size={self.dllm_block_size}, "
            f"half_life_ratio={self.dllm_half_life_ratio}, "
            f"normalization_mode={self.dllm_strategy.normalization_mode}"
        )

        # --- Wrap dataloader collate to produce unshifted format ---
        self._wrap_dataloader_collate()

        # Buffers for dLLM-specific metrics
        self._dllm_loss_buffer = []

    def _wrap_dataloader_collate(self):
        """Replace dataloader collate functions with the dLLM single-pass collater.

        Uses :class:`DLLMCollator` which goes directly from
        variable-length sample lists to block-aligned tensors in one pass.

        Requires datasets to produce unshifted format (``input_ids`` +
        ``loss_mask``, via ``_package_tokenized_example(unshifted=True)``).
        """
        pad_token_id = 0
        if (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "pad_token_id")
            and self.tokenizer.pad_token_id is not None
        ):
            pad_token_id = self.tokenizer.pad_token_id

        eos_token_id = None
        if (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            eos_token_id = self.tokenizer.eos_token_id

        max_seq_len = self.cfg.get("dataset.seq_length", None)
        if max_seq_len is not None:
            max_seq_len = int(max_seq_len)

        dllm_cfg = self.cfg.get("dllm", {})
        supervise_padding = bool(dllm_cfg.get("supervise_padding", False))

        collator = DLLMCollator(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            block_size=self.dllm_pad_block_size,
            pad_seq_len_divisible=self.dllm_pad_seq_len_divisible,
            max_seq_len=max_seq_len,
            supervise_padding=supervise_padding,
        )

        self.dataloader.collate_fn = collator
        for _name, val_dl in self.val_dataloaders.items():
            val_dl.collate_fn = collator

    def _apply_corruption(self, input_ids, loss_mask):
        """Apply token corruption via the configured strategy.

        Args:
            input_ids: Clean token IDs, shape [B, L].
            loss_mask: Supervised positions mask, shape [B, L].

        Returns:
            Tuple of (noisy_input_ids, noise_mask, p_mask).
        """
        return self.dllm_strategy.apply_corruption(
            input_ids,
            loss_mask,
            self.mask_token_id,
            eps=self.dllm_eps,
            block_size=self.dllm_block_size,
            half_life_ratio=self.dllm_half_life_ratio,
        )

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_diffusion_tokens,
        num_batches,
        is_train: bool = True,
    ):
        """Override: apply dLLM corruption and compute dLLM loss."""
        # Move batch to device
        batch = {
            k: (
                {dk: dv.to(self.dist_env.device, non_blocking=True) for dk, dv in v.items() if dv is not None}
                if isinstance(v, dict)
                else (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            )
            for k, v in batch.items()
        }

        # Use pre-computed corruption if available (from _run_train_optim_step),
        # otherwise compute on the fly (validation path).
        if "_noise_mask" in batch:
            noisy_input_ids = batch.pop("_noisy_input_ids")
            noise_mask = batch.pop("_noise_mask")
            p_mask = batch.pop("_p_mask")
            clean_input_ids = batch.pop("_clean_input_ids")
            loss_mask = batch.pop("loss_mask")
        else:
            loss_mask = batch.pop("loss_mask")
            clean_input_ids = batch["input_ids"].clone()
            noisy_input_ids, noise_mask, p_mask = self._apply_corruption(clean_input_ids, loss_mask)

        batch = self.dllm_strategy.prepare_batch(batch, noisy_input_ids, noise_mask, clean_input_ids)

        model = self.model_parts[0]

        # Context parallel setup (no labels to pass for dLLM)
        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
        fp8_ctx = self.te_fp8.maybe_te_autocast() if self.te_fp8 is not None else nullcontext()
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )

        autocast_dtype = getattr(self.distributed_config, "autocast_dtype", None)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
        )

        with train_ctx(), sync_ctx, fp8_ctx, autocast_ctx:
            batch = filter_forward_kwargs(model, batch)
            out = model(**batch)
            logits = getattr(out, "logits", out)
            del out

            # Compute dLLM loss (unified interface via DLLMLossOutput)
            loss_result = self.dllm_loss_fn(
                logits=logits,
                target_ids=clean_input_ids,
                noise_mask=noise_mask,
                p_mask=p_mask,
                loss_mask=loss_mask,
                num_diffusion_tokens=num_diffusion_tokens,
            )
            microbatch_loss = loss_result.total_loss
            dllm_loss = loss_result.dllm_loss.detach().clone()

            loss_buffer.append(microbatch_loss.clone().detach())
            self._dllm_loss_buffer.append(dllm_loss)

            if is_train:
                (microbatch_loss * self._get_dp_group_size(include_cp=True)).backward()

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step with dLLM loss.

        Follows the parent pattern but uses loss_mask from the collate wrapper
        instead of labels != -100 for token counting.
        """
        # Pre-corrupt all microbatches so we can count noise tokens globally
        # before any forward pass.
        num_noise_tokens = 0  # diffusion loss denominator (corrupted positions)
        num_supervised_tokens = 0  # total supervised tokens (all loss_mask==1 positions)

        for batch in batches:
            noisy_input_ids, noise_mask, p_mask = self._apply_corruption(batch["input_ids"], batch["loss_mask"])
            batch["_noisy_input_ids"] = noisy_input_ids
            batch["_noise_mask"] = noise_mask
            batch["_p_mask"] = p_mask
            batch["_clean_input_ids"] = batch["input_ids"].clone()
            num_noise_tokens += noise_mask.sum().item()
            num_supervised_tokens += batch["loss_mask"].sum().item()

        num_noise_tokens = self._dp_allreduce(torch.tensor(num_noise_tokens, dtype=torch.long)).item()
        num_supervised_tokens = self._dp_allreduce(torch.tensor(num_supervised_tokens, dtype=torch.long)).item()

        # Select denominator based on strategy (MDLM -> supervised, future models may use noise)
        if self.dllm_strategy.normalization_mode == "noise":
            num_diffusion_tokens = num_noise_tokens
        else:
            num_diffusion_tokens = num_supervised_tokens

        loss_buffer = []

        # Count total tokens excluding tail padding
        num_tokens_in_batch = torch.tensor(sum(batch["input_ids"].numel() for batch in batches), dtype=torch.long)
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        num_batches = len(batches)
        prepare_for_grad_accumulation(self.model_parts, pp_enabled=self.pp_enabled)

        for i, batch in enumerate(batches):
            if i == num_batches - 1:
                prepare_for_final_backward(self.model_parts, pp_enabled=self.pp_enabled)

            self._forward_backward_step(
                i,
                batch,
                loss_buffer=loss_buffer,
                num_diffusion_tokens=num_diffusion_tokens,
                num_batches=num_batches,
            )

            if i == 0:
                prepare_after_first_microbatch()

        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm,
            self.model_parts,
            norm_type=2.0,
            pp_enabled=self.pp_enabled,
            device_mesh=self.device_mesh,
            ep_axis_name="ep" if getattr(self, 'dist_setup', None) is not None and self.dist_setup.moe_mesh is not None and "ep" in self.dist_setup.moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp" if self.pp_enabled else None,
            foreach=True,
            num_label_tokens=num_diffusion_tokens,
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and not self.pp_enabled
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta

        mfu = None
        mfu_calculator = getattr(self, "mfu_calculator", None)
        if batches and mfu_calculator is not None:
            step_flops = 0.0
            flops_supported = True
            for batch in batches:
                input_ids = batch.get("input_ids")
                if input_ids is None:
                    flops_supported = False
                    break
                batch_flops = mfu_calculator.get_flops(input_ids)
                if batch_flops is None:
                    flops_supported = False
                    break
                step_flops += float(batch_flops)

            if flops_supported:
                step_flops = self._dp_allreduce(
                    torch.tensor(step_flops, dtype=torch.float64, device=self.dist_env.device), include_cp=True
                ).item()
                mfu = calculate_mfu(step_flops / 1e12, self.dist_env.world_size, time_delta)

        total_loss = torch.sum(torch.stack(loss_buffer))
        total_loss = self._dp_allreduce(total_loss, include_cp=True).cpu().item()

        dllm_loss = self._dp_allreduce(torch.stack(self._dllm_loss_buffer).sum(), include_cp=True).item()
        self._dllm_loss_buffer.clear()

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": total_loss,
                "Loss/Train_Total": total_loss,
                "Loss/Train_DLLM": dllm_loss,
                "grad_norm": grad_norm,
                "Train/grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "Train/lr": self.optimizer[0].param_groups[0]["lr"],
                "Train/mem": torch.cuda.max_memory_allocated() / 1024**3,
                "Train/tps": tps,
                "Train/tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
                "Train/mfu": mfu,
                "Train/tokens_per_step": num_tokens_in_batch,
                "Train/supervised_tokens": num_supervised_tokens,
                "Train/mode": self.dllm_mode,
            },
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one validation pass with dLLM corruption and loss.

        Computes per-batch loss with proper denominators, then accumulates
        weighted by noise token count to produce a per-noise-token average
        across the val set.
        """
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_weighted_loss = torch.tensor(0.0, dtype=torch.float32, device=self.dist_env.device)
            total_norm_tokens = 0
            use_noise = self.dllm_strategy.normalization_mode == "noise"

            for batch in val_dataloader:
                # Pre-corrupt this val batch (same as training path)
                input_ids = batch["input_ids"]
                loss_mask = batch["loss_mask"]
                noisy_input_ids, noise_mask, p_mask = self._apply_corruption(input_ids, loss_mask)
                batch["_noisy_input_ids"] = noisy_input_ids
                batch["_noise_mask"] = noise_mask
                batch["_p_mask"] = p_mask
                batch["_clean_input_ids"] = input_ids.clone()

                # Count tokens for this batch (all-reduce across DP for this batch)
                num_noise = self._dp_allreduce(torch.tensor(noise_mask.sum().item(), dtype=torch.long)).item()
                num_supervised = self._dp_allreduce(torch.tensor(loss_mask.sum().item(), dtype=torch.long)).item()
                num_norm = num_noise if use_noise else num_supervised

                loss_buffer = []
                self._forward_backward_step(
                    0,
                    batch,
                    loss_buffer=loss_buffer,
                    num_diffusion_tokens=num_norm,
                    num_batches=1,
                    is_train=False,
                )

                # Accumulate: per-token-avg loss * norm_count
                batch_loss = torch.sum(torch.stack(loss_buffer)).item()
                batch_loss = self._dp_allreduce(
                    torch.tensor(batch_loss, dtype=torch.float32, device=self.dist_env.device),
                    include_cp=True,
                ).item()
                total_weighted_loss += batch_loss * num_norm
                total_norm_tokens += num_norm

        val_loss = total_weighted_loss / max(total_norm_tokens, 1e-8)
        val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

        # Clear dLLM loss buffer from validation
        self._dllm_loss_buffer.clear()

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_norm_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_train_metrics(self, log_data):
        """Log dLLM-specific training metrics."""
        if not self.dist_env.is_main:
            return

        if self.step_scheduler.is_remote_logging_step:
            # Filter out step/epoch/timestamp — they're redundant with the
            # x-axis and would create separate wandb panels.
            remote_metrics = {k: v for k, v in log_data.to_dict().items() if k not in ("step", "epoch", "timestamp")}
            if wandb.run is not None:
                wandb.log(remote_metrics, step=self.step_scheduler.step)
            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics(remote_metrics, step=log_data.step)
            if self.comet_logger is not None:
                self.comet_logger.log_metrics(remote_metrics, step=log_data.step)

        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | dllm_loss {:.4f} | grad_norm {:.4f} | "
            "lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | mode {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["Loss/Train_Total"],
                log_data.metrics["Loss/Train_DLLM"],
                log_data.metrics["Train/grad_norm"],
                log_data.metrics["Train/lr"],
                log_data.metrics["Train/mem"],
                log_data.metrics["Train/tps"],
                log_data.metrics["Train/tps_per_gpu"],
                log_data.metrics["Train/mode"],
            )
        )
        torch.cuda.reset_peak_memory_stats()


# Entry point
def main(config_path=None):
    """Main entry point for dLLM SFT recipe."""
    if config_path is None:
        config_path = "examples/dllm_sft/mdlm_sft.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = DiffusionLMSFTRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
