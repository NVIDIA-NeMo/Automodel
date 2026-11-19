# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Single-stage SFT for native GS1 checkpoints with distribution targets."""

from __future__ import annotations

import json
import logging
import math
import random
from copy import copy
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from nemo_automodel.components.checkpoint.utils import unwrap_ddp_model
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.models.gs1.training import (
    GS1DecisionExample,
    GS1DecisionObjective,
    GS1TrainingConfig,
    collate_gs1_examples,
    ensure_gs1_transformers_compat,
    load_gs1_examples,
)
from nemo_automodel.components.training.utils import clip_grad_norm
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)


class TrainGS1DecisionRecipe(BaseRecipe):
    """Fine-tune the native GS1 model with probability-valued SFT targets."""

    def __init__(self, cfg: RecipeConfig | ConfigNode) -> None:
        """Store the recipe configuration at the YAML boundary."""
        self.cfg = cfg if isinstance(cfg, RecipeConfig) else RecipeConfig(cfg)

    def setup(self) -> None:
        """Load a native checkpoint, compile examples, and build the optimizer."""
        available, _ = safe_import(
            "typesafe_sdk",
            msg="GS1 remote model code requires typesafe-sdk==0.7.0; install it with uv.",
        )
        if not available:
            raise RuntimeError("Install typesafe-sdk==0.7.0 before loading a GS1 checkpoint")
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env.backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env.timeout_minutes", 30),
        )
        setup_logging()
        self.seed = int(self.cfg.get("seed", 1234))
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = self.dist_env.device
        self.output_dir = Path(self.cfg.get("output_dir", "runs/gs1_sft"))
        self.max_length = int(self.cfg.get("data.max_length", 16384))

        (
            self.distributed_setup,
            self.mesh_context,
            self.distributed_config,
            self.device_mesh,
            self.moe_mesh,
            self.pp_enabled,
            self.pipeline_config,
            self.moe_parallel_config,
            self.activation_checkpointing,
        ) = self._distributed_setup_attributes(
            create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)
        )
        if self.pp_enabled or self.mesh_context.tp_size > 1 or self.mesh_context.cp_size > 1:
            raise ValueError("The GS1 decision recipe currently supports data parallelism only")

        ensure_gs1_transformers_compat()
        model_path = str(self.cfg.get("model.pretrained_model_name_or_path"))
        revision = self.cfg.get("model.revision", None)
        self.model = self.cfg.model.instantiate(distributed_setup=self.distributed_setup)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            code_revision=self.cfg.get("model.code_revision", revision),
            trust_remote_code=True,
        )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("GS1 tokenizer must define pad_token_id or eos_token_id")
        self.pad_token_id = int(pad_token_id)

        training_config = GS1TrainingConfig(
            model_id=model_path,
            revision=revision,
            code_revision=self.cfg.get("model.code_revision", revision),
            max_context=self.max_length,
        )
        compiler, request_type = training_config.build(unwrap_ddp_model(self.model), self.tokenizer)
        self.examples = load_gs1_examples(
            self.cfg.get("data.train"),
            compiler=compiler,
            request_type=request_type,
            model_id=model_path,
            max_examples=int(self.cfg.get("data.max_examples", 3000)),
            seed=self.seed,
        )
        self.objective = GS1DecisionObjective(brier_weight=float(self.cfg.get("loss.brier_weight", 1.0)))
        self.local_batch_size = int(self.cfg.get("train.local_batch_size", 1))
        self.epochs = int(self.cfg.get("train.epochs", 1))
        self.max_grad_norm = float(self.cfg.get("train.max_grad_norm", 1.0))
        if self.local_batch_size < 1 or self.epochs < 1 or self.max_grad_norm <= 0:
            raise ValueError("batch size, epochs, and max_grad_norm must be positive")
        if self.cfg.optimizer is None:
            raise ValueError("optimizer is required for GS1 decision training")
        self.optimizers = self.cfg.optimizer.build(self.model, device_mesh=self.device_mesh)
        checkpoint_config = copy(self.cfg.checkpoint)
        checkpoint_config.checkpoint_dir = str(self.output_dir)
        self.checkpointer = checkpoint_config.build(
            dp_rank=self._get_dp_rank(),
            tp_rank=0,
            pp_rank=0,
            moe_mesh=self.moe_mesh,
            process_group=self._get_dp_group(),
        )

    def _shard_indices(self, epoch: int) -> list[int]:
        """Return an equal-size per-rank shard with deterministic padding."""
        indices = list(range(len(self.examples)))
        random.Random(self.seed + epoch).shuffle(indices)
        world = self._get_dp_group_size()
        total = math.ceil(len(indices) / world) * world
        indices.extend(indices[index % len(indices)] for index in range(total - len(indices)))
        return indices[self._get_dp_rank() : total : world]

    def _forward_loss(self, examples: list[GS1DecisionExample]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the model-owned branch readout and exact soft-target objective."""
        batch = collate_gs1_examples(examples, pad_token_id=self.pad_token_id, device=self.device)
        output = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            branches=list(batch.branches),
            state_lengths=list(batch.state_lengths),
        )
        if output.logits is None:
            raise ValueError("GS1 model returned no decision logits")
        return self.objective(output.logits, batch.target_probs, batch.option_mask)

    def run_train_validation_loop(self) -> None:
        """Run one-stage SFT and publish the final native-compatible checkpoint."""
        if not self.examples:
            raise ValueError("GS1 training data produced no examples")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log = (self.output_dir / "training_log.jsonl").open("w", encoding="utf-8") if self.dist_env.is_main else None
        step = 0
        for epoch in range(self.epochs):
            indices = self._shard_indices(epoch)
            self.model.train()
            for start in range(0, len(indices), self.local_batch_size):
                examples = [self.examples[index] for index in indices[start : start + self.local_batch_size]]
                for optimizer in self.optimizers:
                    optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    loss, metrics = self._forward_loss(examples)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("GS1 training objective became nonfinite")
                loss.backward()
                grad_norm = clip_grad_norm(self.max_grad_norm, [self.model], device_mesh=self.device_mesh)
                for optimizer in self.optimizers:
                    optimizer.step()
                step += 1
                if log is not None:
                    values = {key: float(value) for key, value in metrics.items()}
                    values.update(
                        {
                            "epoch": epoch,
                            "step": step,
                            "loss": float(loss.detach()),
                            "grad_norm": float(grad_norm),
                        }
                    )
                    log.write(json.dumps(values) + "\n")
                    if step % int(self.cfg.get("train.log_interval", 10)) == 0:
                        logger.info("GS1 step %d: %s", step, values)
        if log is not None:
            log.close()
        self._save_final_checkpoint()
        self.checkpointer.finalize()
        if dist.is_initialized():
            dist.barrier()

    def _save_final_checkpoint(self) -> None:
        """Publish final model weights through NeMo's checkpoint lifecycle."""
        if not self.checkpointer.config.enabled:
            return
        path = self.output_dir / "final"
        self.checkpointer.lifecycle.reserve(str(path))
        self.checkpointer.save_model(
            unwrap_ddp_model(self.model),
            str(path),
            tokenizer=self.tokenizer,
            is_final_checkpoint=True,
        )
        self.checkpointer.async_wait()
        self.checkpointer.lifecycle.publish(str(path), best_val_metric=None, metric_key=None)


def main(config_path: str | None = None) -> None:
    """Run the GS1 decision SFT recipe from an AutoModel YAML config."""
    cfg = parse_args_and_load_config(config_path)
    recipe = TrainGS1DecisionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
