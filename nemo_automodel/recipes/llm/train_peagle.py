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

"""P-EAGLE training recipe (Parallel-Drafting EAGLE, arXiv:2602.01469).

Reuses the EAGLE-3 recipe end-to-end (online frozen target, dataloader,
draft-vocab mapping, optimizer/LR schedule, checkpointing, train loop) and only
swaps the draft model and trainer module: the autoregressive test-time-training
core is replaced by the parallel COD core (:class:`PEagleTrainerModule`), which
predicts ``num_depths`` tokens per position in one forward pass.
"""

from __future__ import annotations

import logging
import math
import pathlib
from types import SimpleNamespace

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.speculative.eagle import PEagleTrainerModule
from nemo_automodel.components.speculative.eagle.registry import resolve_peagle_draft_spec
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe, _optim_steps_per_epoch

logger = logging.getLogger(__name__)


class TrainPEagleRecipe(TrainEagle3Recipe):
    """P-EAGLE training recipe for Llama-style dense targets (Llama, Phi-3, Qwen3)."""

    def setup(self):
        """Build the frozen target, the P-EAGLE draft, data, optimizer, and trainer module."""
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env", {}).get("backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env", {}).get("timeout_minutes", 30),
        )
        setup_logging()

        recipe_cfg = self.cfg.recipe_args
        self.device = self.dist_env.device or torch.device("cpu")

        target_path = recipe_cfg.target_model_name_or_path
        target_config = AutoConfig.from_pretrained(
            target_path, trust_remote_code=recipe_cfg.get("trust_remote_code", False)
        )
        architectures = getattr(target_config, "architectures", []) or []
        draft_spec = resolve_peagle_draft_spec(architectures)

        from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(
            target_path,
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
        )
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.cached_target_path = recipe_cfg.get("cached_target_path", None)
        self.dist_setup = None
        self.distributed_config = None
        self.device_mesh = None
        self.moe_mesh = None
        if self.cached_target_path is None:
            selected_token_ids, selected_token_mask = self._setup_online_target(recipe_cfg, target_path, target_config)
        else:
            selected_token_ids, selected_token_mask = self._setup_cached_target(recipe_cfg, target_config)

        # Depth>=1 positions are fed a mask token; default to pad/eos when unset.
        mask_token_id = recipe_cfg.get("mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(self.tokenizer, "eos_token_id", None) or 0

        draft_config = target_config.to_dict()
        draft_config["draft_vocab_size"] = int(selected_token_ids.numel())
        draft_config["target_hidden_size"] = target_config.hidden_size
        draft_config["architectures"] = ["LlamaPEagleDraftModel"]
        draft_config["tie_word_embeddings"] = False
        draft_config["num_depths"] = int(recipe_cfg.get("num_depths", 8))
        draft_config["down_sample_ratio"] = float(recipe_cfg.get("down_sample_ratio", 0.7))
        draft_config["down_sample_ratio_min"] = float(recipe_cfg.get("down_sample_ratio_min", 0.2))
        draft_config["mask_token_id"] = int(mask_token_id)
        draft_config_obj = type(target_config).from_dict(draft_config)
        self.draft_model = draft_spec.draft_cls(draft_config_obj).to(device=self.device, dtype=self.compute_dtype)

        embed_source = (
            self.target_wrapper.get_input_embeddings() if self.target_wrapper is not None else self._cached_embed_source
        )
        self.draft_model.copy_embeddings_from_target(embed_source)
        # P-EAGLE trains the draft embeddings (and the mask-token embedding) by
        # default, so freezing is opt-in here (the EAGLE-3 default is the reverse).
        if recipe_cfg.get("freeze_embeddings", False):
            self.draft_model.freeze_embeddings()

        trainer_module = PEagleTrainerModule(
            self.draft_model,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            num_depths=int(recipe_cfg.get("num_depths", 8)),
            down_sample_ratio=float(recipe_cfg.get("down_sample_ratio", 0.7)),
            down_sample_ratio_min=float(recipe_cfg.get("down_sample_ratio_min", 0.2)),
            mask_token_id=int(mask_token_id),
        ).to(self.device)
        if self.dist_env.world_size > 1:
            trainer_module = DistributedDataParallel(
                trainer_module,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        self.trainer_module = trainer_module

        opt_cfg = self.cfg.optimizer
        self.peak_lr = float(opt_cfg.lr)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.trainer_module.parameters() if p.requires_grad],
            lr=self.peak_lr,
            betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
        self.grad_accumulation_steps = recipe_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = recipe_cfg.get("max_grad_norm", 1.0)
        self.num_epochs = recipe_cfg.num_epochs
        self.log_every_steps = recipe_cfg.get("log_every_steps", 10)
        self.output_dir = pathlib.Path(recipe_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            num_batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            num_batches_per_epoch = 0
        total_optim_steps = max(
            1,
            self.num_epochs * _optim_steps_per_epoch(num_batches_per_epoch, self.grad_accumulation_steps),
        )
        warmup_ratio = float(opt_cfg.get("warmup_ratio", 0.05))
        min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.1))
        warmup_steps = max(1, int(warmup_ratio * total_optim_steps))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        self.total_optim_steps = total_optim_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

        self.runtime = SimpleNamespace(global_step=0)
        self._resume_epoch = 0

        self.rng = StatefulRNG(
            seed=int(recipe_cfg.get("shuffle_seed", 42)),
            ranked=self.dist_env.world_size > 1,
        )
        self._build_checkpointer(target_path)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))


def main(config_path=None):
    """Main entry point for the P-EAGLE recipe."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainPEagleRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
