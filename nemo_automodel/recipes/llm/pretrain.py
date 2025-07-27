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

"""Pretraining recipe for next-token prediction using NanoGPT-style datasets.

This module reuses the bulk of the fine-tuning recipe implementation
(`finetune.py`) but removes PEFT-specific paths and ensures the embedding
layers remain **trainable**.  It is designed to work out-of-the-box with the
``BinTokenDataset`` that streams token sequences from the binary shards
produced by the *modded-NanoGPT* preprocessing script.
"""
from __future__ import annotations

import logging

import torch.nn as nn

from nemo_automodel.recipes.llm.finetune import (
    FinetuneRecipeForNextTokenPrediction,
    build_model_and_optimizer,
)

logger = logging.getLogger(__name__)


class PretrainRecipeForNextTokenPrediction(FinetuneRecipeForNextTokenPrediction):
    """Minimal extension over *FinetuneRecipeForNextTokenPrediction* for pre-training.

    The only behavioural change is that **all** parameters – including token
    embeddings – remain trainable.  This is achieved by overriding the
    ``setup`` method and calling the shared ``build_model_and_optimizer``
    helper with ``freeze_embeddings=False``.
    """

    def setup(self):  # noqa: C901 – mirrors parent implementation
        import torch  # local import to avoid unnecessary dependency at module import time
        import time  # noqa: F401 – imported for parity with parent recipe signature

        from nemo_automodel.components.loggers.log_utils import setup_logging
        from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
        from nemo_automodel.components.training.rng import StatefulRNG

        from .finetune import (
            build_checkpoint_config,
            build_dataloader,
            build_distributed,
            build_loss_fn,
            build_lr_scheduler,
            build_step_scheduler,
            build_wandb,
        )

        # ---------------- basic setup copied from parent ----------------
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()

        # Distributed model wrapper (FSDP / TP / CP)
        self.device_mesh = None
        self.model_wrapper = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(
                world_size=self.dist_env.world_size
            )
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)

        # W&B (main rank only)
        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at %s", run.url)

        # If packed sequences are in use, switch to HF Flash-Attention-2
        use_hf_fa2 = self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0

        # Pretraining → usually no PEFT, but respect config if provided
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()

        # ---- Build model & optimizer (embeddings *not* frozen) ----
        self.model, self.optimizer = build_model_and_optimizer(
            self.dist_env.device,
            self.cfg.model,
            self.cfg.optimizer,
            use_hf_fa2,
            self.peft_config,
            self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
            freeze_embeddings=False,  # key difference vs. finetune
        )

        # Sanity-check that embeddings are trainable
        frozen = [p for n, p in self.model.named_parameters() if isinstance(p, nn.Parameter) and not p.requires_grad]
        if frozen:
            logger.warning(
                "Detected %d frozen parameters even though freeze_embeddings=False. Check configuration.",
                len(frozen),
            )

        # ---------------- remaining components ----------------
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)
        self.dataloader, self.tokenizer = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            self.cfg.get("packed_sequence", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
        )

        # Optional validation loader
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                self.cfg.model,
                cfg_ps=None,
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
            )

        from nemo_automodel.components.training.rng import StatefulRNG

        self.total_local_num_loss_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        self.forward_data_store = []

        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None), self.dataloader
        )
        self.lr_scheduler = build_lr_scheduler(
            self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler
        )

        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self.checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            self.cfg.model.pretrained_model_name_or_path
            if hasattr(self.cfg.model, "pretrained_model_name_or_path")
            else None,
            True if self.cfg.get("peft", None) else False,
        )

        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)
        self.load_checkpoint(restore_from) 