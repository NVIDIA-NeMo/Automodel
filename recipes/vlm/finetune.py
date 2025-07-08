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

from __future__ import annotations

import logging
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from nemo_automodel.config.cli import parse_args_and_load_config
from nemo_automodel.datasets.vlm.collate_fns import COLLATE_FNS
from nemo_automodel.distributed.nvfsdp import NVFSDPManager
from nemo_automodel.loggers.log_utils import setup_logging
from nemo_automodel.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.training.rng import StatefulRNG
from nemo_automodel.utils.model_utils import apply_parameter_freezing, print_trainable_parameters
from recipes.llm.finetune import (
    FinetuneRecipeForNextTokenPrediction,
    build_checkpoint_config,
    build_distributed,
    build_loss_fn,
    build_step_scheduler,
    build_wandb,
)


logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------


def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_opt,
    cfg_freeze,
    cfg_peft,
    model_wrapper,
    seed,
    tp_size=1,
) -> tuple[nn.Module, "Optimizer"]:  # noqa: F821
    """Build and initialize a model for VLM."""
    with StatefulRNG(seed=seed, ranked=True):
        model = cfg_model.instantiate()

        if cfg_freeze is not None:
            apply_parameter_freezing(model, cfg_freeze)
        else:
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.requires_grad = False

        # Optionally apply PEFT (e.g., LoRA/DoRA, etc)
        if cfg_peft is not None:
            opts = cfg_peft.to_dict()
            peft_fn = opts.pop("peft_fn")
            peft_fn(model, **opts)

        print_trainable_parameters(model)

        if callable(getattr(model_wrapper, "parallelize", None)):
            if isinstance(model_wrapper, NVFSDPManager):
                trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
                assert len(trainable_params) > 0, "trainable_params cannot be empty"
                if tp_size > 1:
                    cfg_opt.foreach = False
                optimizer = cfg_opt.instantiate(params=trainable_params)
                model, optimizer = model_wrapper.parallelize(model, optimizer)
                return model, optimizer
            else:
                model = model_wrapper.parallelize(model)
        else:
            model = model.to(device)

        trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
        assert len(trainable_params) > 0, "trainable_params cannot be empty"
        if tp_size > 1:
            cfg_opt.foreach = False
        optimizer = cfg_opt.instantiate(params=trainable_params)

        return model, optimizer


def build_dataloader(cfg_ds, cfg_dl, cfg_model, cfg_processor, device_mesh, seed) -> DataLoader:
    """Build a VLM dataloader."""
    dist_sampler_kwargs = {
        "shuffle": cfg_dl.get("shuffle", True),
    }
    if device_mesh is not None:
        dist_sampler_kwargs |= {
            "num_replicas": device_mesh["data_parallel"].size(),
            "rank": device_mesh["data_parallel"].get_local_rank(),
        }

    with StatefulRNG(seed=seed, ranked=True):
        if hasattr(cfg_ds, "_target_") and "vlm" in cfg_ds._target_:
            processor = AutoProcessor.from_pretrained(cfg_model.pretrained_model_name_or_path)

        ds = cfg_ds.instantiate(path_or_dataset=cfg_ds.path_or_dataset)

        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            **dist_sampler_kwargs,
        )
        collate_cfg = cfg_dl.get("collate_fn", None)
        if collate_cfg:
            collate_fn = lambda examples: collate_cfg.instantiate(examples=examples, processor=processor)
        else:
            processor_type = type(processor).__name__
            if processor_type not in COLLATE_FNS:
                processor_type = "default"
                logging.warning(f"You are using {processor_type} with default collate function.")
            collate_fn = lambda examples: COLLATE_FNS[processor_type](examples, processor)

        return cfg_dl.instantiate(dataset=ds, sampler=sampler, collate_fn=collate_fn)


class FinetuneRecipeForVLM(FinetuneRecipeForNextTokenPrediction):
    """VLM Recipe that inherits from LLM Recipe and overrides specific VLM behavior."""

    def setup(self):
        """Override setup to use VLM-specific builders."""
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()

        self.device_mesh = None
        self.model_wrapper = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        # Build components with VLM-specific functions
        self.model, self.optimizer = build_model_and_optimizer(
            self.dist_env.device,
            self.cfg.model,
            self.cfg.optimizer,
            self.cfg.get("freeze_config", None),  # VLM-specific
            self.cfg.get("peft", None),
            self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
        )
        self.loss_fn = build_loss_fn(self.dist_env.device, self.cfg.loss_fn)
        self.dataloader = build_dataloader(  # VLM-specific
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            self.cfg.get("processor", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader = build_dataloader(  # VLM-specific
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                self.cfg.model,
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
            )

        # Initialize metrics required for calculating loss
        self.total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        self.forward_data_store = []

        # Scheduler
        self.step_scheduler = build_step_scheduler(self.cfg.get("step_scheduler", None), self.dataloader)

        # Build checkpointing config
        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self.checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            self.cfg.model.pretrained_model_name_or_path,
            True if self.cfg.get("peft", None) else False,
        )

        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        # Optionally resume
        self.load_checkpoint(restore_from)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    script_path = pathlib.Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "gemma_3_vl_4b_cord_v2.yaml")
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
