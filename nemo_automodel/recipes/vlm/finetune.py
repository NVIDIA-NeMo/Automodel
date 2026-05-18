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

import warnings

# Suppress pydantic v2 UnsupportedFieldAttributeWarning before heavy imports
# (transformers, huggingface_hub) trigger schema generation.
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass

import logging
import pathlib
import time
from typing import TYPE_CHECKING, Optional

import torch
import wandb
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.shared.te_patches import apply_te_patches
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed import build_distributed
from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import (
    MetricsSample,
    build_metric_logger,
    log_training_metrics,
    log_validation_metrics,
)
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.quantization.fp8 import build_fp8_config
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.components.utils.compile_utils import build_compile_config
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep
from nemo_automodel.recipes._component_builders import (
    build_checkpoint_config,
    build_loss_fn,
    build_lr_scheduler,
    build_optimizer,  # noqa: F401 — re-exported for vlm/kd.py + tests
    build_step_scheduler,
    build_wandb,
)
from nemo_automodel.recipes._dist_setup import setup_distributed
from nemo_automodel.recipes.base_recipe import BaseRecipe

# Re-exports (back-compat for vlm/kd.py and external test imports).
# Builders/helpers were moved out of this recipe in the engine refactor.
from nemo_automodel._transformers.auto_tokenizer import _get_model_name  # noqa: E402, F401
from nemo_automodel.components.datasets.vlm.build import (  # noqa: E402, F401
    build_vlm_dataloader as build_dataloader,
)
from nemo_automodel.components.loss.calculate import calculate_loss  # noqa: E402, F401
from nemo_automodel.components.training.build import build_model as _build_model_impl  # noqa: E402
from nemo_automodel.vlm_engine import chunk_vlm_media as _chunk_vlm_media  # noqa: E402, F401


def build_model(
    cfg_model,
    cfg_freeze,
    cfg_peft,
    seed,
    cfg_fp8=None,
    cfg_compile=None,
    device_mesh=None,
    moe_mesh=None,
    distributed_config=None,
    pipeline_config=None,
    cfg_moe=None,
    activation_checkpointing=False,
):
    """Recipe-layer wrapper. Translates the VLM recipe's ``cfg_*`` ConfigNodes
    into typed kwargs for :func:`components.training.build.build_model`, and
    enforces that the model target is one of the ``NeMoAutoModelFor*``
    classmethods (VLM has no bare-model fallback)."""
    from nemo_automodel._transformers.auto_model import is_nemo_auto_factory
    from nemo_automodel.components.moe.config import MoEParallelizerConfig

    target = cfg_model.get("_target_", None)
    if not is_nemo_auto_factory(target):
        raise ValueError(
            f"VLM finetuning requires NeMoAutoModelForImageTextToText. "
            f"Got model target: {target}"
        )

    return _build_model_impl(
        model_factory=cfg_model.instantiate,
        model_kwargs={},
        is_nemo_auto_model=True,
        seed=seed,
        peft_config=cfg_peft,
        freeze_config=cfg_freeze.to_dict() if cfg_freeze is not None else None,
        fp8_config=build_fp8_config(cfg_fp8) if cfg_fp8 is not None else None,
        compile_config=build_compile_config(cfg_compile) if cfg_compile is not None else None,
        moe_config=MoEParallelizerConfig.coerce(cfg_moe),
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=distributed_config,
        pipeline_config=pipeline_config,
        activation_checkpointing=activation_checkpointing,
    )

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------


class FinetuneRecipeForVLM(BaseRecipe):
    """Recipe for fine-tuning a VLM model."""

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg

    # ------------------ build phase ------------------
    def setup(self):
        """Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        self._setup_distributed_env()

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        # Log experiment details on main rank
        self._log_experiment_details()
        self._log_library_versions()

        # Build loss_fn (will be set on pipeline_config if PP enabled)
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)

        # Pipeline runtime fields: override pp_batch_size and pp_microbatch_size
        if self.pp_enabled:
            pp_batch_size = self.cfg.step_scheduler.local_batch_size
            pp_microbatch_size = self.cfg.get("distributed.pipeline.pp_microbatch_size", 1)

            assert pp_batch_size // pp_microbatch_size >= self.dist_setup.pp_size, (
                f"pp_batch_size {pp_batch_size} // pp_microbatch_size {pp_microbatch_size} must be >= pp_size {self.dist_setup.pp_size}"
            )

            assert not isinstance(self.distributed_config, MegatronFSDPConfig), (
                "MegatronFSDPConfig is not supported when pipeline parallelism is enabled"
            )

            # Update pipeline_config runtime fields
            self.pipeline_config.pp_batch_size = pp_batch_size
            self.pipeline_config.pp_microbatch_size = pp_microbatch_size
            self.pipeline_config.patch_stage_backward_maybe_with_nosync = self.cfg.get(
                "model.backend.enable_fsdp_optimizations", False
            )
            self.pipeline_config.loss_fn = self.loss_fn

        # Build components with VLM-specific functions
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()

        # Build checkpoint config
        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            _get_model_name(self.cfg.model),
            True if self.cfg.get("peft", None) else False,
        )

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified in config, using default value of 1.0")
            self.max_grad_norm = 1.0

        # Create Checkpointer instance
        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        # Disable fused RoPE when context parallelism is enabled (cp > 1)
        if self.dist_setup.cp_size > 1 and self.cfg.get("model.backend.rope_fusion", False):
            logging.info("Disabling rope_fusion because cp_size=%d > 1", self.dist_setup.cp_size)
            self.cfg.model.backend.rope_fusion = False

        # lr_scheduler is built later from step_scheduler; attach to engine after.
        from nemo_automodel.engine import Engine
        from nemo_automodel.vlm_engine import VLMEngine

        self.engine = VLMEngine(Engine.Config(
            model=self.cfg.model,
            distributed=self.dist_setup,
            optimizer=self.cfg.optimizer,
            lr_scheduler=None,
            peft=self.peft_config,
            fp8=self.cfg.get("fp8", None),
            compile=self.cfg.get("compile", None),
            freeze_config=self.cfg.get("freeze_config", None),
            seed=self.cfg.get("seed", 42),
            max_grad_norm=self.max_grad_norm,
            defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
        ))
        model = self.engine.model
        self.optimizer = [self.engine.optimizer]

        if not _supports_logits_to_keep(model) and not isinstance(self.loss_fn, MaskedCrossEntropy):
            logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
            self.loss_fn = MaskedCrossEntropy()

        if isinstance(model, AutoPipeline):
            self.model_parts = model.parts
            self.pp = model
            if self.enable_nvtx:
                import nemo_automodel.autonvtx as autonvtx

                for i, part in enumerate(self.model_parts):
                    autonvtx.patch(part, name=f"PipelineStage_{i}")
        else:
            if self.enable_nvtx:
                import nemo_automodel.autonvtx as autonvtx

                autonvtx.patch(model, name=model.__class__.__name__)
            self.model_parts = [model]
            self.pp = None

        # Extract mRoPE position-id builder from the model so VLM neat packing can
        # produce 3D position_ids per sample. Without this, packed Qwen2.5-VL /
        # Qwen3-VL training silently degrades mRoPE to plain 1D positions.
        get_rope_index = getattr(self.model_parts[0], "get_rope_index", None)

        self.dataloader, self.processor = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            _get_model_name(self.cfg.model),
            self.cfg.get("processor", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            cfg_model=self.cfg.model,
            cfg_ps=self.cfg.get("packed_sequence", None),
            get_rope_index=get_rope_index,
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                _get_model_name(self.cfg.model),
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
                get_rope_index=get_rope_index,
            )

        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        # Scheduler
        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self._get_dp_group_size(),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )
        self._setup_garbage_collection(self.step_scheduler)

        # Build learning rate scheduler
        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        # Log model, parameter counts, norms, optimizer and scheduler
        self._log_model_and_optimizer_details(self.model_parts, self.optimizer, self.lr_scheduler)

        restore_from = self.cfg.get("checkpoint.restore_from", None)

        # Initialize JSONL loggers
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )

        # Optionally resume
        self.load_checkpoint(restore_from)

        # Log step scheduler details
        self._log_step_scheduler_details(self.step_scheduler)

        self.engine.lr_scheduler = (
            self.lr_scheduler[0] if isinstance(self.lr_scheduler, list) and self.lr_scheduler else None
        )
        # VLM neat-packing collators are THD-style; engine reads CP knobs from
        # the recipe's own config helpers (same as train_ft.py).
        self.engine.cp_padding_token_id = (
            getattr(self.processor, "tokenizer", None) and getattr(self.processor.tokenizer, "pad_token_id", 0)
        ) or 0
        # FP8 autocast (TE) if configured.
        self.engine.fp8_autocast = getattr(self.model_parts[0], "te_fp8", None)
        if self.engine.fp8_autocast is not None:
            self.engine.fp8_autocast = self.engine.fp8_autocast.maybe_te_autocast

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        """Run the training loop over all epochs and batches.

        For each batch, perform a forward pass, compute loss, backpropagate,
        and update model parameters when necessary. Also prints loss every gradient step.
        """
        for mp in self.model_parts:
            mp.train()
        self.timestamp = time.perf_counter()

        pbar = self._make_progress_bar()
        try:
            for epoch in self.step_scheduler.epochs:
                self.step_scheduler.set_epoch(epoch)
                for batch_idx, batches in enumerate(self.step_scheduler):
                    log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                    # log
                    self.log_train_metrics(log_data)
                    self._update_progress_bar(pbar, log_data.metrics)

                    val_loss = {}
                    if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                        if self.pp_enabled:
                            logger.warning("Validation is not supported for pipeline parallelism")
                        else:
                            val_log_data = self._run_validation_epoch(self.val_dataloader)
                            val_loss["val_loss"] = val_log_data.metrics["val_loss"]
                            self.log_val_metrics(val_log_data)
                        for mp in self.model_parts:
                            mp.train()

                    if self.step_scheduler.is_ckpt_step:
                        self.save_checkpoint(
                            epoch,
                            self.step_scheduler.step,
                            log_data.metrics["loss"],
                            val_loss,
                            best_metric_key=self.best_metric_key,
                        )
                    self._maybe_collect_garbage()
        finally:
            if pbar is not None:
                pbar.close()

        # Close JSONL loggers after training loop completes
        self.metric_logger_train.close()
        self.metric_logger_valid.close()

        self.checkpointer.close()

    # ------------------ helpers ------------------
    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step.

        Args:
            batches: List of batches of training data.
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.
        """
        num_label_tokens = torch.tensor(
            sum((batch["labels"] != -100).sum().item() for batch in batches), dtype=torch.long
        )
        num_label_tokens = self._dp_allreduce(num_label_tokens).item()

        # number of tokens in the batch, excluding any tail padding (for TPS).
        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        if max_grad_norm is not None:
            self.engine.max_grad_norm = max_grad_norm

        # VLMEngine drives: prepare_for_grad_accumulation → MoE aux-loss scale →
        # microbatch loop with VLM CP pre-embed (_pre_cp_hook) and PP media
        # chunking (_pre_pp_schedule_hook) → prepare_for_final_backward.
        self.engine.zero_grad()
        for opt in self.optimizer[1:] if isinstance(self.optimizer, list) and len(self.optimizer) > 1 else []:
            opt.zero_grad(set_to_none=True)

        result = self.engine.forward_backward(
            batches, loss_fn=self.loss_fn, num_label_tokens=num_label_tokens,
        )
        loss_buffer = result["losses"]

        self.checkpointer.maybe_wait_for_staging()

        ok, grad_norm = self.engine.optimizer_step(num_label_tokens=num_label_tokens)
        for opt in self.optimizer[1:] if isinstance(self.optimizer, list) and len(self.optimizer) > 1 else []:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if hasattr(self.model_parts[0], "update_moe_gate_bias"):
            for mp in self.model_parts:
                mp.update_moe_gate_bias()

        self.engine.lr_scheduler_step()
        if isinstance(self.lr_scheduler, list) and len(self.lr_scheduler) > 1:
            for scheduler in self.lr_scheduler[1:]:
                scheduler.step(1)

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.install_optimized_model_weights()
        # self.model.zero_grad_buffer()

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
        if self.pp_enabled:
            # PP uses sum reduction per microbatch (no internal normalization).
            # Divide by num_label_tokens to get the mean loss, same as non-PP.
            reporting_loss = reporting_loss / num_label_tokens if num_label_tokens > 0 else reporting_loss * 0.0
            reporting_loss = reporting_loss.float().to(self.dist_env.device)
            # Send loss to first rank from the last PP stage of rank0's mesh coords.
            # This avoids picking a global-rank sender from a different EP/PP group.
            if self.device_mesh is not None and "pp" in self.device_mesh.mesh_dim_names:
                dim_names = list(self.device_mesh.mesh_dim_names)
                mesh = self.device_mesh.mesh
                idx = []
                for name in dim_names:
                    if name == "pp":
                        idx.append(-1)
                    else:
                        idx.append(0)
                src_rank = mesh[tuple(idx)].item()
            else:
                src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
            if self.dist_env.rank == src_rank:
                torch.distributed.send(reporting_loss, dst=0)
            elif self.dist_env.is_main:
                torch.distributed.recv(reporting_loss, src=src_rank)

        reporting_loss = reporting_loss.item()
        # fix reporting_loss, tps across ranks

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
            },
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one pass over `self.val_dataloader`."""
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_loss = 0.0
            total_tokens = 0
            total_num_label_tokens = 0
            for batch in val_dataloader:
                num_label_tokens = (batch["labels"] != -100).sum().item()
                # VLMEngine's _pre_cp_hook handles multimodal CP pre-embed and
                # make_cp_batch_and_ctx is applied inside engine.forward_backward.
                result = self.engine.forward_backward(
                    [batch],
                    loss_fn=self.loss_fn,
                    forward_only=True,
                    num_label_tokens=num_label_tokens,
                )
                local_loss = result["losses"][0] if result["losses"] else torch.tensor(0.0)
                total_num_label_tokens += num_label_tokens
                total_loss += float(local_loss.item()) * num_label_tokens
                total_tokens += num_label_tokens

        # Aggregate across ranks if distributed is initialized
        total_loss = self._dp_allreduce(torch.FloatTensor([total_loss]), include_cp=True).item()
        total_tokens = self._dp_allreduce(torch.LongTensor([total_tokens]), include_cp=True).item()
        total_num_label_tokens = self._dp_allreduce(torch.LongTensor([total_num_label_tokens])).item()

        val_loss = total_loss / max(total_tokens, 1e-8)

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_num_label_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_val_metrics(self, log_data):
        log_validation_metrics(
            log_data,
            is_main=self.dist_env.is_main,
            metric_logger=self.metric_logger_valid,
            wandb_run=wandb.run,
        )

    def log_train_metrics(self, log_data) -> None:
        log_training_metrics(
            log_data,
            is_main=self.dist_env.is_main,
            is_remote_logging_step=self.step_scheduler.is_remote_logging_step,
            step=self.step_scheduler.step,
            metric_logger=self.metric_logger_train,
            wandb_run=wandb.run,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path=None):
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    if config_path is None:
        config_path = pathlib.Path(__file__).parent.resolve() / "gemma3" / "gemma3_vl_4b_cord_v2.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
