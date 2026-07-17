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
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import mlflow
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers.processing_utils import ProcessorMixin

from nemo_automodel._transformers import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    NeMoAutoModelForMultimodalLM,
)
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.vlm.pp_media import stage_vlm_media_for_pp
from nemo_automodel.components.distributed.config import DistributedSetup, MegatronFSDPConfig
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.magi_attn_utils import MagiState, setup_magi
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import FirstRankPerNode, get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.mlflow_utils import (
    end_mlflow_active_run_as_killed,
    to_float_metrics,
)
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mtp import calculate_mtp_loss
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.quantization.fp8 import build_fp8_config
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.utils import (
    count_tail_padding,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.compile_utils import build_compile_config
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS, _supports_logits_to_keep, filter_forward_kwargs
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config, shard_optimizers_for_megatron_fsdp
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.shared.te_patches import apply_te_patches

if TYPE_CHECKING:
    from torch.optim import Optimizer


logger = logging.getLogger(__name__)

try:
    from megatron_fsdp import MegatronFSDP
    from megatron_fsdp.fully_shard import fully_shard_optimizer
except (ImportError, FileNotFoundError, OSError):
    MegatronFSDP = None
    fully_shard_optimizer = None

# ---------------------------
#  Stateless helper functions
# ---------------------------


def _get_model_name(cfg_model):
    if cfg_model.get("pretrained_model_name_or_path", None) is not None:
        return cfg_model.pretrained_model_name_or_path
    elif cfg_model.get("config", None) is not None:
        if isinstance(cfg_model.config, str):
            return cfg_model.config
        return cfg_model.config.get("pretrained_model_name_or_path", None)
    else:
        return None


def build_model(
    cfg_model,
    cfg_freeze,
    cfg_peft,
    seed,
    cfg_fp8=None,
    cfg_compile=None,
    distributed_setup: DistributedSetup | None = None,
    cfg_quantization=None,
) -> tuple[nn.Module | AutoPipeline, list["Optimizer"]]:  # noqa: F821
    """Build and initialize a model for VLM.

    Returns:
        The instantiated model and optimizer.
    """
    with ScopedRNG(seed=seed, ranked=True):
        # Build infrastructure kwargs
        kwargs = {
            "peft_config": cfg_peft,
            "freeze_config": cfg_freeze.to_dict() if cfg_freeze is not None else None,
        }
        if distributed_setup is not None:
            kwargs["distributed_setup"] = distributed_setup

        if cfg_fp8 is not None:
            fp8_config = build_fp8_config(cfg_fp8)
            kwargs["fp8_config"] = fp8_config
        if cfg_compile is not None:
            kwargs["compile_config"] = build_compile_config(cfg_compile)
        if cfg_quantization is not None:
            logger.info("Model weight quantization enabled with BitsAndBytes")
            from nemo_automodel.components.quantization.qlora import create_bnb_config

            kwargs["quantization_config"] = create_bnb_config(cfg_quantization)

        if _is_recipe_target(cfg_model.get("_target_", None)):
            model = cfg_model.instantiate(**kwargs)
        else:
            raise ValueError(
                "VLM finetuning requires a recipe-compatible model target. "
                "Add the entrypoint to `_accepted_targets()` in this module "
                "if you're onboarding a new wrapper that absorbs the recipe's "
                "infrastructure kwargs. "
                f"Got model target: {cfg_model.get('_target_', None)}"
            )
    return model


def _accepted_targets() -> set:
    """Return the set of model ``_target_`` callables this recipe accepts.

    These are the wrapper-layer entrypoints that know how to absorb the
    recipe's infrastructure kwargs (``device_mesh``, ``distributed_config``,
    ``peft_config``, ``freeze_config``, ``pipeline_config``, plus the
    optional ``moe_config`` / ``fp8_config`` / ``compile_config``). Anything
    not on this list is rejected with a clear error -- vanilla
    ``transformers.AutoModelFor*`` does not handle these kwargs and would
    otherwise fail deep inside HF code.

    New infra-aware composites (e.g. Gemma4WithDrafter) opt in by adding their ``.from_pretrained``
    (and ``.from_config`` if applicable) here.

    The Gemma4 joint composite is added behind a try/except because it
    requires the optional ``transformers.models.gemma4_assistant`` module
    that ships with ``transformers>=5.8.0.dev``.
    """
    accepted = {
        NeMoAutoModelForCausalLM.from_pretrained,
        NeMoAutoModelForCausalLM.from_config,
        NeMoAutoModelForImageTextToText.from_pretrained,
        NeMoAutoModelForImageTextToText.from_config,
        NeMoAutoModelForMultimodalLM.from_pretrained,
        NeMoAutoModelForMultimodalLM.from_config,
    }
    try:
        from nemo_automodel.components.models.gemma4_drafter.composite import (
            Gemma4WithDrafter,
        )

        accepted.add(Gemma4WithDrafter.from_pretrained)
    except ImportError:
        pass
    return accepted


def _is_recipe_target(target) -> bool:
    """True if ``target`` is on this recipe's allowlist of model entrypoints."""
    if target is None:
        return False
    return target in _accepted_targets()


def _shift_labels_left(labels: torch.Tensor, k: int) -> torch.Tensor:
    """Shift ``labels`` left by ``k`` positions, padding the tail with ``-100``.

    Used to build drafter-step targets in joint base + drafter training.

    The VLM collate pipeline already pre-shifts labels by 1 so that
    ``labels[t] == input_ids[t + 1]`` (the next-token target). Drafter step ``k``
    predicts position ``t + 1 + k`` of the original sequence, which corresponds
    to ``labels[t + k]`` in the pre-shifted convention. So for step ``k``:

    * ``k = 0`` (one-step drafter) -> no shift; reuse ``labels`` as-is.
    * ``k = 1`` -> shift labels left by 1 (drafter predicts two tokens ahead).
    * ``k = n`` -> shift labels left by ``n``.

    Args:
        labels: ``[B, S]`` LongTensor of label ids (``-100`` marks ignored
            positions).
        k: Number of positions to shift to the left. ``k <= 0`` is a no-op.

    Returns:
        A new ``[B, S]`` LongTensor with ``labels[:, k:]`` in the leading slice
        and ``-100`` in the trailing ``k`` columns. When ``k <= 0``, the input
        is returned unchanged.
    """
    if k <= 0:
        return labels
    shifted = torch.full_like(labels, fill_value=-100)
    if k < labels.size(-1):
        shifted[..., : labels.size(-1) - k] = labels[..., k:]
    return shifted


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) if v is not None else None for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def build_dataloader(
    cfg_ds,
    cfg_dl,
    pretrained_model_name_or_path,
    cfg_processor,
    device_mesh,
    seed,
    local_batch_size,
    cfg_model=None,
    cfg_ps=None,
    get_rope_index=None,
    pp_n_microbatches=None,
) -> tuple[DataLoader, ProcessorMixin]:
    """Build a DataLoader for the VLM dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        pretrained_model_name_or_path: Pretrained model name or path for processor loading.
        cfg_processor: Processor configuration or None.
        device_mesh: Device mesh for distributed training.
        seed: Random seed.
        local_batch_size: Local batch size.
        cfg_model: Model configuration (used to detect attention backend).
        cfg_ps: Packed sequence configuration (top-level ``packed_sequence:`` section).
            When provided, takes precedence over ``dataset.packing``.
        get_rope_index: Optional ``model.get_rope_index`` callable. When provided,
            VLM neat packing computes mRoPE 3D position IDs per sample so packed
            mRoPE-aware models (Qwen2.5-VL, Qwen3-VL, ...) preserve multimodal
            position semantics across pack boundaries instead of falling back to
            plain 1D positions.
        pp_n_microbatches: When set, wrap collate so VLM media tensors are
            pre-chunked for this many PP microbatches before entering the train loop.

    Returns:
        The instantiated DataLoader and processor.
    """
    warnings.warn(
        "build_dataloader is deprecated; resolve RecipeConfig.vlm_dataloader and call its build() method",
        DeprecationWarning,
        stacklevel=2,
    )
    config = RecipeConfig.resolve_vlm_dataloader(
        cfg_ds,
        cfg_dl,
        processor_node=cfg_processor,
        packed_sequence_node=cfg_ps,
    )
    dp_rank = 0
    dp_world_size = 1
    cp_size = 1
    if device_mesh is not None:
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        dp_mesh = get_flat_mesh(device_mesh, "dp")
        dp_rank = dp_mesh.get_local_rank()
        dp_world_size = dp_mesh.size()
        if "cp" in getattr(device_mesh, "mesh_dim_names", ()):
            cp_size = device_mesh["cp"].size()

    from nemo_automodel.components.models.common.packing import configure_packing, get_attn_implementation

    packing_attn_implementation = config.resolve_packing_attn_implementation(
        model_attn_implementation=get_attn_implementation(cfg_model),
        cp_size=cp_size,
    )
    if config.packing is not None and config.packing.packing_format != "thd":
        configure_packing(attn_implementation=packing_attn_implementation)

    with ScopedRNG(seed=seed, ranked=True):
        result = config.build(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            dataset_build_context=FirstRankPerNode(),
            get_rope_index=get_rope_index,
            packing_attn_implementation=packing_attn_implementation,
            pp_n_microbatches=pp_n_microbatches,
        )
    return result.dataloader, result.processor


def calculate_loss(loss_fn, **kwargs) -> torch.Tensor:
    """Calculate the loss.

    Args:
        loss_fn: Loss function.
        **kwargs: Keyword arguments for the loss function.

    Returns:
        The loss.
    """
    loss_fn_kwargs = {"num_label_tokens": kwargs.pop("num_label_tokens", None)}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")

        # find the lm_head in the model
        lm_head = None
        if hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings().weight
        else:
            for n, p in model.named_parameters(remove_duplicate=False):
                if "lm_head" in n and n.endswith(".weight"):
                    lm_head = p
                    break
        if lm_head is None:
            raise ValueError("lm_head.weight not found in model")

        # unshard the possibly sharded lm_head
        lm_head = lm_head.full_tensor() if hasattr(lm_head, "full_tensor") else lm_head
        loss_fn_kwargs.update(
            {
                "hidden_states": kwargs.pop("hidden_states"),
                "labels": labels,
                "lm_weight": lm_head,
            }
        )
    else:
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
            }
        )

    return loss_fn(**loss_fn_kwargs)


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------


class FinetuneRecipeForVLM(BaseRecipe):
    """Recipe for fine-tuning a VLM model."""

    # MagiAttention is disabled until setup() resolves it from config; this
    # disabled default keeps the train step working if setup() is skipped (e.g.
    # unit tests that exercise the step directly). It is read-only.
    magi = MagiState()

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg if isinstance(cfg, RecipeConfig) else RecipeConfig(cfg)

    # ------------------ build phase ------------------
    def setup(self):
        """Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env", {}).get("backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env", {}).get("timeout_minutes", 1),
        )
        setup_logging()

        apply_cache_compatibility_patches()

        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

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

        # MagiAttention (FFA) backend for the language backbone; the vision tower
        # stays on SDPA. Enabled via model.attn_implementation="magi" (HF VLMs) or
        # model.backend.attn="magi" (custom VLMs, e.g. qwen3_vl_moe).
        self.magi = setup_magi(self.cfg, self.device_mesh, label="VLM language backbone")

        if self.dist_env.is_main and self.cfg.wandb is not None:
            suppress_wandb_log_messages()
            run = self.cfg.wandb.build(run_config=self.cfg.to_dict(), model_name=_get_model_name(self.cfg.model))
            logging.info("🚀 View run at {}".format(run.url))

        if self.dist_env.is_main and self.cfg.mlflow is not None:
            run_config = self.cfg.to_yaml_dict(use_orig_values=True)
            checkpoint_dir = self.cfg.get("checkpoint.checkpoint_dir", None)
            if self.cfg.mlflow.build(checkpoint_dir=checkpoint_dir, run_config=run_config) is not None:
                logging.info("MLflow experiment tracking enabled")

        # Log experiment details on main rank
        self._log_experiment_details()
        self._log_library_versions()

        # Build loss_fn (will be set on pipeline_config if PP enabled)
        self.loss_fn = self.cfg.loss_fn.build()

        # Pipeline runtime fields: override pp_batch_size and pp_microbatch_size
        if self.pp_enabled:
            pp_batch_size = self.cfg.get("step_scheduler.local_batch_size", 1)
            pp_microbatch_size = self.cfg.get("distributed.pipeline.pp_microbatch_size", 1)

            assert pp_batch_size // pp_microbatch_size >= self.mesh_context.pp_size, (
                f"pp_batch_size {pp_batch_size} // pp_microbatch_size {pp_microbatch_size} must be >= pp_size {self.mesh_context.pp_size}"
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

        # Checkpoint config (model-derived fields are filled in by RecipeConfig)
        checkpoint_config = self.cfg.checkpoint

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified in config, using default value of 1.0")
            self.max_grad_norm = 1.0

        # Build the checkpointer from its config
        self.checkpointer = checkpoint_config.build(
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        # Disable fused RoPE when context parallelism is enabled (cp > 1)
        if self.mesh_context.cp_size > 1 and self.cfg.get("model.backend.rope_fusion", False):
            logging.info("Disabling rope_fusion because cp_size=%d > 1", self.mesh_context.cp_size)
            self.cfg.model.backend.rope_fusion = False

        # fp32 master-weight default planned to be enabled in follow-up PR (resolve_storage_dtype).

        model = build_model(
            self.cfg.model,
            self.cfg.get("freeze_config", None),
            self.peft_config,
            seed=self.cfg.get("seed", 42),
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
            distributed_setup=self.distributed_setup,
            cfg_quantization=self.cfg.get("quantization", None),
        )
        apply_te_patches()
        optimizer = self.cfg.optimizer.build(model, device_mesh=self.device_mesh, is_peft=self.peft_config is not None)
        allow_megatron_fsdp_sharding = getattr(self.cfg.optimizer, "supports_megatron_fsdp_sharding", True)
        self.optimizer = shard_optimizers_for_megatron_fsdp(
            model, optimizer, self.distributed_config, allow=allow_megatron_fsdp_sharding
        )

        if not _supports_logits_to_keep(model) and not isinstance(self.loss_fn, MaskedCrossEntropy):
            logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
            self.loss_fn = MaskedCrossEntropy()

        if isinstance(model, AutoPipeline):
            self.model_parts = model.parts
            self.pp = model
        else:
            self.model_parts = [model]
            self.pp = None
        if self.pp_enabled:
            self._configure_pipeline_loss_fn()

        # Extract mRoPE position-id builder from the model so VLM neat packing can
        # produce 3D position_ids per sample. Without this, packed multimodal
        # training silently degrades mRoPE to plain 1D positions.
        get_rope_index = getattr(self.model_parts[0], "get_rope_index", None)
        pp_n_microbatches = None
        pp_cp_preembed = (
            self.pp_enabled
            and self.mesh_context.cp_size > 1
            and hasattr(self.model_parts[0], "prepare_model_inputs_for_cp")
        )
        if self.pp_enabled and not pp_cp_preembed:
            pp_n_microbatches = self.pp.pp_batch_size // self.pp.pp_microbatch_size

        dataloader_config = self.cfg.vlm_dataloader
        if dataloader_config is None:
            raise ValueError("VLM training requires a dataset config")
        from nemo_automodel.components.models.common.packing import configure_packing, get_attn_implementation

        packing_attn_implementation = dataloader_config.resolve_packing_attn_implementation(
            model_attn_implementation=get_attn_implementation(self.cfg.model),
            cp_size=self.mesh_context.cp_size,
        )
        if dataloader_config.packing is not None and dataloader_config.packing.packing_format != "thd":
            configure_packing(attn_implementation=packing_attn_implementation)
        with ScopedRNG(seed=self.cfg.get("seed", 42), ranked=True):
            dataloader_build = dataloader_config.build(
                pretrained_model_name_or_path=_get_model_name(self.cfg.model),
                dp_rank=self._get_dp_rank(),
                dp_world_size=self._get_dp_group_size(),
                batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
                dataset_build_context=FirstRankPerNode(),
                get_rope_index=get_rope_index,
                packing_attn_implementation=packing_attn_implementation,
                pp_n_microbatches=pp_n_microbatches,
            )
        self.dataloader = dataloader_build.dataloader
        self.processor = dataloader_build.processor

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        validation_config = self.cfg.vlm_validation_dataloader
        if validation_config is not None:
            with ScopedRNG(seed=self.cfg.get("seed", 42), ranked=True):
                validation_build = validation_config.build(
                    pretrained_model_name_or_path=_get_model_name(self.cfg.model),
                    dp_rank=self._get_dp_rank(),
                    dp_world_size=self._get_dp_group_size(),
                    batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
                    dataset_build_context=FirstRankPerNode(),
                    get_rope_index=get_rope_index,
                )
            self.val_dataloader = validation_build.dataloader

        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        # Scheduler
        self.step_scheduler = self.cfg.step_scheduler.build(
            self.dataloader,
            self._get_dp_group_size(),
            self.cfg.get("step_scheduler.local_batch_size", 1),
        )
        self._setup_garbage_collection(self.step_scheduler)

        # Build learning rate scheduler
        self.lr_scheduler = (
            self.cfg.lr_scheduler.build(self.optimizer, self.step_scheduler)
            if self.cfg.lr_scheduler is not None
            else None
        )

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

        # Mark the MLflow run KILLED if training exited via SIGTERM.
        if self.step_scheduler.sigterm_flag:
            end_mlflow_active_run_as_killed()

    # ------------------ helpers ------------------
    def _maybe_add_drafter_loss(
        self,
        *,
        out: Any,
        base_loss: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
        num_label_tokens: int,
        log: bool = False,
    ) -> torch.Tensor:
        """Return ``base_loss + lambda * sum_k CE(drafter_logits[k], shifted_labels_k)``.

        If ``out`` does not carry a non-empty ``drafter_logits`` attribute (i.e. the
        model isn't a joint composite), returns ``base_loss`` unchanged.

        For drafter step ``k``, labels are shifted left by ``k`` positions to match
        the VLM collate's pre-shifted convention (``labels[t] == input_ids[t+1]``).
        ``log=True`` emits a one-line breakdown on rank 0; callers should gate this
        on the appropriate step / microbatch index to avoid log spam.
        """
        drafter_logits = getattr(out, "drafter_logits", None)
        if drafter_logits is None or len(drafter_logits) == 0:
            return base_loss

        drafter_loss_weight = getattr(out, "drafter_loss_weight", 1.0)
        drafter_loss_total = None
        for k, dl in enumerate(drafter_logits):
            shifted_labels = _shift_labels_left(labels, k)
            l_k = calculate_loss(
                self.loss_fn,
                logits=dl,
                labels=shifted_labels,
                model=model,
                hidden_states=None,
                num_label_tokens=num_label_tokens,
            )
            drafter_loss_total = l_k if drafter_loss_total is None else drafter_loss_total + l_k

        total_loss = base_loss + drafter_loss_weight * drafter_loss_total
        if log and self.dist_env.is_main:
            logger.info(
                "[joint-drafter] L_base=%.4f L_drafter=%.4f L_total=%.4f (lambda=%.3f)",
                base_loss.detach().item(),
                drafter_loss_total.detach().item(),
                total_loss.detach().item(),
                drafter_loss_weight,
            )
        return total_loss

    def _maybe_set_pp_first_stage_embed_input_meta(self, model_input: torch.Tensor) -> None:
        if (
            not self.pp_enabled
            or not getattr(self.pp.info, "has_first_stage", False)
            or not model_input.dtype.is_floating_point
            or model_input.ndim != 3
        ):
            return

        for stage in self.pp.info.stages:
            if stage.is_first:
                stage.inputs_meta = (
                    torch.empty(
                        self.pp.pp_microbatch_size,
                        model_input.shape[1],
                        model_input.shape[2],
                        device="meta",
                        dtype=model_input.dtype,
                    ),
                )

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_label_tokens,
        num_batches,
        is_train: bool = True,
    ):
        batch = {k: _move_to_device(v, self.dist_env.device) for k, v in batch.items()}

        # Routed through __call__ so FSDP2 forward pre-hook fires and
        # unshards the vision tower's weights before the embed/scatter.
        _model = self.model_parts[0]
        _cp_active = (
            self.device_mesh is not None
            and "cp" in getattr(self.device_mesh, "mesh_dim_names", ())
            and self.device_mesh["cp"].size() > 1
        )
        if _cp_active and hasattr(_model, "prepare_model_inputs_for_cp"):
            if not self.pp_enabled or getattr(self.pp.info, "has_first_stage", False):
                mm_kwargs = {k: batch[k] for k in VLM_INPUT_KEYS if batch.get(k) is not None}
                prepared = _model(_pre_embed_only=True, **mm_kwargs)
                for k in VLM_INPUT_KEYS:
                    batch.pop(k, None)
                batch.update(prepared)
            else:
                for k in VLM_INPUT_KEYS:
                    if k != "input_ids":
                        batch.pop(k, None)

        if self.magi.enabled:
            # magi manages the language-backbone attention itself (vision stays on
            # SDPA); skip the torch-native DTensor CP context.
            train_ctx, batch = self.magi.prepare_vlm_batch(
                self.model_parts[0], batch
            )  # pragma: no cover - requires GPU + magi_attention
        elif batch.get("qkv_format", None) == "thd":
            # THD packed VLM inputs use TE sequence metadata even without context parallelism.
            if self.mesh_context.cp_size > 1:
                raise NotImplementedError(
                    "THD packing (packing_format='thd') for VLM currently supports cp_size=1 only; "
                    "context-parallel THD for mRoPE VLMs is not yet implemented."
                )
            padding_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", 0) or 0
            train_ctx, batch = make_cp_batch_and_ctx(
                self.device_mesh,
                batch,
                use_te=True,
                padding_token_id=padding_id,
            )
        else:
            train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
        labels = batch.pop("labels")

        if self.pp_enabled:
            if not is_train:
                logging.info("Skipping forward pass for validation because pipeline parallelism is enabled")
                return

            with train_ctx():
                losses = [] if self.pp.info.has_last_stage else None
                if self.pp.info.has_last_stage:
                    masked_labels = labels.clone()
                    targets = masked_labels
                else:
                    targets = None

                model_input_key = "inputs_embeds" if "inputs_embeds" in batch else "input_ids"
                model_input = batch.pop(model_input_key)
                self.pp.update_seq_len(model_input.shape[1])
                self._maybe_set_pp_first_stage_embed_input_meta(model_input)

                with stage_vlm_media_for_pp(self.pp, self.model_parts, batch):
                    if self.pp.info.has_first_stage:
                        self.pp.info.schedule.step(model_input, target=targets, losses=losses, **batch)
                    else:
                        self.pp.info.schedule.step(target=targets, losses=losses, **batch)

            if self.pp.info.has_last_stage:
                local_loss = torch.sum(torch.stack(losses))
            else:
                local_loss = torch.tensor(0.0, device=self.dist_env.device)

            loss_buffer.append(local_loss.clone().detach())
        else:
            model = self.model_parts[0]
            sync_ctx = (
                get_sync_ctx(
                    model,
                    idx == num_batches - 1,
                    defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
                )
                if is_train
                else nullcontext()
            )
            with sync_ctx, train_ctx():
                batch = filter_forward_kwargs(model, batch)
                if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                    # use num_logits_to_keep to avoid full logits matrix in memory
                    out = model(logits_to_keep=1, **batch)
                    if "hidden_states" not in out:
                        raise ValueError(
                            "FusedLinearCrossEntropy requires the model to output hidden states. "
                            "Set `model.text_config.output_hidden_states=True` in the config."
                        )
                else:
                    out = model(**batch)

                local_loss = calculate_loss(
                    self.loss_fn,
                    logits=getattr(out, "logits", out),
                    labels=labels,
                    model=model,
                    hidden_states=get_final_hidden_states(out),
                    num_label_tokens=num_label_tokens,
                )
                # DSV4-style MTP loss (from main): triggers when the model emits
                # ``mtp_per_depth_h`` / ``mtp_per_depth_logits``.
                mtp_per_depth_h = getattr(out, "mtp_per_depth_h", None)
                mtp_per_depth_logits = getattr(out, "mtp_per_depth_logits", None)
                if mtp_per_depth_h is not None or mtp_per_depth_logits is not None:
                    mtp_cfg = self.cfg.mtp
                    scaling_factor = (
                        mtp_cfg.scaling_factor if mtp_cfg.scaling_factor is not None else out.mtp_loss_scaling_factor
                    )
                    local_loss = local_loss + calculate_mtp_loss(
                        self.loss_fn,
                        mtp_per_depth_h=mtp_per_depth_h,
                        mtp_per_depth_logits=mtp_per_depth_logits,
                        labels=labels,
                        model=model,
                        scaling_factor=scaling_factor,
                        num_label_tokens=num_label_tokens,
                        ignore_index=mtp_cfg.ignore_index,
                    )

                # Joint base + drafter co-training (Gemma4WithDrafter and
                # similar): detect by presence of ``drafter_logits`` on the
                # model output and add
                # ``drafter_loss_weight * sum_k CE(drafter_logits[k], shifted_labels_k)``
                # to the base loss. See ``_shift_labels_left`` for the shift
                # convention. Mutually exclusive with the DSV4-style MTP path
                # above -- only one of ``drafter_logits`` /
                # ``mtp_per_depth_*`` is set per model.
                local_loss = self._maybe_add_drafter_loss(
                    out=out,
                    base_loss=local_loss,
                    labels=labels,
                    model=model,
                    num_label_tokens=num_label_tokens,
                    # Log once per remote-logging step on the first microbatch.
                    log=(idx == 0 and self.step_scheduler.is_remote_logging_step),
                )

                loss_buffer.append(local_loss.clone().detach())
                if is_train:
                    (local_loss * self._get_dp_group_size(include_cp=True)).backward()

    def _configure_pipeline_loss_fn(self):
        if self.pp is None or not self.pp.info.has_last_stage:
            return

        last_stage_model = None
        for model_part, stage in zip(self.model_parts, self.pp.info.stages):
            if stage.is_last:
                last_stage_model = model_part
                break
        if last_stage_model is None:
            raise RuntimeError("Pipeline reports a last stage, but no last-stage model part was found")

        self.pp.info.schedule._loss_fn = self.cfg.mtp.build(self.loss_fn, last_stage_model)

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

        # MoE aux loss gradients are injected via MoEAuxLossAutoScaler, which
        # multiplies them by main_loss_backward_scale during backward.  This
        # counteracts the unwanted scaling that FSDP and PP post-hoc rescaling
        # apply to *all* gradients (including aux loss):
        #
        #   Non-PP: FSDP allreduce divides grads by dp_group_size.
        #           Scale = dp_group_size  →  net = 1.
        #
        #   PP:     FSDP divides by dp_group_size, then
        #           scale_grads_and_clip_grad_norm divides by
        #           (num_label_tokens / dp_group_size).  The dp_group_size
        #           factors cancel, leaving net 1/num_label_tokens.
        #           Scale = num_label_tokens  →  net = 1.
        if self.pp_enabled:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(float(num_label_tokens))
        else:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                float(self._get_dp_group_size(include_cp=True))
            )

        loss_buffer = []

        # number of tokens in the batch, excluding any tail padding.
        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        num_batches = len(batches)
        prepare_for_grad_accumulation(self.model_parts, pp_enabled=self.pp_enabled)

        for i, batch in enumerate(batches):
            if i == num_batches - 1:
                prepare_for_final_backward(self.model_parts, pp_enabled=self.pp_enabled)

            self._forward_backward_step(
                i, batch, loss_buffer=loss_buffer, num_label_tokens=num_label_tokens, num_batches=num_batches
            )

            if i == 0:
                prepare_after_first_microbatch()

        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm=max_grad_norm,
            model_parts=self.model_parts,
            norm_type=2.0,
            pp_enabled=self.pp_enabled,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp" if self.pp_enabled else None,
            foreach=True,
            num_label_tokens=num_label_tokens,
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.finish_grad_sync()

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if hasattr(self.model_parts[0], "update_moe_gate_bias"):
            for mp in self.model_parts:
                mp.update_moe_gate_bias()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
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
                batch = {
                    k: (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                num_label_tokens = (batch["labels"] != -100).sum().item()

                _model = self.model_parts[0]
                _cp_active = (
                    self.device_mesh is not None
                    and "cp" in getattr(self.device_mesh, "mesh_dim_names", ())
                    and self.device_mesh["cp"].size() > 1
                    and not self.pp_enabled
                )
                if _cp_active and hasattr(_model, "prepare_model_inputs_for_cp"):
                    mm_kwargs = {k: batch[k] for k in VLM_INPUT_KEYS if batch.get(k) is not None}
                    with torch.no_grad():
                        prepared = _model(_pre_embed_only=True, **mm_kwargs)
                    for k in VLM_INPUT_KEYS:
                        batch.pop(k, None)
                    batch.update(prepared)

                train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
                labels = batch.pop("labels")
                with train_ctx():
                    batch = filter_forward_kwargs(self.model_parts[0], batch)
                    if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                        out = self.model_parts[0](logits_to_keep=1, **batch)
                    else:
                        out = self.model_parts[0](**batch)
                    local_loss = calculate_loss(
                        self.loss_fn,
                        logits=getattr(out, "logits", out),
                        labels=labels,
                        model=self.model_parts[0],
                        hidden_states=out.hidden_states[-1]
                        if getattr(out, "hidden_states", None) is not None
                        else None,
                        num_label_tokens=num_label_tokens,
                    )
                    # Mirror training: include the drafter term so validation
                    # reflects drafter drift, not just the base.
                    local_loss = self._maybe_add_drafter_loss(
                        out=out,
                        base_loss=local_loss,
                        labels=labels,
                        model=self.model_parts[0],
                        num_label_tokens=num_label_tokens,
                    )
                    total_num_label_tokens += num_label_tokens

                total_loss += local_loss.item() * num_label_tokens
                total_tokens += num_label_tokens

        # Aggregate across ranks if distributed is initialized
        total_loss = self._dp_allreduce(torch.FloatTensor([total_loss]), include_cp=True).item()
        # `num_label_tokens` is measured before CP sharding, so each CP rank
        # contributes the full sequence token count while `total_loss` is
        # reconstructed from CP-sharded loss sums. Do not sum tokens over CP.
        total_tokens = self._dp_allreduce(torch.LongTensor([total_tokens])).item()
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
        """Log metrics to wandb and other loggers
        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "val_loss": Validation loss.
                    "lr": Learning rate.
                    "num_label_tokens": Number of label tokens.
                    "mem": Memory allocated.
        """

        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)

        if mlflow.active_run() is not None:
            mlflow.log_metrics(to_float_metrics(log_data.to_dict()), step=log_data.step)

        # JSONL validation log
        self.metric_logger_valid.log(log_data)

        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
            )
        )

    def log_train_metrics(self, log_data) -> float:
        """Log metrics to wandb.

        Args:
            train_loss: Training loss.
            grad_norm: Grad norm from the training step.
            num_tokens_in_batch: Total number of loss tokens.
            tps: Tokens per second.
        """
        if not self.dist_env.is_main:
            return

        # Log to remote services (WandB, MLflow) according to step_scheduler frequency
        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=self.step_scheduler.step)
            if mlflow.active_run() is not None:
                mlflow.log_metrics(to_float_metrics(log_data.to_dict()), step=self.step_scheduler.step)

        # JSONL training log (always log for detailed local records)
        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["loss"],
                log_data.metrics["grad_norm"],
                log_data.metrics["lr"],
                log_data.metrics["mem"],
                log_data.metrics["tps"],
                log_data.metrics["tps_per_gpu"],
                log_data.metrics["num_label_tokens"],
            )
        )
        torch.cuda.reset_peak_memory_stats()


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
