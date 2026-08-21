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
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, cast

import mlflow
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from transformers.processing_utils import ProcessorMixin

from nemo_automodel._transformers import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    NeMoAutoModelForMultimodalLM,
)
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches, resolve_get_rope_index
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.datum import Datum, LossInputLayout
from nemo_automodel.components.datasets.vlm.pp_media import VLM_PP_MEDIA_KEY, stage_vlm_media_for_pp
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config, MegatronFSDPConfig
from nemo_automodel.components.distributed.context_parallel.magi import MagiState, setup_magi
from nemo_automodel.components.distributed.cp_vision_frame_shard import (
    CpVisionFrameShardingConfig,
    reset_cp_vision_group,
    set_cp_vision_group,
)
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import FirstRankPerNode
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
from nemo_automodel.components.loss.utils import _get_lm_head_weight, calculate_loss
from nemo_automodel.components.quantization.fp8 import build_fp8_config
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.components.utils.compile_utils import build_compile_config
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS, _supports_logits_to_keep
from nemo_automodel.engine import Engine, collate_prebatched
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


class _CpVisionFrameShardingCapability(Protocol):
    """Model capability required by the VLM vision frame-sharding recipe policy."""

    @property
    def supports_cp_vision_frame_sharding(self) -> bool:
        """Whether the model owns a verified CP vision frame-sharding integration."""
        ...


class _CpPackingCapability(Protocol):
    """Model capability required by packed VLM context parallelism."""

    @property
    def supports_cp_with_sequence_packing(self) -> bool:
        """Whether the model's active backend owns packed CP routing."""
        ...


def _validate_cp_vision_frame_sharding_support(
    model: _CpVisionFrameShardingCapability,
    config: CpVisionFrameShardingConfig,
) -> None:
    """Reject enabled vision frame sharding when the model has no production integration."""
    if not config.enabled or model.supports_cp_vision_frame_sharding:
        return

    model_name = type(model).__name__
    raise ValueError(
        "distributed.multimodal.vision.frame_sharding.enabled=true requires a model-owned integration "
        f"for sharding vision frames over CP ranks, but {model_name} declares "
        "supports_cp_vision_frame_sharding=False. "
        "Disable the policy with distributed.multimodal.vision.frame_sharding.enabled=false "
        "or use a supported model."
    )


def _validate_cp_packing_support(
    model: _CpPackingCapability,
    *,
    packing_enabled: bool,
    cp_size: int,
) -> None:
    """Reject packed CP before dataloader construction when routing is unsupported."""
    if cp_size <= 1 or not packing_enabled or model.supports_cp_with_sequence_packing:
        return

    raise ValueError(
        f"Context parallelism (cp_size={cp_size}) with VLM sequence packing is not supported "
        f"for {type(model).__name__} with its active attention backend. Disable sequence "
        "packing, use cp_size=1, or select a model-supported packed-CP backend."
    )


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
            cp_size=cp_size,
        )
    return result.dataloader, result.processor


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------


class FinetuneRecipeForVLM(BaseRecipe):
    """Recipe for fine-tuning a VLM model."""

    # MagiAttention is disabled until setup() resolves it from config. It is read-only.
    magi = MagiState()

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg if isinstance(cfg, RecipeConfig) else RecipeConfig(cfg)

    # ------------------ build phase ------------------
    def _create_distributed_setup(self) -> DistributedSetup:
        """Create the distributed setup used by this recipe rank."""
        return create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)

    def _should_setup_training_components(self) -> bool:
        """Whether this rank owns the trainable model and its components."""
        return True

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
        ) = self._distributed_setup_attributes(self._create_distributed_setup())
        self.cp_vision_frame_sharding = (
            self.distributed_config.multimodal.vision.frame_sharding
            if isinstance(self.distributed_config, FSDP2Config)
            else CpVisionFrameShardingConfig()
        )

        if not self._should_setup_training_components():
            return

        if self.pp_enabled and getattr(self.pipeline_config, "scale_grads_in_schedule", False):
            raise ValueError("Engine-backed VLM finetuning requires distributed.pipeline.scale_grads_in_schedule=False")

        # MagiAttention (FFA) backend for the language backbone; the vision tower
        # stays on SDPA. Enabled via model.attn_implementation="magi" (HF VLMs) or
        # model.backend.attn="magi" (custom VLMs, e.g. qwen3_vl_moe).
        self.magi = setup_magi(self.cfg, self.device_mesh, domain="vlm", label="VLM language backbone")

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

            if self.magi.enabled:
                if pp_batch_size != 1 or pp_microbatch_size != 1:
                    raise ValueError(
                        "Magi pipeline training requires local_batch_size=1 and pp_microbatch_size=1; "
                        "use outer gradient accumulation for larger optimizer windows"
                    )
            else:
                if pp_batch_size // pp_microbatch_size < self.mesh_context.pp_size:
                    raise ValueError(
                        f"pp_batch_size {pp_batch_size} // pp_microbatch_size {pp_microbatch_size} "
                        f"must be >= pp_size {self.mesh_context.pp_size}"
                    )

            if isinstance(self.distributed_config, MegatronFSDPConfig):
                raise ValueError("MegatronFSDPConfig is not supported when pipeline parallelism is enabled")

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
            process_group=getattr(self.mesh_context, "process_group", None),
            pp_group=self._get_pp_group(),
        )

        # Disable fused RoPE when context parallelism is enabled (cp > 1)
        if self.mesh_context.cp_size > 1 and self.cfg.get("model.backend.rope_fusion", False):
            logging.info("Disabling rope_fusion because cp_size=%d > 1", self.mesh_context.cp_size)
            self.cfg.model.backend.rope_fusion = False

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
        capability_model = model.parts[0] if isinstance(model, AutoPipeline) else model
        self._has_joint_drafter = getattr(capability_model, "drafter", None) is not None
        _validate_cp_vision_frame_sharding_support(capability_model, self.cp_vision_frame_sharding)
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
        self.pipeline_loss_fn = None
        if self.pp_enabled:
            self._configure_pipeline_loss_fn()

        # Optional setup-time prewarms (cuBLAS workspaces, Triton autotune
        # caches, NCCL communicators) while the allocator pool is still small,
        # instead of lazily at step-1 peak memory.
        if self.cfg.prewarm is not None:
            self.cfg.prewarm.apply(
                model_parts=self.model_parts,
                device=self.dist_env.device,
                batch_size=(
                    self.pp.pp_microbatch_size
                    if self.pp is not None
                    else self.cfg.get("step_scheduler.local_batch_size", 1)
                ),
                pp_mesh=(self.device_mesh["pp"] if self.pp_enabled and self.device_mesh is not None else None),
            )

        # Extract mRoPE position-id builder from the model so VLM neat packing can
        # produce 3D position_ids per sample. Without this, packed multimodal
        # training silently degrades mRoPE to plain 1D positions.
        get_rope_index = resolve_get_rope_index(self.model_parts[0])
        pp_n_microbatches = None
        # Under PP, media is staged per microbatch: every VLM here embeds + shards
        # inside its own forward and pulls media from the PP side channel, so raw
        # pixel_values/image_grid_thw must not ride schedule.step -- otherwise torch
        # pipelining row-chunks them independently and the vision RoPE positions
        # desync (156-vs-160 patch mismatch).
        if self.pp_enabled:
            pp_n_microbatches = self.pp.pp_batch_size // self.pp.pp_microbatch_size

        dataloader_config = self.cfg.vlm_dataloader
        if dataloader_config is None:
            raise ValueError("VLM training requires a dataset config")
        _validate_cp_packing_support(
            self.model_parts[0],
            packing_enabled=dataloader_config.packing is not None,
            cp_size=self.mesh_context.cp_size,
        )
        from nemo_automodel.components.models.common.packing import configure_packing, get_attn_implementation

        model_attn_implementation = get_attn_implementation(self.cfg.model, model=self.model_parts[0])
        packing_attn_implementation = dataloader_config.resolve_packing_attn_implementation(
            model_attn_implementation=model_attn_implementation,
            cp_size=self.mesh_context.cp_size,
        )
        if dataloader_config.packing is not None and dataloader_config.packing.packing_format != "thd":
            configure_packing(attn_implementation=packing_attn_implementation)
        process_group = getattr(self.mesh_context, "process_group", None)
        dataset_build_context = FirstRankPerNode(group=process_group)
        with ScopedRNG(seed=self.cfg.get("seed", 42), ranked=True):
            dataloader_build = dataloader_config.build(
                pretrained_model_name_or_path=_get_model_name(self.cfg.model),
                dp_rank=self._get_dp_rank(),
                dp_world_size=self._get_dp_group_size(),
                batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
                dataset_build_context=dataset_build_context,
                get_rope_index=get_rope_index,
                packing_attn_implementation=packing_attn_implementation,
                pp_n_microbatches=pp_n_microbatches,
                cp_size=self.mesh_context.cp_size,
            )
        self.dataloader = dataloader_build.dataloader
        self.processor = dataloader_build.processor

        if getattr(self.loss_fn, "reduction", None) != "sum":
            raise ValueError("Engine-backed VLM finetuning requires a loss with reduction='sum'")
        padding_token_id = getattr(getattr(getattr(self, "processor", None), "tokenizer", None), "pad_token_id", 0) or 0

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        validation_config = self.cfg.vlm_validation_dataloader
        if validation_config is not None:
            if self.pp_enabled and not validation_config.drop_last:
                raise ValueError(
                    "Pipeline-parallel VLM validation requires validation_dataloader.drop_last=true because "
                    "AutoPipeline uses a fixed outer batch size. Enable drop_last or remove validation_dataset."
                )
            _validate_cp_packing_support(
                self.model_parts[0],
                packing_enabled=validation_config.packing is not None,
                cp_size=self.mesh_context.cp_size,
            )
            validation_packing_attn_implementation = validation_config.resolve_packing_attn_implementation(
                model_attn_implementation=model_attn_implementation,
                cp_size=self.mesh_context.cp_size,
            )
            if validation_config.packing is not None and validation_config.packing.packing_format != "thd":
                configure_packing(attn_implementation=validation_packing_attn_implementation)
            validation_build_context = FirstRankPerNode(group=process_group)
            with ScopedRNG(seed=self.cfg.get("seed", 42), ranked=True):
                validation_build = validation_config.build(
                    pretrained_model_name_or_path=_get_model_name(self.cfg.model),
                    dp_rank=self._get_dp_rank(),
                    dp_world_size=self._get_dp_group_size(),
                    batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
                    dataset_build_context=validation_build_context,
                    get_rope_index=get_rope_index,
                    packing_attn_implementation=validation_packing_attn_implementation,
                    pp_n_microbatches=pp_n_microbatches,
                    cp_size=self.mesh_context.cp_size,
                )
            self.val_dataloader = validation_build.dataloader

        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        # Scheduler
        self.step_scheduler = self.cfg.step_scheduler.build(
            self.dataloader,
            self._get_dp_group_size(),
            self.cfg.get("step_scheduler.local_batch_size", 1),
            process_group=getattr(self, "_training_process_group", None),
        )
        self._setup_garbage_collection(self.step_scheduler)

        # Build learning rate scheduler
        self.lr_scheduler = (
            self.cfg.lr_scheduler.build(self.optimizer, self.step_scheduler)
            if self.cfg.lr_scheduler is not None
            else None
        )

        self.engine = Engine(
            self.pp if self.pp_enabled else self.model_parts[0],
            device=self.dist_env.device,
            mesh_context=self.mesh_context,
            microbatch_size=1,
            collate_fn=collate_prebatched,
            padding_token_id=padding_token_id,
            mtp_ignore_index=self.cfg.mtp.ignore_index,
            context_fn=self._engine_context,
            defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            optimizers=self.optimizer,
            lr_schedulers=self.lr_scheduler,
            max_grad_norm=self.max_grad_norm,
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
                    log_data = self._run_train_optim_step(batches)
                    # log
                    self.log_train_metrics(log_data)
                    self._update_progress_bar(pbar, log_data.metrics)

                    val_loss = {}
                    if self.step_scheduler.is_val_step and self.val_dataloader is not None:
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

        self._finalize_and_close_checkpointer()

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
        num_label_tokens: int | None,
        log: bool = False,
        log_denominator: int | float | None = None,
    ) -> torch.Tensor:
        """Return ``base_loss + lambda * sum_k CE(drafter_logits[k], shifted_labels_k)``.

        If ``out`` does not carry a non-empty ``drafter_logits`` attribute (i.e. the
        model isn't a joint composite), returns ``base_loss`` unchanged.

        For drafter step ``k``, labels are shifted left by ``k`` positions to match
        the VLM collate's pre-shifted convention (``labels[t] == input_ids[t+1]``).
        ``log=True`` emits a one-line breakdown on rank 0; callers should gate this
        on the appropriate step / microbatch index to avoid log spam. A caller
        that supplies unnormalized sums can set ``log_denominator`` to keep the
        reported breakdown in mean-loss units.
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
            log_scale = 1.0 if log_denominator is None else 1.0 / log_denominator
            logger.info(
                "[joint-drafter] L_base=%.4f L_drafter=%.4f L_total=%.4f (lambda=%.3f)",
                base_loss.detach().item() * log_scale,
                drafter_loss_total.detach().item() * log_scale,
                total_loss.detach().item() * log_scale,
                drafter_loss_weight,
            )
        return total_loss

    def _compute_vlm_loss(
        self,
        *,
        out: Any,
        labels: torch.Tensor,
        num_label_tokens: int | None,
        is_train: bool,
        cu_seqlens: torch.Tensor | None = None,
        mtp_per_depth_targets: Sequence[torch.Tensor] | None = None,
        log_drafter: bool = False,
        log_denominator: int | float | None = None,
    ) -> torch.Tensor:
        """Compute base, MTP, and optional joint-drafter losses.

        Args:
            out: Model output with base logits shaped [batch, sequence, vocab]
                or [tokens, vocab], plus optional per-depth MTP outputs in the
                matching token layout.
            labels: Tensor of shape [batch, sequence] or [tokens].
            num_label_tokens: Optional supervised-token count consumed by the
                configured loss; Engine supplies ``None`` because it normalizes globally.
            is_train: Whether distributed training-loss reduction is active.
            cu_seqlens: Optional tensor of shape [num_sequences + 1] or
                [1, num_sequences + 1] describing packed THD sequence boundaries.
            mtp_per_depth_targets: Optional CP-local target tensors, one per MTP
                depth, each with the same shape and token layout as ``labels``.
            log_drafter: Whether to log the optional drafter loss breakdown.
            log_denominator: Optional denominator used only for drafter logging.

        Returns:
            Scalar local loss-sum tensor.
        """
        model = self.model_parts[0]
        grad_reduce_group = self._get_dp_group(include_cp=True) if is_train else None
        hidden_states = get_final_hidden_states(out)
        if isinstance(self.loss_fn, FusedLinearCrossEntropy) and hidden_states is None:
            raise ValueError("FusedLinearCrossEntropy requires the model to output hidden states")

        lm_weight = (
            self.loss_fn.materialize_lm_weight(
                _get_lm_head_weight(model),
                grad_reduce_group=grad_reduce_group,
            )
            if isinstance(self.loss_fn, FusedLinearCrossEntropy)
            else None
        )
        loss = calculate_loss(
            self.loss_fn,
            logits=getattr(out, "logits", out),
            labels=labels,
            model=model,
            hidden_states=hidden_states,
            lm_weight=lm_weight,
            num_label_tokens=num_label_tokens,
            grad_reduce_group=grad_reduce_group,
        )

        mtp_hidden = getattr(out, "mtp_per_depth_h", None)
        mtp_logits = getattr(out, "mtp_per_depth_logits", None)
        if mtp_hidden is not None or mtp_logits is not None:
            mtp_config = getattr(getattr(self, "cfg", None), "mtp", None)
            if mtp_config is None:
                raise ValueError("MTP model output requires an MTP loss config")
            if self._get_cp_group_size() > 1 and mtp_per_depth_targets is None:
                raise RuntimeError("MTP with context parallelism requires globally prepared per-depth targets")
            scaling_factor = (
                mtp_config.scaling_factor if mtp_config.scaling_factor is not None else out.mtp_loss_scaling_factor
            )
            loss = loss + calculate_mtp_loss(
                self.loss_fn,
                mtp_per_depth_h=mtp_hidden,
                mtp_per_depth_logits=mtp_logits,
                mtp_per_depth_targets=mtp_per_depth_targets,
                labels=labels,
                model=model,
                scaling_factor=scaling_factor,
                num_label_tokens=num_label_tokens,
                ignore_index=mtp_config.ignore_index,
                cu_seqlens=None if mtp_per_depth_targets is not None else cu_seqlens,
                lm_weight=lm_weight,
                grad_reduce_group=grad_reduce_group,
            )

        return self._maybe_add_drafter_loss(
            out=out,
            base_loss=loss,
            labels=labels,
            model=model,
            num_label_tokens=num_label_tokens,
            log=log_drafter,
            log_denominator=log_denominator,
        )

    def _make_engine_datum(self, batch: dict[str, Any]) -> Datum:
        """Wrap one processor-collated VLM batch for ``collate_prebatched``.

        Args:
            batch: Model and media inputs plus labels. Padded labels have shape
                [batch, sequence]; packed labels have shape [tokens].

        Returns:
            A Datum preserving model/media layouts with labels and weights on
            matching token axes. Non-first PP stages omit raw media tensors.
        """
        labels = batch["labels"]
        model_inputs = {key: value for key, value in batch.items() if key != "labels"}
        if self.pp_enabled and not self.pp.info.has_first_stage:
            for key in VLM_INPUT_KEYS:
                if key != "input_ids":
                    model_inputs.pop(key, None)
            model_inputs.pop(VLM_PP_MEDIA_KEY, None)
        if isinstance(self.loss_fn, FusedLinearCrossEntropy):
            model_inputs["logits_to_keep"] = 1
        return Datum(
            model_inputs=model_inputs,
            loss_fn_inputs={"labels": labels, "weights": labels.ne(-100)},
            loss_fn_input_layouts={
                "labels": LossInputLayout.PER_TOKEN,
                "weights": LossInputLayout.PER_TOKEN,
            },
        )

    def _engine_loss_fn(
        self,
        out: Any,
        loss_inputs: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...]],
        *,
        is_train: bool = True,
        log_drafter: bool = False,
        log_denominator: int | float | None = None,
    ) -> torch.Tensor:
        """Compute a local summed VLM loss for Engine normalization.

        Args:
            out: Model output with logits shaped [batch, sequence, vocab],
                packed logits shaped [tokens, vocab], or the PP/MTP tuple contract.
            loss_inputs: Mapping whose CP-local ``labels`` and ``weights`` tensors
                have shape [batch, sequence] or [tokens]. Optional ``cu_seqlens``
                has shape [num_sequences + 1] or [1, num_sequences + 1], and each
                ``mtp_per_depth_targets`` tensor matches the labels' local layout.
            is_train: Whether the loss participates in training gradients.

        Returns:
            Scalar local loss-sum tensor.
        """
        labels = cast(torch.Tensor, loss_inputs["labels"])
        cu_seqlens = cast(torch.Tensor | None, loss_inputs.get("cu_seqlens"))
        if self.pp_enabled:
            if self.pipeline_loss_fn is None:
                raise RuntimeError("The last pipeline stage has no configured causal-LM loss")
            self.pipeline_loss_fn.cu_seqlens = cu_seqlens
            return self.pipeline_loss_fn(out, labels)
        mtp_per_depth_targets = cast(
            tuple[torch.Tensor, ...] | None,
            loss_inputs.get("mtp_per_depth_targets"),
        )
        return self._compute_vlm_loss(
            out=out,
            labels=labels,
            num_label_tokens=None,
            is_train=is_train,
            cu_seqlens=cu_seqlens,
            mtp_per_depth_targets=mtp_per_depth_targets,
            log_drafter=log_drafter,
            log_denominator=log_denominator,
        )

    def _engine_validation_loss_fn(
        self,
        out: Any,
        loss_inputs: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """Compute the validation loss numerator without training reductions."""
        return self._engine_loss_fn(out, loss_inputs, is_train=False)

    @contextmanager
    def _cp_vision_frame_sharding_context(self):
        """Publish the CP-only group while a VLM forward may run its vision tower."""
        if self.device_mesh is None:
            yield
            return

        mesh_dim = self.cp_vision_frame_sharding.mesh_dims[0]
        cp_active = mesh_dim in self.device_mesh.mesh_dim_names and self.device_mesh[mesh_dim].size() > 1
        if not cp_active:
            yield
            return

        token = set_cp_vision_group(
            self.device_mesh[mesh_dim].get_group(),
            config=self.cp_vision_frame_sharding,
        )
        try:
            yield
        finally:
            reset_cp_vision_group(token)

    @contextmanager
    def _engine_context(self, model_inputs: dict[str, Any]):
        """Install VLM runtime state around one Engine forward/backward call.

        Args:
            model_inputs: CP-local outer-batch mapping. Padded token tensors use
                shape [batch, sequence]; THD tensors use the sharder-produced
                packed layout. PP media is stored as per-microbatch tensor lists.
        """
        if not self.pp_enabled:
            with self._cp_vision_frame_sharding_context():
                yield
            return

        with self._cp_vision_frame_sharding_context(), stage_vlm_media_for_pp(self.pp, self.model_parts, model_inputs):
            yield

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

        if isinstance(self.loss_fn, FusedLinearCrossEntropy):
            last_stage_model._pp_return_hidden_states = True

        self.pipeline_loss_fn = self.cfg.mtp.build(
            self.loss_fn,
            last_stage_model,
            grad_reduce_group=self._get_dp_group(include_cp=True),
        )
        self.pp.info.schedule._loss_fn = self.pipeline_loss_fn

    def _run_train_optim_step(self, batches: list[dict[str, Any]]) -> MetricsSample:
        """Execute a single training step.

        Args:
            batches: Processor-collated optimizer window. Padded token tensors
                use shape [batch, sequence]; packed tensors use their THD token
                layout, and media tensors retain model-specific layouts.
        Returns:
            Metrics for the completed optimizer step.
        """
        # number of tokens in the batch, excluding any tail padding.
        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        log_drafter = self.step_scheduler.is_remote_logging_step and self._has_joint_drafter
        log_denominator = None
        if log_drafter:
            # The optional drafter breakdown is emitted from inside the first
            # loss callback, before forward_backward can return its weight_sum.
            # Preserve its exact global mean only on logging steps; ordinary
            # steps avoid this otherwise-duplicate collective.
            local_label_tokens = torch.tensor(
                sum((batch["labels"] != -100).sum().item() for batch in batches), dtype=torch.long
            )
            log_denominator = max(self._dp_allreduce(local_label_tokens).item(), 1)

        def engine_loss_fn(
            out: Any,
            loss_inputs: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...]],
        ) -> torch.Tensor:
            nonlocal log_drafter
            should_log = log_drafter
            log_drafter = False
            return self._engine_loss_fn(
                out,
                loss_inputs,
                log_drafter=should_log,
                log_denominator=log_denominator,
            )

        forward_backward_result = self.engine.forward_backward(
            [self._make_engine_datum(batch) for batch in batches],
            engine_loss_fn,
        )
        reporting_loss = forward_backward_result.loss
        num_label_tokens = int(forward_backward_result.weight_sum.item())

        step_result = self.engine.optim_step(
            before_optimizer_step=self.checkpointer.maybe_wait_for_staging,
        )

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.install_optimized_model_weights()
        # self.model.zero_grad_buffer()

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = reporting_loss.item()
        # fix reporting_loss, tps across ranks

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "grad_norm": step_result.grad_norm,
                "lr": step_result.learning_rates[0],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
            },
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one recipe-owned pass over a VLM validation dataloader.

        Args:
            val_dataloader: Iterable of processor-collated batches. Padded token
                tensors have shape [batch, sequence], while packed tensors use
                their THD token layout; media tensors retain model-specific layouts.

        Returns:
            Metrics for the completed validation epoch.
        """
        with ScopedRNG(seed=1, ranked=True):
            total_loss = torch.zeros((), dtype=torch.float64, device=self.dist_env.device)
            total_num_label_tokens = torch.zeros((), dtype=torch.float64, device=self.dist_env.device)
            for batch in val_dataloader:
                result = self.engine.forward([self._make_engine_datum(batch)], self._engine_validation_loss_fn)
                total_loss += result.loss_sum
                total_num_label_tokens += result.weight_sum

        # Engine.forward has already reconstructed CP shards and synchronized
        # PP stages. Only independent DP validation shards remain to combine.
        total_loss = self._dp_allreduce(total_loss).item()
        total_num_label_tokens = int(self._dp_allreduce(total_num_label_tokens).item())
        if total_num_label_tokens <= 0:
            raise ValueError(
                "VLM validation produced no supervised label tokens after DP aggregation. "
                "With pipeline parallelism, validation_dataloader.drop_last=true may have removed every batch "
                "because each DP shard is smaller than the local batch size; otherwise verify that labels are not "
                "all masked."
            )
        val_loss = total_loss / total_num_label_tokens

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
