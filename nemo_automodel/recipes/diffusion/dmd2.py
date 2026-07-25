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

"""AutoModel-native DMD2 training.

Model Optimizer owns the DMD2 math, while a declarative model adapter owns
architecture-specific calls. This module owns only training-framework concerns:
model construction, the student/fake-score/discriminator update schedule,
distributed gradient synchronization, data movement, and checkpointing.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import torch
import torch.distributed as dist
import wandb
from torch import nn

from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.distributed.parallelizer import register_full_block_checkpointing_strategy
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from nemo_automodel.recipes.diffusion.train import (
    TrainDiffusionRecipe,
    _build_diffusion_parallel_manager_args,
    _build_optimizer,
    _count_local_batch_group_samples,
    build_lr_scheduler,
    is_main_process,
)
from nemo_automodel.shared.import_utils import safe_import

_CHECKPOINT_FORMAT_VERSION = 1
_MODELOPT_INSTALL_ERROR = (
    "DMD2 training requires a Model Optimizer build that provides `modelopt.torch.fastgen`. "
    "Install the DMD2 extra "
    "with `uv sync --extra dmd2`, or install a current Model-Optimizer source build."
)
_DMD2_TRAINER_CONFIG_FIELDS = {
    "discriminator_betas",
    "discriminator_lr",
    "discriminator_optimizer",
    "discriminator_weight_decay",
    "fake_score_lr",
    "model_adapter",
    "negative_prompt_embedding_path",
    "recipe_path",
}


@dataclass(frozen=True)
class _ModelOptDMD2API:
    """Model Optimizer symbols used by the thin AutoModel integration."""

    dmd_config_cls: type
    load_dmd_config: Callable[[str], Any]


class _DMD2ModelAdapter(Protocol):
    """Typed boundary for model-owned DMD2 behavior."""

    def require_modelopt_dependencies(self) -> None:
        """Resolve optional model-specific Model Optimizer symbols."""
        ...

    @property
    def parallel_model_class_name(self) -> str:
        """Return the model class key used by AutoModel parallelization."""
        ...

    def checkpoint_transformer_blocks(self, model: nn.Module) -> int:
        """Apply the model-owned full-block activation-checkpoint boundary."""
        ...

    def validate_transformer(self, model: nn.Module, *, name: str) -> None:
        """Validate one loaded transformer against the model contract."""
        ...

    def configure_transformer(
        self,
        model: nn.Module,
        *,
        name: str,
        attention_backend: str | None,
    ) -> None:
        """Validate a transformer and configure its attention backend."""
        ...

    def validate_dmd_config(self, config: Any) -> None:
        """Validate model-specific DMD configuration."""
        ...

    def normalize_text_mask(
        self,
        mask: torch.Tensor | None,
        *,
        attention_backend: str | None,
        prompt_kind: Literal["positive", "negative"],
    ) -> torch.Tensor | None:
        """Normalize a model-owned text mask.

        Args:
            mask: Tensor of shape ``[sequence]`` or ``[batch, sequence]``, or
                ``None``.
            attention_backend: Configured attention backend.
            prompt_kind: Whether the mask belongs to positive or negative text.

        Returns:
            A mask with the input shape, or ``None`` when the model backend does
            not consume an explicit all-valid mask.
        """
        ...

    def build_discriminator(
        self,
        config: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module | None:
        """Build the model-owned DMD2 discriminator when configured."""
        ...

    def attach_feature_capture(
        self,
        teacher: nn.Module,
        *,
        height: int,
        width: int,
    ) -> None:
        """Attach the model-owned GAN feature hook."""
        ...

    def build_pipeline(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        fake_score: nn.Module,
        config: Any,
        discriminator: nn.Module | None,
    ) -> Any:
        """Build the model-owned Model Optimizer DMD2 pipeline."""
        ...


@dataclass(frozen=True)
class _PreparedDMD2Batch:
    """Device-resident inputs for one DMD2 microbatch.

    Attributes:
        latents: Clean image latents with shape ``[B, C, H, W]``.
        noise: Gaussian student-input noise with shape ``[B, C, H, W]``.
        text_embeddings: Positive text conditioning with shape ``[B, S, D]``.
        text_mask: Optional positive-text mask with shape ``[B, S]``.
        negative_text_embeddings: Optional CFG conditioning with shape
            ``[B, S_neg, D]``.
        negative_text_mask: Optional CFG mask with shape ``[B, S_neg]``.
    """

    latents: torch.Tensor
    noise: torch.Tensor
    text_embeddings: torch.Tensor
    text_mask: torch.Tensor | None
    negative_text_embeddings: torch.Tensor | None
    negative_text_mask: torch.Tensor | None


def _require_modelopt_dmd2() -> _ModelOptDMD2API:
    """Resolve the optional Model Optimizer DMD2 API or fail actionably."""
    available, fastgen = safe_import("modelopt.torch.fastgen", msg=_MODELOPT_INSTALL_ERROR)
    if not available:
        raise ImportError(_MODELOPT_INSTALL_ERROR)

    required_fastgen_symbols = ("DMDConfig", "load_dmd_config")
    missing = [name for name in required_fastgen_symbols if not hasattr(fastgen, name)]
    if missing:
        raise ImportError(f"{_MODELOPT_INSTALL_ERROR} Missing symbols: {', '.join(missing)}.")

    return _ModelOptDMD2API(
        dmd_config_cls=fastgen.DMDConfig,
        load_dmd_config=fastgen.load_dmd_config,
    )


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _load_negative_prompt_embedding(path: str | os.PathLike[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Safely load one static negative-prompt embedding.

    The file may contain a bare ``[S, D]`` tensor or a mapping with an ``embed``
    tensor and an optional ``mask`` (also accepting ``prompt_embeds_mask`` or
    ``text_mask``). A leading singleton batch dimension is removed.

    Args:
        path: Path to a tensor-only PyTorch checkpoint.

    Returns:
        A CPU embedding of shape ``[S, D]`` and mask of shape ``[S]``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the payload does not contain tensors.
        ValueError: If embedding or mask shapes are invalid.
    """
    embedding_path = Path(path).expanduser()
    if not embedding_path.is_file():
        raise FileNotFoundError(f"DMD2 negative prompt embedding does not exist: {embedding_path}")

    payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
    embedding = payload.get("embed") if isinstance(payload, dict) else payload
    if not torch.is_tensor(embedding):
        raise TypeError(
            "DMD2 negative prompt payload must be a tensor or a mapping with a tensor "
            f"under `embed`; got {type(embedding).__name__} from {embedding_path}."
        )
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    if embedding.ndim != 2:
        raise ValueError(
            "DMD2 negative prompt embedding must have shape [S, D] or [1, S, D], "
            f"got {tuple(embedding.shape)} from {embedding_path}."
        )
    if not embedding.is_floating_point():
        raise TypeError(f"DMD2 negative prompt embedding must be floating point, got {embedding.dtype}.")
    if not torch.isfinite(embedding).all():
        raise ValueError(f"DMD2 negative prompt embedding contains non-finite values: {embedding_path}")

    mask = None
    if isinstance(payload, dict):
        for key in ("mask", "prompt_embeds_mask", "text_mask"):
            candidate = payload.get(key)
            if candidate is not None:
                mask = candidate
                break
    if mask is None:
        mask = torch.ones(embedding.shape[0], dtype=torch.long)
    if not torch.is_tensor(mask):
        raise TypeError(f"DMD2 negative prompt mask must be a tensor, got {type(mask).__name__}.")
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim != 1 or mask.shape[0] != embedding.shape[0]:
        raise ValueError(
            "DMD2 negative prompt mask must have shape [S] matching the embedding; "
            f"got mask={tuple(mask.shape)}, embedding={tuple(embedding.shape)}."
        )
    if mask.is_floating_point() and not torch.isfinite(mask).all():
        raise ValueError(f"DMD2 negative prompt mask contains non-finite values: {embedding_path}")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("DMD2 negative prompt mask must contain only zeros and ones.")

    return embedding.contiguous(), mask.to(dtype=torch.long).contiguous()


def _expand_negative_conditioning(
    embedding: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Broadcast static CFG conditioning to one device-resident microbatch.

    Args:
        embedding: Static negative embedding with shape ``[S, D]``.
        mask: Optional static text mask with shape ``[S]``.
        batch_size: Microbatch size ``B``.
        device: Destination device.
        dtype: Destination floating-point dtype for the embedding.

    Returns:
        Embedding with shape ``[B, S, D]`` and optional mask with shape
        ``[B, S]``.
    """
    if batch_size <= 0:
        raise ValueError(f"DMD2 conditioning batch_size must be positive, got {batch_size}.")
    expanded_embedding = embedding.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(device=device, dtype=dtype)
    expanded_mask = None
    if mask is not None:
        expanded_mask = mask.unsqueeze(0).expand(batch_size, -1).contiguous().to(device=device)
    return expanded_embedding, expanded_mask


class _DMD2CheckpointState:
    """Atomically checkpoint all trainable DMD2 auxiliary state through DCP."""

    use_distributed_checkpointing = True

    def __init__(
        self,
        *,
        fake_score: nn.Module,
        fake_score_optimizer: torch.optim.Optimizer,
        discriminator: nn.Module | None,
        discriminator_optimizer: torch.optim.Optimizer | None,
        dmd_pipeline: Any,
        student_update_freq: int,
        cpu_offload: bool,
    ) -> None:
        """Create state wrappers without taking ownership of the live objects."""
        self._fake_score = ModelState(fake_score, cpu_offload=cpu_offload)
        self._fake_score_optimizer = OptimizerState(
            fake_score,
            fake_score_optimizer,
            cpu_offload=cpu_offload,
        )
        self._discriminator = ModelState(discriminator, cpu_offload=cpu_offload) if discriminator is not None else None
        self._discriminator_optimizer = (
            OptimizerState(
                discriminator,
                discriminator_optimizer,
                cpu_offload=cpu_offload,
            )
            if discriminator is not None and discriminator_optimizer is not None
            else None
        )
        self._dmd_pipeline = dmd_pipeline
        self._student_update_freq = student_update_freq
        self.student_update_count = 0

    def _topology_manifest(self) -> torch.Tensor:
        """Return the fixed-shape DMD2 state topology expected by this run.

        Returns:
            CPU int64 tensor of shape ``[4]`` containing format version, student
            update frequency, GAN presence, and EMA presence in that order.
        """
        return torch.tensor(
            [
                _CHECKPOINT_FORMAT_VERSION,
                self._student_update_freq,
                int(self._discriminator is not None),
                int(self._dmd_pipeline.ema is not None),
            ],
            dtype=torch.int64,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a nested DCP state dictionary for all DMD2 auxiliaries.

        Returns:
            Mapping containing scalar CPU int64 format/counter tensors, the
            int64 topology tensor of shape ``[4]``, and nested DCP-compatible
            fake-score, discriminator, optimizer, and optional EMA tensor state.
            FSDP model entries retain their global DTensor shapes and placements.
        """
        state: dict[str, Any] = {
            "format_version": torch.tensor(_CHECKPOINT_FORMAT_VERSION, dtype=torch.int64),
            "topology": self._topology_manifest(),
            "student_update_count": torch.tensor(self.student_update_count, dtype=torch.int64),
            "fake_score": self._fake_score.state_dict(),
            "fake_score_optimizer": self._fake_score_optimizer.state_dict(),
        }
        if self._discriminator is not None:
            state["discriminator"] = self._discriminator.state_dict()
        if self._discriminator_optimizer is not None:
            state["discriminator_optimizer"] = self._discriminator_optimizer.state_dict()
        if self._dmd_pipeline.ema is not None:
            state["ema"] = self._dmd_pipeline.ema.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore DMD2 auxiliaries from a DCP-populated state dictionary.

        Args:
            state: Mapping with scalar CPU int64 format/counter tensors, an
                int64 topology tensor of shape ``[4]``, and nested model,
                optimizer, and optional EMA tensors matching ``state_dict()``.
                FSDP model entries use their current global DTensor shapes and
                placements.
        """
        version = int(state["format_version"].item())
        if version != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"Unsupported DMD2 checkpoint format {version}; expected {_CHECKPOINT_FORMAT_VERSION}.")

        loaded_topology = state["topology"].to(device="cpu")
        expected_topology = self._topology_manifest()
        if loaded_topology.shape != expected_topology.shape or not torch.equal(loaded_topology, expected_topology):
            raise ValueError(
                "DMD2 checkpoint topology does not match the current run "
                f"(checkpoint={loaded_topology.tolist()}, current={expected_topology.tolist()})."
            )

        self._fake_score.load_state_dict(state["fake_score"])
        self._fake_score_optimizer.load_state_dict(state["fake_score_optimizer"])

        has_discriminator_state = "discriminator" in state
        if (self._discriminator is not None) != has_discriminator_state:
            raise ValueError("DMD2 checkpoint discriminator state does not match the current GAN configuration.")
        has_discriminator_optimizer_state = "discriminator_optimizer" in state
        if (self._discriminator_optimizer is not None) != has_discriminator_optimizer_state:
            raise ValueError(
                "DMD2 checkpoint discriminator optimizer state does not match the current GAN configuration."
            )
        has_ema_state = "ema" in state
        if (self._dmd_pipeline.ema is not None) != has_ema_state:
            raise ValueError("DMD2 checkpoint EMA state does not match the current EMA configuration.")

        if self._discriminator is not None:
            self._discriminator.load_state_dict(state["discriminator"])
        if self._discriminator_optimizer is not None:
            self._discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        if self._dmd_pipeline.ema is not None:
            self._dmd_pipeline.ema.load_state_dict(state["ema"])

        self.student_update_count = int(state["student_update_count"].item())


class DMD2DiffusionRecipe(TrainDiffusionRecipe):
    """Train a diffusion student with Model Optimizer's complete DMD2 losses.

    AutoModel supplies its native loader, dataset, FSDP2 wrapping, optimizer
    configuration, accumulation schedule, and checkpoint lifecycle. Model
    Optimizer supplies VSD/DSM, CFG, GAN/R1, multi-step simulation, and EMA.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the recipe while deferring checkpoint restore until setup."""
        self._dmd2_state_ready = False
        self._restore_was_deferred = False
        self._deferred_restore_from: str | None = None
        super().__init__(cfg)

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Defer parent restore until every DMD2 stateful object is registered."""
        if not self._dmd2_state_ready:
            self.__dict__["_restore_was_deferred"] = True
            self.__dict__["_deferred_restore_from"] = restore_from
            return
        super().load_checkpoint(restore_from)
        self._validate_restored_dmd2_state()

    def setup(self) -> None:
        """Build the native student, then attach the thin DMD2 orchestration."""
        self._validate_recipe_scope()
        adapter_config = self.cfg.get("dmd2.model_adapter")
        self._model_adapter: _DMD2ModelAdapter = adapter_config.instantiate()
        self._modelopt = _require_modelopt_dmd2()
        self._model_adapter.require_modelopt_dependencies()
        register_full_block_checkpointing_strategy(
            model_class_name=self._model_adapter.parallel_model_class_name,
            checkpoint_blocks=self._model_adapter.checkpoint_transformer_blocks,
        )

        super().setup()
        self._model_adapter.validate_transformer(self.model, name="student")

        self._dmd_config = self._resolve_dmd_config()
        self._rebuild_student_lr_scheduler()
        self._negative_prompt_embedding, self._negative_prompt_mask = self._load_cfg_conditioning()

        parallel_scheme = self._build_auxiliary_parallel_scheme()
        self.teacher_model = self._load_auxiliary_transformer(
            parallel_scheme,
            trainable=False,
            name="teacher",
        )
        self.fake_score_model = self._load_auxiliary_transformer(
            parallel_scheme,
            trainable=True,
            name="fake score",
        )
        self.fake_score_optimizer = self._build_fake_score_optimizer()

        self.discriminator = self._build_discriminator()
        self.discriminator_optimizer = self._build_discriminator_optimizer()
        self._gan_capture_shape: tuple[int, int] | None = None

        self._dmd_pipeline = self._model_adapter.build_pipeline(
            student=self.model,
            teacher=self.teacher_model,
            fake_score=self.fake_score_model,
            config=self._dmd_config,
            discriminator=self.discriminator,
        )

        if hasattr(self, "flow_matching_pipeline"):
            del self.flow_matching_pipeline

        # BaseRecipe otherwise picks the last tracked model/optimizer and would
        # replace the primary student checkpoint with an auxiliary object.
        self.untrack_state(
            "teacher_model",
            "fake_score_model",
            "fake_score_optimizer",
            "discriminator",
            "discriminator_optimizer",
        )
        self.dmd2_state = _DMD2CheckpointState(
            fake_score=self.fake_score_model,
            fake_score_optimizer=self.fake_score_optimizer,
            discriminator=self.discriminator,
            discriminator_optimizer=self.discriminator_optimizer,
            dmd_pipeline=self._dmd_pipeline,
            student_update_freq=self._dmd_config.student_update_freq,
            cpu_offload=self.cpu_offload,
        )

        self.__dict__["_dmd2_state_ready"] = True
        if self._restore_was_deferred:
            self.load_checkpoint(self._deferred_restore_from)

        if is_main_process():
            logging.info("[DMD2] Initialized full DMD2: %s", self._dmd_config_summary())

    def _validate_restored_dmd2_state(self) -> None:
        """Reject a partial or phase-inconsistent DMD2 checkpoint restore."""
        completed_outer_steps = int(self.step_scheduler.step)
        update_frequency = int(self._dmd_config.student_update_freq)
        expected_student_updates = (completed_outer_steps + update_frequency - 1) // update_frequency
        restored_student_updates = int(self.dmd2_state.student_update_count)
        if restored_student_updates != expected_student_updates:
            raise ValueError(
                "DMD2 checkpoint phase is inconsistent: "
                f"step={completed_outer_steps}, student_update_freq={update_frequency}, "
                f"expected student_update_count={expected_student_updates}, "
                f"restored={restored_student_updates}."
            )

        if self.lr_scheduler is not None:
            restored_scheduler_steps = int(self.lr_scheduler[0].num_steps)
            if restored_scheduler_steps != restored_student_updates:
                raise ValueError(
                    "DMD2 checkpoint student LR scheduler is inconsistent: "
                    f"scheduler_steps={restored_scheduler_steps}, "
                    f"student_update_count={restored_student_updates}."
                )

    def _validate_recipe_scope(self) -> None:
        """Reject configurations outside this production integration's contract."""
        if self.cfg.get("model.mode", "finetune").lower() != "finetune":
            raise ValueError("DMD2 currently requires model.mode=finetune.")
        if self.cfg.get("peft", None) is not None:
            raise ValueError("DMD2 currently supports full-parameter training only; remove the `peft` block.")
        if self.cfg.get("ddp", None) is not None:
            raise ValueError("DMD2 currently requires FSDP2; DDP is not supported.")

        fsdp_cfg = self.cfg.get("fsdp", {}) or {}
        if not fsdp_cfg:
            raise ValueError("DMD2 requires an `fsdp` configuration block.")
        unsupported = {
            name: int(fsdp_cfg.get(name, 1))
            for name in ("tp_size", "cp_size", "pp_size")
            if int(fsdp_cfg.get(name, 1)) != 1
        }
        if unsupported:
            rendered = ", ".join(f"{key}={value}" for key, value in unsupported.items())
            raise ValueError(
                f"DMD2 currently supports data parallel FSDP2 only (tp_size=cp_size=pp_size=1); got {rendered}."
            )

        adapter_config = self.cfg.get("dmd2.model_adapter", None)
        if adapter_config is None or not hasattr(adapter_config, "instantiate"):
            raise ValueError("DMD2 requires a declarative `dmd2.model_adapter._target_` configuration.")

        if bool(self.cfg.get("data.dataloader.train_text_encoder", False)):
            raise ValueError("DMD2 requires precomputed text embeddings (`train_text_encoder: false`).")
        if self.cfg.get("model.stage", None) is not None:
            raise ValueError("DMD2 does not support `model.stage`; remove that setting.")
        if bool(self.cfg.get("model.transformer_engine_fp8", False)):
            raise ValueError(
                "DMD2 does not yet support Transformer Engine FP8 because "
                "the frozen teacher's mutable FP8 history is not checkpointed exactly."
            )

    def _resolve_dmd_config(self) -> Any:
        """Load a ModelOpt DMD recipe and deep-merge inline method overrides."""
        config_node = self.cfg.get("dmd2", None)
        if config_node is None:
            raise ValueError("DMD2 requires a `dmd2` configuration block.")
        values = config_node.to_dict() if hasattr(config_node, "to_dict") else dict(config_node)
        recipe_path = values.pop("recipe_path", None)
        if not recipe_path:
            raise ValueError("DMD2 requires a Model Optimizer recipe path in `dmd2.recipe_path`.")

        base = self._modelopt.load_dmd_config(recipe_path)
        field_names = set(self._modelopt.dmd_config_cls.model_fields)
        unknown = sorted(set(values) - field_names - _DMD2_TRAINER_CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"Unknown DMD2 configuration fields: {', '.join(unknown)}.")
        overrides = {key: value for key, value in values.items() if key in field_names}
        merged = _deep_merge_dicts(base.model_dump(), overrides)
        config = self._modelopt.dmd_config_cls.model_validate(merged)
        self._validate_resolved_dmd_config(config)
        return config

    def _validate_resolved_dmd_config(self, config: Any) -> None:
        """Validate scheduling and GAN invariants required by this trainer."""
        if config.student_update_freq < 1:
            raise ValueError(f"DMD2 student_update_freq must be at least 1, got {config.student_update_freq}.")
        if config.student_sample_steps < 1:
            raise ValueError(f"DMD2 student_sample_steps must be at least 1, got {config.student_sample_steps}.")
        if config.student_sample_steps > 1:
            timesteps = config.sample_t_cfg.t_list
            expected_length = config.student_sample_steps + 1
            if timesteps is None or len(timesteps) != expected_length:
                actual_length = None if timesteps is None else len(timesteps)
                raise ValueError(
                    "Multi-step DMD2 requires len(sample_t_cfg.t_list) == "
                    f"student_sample_steps + 1; got {actual_length} vs {expected_length}."
                )
            if any(not 0.0 <= timestep <= 1.0 for timestep in timesteps):
                raise ValueError("DMD2 sample_t_cfg.t_list values must lie in [0, 1].")
            if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
                raise ValueError("DMD2 sample_t_cfg.t_list must be strictly decreasing.")

        if config.gan_loss_weight_gen < 0:
            raise ValueError(f"DMD2 gan_loss_weight_gen must be non-negative, got {config.gan_loss_weight_gen}.")
        if config.ema is not None:
            if not config.ema.fsdp2 or config.ema.mode != "full_tensor":
                raise ValueError(
                    "DMD2 exact checkpoint resume currently requires EMA with "
                    "`fsdp2: true` and `mode: full_tensor`; set `ema: null` for "
                    "the memory-efficient production recipe."
                )
        self._model_adapter.validate_dmd_config(config)

    def _rebuild_student_lr_scheduler(self) -> None:
        """Scale the native LR schedule to the number of student updates."""
        if self.lr_scheduler is None:
            return
        outer_steps = min(
            int(self.step_scheduler.max_steps),
            int(self.num_epochs) * int(self.steps_per_epoch),
        )
        frequency = int(self._dmd_config.student_update_freq)
        student_steps = (outer_steps + frequency - 1) // frequency
        previous_scheduler = self.lr_scheduler[0]
        self.optimizer.param_groups[0]["lr"] = float(previous_scheduler.max_lr)
        scheduler = build_lr_scheduler(
            self.cfg.get("lr_scheduler", None),
            self.optimizer,
            student_steps,
        )
        if scheduler is None:
            raise RuntimeError("DMD2 expected the configured student LR scheduler to be buildable.")
        self.lr_scheduler[0] = scheduler
        if is_main_process():
            logging.info(
                "[DMD2] Student LR schedule uses %d optimizer updates across %d outer steps.",
                student_steps,
                outer_steps,
            )

    def _load_cfg_conditioning(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Load static CFG conditioning when the DMD config enables guidance."""
        path = self.cfg.get("dmd2.negative_prompt_embedding_path", None)
        if self._dmd_config.guidance_scale is None:
            if path and is_main_process():
                logging.warning(
                    "[DMD2] negative_prompt_embedding_path is set but guidance_scale is null; "
                    "the embedding will not be loaded."
                )
            return None, None
        if not path:
            raise ValueError(
                "DMD2 guidance_scale is enabled, so `dmd2.negative_prompt_embedding_path` "
                "must point to a user-generated empty/negative-prompt embedding."
            )

        embedding, mask = _load_negative_prompt_embedding(path)
        mask = self._model_adapter.normalize_text_mask(
            mask,
            attention_backend=getattr(self, "attention_backend", None),
            prompt_kind="negative",
        )
        if is_main_process():
            logging.info(
                "[DMD2] Loaded negative prompt conditioning from %s: embed=%s mask=%s dtype=%s",
                path,
                tuple(embedding.shape),
                None if mask is None else tuple(mask.shape),
                embedding.dtype,
            )
        return embedding, mask

    def _build_auxiliary_parallel_scheme(self) -> dict[str, dict[str, Any]]:
        """Recreate the student's current AutoModel parallel-manager arguments."""
        manager_args = _build_diffusion_parallel_manager_args(
            fsdp_cfg=self.cfg.get("fsdp", None),
            ddp_cfg=None,
            world_size=self.world_size,
            dtype=self.model_dtype,
            compute_dtype=self.compute_dtype,
            lora_enabled=False,
        )
        return {"transformer": manager_args}

    def _load_auxiliary_transformer(
        self,
        parallel_scheme: dict[str, dict[str, Any]],
        *,
        trainable: bool,
        name: str,
    ) -> nn.Module:
        """Load one auxiliary transformer through native AutoModel FSDP2."""
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.model_dtype,
            device=self.device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=trainable,
            low_cpu_mem_usage=True,
            active_transformer=self.active_transformer,
            transformer_engine_linear=self.transformer_engine_linear,
            transformer_engine_fp8_safe_only=self.transformer_engine_fp8,
            fuse_qkv_projections=self.fuse_qkv_projections,
            compact_fused_qkv_projections=self.compact_fused_qkv_projections,
        )
        model = pipe.transformer
        del pipe
        self._model_adapter.configure_transformer(
            model,
            name=name,
            attention_backend=self.attention_backend,
        )

        model.train(trainable)
        for parameter in model.parameters():
            parameter.requires_grad_(trainable)

        if is_main_process():
            logging.info("[DMD2] Loaded %s transformer (trainable=%s)", name, trainable)
        return model

    def _build_fake_score_optimizer(self) -> torch.optim.Optimizer:
        """Build the fake-score optimizer from the native optimizer configuration."""
        learning_rate = float(self.cfg.get("dmd2.fake_score_lr", self.learning_rate))
        trainable_parameters = [
            parameter for parameter in self.fake_score_model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("DMD2 fake-score transformer has no trainable parameters.")
        return _build_optimizer(
            trainable_parameters,
            self.cfg.get("optim.optimizer", {}),
            learning_rate,
        )

    def _build_discriminator(self) -> nn.Module | None:
        """Build and synchronize the model-owned discriminator."""
        discriminator = self._model_adapter.build_discriminator(
            self._dmd_config,
            device=self.device,
            dtype=self.compute_dtype,
        )
        if discriminator is None:
            return None
        self._broadcast_discriminator_state(discriminator)
        return discriminator

    def _broadcast_discriminator_state(self, discriminator: nn.Module) -> None:
        """Make replicated discriminator initialization identical on every DP rank."""
        if not dist.is_initialized() or self._get_dp_group_size() == 1:
            return
        process_group = self._get_dp_group()
        with torch.no_grad():
            for tensor in (*discriminator.parameters(), *discriminator.buffers()):
                dist.broadcast(tensor, src=0, group=process_group)

    def _build_discriminator_optimizer(self) -> torch.optim.Optimizer | None:
        """Build the replicated discriminator optimizer from native config."""
        if self.discriminator is None:
            return None
        configured = self.cfg.get(
            "dmd2.discriminator_optimizer",
            self.cfg.get("optim.optimizer", {}),
        )
        optimizer_config = configured.to_dict() if hasattr(configured, "to_dict") else dict(configured or {})
        weight_decay = self.cfg.get("dmd2.discriminator_weight_decay", None)
        betas = self.cfg.get("dmd2.discriminator_betas", None)
        if weight_decay is not None:
            optimizer_config["weight_decay"] = float(weight_decay)
        if betas is not None:
            optimizer_config["betas"] = tuple(betas)
        return _build_optimizer(
            list(self.discriminator.parameters()),
            optimizer_config,
            float(self.cfg.get("dmd2.discriminator_lr", self.learning_rate)),
        )

    def _ensure_gan_feature_capture(self, height: int, width: int) -> None:
        """Attach or refresh the model-owned GAN hook for this resolution."""
        if self.discriminator is None or self._gan_capture_shape == (height, width):
            return
        self._model_adapter.attach_feature_capture(
            self.teacher_model,
            height=height,
            width=width,
        )
        self._gan_capture_shape = (height, width)

    def _prepare_micro_batch(self, micro_batch: dict[str, Any]) -> _PreparedDMD2Batch:
        """Move one native text-to-image batch to the training device.

        Args:
            micro_batch: Mapping containing channels-first ``image_latents`` of
                shape ``[B, C, H, W]``, ``text_embeddings`` of shape
                ``[B, S, D]`` (or ``[S, D]`` when ``B=1``), and an optional
                ``text_embeddings_mask`` of shape ``[B, S]`` (or ``[S]`` when
                ``B=1``).

        Returns:
            Device-resident DMD2 inputs with latents/noise of shape
            ``[B, C, H, W]``, positive text conditioning of shape ``[B, S, D]``,
            and optional expanded negative conditioning of shape
            ``[B, S_neg, D]``.
        """
        if "image_latents" not in micro_batch:
            raise KeyError(f"DMD2 batches require `image_latents`; got keys {sorted(micro_batch)}.")
        if "text_embeddings" not in micro_batch:
            raise KeyError(f"DMD2 batches require `text_embeddings`; got keys {sorted(micro_batch)}.")

        latents = micro_batch["image_latents"].to(
            device=self.device,
            dtype=self.compute_dtype,
            non_blocking=True,
        )
        if latents.ndim != 4:
            raise ValueError(f"DMD2 expects image latents [B, C, H, W], got {tuple(latents.shape)}.")

        text_embeddings = micro_batch["text_embeddings"].to(
            device=self.device,
            dtype=self.compute_dtype,
            non_blocking=True,
        )
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        if text_embeddings.ndim != 3 or text_embeddings.shape[0] != latents.shape[0]:
            raise ValueError(
                "DMD2 text embeddings must have shape [B, S, D] matching "
                f"latents; got text={tuple(text_embeddings.shape)}, latents={tuple(latents.shape)}."
            )

        text_mask = micro_batch.get("text_embeddings_mask")
        if text_mask is not None:
            text_mask = text_mask.to(device=self.device, non_blocking=True)
            if text_mask.ndim == 1:
                text_mask = text_mask.unsqueeze(0)
            expected_mask_shape = text_embeddings.shape[:2]
            if text_mask.ndim != 2 or text_mask.shape != expected_mask_shape:
                raise ValueError(
                    "DMD2 text mask must have shape [B, S] matching text embeddings; "
                    f"got mask={tuple(text_mask.shape)}, expected={tuple(expected_mask_shape)}."
                )
            text_mask = self._model_adapter.normalize_text_mask(
                text_mask,
                attention_backend=getattr(self, "attention_backend", None),
                prompt_kind="positive",
            )

        negative_embeddings = None
        negative_mask = None
        if self._negative_prompt_embedding is not None:
            if self._negative_prompt_embedding.shape[-1] != text_embeddings.shape[-1]:
                raise ValueError(
                    "Negative and positive text embedding dimensions differ: "
                    f"{self._negative_prompt_embedding.shape[-1]} vs {text_embeddings.shape[-1]}."
                )
            negative_embeddings, negative_mask = _expand_negative_conditioning(
                self._negative_prompt_embedding,
                self._negative_prompt_mask,
                batch_size=latents.shape[0],
                device=self.device,
                dtype=self.compute_dtype,
            )

        self._ensure_gan_feature_capture(latents.shape[-2], latents.shape[-1])
        return _PreparedDMD2Batch(
            latents=latents,
            noise=torch.randn_like(latents),
            text_embeddings=text_embeddings,
            text_mask=text_mask,
            negative_text_embeddings=negative_embeddings,
            negative_text_mask=negative_mask,
        )

    def _set_phase(self, *, student_phase: bool) -> None:
        """Set train/eval modes and replicated discriminator trainability."""
        self.model.train(student_phase)
        self.fake_score_model.train(not student_phase)
        self.teacher_model.eval()
        if self.discriminator is not None:
            self.discriminator.train(not student_phase)
            self.discriminator.requires_grad_(not student_phase)

    def _synchronize_discriminator_gradients(self) -> None:
        """Average replicated discriminator gradients over the data-parallel group."""
        if self.discriminator is None or not dist.is_initialized():
            return
        process_group = self._get_dp_group()
        dp_size = self._get_dp_group_size()
        if dp_size == 1:
            return
        for parameter in self.discriminator.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=process_group)
                parameter.grad.div_(dp_size)

    def run_train_validation_loop(self) -> None:
        """Run complete DMD2 student and fake-score/discriminator alternation."""
        dmd = self._dmd_pipeline
        dmd_config = self._dmd_config

        logging.info(
            "[DMD2] Starting training: global_batch=%s local_batch=%s dp=%s",
            self.global_batch_size,
            self.local_batch_size,
            self.dp_size,
        )

        self._sync_device()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        performance_window_start = time.perf_counter()
        performance_window_steps = 0
        performance_window_local_samples = 0

        for epoch in self.step_scheduler.epochs:
            if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epoch)

            if is_main_process():
                from tqdm import tqdm

                self.step_scheduler.dataloader = tqdm(
                    self.dataloader,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                )
            else:
                self.step_scheduler.dataloader = self.dataloader

            phase_loss_sums = {"student": 0.0, "fake_score": 0.0}
            phase_steps = {"student": 0, "fake_score": 0}

            for batch_group in self.step_scheduler:
                global_step = int(self.step_scheduler.step)
                student_phase = global_step % dmd_config.student_update_freq == 0
                phase = "student" if student_phase else "fake_score"
                active_model = self.model if student_phase else self.fake_score_model
                active_optimizer = self.optimizer if student_phase else self.fake_score_optimizer

                self._set_phase(student_phase=student_phase)
                active_optimizer.zero_grad(set_to_none=True)
                if not student_phase and self.discriminator_optimizer is not None:
                    self.discriminator_optimizer.zero_grad(set_to_none=True)

                prepare_for_grad_accumulation([active_model], pp_enabled=False)
                num_microbatches = len(batch_group)
                if num_microbatches == 0:
                    raise RuntimeError("DMD2 received an empty gradient-accumulation group.")

                loss_values: list[float] = []
                component_sums: dict[str, float] = {}
                discriminator_values: list[float] = []

                for microbatch_index, micro_batch in enumerate(batch_group):
                    if microbatch_index == num_microbatches - 1:
                        prepare_for_final_backward([active_model], pp_enabled=False)

                    prepared = self._prepare_micro_batch(micro_batch)
                    common_kwargs = {
                        "encoder_hidden_states": prepared.text_embeddings,
                        "encoder_hidden_states_mask": prepared.text_mask,
                    }

                    with self._transformer_engine_fp8_context():
                        if student_phase:
                            losses = dmd.compute_student_loss(
                                prepared.latents,
                                prepared.noise,
                                negative_encoder_hidden_states=prepared.negative_text_embeddings,
                                negative_encoder_hidden_states_mask=prepared.negative_text_mask,
                                **common_kwargs,
                            )
                        else:
                            losses = dmd.compute_fake_score_loss(
                                prepared.latents,
                                prepared.noise,
                                **common_kwargs,
                            )
                    if self.check_loss and not torch.isfinite(losses["total"]).all():
                        raise FloatingPointError(f"Non-finite DMD2 {phase} loss at global step {global_step}.")
                    (losses["total"] / num_microbatches).backward()

                    loss_values.append(float(losses["total"].detach().item()))
                    for name, value in losses.items():
                        component_sums[name] = component_sums.get(name, 0.0) + float(value.detach().item())

                    if not student_phase and self.discriminator_optimizer is not None:
                        with self._transformer_engine_fp8_context():
                            discriminator_losses = dmd.compute_discriminator_loss(
                                prepared.latents,
                                prepared.noise,
                                **common_kwargs,
                            )
                        if self.check_loss and not torch.isfinite(discriminator_losses["total"]).all():
                            raise FloatingPointError(
                                f"Non-finite DMD2 discriminator loss at global step {global_step}."
                            )
                        (discriminator_losses["total"] / num_microbatches).backward()
                        discriminator_values.append(float(discriminator_losses["total"].detach().item()))
                        for name, value in discriminator_losses.items():
                            key = f"discriminator_{name}"
                            component_sums[key] = component_sums.get(key, 0.0) + float(value.detach().item())

                    if microbatch_index == 0:
                        prepare_after_first_microbatch()

                grad_norm = clip_grad_norm(
                    self.clip_grad_max_norm,
                    [active_model],
                    foreach=self.grad_clip_foreach,
                )
                grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm)

                discriminator_grad_norm = None
                if not student_phase and self.discriminator_optimizer is not None:
                    self._synchronize_discriminator_gradients()
                    discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        max_norm=self.clip_grad_max_norm,
                    )
                    discriminator_grad_norm = float(discriminator_grad_norm)

                active_optimizer.step()
                if not student_phase and self.discriminator_optimizer is not None:
                    self.discriminator_optimizer.step()

                if student_phase:
                    if self.lr_scheduler is not None:
                        self.lr_scheduler[0].step(1)
                    self.dmd2_state.student_update_count += 1
                    dmd.update_ema(iteration=self.dmd2_state.student_update_count)

                # Release inactive-model gradient shards immediately. Otherwise
                # student gradients survive all intervening fake-score phases
                # (and vice versa), substantially increasing peak memory.
                active_optimizer.zero_grad(set_to_none=True)
                if not student_phase and self.discriminator_optimizer is not None:
                    self.discriminator_optimizer.zero_grad(set_to_none=True)

                group_loss = sum(loss_values) / len(loss_values)
                phase_loss_sums[phase] += group_loss
                phase_steps[phase] += 1

                performance_window_steps += 1
                performance_window_local_samples += _count_local_batch_group_samples(batch_group)
                should_log = bool(self.log_every and global_step % self.log_every == 0)
                if should_log:
                    elapsed_seconds, window_end = self._elapsed_seconds_since(performance_window_start)
                    global_samples = self._count_global_samples(performance_window_local_samples)
                    throughput = {
                        "step_time": elapsed_seconds / max(performance_window_steps, 1),
                        "samples_per_sec": global_samples / max(elapsed_seconds, 1e-12),
                    }
                    memory = self._get_memory_metrics()
                    performance_window_start = window_end
                    performance_window_steps = 0
                    performance_window_local_samples = 0

                    averaged_components = {
                        f"{phase}/{name}": value / num_microbatches for name, value in component_sums.items()
                    }
                    metrics = {
                        f"{phase}/loss": group_loss,
                        f"{phase}/grad_norm": grad_norm,
                        "global_step": global_step,
                        "lr/student": self.optimizer.param_groups[0]["lr"],
                        "lr/fake_score": self.fake_score_optimizer.param_groups[0]["lr"],
                        **averaged_components,
                        **throughput,
                        **memory,
                    }
                    if discriminator_grad_norm is not None:
                        metrics["discriminator/grad_norm"] = discriminator_grad_norm
                        metrics["discriminator/loss"] = sum(discriminator_values) / len(discriminator_values)

                    if is_main_process():
                        if wandb.run is not None:
                            wandb.log(metrics, step=global_step)
                        logging.info(
                            "[DMD2] step=%d phase=%s loss=%.6f grad_norm=%.4f lr=%.3e "
                            "step_time=%.3fs samples_per_sec=%.2f mem=%.2fGB",
                            global_step,
                            phase,
                            group_loss,
                            grad_norm,
                            active_optimizer.param_groups[0]["lr"],
                            throughput["step_time"],
                            throughput["samples_per_sec"],
                            memory["max_memory_allocated_gb"],
                        )

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, global_step, group_loss)

            if is_main_process():
                summaries = []
                for phase, count in phase_steps.items():
                    average = phase_loss_sums[phase] / count if count else float("nan")
                    summaries.append(f"{phase}={average:.6f} ({count} steps)")
                logging.info("[DMD2] Epoch %d complete: %s", epoch + 1, ", ".join(summaries))

        if is_main_process() and wandb.run is not None:
            wandb.finish()
        self._finalize_and_close_checkpointer()
        logging.info("[DMD2] Training complete at step %d", int(self.step_scheduler.step))

    def _dmd_config_summary(self) -> str:
        """Return a concise summary of the resolved method configuration."""
        config = self._dmd_config
        return (
            f"guidance_scale={config.guidance_scale} "
            f"student_sample_steps={config.student_sample_steps} "
            f"backward_simulation={config.backward_simulation} "
            f"student_update_freq={config.student_update_freq} "
            f"gan_loss_weight_gen={config.gan_loss_weight_gen} "
            f"gan_r1_reg_weight={config.gan_r1_reg_weight} "
            f"ema={'enabled' if config.ema is not None else 'disabled'}"
        )
