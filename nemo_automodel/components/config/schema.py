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

"""
Pydantic schema models for AutoModel recipe configs.

These models provide fail-fast validation of YAML configs before any GPU work begins.
The validation layer sits on top of the existing ConfigNode system: YAML is validated
against these schemas, then wrapped in ConfigNode for backwards-compatible consumption
by recipes.

Design principles:
- Structured sections (step_scheduler, distributed, checkpoint, etc.) are fully typed.
- Dynamic _target_ sections (model, optimizer, dataset, etc.) use TargetConfig with
  extra="allow" to accept arbitrary kwargs while still validating the _target_ field.
- All fields use the same names as YAML keys for zero-migration-cost.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Base for _target_ sections (dynamic, accepts arbitrary kwargs)
# ---------------------------------------------------------------------------


class TargetConfig(BaseModel):
    """Base model for any config section that uses _target_ for instantiation.

    Accepts arbitrary extra kwargs which are passed to the target callable.
    Validates that _target_ is present and is a non-empty string.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    target_: str = Field(alias="_target_", description="Dotted path to a Python callable to instantiate.")

    @model_validator(mode="before")
    @classmethod
    def _ensure_target_is_string(cls, data: Any) -> Any:
        if isinstance(data, dict):
            target = data.get("_target_")
            if target is not None and not isinstance(target, str):
                raise ValueError(f"_target_ must be a string, got {type(target).__name__}: {target}")
        return data


class OptionalTargetConfig(BaseModel):
    """Config section that may or may not have a _target_ field.

    Used for sections like `collate_fn` which can be either a _target_ config
    or a plain dotted string path.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    target_: Optional[str] = Field(default=None, alias="_target_")


# ---------------------------------------------------------------------------
# Structured config sections (fully typed, known schemas)
# ---------------------------------------------------------------------------


class StepSchedulerConfig(BaseModel):
    """Training step scheduling configuration."""

    model_config = ConfigDict(extra="allow")

    global_batch_size: int = Field(gt=0, description="Total batch size across all GPUs.")
    local_batch_size: int = Field(gt=0, description="Batch size per GPU per gradient accumulation step.")
    num_epochs: int = Field(default=1, ge=1, description="Number of training epochs.")
    max_steps: Optional[int] = Field(
        default=None,
        description="Max training steps. If set, training stops after this many steps regardless of epochs.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_none_strings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("max_steps",):
                val = data.get(key)
                if isinstance(val, str) and val.lower() == "none":
                    data[key] = None
        return data

    ckpt_every_steps: Optional[int] = Field(default=None, ge=1, description="Save a checkpoint every N gradient steps.")
    ckpt_every_n_steps: Optional[int] = Field(default=None, ge=1, description="Alias for ckpt_every_steps.")
    val_every_steps: Optional[int] = Field(default=None, ge=1, description="Run validation every N gradient steps.")
    log_remote_every_steps: Optional[int] = Field(
        default=None, ge=1, description="Log to remote (WandB/MLflow) every N steps."
    )
    gc_every_steps: Optional[int] = Field(default=None, ge=1, description="Run garbage collection every N steps.")


class DistEnvConfig(BaseModel):
    """Distributed environment configuration."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="nccl", description="Distributed backend: 'nccl' or 'gloo'.")
    timeout_minutes: int = Field(default=5, ge=1, description="Timeout for distributed operations in minutes.")


class PipelineConfig(BaseModel):
    """Pipeline parallelism configuration."""

    model_config = ConfigDict(extra="allow")

    pp_schedule: Optional[str] = Field(
        default=None,
        description="Pipeline schedule type, e.g. '1f1b', 'interleaved_1f1b', 'looped_bfs'.",
    )
    pp_microbatch_size: Optional[int] = Field(default=None, ge=1, description="Micro-batch size for pipeline stages.")
    pp_n_virtual_stages: Optional[int] = Field(
        default=None, ge=1, description="Number of virtual pipeline stages (for interleaved schedules)."
    )


class DistributedConfig(BaseModel):
    """Distributed training strategy configuration."""

    model_config = ConfigDict(extra="allow")

    strategy: str = Field(
        default="fsdp2",
        description="Distributed strategy: 'fsdp2', 'megatron_fsdp', or 'ddp'.",
    )
    dp_size: Optional[Union[int, str]] = Field(
        default=None,
        description="Data parallelism degree. 'none' or null for auto-inference from world_size.",
    )
    tp_size: int = Field(default=1, ge=1, description="Tensor parallelism degree.")
    cp_size: int = Field(default=1, ge=1, description="Context parallelism degree.")
    pp_size: Optional[int] = Field(default=None, ge=1, description="Pipeline parallelism degree.")
    ep_size: Optional[int] = Field(default=None, ge=1, description="Expert parallelism degree.")
    dp_replicate_size: Optional[int] = Field(default=None, ge=1, description="FSDP2 replication degree (HSDP).")
    sequence_parallel: bool = Field(default=False, description="Enable sequence parallelism with TP.")
    pipeline: Optional[PipelineConfig] = Field(default=None, description="Pipeline parallelism settings.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_none_strings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize "none"/"None" strings to None for size fields
            for key in ("dp_size", "dp_replicate_size", "ep_size", "pp_size"):
                val = data.get(key)
                if isinstance(val, str) and val.lower() == "none":
                    data[key] = None
        return data


class CheckpointConfig(BaseModel):
    """Checkpointing configuration."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(default=True, description="Enable checkpointing.")
    checkpoint_dir: Optional[str] = Field(default=None, description="Directory to save checkpoints.")
    model_save_format: str = Field(
        default="safetensors",
        description="Checkpoint format: 'safetensors', 'torch_save', or 'dcp'.",
    )
    save_consolidated: bool = Field(
        default=True,
        description="Save a single consolidated checkpoint file (vs sharded).",
    )
    restore_from: Optional[str] = Field(
        default=None,
        description="Path to restore from, or 'LATEST' for auto-detection.",
    )
    best_metric_key: Optional[str] = Field(
        default=None,
        description="Metric key for best-checkpoint tracking (e.g. 'val_loss').",
    )
    async_checkpoint: Optional[bool] = Field(
        default=None,
        description="Enable async checkpoint staging.",
    )


class LrSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    model_config = ConfigDict(extra="allow")

    lr_decay_style: str = Field(
        default="cosine",
        description="LR decay style: 'cosine', 'linear', 'constant', 'inverse-square-root', 'WSD'.",
    )
    lr_warmup_steps: Union[int, float] = Field(
        default=0,
        description="Warmup steps. If < 1, treated as fraction of total steps.",
    )
    lr_decay_steps: Optional[int] = Field(default=None, description="Total decay steps.")
    min_lr: float = Field(default=0.0, ge=0.0, description="Minimum learning rate after decay.")
    init_lr: Optional[float] = Field(default=None, description="Initial LR during warmup (default: 10% of max).")
    wd_incr_style: Optional[str] = Field(default=None, description="Weight decay schedule style.")
    wsd_decay_steps: Optional[int] = Field(default=None, description="WSD-specific decay steps.")
    lr_wsd_decay_style: Optional[str] = Field(default=None, description="WSD decay curve style.")


class ClipGradNormConfig(BaseModel):
    """Gradient clipping configuration."""

    max_norm: float = Field(default=1.0, gt=0.0, description="Max gradient norm for clipping.")


class PackedSequenceConfig(BaseModel):
    """Packed sequence configuration for efficient training."""

    model_config = ConfigDict(extra="allow")

    packed_sequence_size: int = Field(default=0, ge=0, description="Pack size. 0 = disabled. >0 enables packing.")
    packing_strategy: Optional[str] = Field(
        default=None,
        description="Packing strategy: 'thd' or 'neat'.",
    )


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    model_config = ConfigDict(extra="allow")

    project: str = Field(description="WandB project name.")
    entity: Optional[str] = Field(default=None, description="WandB entity (team or user).")
    name: Optional[str] = Field(default=None, description="Run name.")
    save_dir: Optional[str] = Field(default=None, description="Local directory for WandB files.")


class MlflowConfig(BaseModel):
    """MLflow logging configuration."""

    model_config = ConfigDict(extra="allow")

    experiment_name: Optional[str] = Field(default=None, description="MLflow experiment name.")
    run_name: Optional[str] = Field(default=None, description="MLflow run name.")
    tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking server URI.")


class CometConfig(BaseModel):
    """Comet ML logging configuration."""

    model_config = ConfigDict(extra="allow")

    project_name: Optional[str] = Field(default=None, description="Comet project name.")
    workspace: Optional[str] = Field(default=None, description="Comet workspace.")


class Fp8Config(BaseModel):
    """FP8 training configuration."""

    model_config = ConfigDict(extra="allow")

    enabled: Optional[bool] = Field(default=None, description="Enable FP8 training.")


class CompileConfig(BaseModel):
    """torch.compile configuration."""

    model_config = ConfigDict(extra="allow")

    enabled: Optional[bool] = Field(default=None, description="Enable torch.compile.")


class QatConfig(BaseModel):
    """Quantization-aware training configuration."""

    model_config = ConfigDict(extra="allow")

    enabled: Optional[bool] = Field(default=None, description="Enable QAT.")
    fake_quant_after_n_steps: Optional[int] = Field(
        default=None, ge=0, description="Enable fake quantization after N steps."
    )


class MoeMetricsConfig(BaseModel):
    """MoE metrics logging configuration."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(default=False, description="Enable MoE metrics logging.")
    mode: Optional[str] = Field(default=None, description="Metrics mode: 'brief' or 'detailed'.")
    top_k_experts: Optional[int] = Field(default=None, ge=1, description="Top-K experts to report.")
    detailed_every_steps: Optional[int] = Field(default=None, ge=1, description="Detailed metrics interval.")


class BenchmarkConfig(BaseModel):
    """Benchmarking configuration."""

    model_config = ConfigDict(extra="allow")

    warmup_steps: int = Field(default=0, ge=0, description="Warmup steps before timing.")
    peak_tflops: Optional[float] = Field(default=None, description="Reference peak TFLOPs for MFU calculation.")
    nsys_start: Optional[int] = Field(default=None, description="NVIDIA nsys profiling start step.")
    nsys_end: Optional[int] = Field(default=None, description="NVIDIA nsys profiling end step.")
    nsys_ranks: Optional[List[int]] = Field(default=None, description="Ranks to profile with nsys.")
    json_output_path: Optional[str] = Field(default=None, description="Path to save benchmark JSON summary.")


# ---------------------------------------------------------------------------
# Root recipe configs
# ---------------------------------------------------------------------------


class LLMRecipeConfig(BaseModel):
    """Root config schema for LLM training recipes (finetune, pretrain)."""

    model_config = ConfigDict(extra="allow")

    # Structured (fully validated)
    step_scheduler: StepSchedulerConfig
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    dist_env: Optional[DistEnvConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
    lr_scheduler: Optional[LrSchedulerConfig] = None
    clip_grad_norm: Optional[ClipGradNormConfig] = None
    packed_sequence: Optional[PackedSequenceConfig] = None
    wandb: Optional[WandbConfig] = None
    mlflow: Optional[MlflowConfig] = None
    comet: Optional[CometConfig] = None
    fp8: Optional[Fp8Config] = None
    compile: Optional[CompileConfig] = None
    qat: Optional[QatConfig] = None
    moe_metrics: Optional[MoeMetricsConfig] = None
    benchmark: Optional[BenchmarkConfig] = None

    # Dynamic _target_ sections
    model: TargetConfig
    optimizer: TargetConfig
    dataset: TargetConfig
    dataloader: TargetConfig

    # Optional dynamic sections (may or may not have _target_)
    loss_fn: Optional[TargetConfig] = None
    peft: Optional[TargetConfig] = None
    rng: Optional[TargetConfig] = None
    validation_dataset: Optional[TargetConfig] = None
    validation_dataloader: Optional[TargetConfig] = None

    # These sections may be plain dicts without _target_
    quantization: Optional[Dict[str, Any]] = None

    # Scalars
    seed: Optional[int] = Field(default=None, description="Global random seed.")


class VLMRecipeConfig(BaseModel):
    """Root config schema for VLM (vision-language model) finetuning recipes."""

    model_config = ConfigDict(extra="allow")

    # Structured
    step_scheduler: StepSchedulerConfig
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    dist_env: Optional[DistEnvConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
    lr_scheduler: Optional[LrSchedulerConfig] = None
    clip_grad_norm: Optional[ClipGradNormConfig] = None
    wandb: Optional[WandbConfig] = None
    mlflow: Optional[MlflowConfig] = None
    fp8: Optional[Fp8Config] = None
    compile: Optional[CompileConfig] = None

    # Dynamic _target_ sections
    model: TargetConfig
    optimizer: TargetConfig
    dataset: TargetConfig
    dataloader: TargetConfig

    # Optional dynamic sections
    loss_fn: Optional[TargetConfig] = None
    peft: Optional[TargetConfig] = None
    processor: Optional[TargetConfig] = None
    rng: Optional[TargetConfig] = None
    validation_dataset: Optional[TargetConfig] = None
    validation_dataloader: Optional[TargetConfig] = None
    quantization: Optional[Dict[str, Any]] = None

    seed: Optional[int] = None


class KDRecipeConfig(LLMRecipeConfig):
    """Root config schema for knowledge distillation recipes.

    Extends LLMRecipeConfig with teacher model and KD-specific fields.
    """

    teacher_model: TargetConfig
    kd_loss_fn: Optional[TargetConfig] = None
    kd_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for KD loss vs CE loss.")
    offload_teacher_model: bool = Field(default=False, description="Offload teacher to CPU between steps.")


class DiffusionRecipeConfig(BaseModel):
    """Root config schema for diffusion model training recipes."""

    model_config = ConfigDict(extra="allow")

    step_scheduler: StepSchedulerConfig
    checkpoint: Optional[CheckpointConfig] = None
    lr_scheduler: Optional[LrSchedulerConfig] = None
    wandb: Optional[WandbConfig] = None

    # Diffusion uses different top-level keys (plain dicts, not _target_ based)
    model: Dict[str, Any]
    optim: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None

    # Diffusion-specific
    flow_matching: Optional[Dict[str, Any]] = None

    # Distributed (diffusion uses fsdp/ddp directly, not the shared distributed key)
    fsdp: Optional[Dict[str, Any]] = None
    ddp: Optional[Dict[str, Any]] = None

    seed: Optional[int] = None


class BiencoderRecipeConfig(BaseModel):
    """Root config schema for biencoder (retrieval) training recipes."""

    model_config = ConfigDict(extra="allow")

    step_scheduler: StepSchedulerConfig
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    dist_env: Optional[DistEnvConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
    lr_scheduler: Optional[LrSchedulerConfig] = None

    model: TargetConfig
    optimizer: TargetConfig
    tokenizer: TargetConfig
    dataloader: TargetConfig

    peft: Optional[TargetConfig] = None
    wandb: Optional[WandbConfig] = None

    seed: Optional[int] = None
    train_n_passages: Optional[int] = Field(default=None, ge=1, description="Passages per query for training.")
    eval_negative_size: Optional[int] = Field(default=None, ge=0, description="Negative samples for evaluation.")
    temperature: Optional[float] = Field(default=None, gt=0.0, description="Contrastive loss temperature.")


# ---------------------------------------------------------------------------
# Schema registry for CLI --show-schema / --validate
# ---------------------------------------------------------------------------

RECIPE_SCHEMAS: Dict[str, type[BaseModel]] = {
    "llm": LLMRecipeConfig,
    "vlm": VLMRecipeConfig,
    "kd": KDRecipeConfig,
    "diffusion": DiffusionRecipeConfig,
    "biencoder": BiencoderRecipeConfig,
}


def get_schema_json(recipe_type: str) -> str:
    """Export JSON Schema for a recipe type."""
    schema_cls = RECIPE_SCHEMAS.get(recipe_type)
    if schema_cls is None:
        available = ", ".join(sorted(RECIPE_SCHEMAS.keys()))
        raise ValueError(f"Unknown recipe type '{recipe_type}'. Available: {available}")
    return json.dumps(schema_cls.model_json_schema(), indent=2)


def validate_config(raw_dict: dict[str, Any], recipe_type: str = "llm") -> BaseModel:
    """Validate a raw YAML dict against a recipe schema.

    Args:
        raw_dict: The raw dictionary loaded from YAML.
        recipe_type: One of 'llm', 'vlm', 'kd', 'diffusion', 'biencoder'.

    Returns:
        The validated Pydantic model instance.

    Raises:
        pydantic.ValidationError: If the config is invalid.
    """
    schema_cls = RECIPE_SCHEMAS.get(recipe_type)
    if schema_cls is None:
        available = ", ".join(sorted(RECIPE_SCHEMAS.keys()))
        raise ValueError(f"Unknown recipe type '{recipe_type}'. Available: {available}")
    return schema_cls.model_validate(raw_dict)
