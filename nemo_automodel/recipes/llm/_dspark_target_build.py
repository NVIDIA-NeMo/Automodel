# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared builders for the large expert-parallel / FSDP DSpark targets.

DeepSeek V4, GLM-5.2, and Kimi K3 targets are too large to load on a single 8x80GB
node, so both the online training recipe (``train_dspark.py``) and the distributed offline
precompute (``precompute_dspark_dist.py``) load them frozen through the same
expert-parallel / FSDP distributed path: ``create_distributed_setup_from_config``
builds the ``device_mesh`` + ``moe_mesh`` from the recipe's ``distributed:`` block
and ``NeMoAutoModelForCausalLM.from_config(..., load_base_model=True)`` shards the
routed experts across ranks while dequantizing any FP8 base weights on load.

These builders are recipe-level (they depend on ``recipes._dist_utils``), so they
live here rather than under ``components/`` to keep the component layer free of
recipe imports.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import torch

from nemo_automodel import NeMoAutoConfig
from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41DSparkTargetConfig
from nemo_automodel.components.speculative.dspark.common import validate_target_layer_ids
from nemo_automodel.components.speculative.dspark.registry import build_target_layer_ids
from nemo_automodel.components.speculative.dspark.target_utils import (
    DEEPSEEK_V4_MODEL_TYPE,
    DEEPSEEK_V41_MODEL_TYPE,
    GEMMA4_MODEL_TYPES,
    GLM_5_2_MODEL_TYPE,
    KIMI_K3_MODEL_TYPES,
    MINIMAX_M3_MODEL_TYPES,
)
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes.llm._spec_train_utils import read_load_metadata_with_gloo

logger = logging.getLogger(__name__)


def distributed_section_dict(cfg) -> dict:
    """Return the ``distributed:`` section of a top-level recipe config as a plain dict.

    A missing block yields ``{}``, i.e. the default FSDP2 configuration; this keeps the
    fail-loud missing-block contract of ``create_distributed_setup_from_config`` intact
    for callers that require an explicit block.
    """
    section = cfg.get("distributed", None)
    if section is None:
        return {}
    return section.to_dict() if hasattr(section, "to_dict") else dict(section)


def chat_template_kwargs_from(recipe_cfg) -> dict[str, Any] | None:
    """Return ``recipe_args.chat_template_kwargs`` as a plain dict, or ``None`` when unset.

    DSpark training and the distributed precompute both read it here, so the kwargs a cache
    records and the ones a cached run compares them with are read the same way. A value that
    is not a mapping is returned unchanged for the dataset to reject.
    """
    kwargs = recipe_cfg.get("chat_template_kwargs", None)
    return kwargs.to_dict() if hasattr(kwargs, "to_dict") else kwargs


def unsupported_parallel_axes(sizes: dict, axes: tuple[str, ...]) -> dict[str, int]:
    """Return the entries of ``axes`` whose parallelism size is not 1.

    Reads a raw ``distributed:`` section or a ``parse_distributed_section`` result the
    same way: an absent key and an explicit YAML null (``tp_size:``) both mean 1.
    """
    resolved = {axis: int(sizes.get(axis) or 1) for axis in axes}
    return {axis: size for axis, size in resolved.items() if size != 1}


def validate_dspark_parallelism_axes(cfg) -> None:
    """Reject the model-parallel axes the DSpark entrypoints cannot honor.

    Both the training recipe and the distributed precompute replicate their work per
    rank and shard the dataset by rank, so only data-parallel-shaped topologies are
    correct. Two axes break that silently rather than loudly:

    - ``tp_size > 1``: every rank of a tensor-parallel group must run the target on the
      SAME micro-batch, but the sampler hands each rank a different one, so the target's
      sharded attention and MLPs combine activations from unrelated samples.
    - ``pp_size > 1``: the target is loaded as an ``AutoPipeline`` instead of a module,
      which the forward-hook hidden-state capture in ``HFDSparkTargetModel`` cannot run.

    Shard a large target with ``ep_size`` / FSDP instead. Context parallelism has its own
    (supported) path in the training recipe.

    Args:
        cfg: The top-level recipe config.

    Raises:
        NotImplementedError: if ``distributed.tp_size`` or ``distributed.pp_size`` > 1.
    """
    # Read the raw section rather than parse_distributed_section: the parser rejects
    # pp_size>1 without a `pipeline` block before this gate could report the axis.
    unsupported = unsupported_parallel_axes(distributed_section_dict(cfg), ("tp_size", "pp_size"))
    if unsupported:
        raise NotImplementedError(
            f"DSpark does not support {unsupported}: a tensor-parallel target is fed a different "
            f"micro-batch on each rank of its TP group, and a pipelined target is an AutoPipeline "
            f"the forward-hook hidden-state capture cannot run. Set tp_size=pp_size=1 and shard a "
            f"large target with ep_size / FSDP instead."
        )


def gather_full_weight_module(module):
    """Return an object exposing a full (non-DTensor) ``.weight`` tensor.

    The expert-parallel / FSDP-sharded DeepSeek V4 / GLM-5.2 / Kimi K3 targets store
    ``embed_tokens`` and ``lm_head`` weights as DTensors, while a draft (or the
    offline cache) wants plain tensors. Gather the sharded weight to a full tensor
    first (an all-gather, so every rank must call this in lockstep); non-sharded
    targets (Qwen3, Gemma4) pass through unchanged.
    """
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "full_tensor"):
        return SimpleNamespace(weight=weight.full_tensor())
    return module


def resolve_reduced_target_layers(checkpoint_num_layers: int, requested) -> int | None:
    """Validate the optional ``target_num_hidden_layers`` diagnostic override.

    Returns the reduced layer count (an int in ``[1, checkpoint_num_layers]``) or
    ``None`` when unset. Loading fewer layers lets the full EP / hidden-capture
    path run on one node (the full DeepSeek-V4-Flash target OOMs at load on a
    single 8x80GB box); a cache produced against any reduced target is not usable.
    """
    if requested is None:
        return None
    n = int(requested)
    if n < 1 or n > checkpoint_num_layers:
        raise ValueError(
            f"target_num_hidden_layers={n} must be in [1, {checkpoint_num_layers}] (the checkpoint's depth)."
        )
    return n


def repair_glm_5_2_qk_rope_head_dim(target_config, raw_config_dict: dict) -> None:
    """Restore a ``qk_rope_head_dim`` clobbered by the HF ``head_dim`` attribute map.

    The published GLM-5.2 config carries ``head_dim: 192`` (the attention-kernel head
    dim) alongside ``qk_rope_head_dim: 64``, and ``GlmMoeDsaConfig``'s
    ``attribute_map = {"head_dim": "qk_rope_head_dim"}`` lets the former clobber the
    latter on load: the loaded config reports ``qk_rope_head_dim=192``, so
    ``kv_a_proj_with_mqa`` builds ``kv_lora_rank + 192 = 704`` wide while the
    checkpoint ships ``512 + 64 = 576`` and shape validation fails. Restore the raw
    checkpoint value (the GLM finetune examples apply the same correction via a
    ``config: head_dim: 64`` override). No-op when the raw config omits the field or
    the loaded value already matches (e.g. a locally repaired checkpoint view).
    """
    raw_qk_rope = raw_config_dict.get("qk_rope_head_dim")
    if raw_qk_rope is None or int(target_config.qk_rope_head_dim) == int(raw_qk_rope):
        return
    logger.warning(
        "GLM-5.2 config head_dim clobbered qk_rope_head_dim (loaded %s) via the HF attribute_map; "
        "restoring qk_rope_head_dim=%s from the raw checkpoint config.",
        target_config.qk_rope_head_dim,
        raw_qk_rope,
    )
    target_config.qk_rope_head_dim = int(raw_qk_rope)


def build_deepseek_v4_backend(recipe_cfg) -> BackendConfig:
    """Build the V4 target BackendConfig (TileLang attention, hybrid-EP, FP8 adapter).

    Matches the V4 finetune recipe's backend: dense linears and fp32 RMSNorm on
    torch, the ``torch_mm`` grouped-expert GEMM, the hybrid-EP token dispatcher, and
    the HF state-dict adapter that dequantizes the FP8 base checkpoint on load.
    """
    return BackendConfig(
        attn=str(recipe_cfg.get("target_attn_backend", "tilelang")),
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        dispatcher=str(recipe_cfg.get("target_dispatcher", "hybridep")),
        experts=str(recipe_cfg.get("target_experts", "torch_mm")),
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=bool(recipe_cfg.get("target_enable_fsdp_optimizations", True)),
    )


def build_deepseek_v4_target(
    *,
    cfg: Any,
    world_size: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    target_path: str,
    recipe_cfg,
    trust_remote_code: bool,
):
    """Load the full DeepSeek V4 target as a frozen, expert-parallel / FSDP model.

    Mirrors the V4 finetune recipe's model build: an expert-parallel / FSDP
    distributed setup (``device_mesh`` + ``moe_mesh``, derived from ``cfg``'s
    ``distributed`` block) shards the 256 experts across ranks while the
    ``enable_hf_state_dict_adapter`` path dequantizes the FP8 base weights on load.
    The target is MTP-free (``num_nextn_predict_layers=0``) because only hidden
    states are consumed, and is returned frozen for inference.

    Returns ``(target_config, target_model, distributed_setup)``.
    """
    if device.type != "cuda":
        raise RuntimeError(
            "DeepSeek V4 DSpark target requires CUDA: the target is loaded "
            "with the expert-parallel / FSDP distributed path."
        )
    # Build the device_mesh + moe_mesh from the recipe's `distributed` block
    # (strategy, ep_size, moe, ...). NeMoAutoConfig resolves the registered local
    # config; pass the resolved revision to both config and weight loading.
    distributed_setup = create_distributed_setup_from_config(cfg, world_size=world_size)
    # Pass name_or_path explicitly (as the V4 finetune recipe does) so from_config
    # resolves the base checkpoint to load and dequantize from.
    revision = NeMoAutoConfig.resolve_revision(target_path)
    target_config = NeMoAutoConfig.from_pretrained(
        target_path, revision=revision, name_or_path=target_path, num_nextn_predict_layers=0
    )
    # Diagnostic / CI knob: a full 43-layer V4-Flash target dequantizes to
    # ~63 GiB of experts per rank at ep_size=8 and does NOT fit on a single
    # 8x80GB node. Shrinking the layer count loads only the first N layers, so the
    # entire EP / hidden-capture path can be exercised on one node. This is a
    # validation aid: a cache produced against a reduced target is not usable.
    n_reduced = resolve_reduced_target_layers(
        target_config.num_hidden_layers, recipe_cfg.get("target_num_hidden_layers", None)
    )
    if n_reduced is not None:
        logger.warning(
            "Reducing the DeepSeek V4 target from %d to %d layers "
            "(target_num_hidden_layers): diagnostic/CI only, not a usable cache.",
            target_config.num_hidden_layers,
            n_reduced,
        )
        target_config.num_hidden_layers = n_reduced
    backend = build_deepseek_v4_backend(recipe_cfg)
    target_model = NeMoAutoModelForCausalLM.from_config(
        config=target_config,
        backend=backend,
        distributed_setup=distributed_setup,
        load_base_model=True,
        load_metadata_with_gloo=read_load_metadata_with_gloo(cfg.get("checkpoint", None)),
        revision=revision,
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    return target_config, target_model, distributed_setup


def build_glm_5_2_backend(recipe_cfg) -> BackendConfig:
    """Build the GLM-5.2 target BackendConfig (mirrors the GLM finetune recipe).

    Dense linears and fp32 RMSNorm on torch, an fp32 router gate (the GLM/DeepSeek-V3
    top-k routing is fp32 for stability), the ``torch_mm`` grouped-expert GEMM, the
    hybrid-EP token dispatcher, and the HF state-dict adapter that maps (and
    dequantizes, when the base checkpoint is FP8) the HF weights on load. The target
    is frozen / forward-only, so SDPA attention is used by default (the DSA indexer
    emits an additive float bias that SDPA's explicit-mask path accepts);
    ``target_attn_backend=tilelang`` switches to the fused sparse kernels when a
    TileLang build is available. ``target_experts`` defaults to the native ``torch_mm``
    backend.
    """
    return BackendConfig(
        attn=str(recipe_cfg.get("target_attn_backend", "sdpa")),
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        gate_precision="float32",
        dispatcher=str(recipe_cfg.get("target_dispatcher", "hybridep")),
        experts=str(recipe_cfg.get("target_experts", "torch_mm")),
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=bool(recipe_cfg.get("target_enable_fsdp_optimizations", True)),
    )


def build_glm_5_2_target(
    *,
    cfg: Any,
    world_size: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    target_path: str,
    recipe_cfg,
    trust_remote_code: bool,
):
    """Load the full GLM-5.2 target as a frozen, expert-parallel / FSDP model.

    Mirrors :func:`build_deepseek_v4_target`: an expert-parallel / FSDP distributed
    setup (derived from ``cfg``'s ``distributed`` block) shards the 256 routed
    experts across ranks. GLM-5.2's ``model_type`` is registered, so ``AutoConfig``
    resolves it directly (unlike DeepSeek V4), but the model must still be built with
    ``from_config`` + ``load_base_model=True``: ``from_pretrained`` re-reads the
    checkpoint's own config and silently rebuilds the full 78-layer target,
    discarding the ``target_num_hidden_layers`` reduction (which OOMs on one node).

    The forward-hook hidden-state capture needs one non-pipelined ``model(...)``
    call, so ``pp_size`` must be 1 in the recipe's ``distributed:`` block; use a
    larger ``ep_size`` instead of PP to shard the parameter memory.

    Returns ``(target_config, target_model, distributed_setup)``.
    """
    if device.type != "cuda":
        raise RuntimeError(
            "GLM-5.2 DSpark target requires CUDA: the target is loaded "
            "with the expert-parallel / FSDP distributed path."
        )
    revision = NeMoAutoConfig.resolve_revision(target_path)
    target_config = NeMoAutoConfig.from_pretrained(target_path, revision=revision, trust_remote_code=trust_remote_code)
    # The published config's head_dim=192 clobbers qk_rope_head_dim on load via the
    # HF attribute_map, breaking checkpoint shape validation (see the helper).
    raw_config_dict, _ = NeMoAutoConfig.get_config_dict(
        target_path, trust_remote_code=trust_remote_code, revision=revision
    )
    repair_glm_5_2_qk_rope_head_dim(target_config, raw_config_dict)
    n_reduced = resolve_reduced_target_layers(
        target_config.num_hidden_layers,
        recipe_cfg.get("target_num_hidden_layers", None),
    )
    if n_reduced is not None:
        logger.warning(
            "Reducing the GLM-5.2 target from %d to %d layers "
            "(target_num_hidden_layers): diagnostic/CI only, not a usable cache.",
            target_config.num_hidden_layers,
            n_reduced,
        )
        target_config.num_hidden_layers = n_reduced
    distributed_setup = create_distributed_setup_from_config(cfg, world_size=world_size)
    backend = build_glm_5_2_backend(recipe_cfg)
    target_model = NeMoAutoModelForCausalLM.from_config(
        config=target_config,
        backend=backend,
        distributed_setup=distributed_setup,
        load_base_model=True,
        load_metadata_with_gloo=read_load_metadata_with_gloo(cfg.get("checkpoint", None)),
        revision=revision,
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    return target_config, target_model, distributed_setup


def build_kimi_k3_backend(recipe_cfg) -> BackendConfig:
    """Build the backend used by the frozen expert-parallel Kimi K3 target.

    Mirrors :func:`build_glm_5_2_backend`: the hybrid-EP token dispatcher shards the
    routed experts and the HF state-dict adapter maps (and dequantizes, when the base
    checkpoint is FP8) the HF weights on load. ``target_experts`` defaults to the
    native ``torch_mm`` backend.

    Two fields differ from the GLM backend by design:

    - ``attn`` defaults to ``eager`` and is inert for this target: ``KimiK3ForCausalLM``
      never reads ``backend.attn`` (its MLA and KDA layers each have a fixed attention
      path), so the field only records that no fused-kernel build is required.
    - no ``gate_precision`` override is passed, because ``KimiK3ForCausalLM`` already
      defaults ``backend.gate_precision`` to ``torch.float32`` itself.
    """
    return BackendConfig(
        attn=str(recipe_cfg.get("target_attn_backend", "eager")),
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        dispatcher=str(recipe_cfg.get("target_dispatcher", "hybridep")),
        experts=str(recipe_cfg.get("target_experts", "torch_mm")),
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=bool(recipe_cfg.get("target_enable_fsdp_optimizations", True)),
    )


def build_kimi_k3_target(
    *,
    cfg: Any,
    world_size: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    target_path: str,
    recipe_cfg,
    trust_remote_code: bool,
):
    """Load Kimi K3 as a frozen, expert-parallel / FSDP target for online DSpark training.

    Mirrors :func:`build_glm_5_2_target`: an expert-parallel / FSDP distributed setup
    (derived from ``cfg``'s ``distributed`` block) shards the routed experts across
    ranks, and the model is built with ``from_config`` + ``load_base_model=True`` so the
    ``target_num_hidden_layers`` reduction survives (``from_pretrained`` would re-read
    the checkpoint's own config and rebuild the full-depth target, which OOMs on one
    node). Unlike GLM, the config may be the multimodal ``kimi_k3`` wrapper, so the text
    sub-config is unwrapped and re-tagged as ``KimiK3ForCausalLM``: DSpark captures
    hidden states from the text backbone only.

    The forward-hook hidden-state capture needs one non-pipelined ``model(...)`` call,
    so ``pp_size`` must be 1 in the recipe's ``distributed:`` block; use a larger
    ``ep_size`` instead of PP to shard the parameter memory.

    Returns ``(text_config, target_model, distributed_setup)``.
    """
    if device.type != "cuda":
        raise RuntimeError(
            "Kimi K3 DSpark target requires CUDA: the target is loaded with the "
            "expert-parallel / FSDP distributed path."
        )
    revision = NeMoAutoConfig.resolve_revision(target_path)
    target_config = NeMoAutoConfig.from_pretrained(target_path, revision=revision, trust_remote_code=trust_remote_code)
    text_config = getattr(target_config, "text_config", target_config)
    n_reduced = resolve_reduced_target_layers(
        text_config.num_hidden_layers,
        recipe_cfg.get("target_num_hidden_layers", None),
    )
    if n_reduced is not None:
        logger.warning(
            "Reducing the Kimi K3 target from %d to %d layers "
            "(target_num_hidden_layers): diagnostic/CI only, not a usable cache.",
            text_config.num_hidden_layers,
            n_reduced,
        )
        text_config.num_hidden_layers = n_reduced
        text_config.linear_attn_config = {
            **text_config.linear_attn_config,
            "kda_layers": [
                layer_idx for layer_idx in text_config.linear_attn_config["kda_layers"] if layer_idx <= n_reduced
            ],
            "full_attn_layers": [
                layer_idx for layer_idx in text_config.linear_attn_config["full_attn_layers"] if layer_idx <= n_reduced
            ],
        }
    text_config.architectures = ["KimiK3ForCausalLM"]
    text_config.name_or_path = target_path
    distributed_setup = create_distributed_setup_from_config(cfg, world_size=world_size)
    target_model = NeMoAutoModelForCausalLM.from_config(
        config=text_config,
        backend=build_kimi_k3_backend(recipe_cfg),
        distributed_setup=distributed_setup,
        load_base_model=True,
        load_metadata_with_gloo=read_load_metadata_with_gloo(cfg.get("checkpoint", None)),
        revision=revision,
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    return text_config, target_model, distributed_setup


def build_dspark_target(
    *,
    cfg,
    recipe_cfg,
    world_size: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    model_type: str,
    target_path: str,
    trust_remote_code: bool,
) -> tuple[Any, torch.nn.Module]:
    """Build the frozen DSpark target for capture outside training, dispatching on model type.

    Used by the entry points that run the target without a draft: the distributed
    offline precompute and the remote target server. DeepSeek V4/V4.1, GLM-5.2, and
    Kimi K3 load through the sharded EP/FSDP path; other text targets (Qwen3, Gemma4)
    load whole on ``device``. MiniMax M3 is rejected because its features need image
    inputs, which these entry points do not carry.

    Returns:
        ``(target_text_config, target_model)``, where ``target_text_config`` is the
        config holding the text stack's ``num_hidden_layers``, ``hidden_size``, and
        ``vocab_size``.
    """
    if model_type in MINIMAX_M3_MODEL_TYPES:
        raise ValueError(
            "MiniMax M3 is a multimodal target; DSpark offline precompute and remote target serving "
            "are text-only. Use the colocated online MiniMax M3 training path instead."
        )
    if model_type == DEEPSEEK_V4_MODEL_TYPE:
        target_config, target_model, _ = build_deepseek_v4_target(
            cfg=cfg,
            world_size=world_size,
            device=device,
            compute_dtype=compute_dtype,
            target_path=target_path,
            recipe_cfg=recipe_cfg,
            trust_remote_code=trust_remote_code,
        )
        return target_config, target_model
    if model_type == DEEPSEEK_V41_MODEL_TYPE:
        target_options = DeepseekV41DSparkTargetConfig(
            target_path=target_path,
            trust_remote_code=trust_remote_code,
            target_num_hidden_layers=recipe_cfg.get("target_num_hidden_layers", None),
        )
        target_options.attn_backend = str(recipe_cfg.get("target_attn_backend", target_options.attn_backend))
        target_options.dispatcher = str(recipe_cfg.get("target_dispatcher", target_options.dispatcher))
        target_options.experts = str(recipe_cfg.get("target_experts", target_options.experts))
        target_options.enable_fsdp_optimizations = bool(
            recipe_cfg.get("target_enable_fsdp_optimizations", target_options.enable_fsdp_optimizations)
        )
        distributed_setup = create_distributed_setup_from_config(cfg, world_size=world_size)
        target_model = target_options.build(
            device=device,
            compute_dtype=compute_dtype,
            distributed_setup=distributed_setup,
        )
        return target_model.config.text_config, target_model
    if model_type == GLM_5_2_MODEL_TYPE:
        target_config, target_model, _ = build_glm_5_2_target(
            cfg=cfg,
            world_size=world_size,
            device=device,
            compute_dtype=compute_dtype,
            target_path=target_path,
            recipe_cfg=recipe_cfg,
            trust_remote_code=trust_remote_code,
        )
        return target_config, target_model
    if model_type in KIMI_K3_MODEL_TYPES:
        target_config, target_model, _ = build_kimi_k3_target(
            cfg=cfg,
            world_size=world_size,
            device=device,
            compute_dtype=compute_dtype,
            target_path=target_path,
            recipe_cfg=recipe_cfg,
            trust_remote_code=trust_remote_code,
        )
        return target_config, target_model
    # Single-process text target (Qwen3, Gemma4): every process holds a full copy.
    target_kwargs = {}
    target_attn_implementation = recipe_cfg.get("target_attn_implementation", None)
    if target_attn_implementation is not None:
        target_kwargs["attn_implementation"] = target_attn_implementation
    target_model = NeMoAutoModelForCausalLM.from_pretrained(
        target_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=compute_dtype,
        force_hf=bool(recipe_cfg.get("target_force_hf", False)),
        **target_kwargs,
    ).to(device)
    text_config = target_model.config.text_config if model_type in GEMMA4_MODEL_TYPES else target_model.config
    return text_config, target_model


def resolve_dspark_target_layer_ids(recipe_cfg, num_target_layers: int) -> list[int]:
    """Return ``recipe_args.target_layer_ids``, or the default spread for the draft depth, validated."""
    target_layer_ids = list(
        recipe_cfg.get("target_layer_ids", None)
        or build_target_layer_ids(num_target_layers, int(recipe_cfg.get("draft_num_hidden_layers", 5)))
    )
    return validate_target_layer_ids(target_layer_ids, num_target_layers)


__all__ = [
    "build_deepseek_v4_backend",
    "build_deepseek_v4_target",
    "build_dspark_target",
    "build_glm_5_2_backend",
    "build_glm_5_2_target",
    "build_kimi_k3_backend",
    "build_kimi_k3_target",
    "distributed_section_dict",
    "gather_full_weight_module",
    "repair_glm_5_2_qk_rope_head_dim",
    "resolve_dspark_target_layer_ids",
    "resolve_reduced_target_layers",
    "unsupported_parallel_axes",
    "validate_dspark_parallelism_axes",
]
