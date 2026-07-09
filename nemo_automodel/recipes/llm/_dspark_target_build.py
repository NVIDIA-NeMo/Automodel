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

DeepSeek V4 and GLM-5.2 targets are too large to load on a single 8x80GB node, so
both the online training recipe (``train_dspark.py``) and the distributed offline
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
from transformers import AutoConfig, PretrainedConfig

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config

logger = logging.getLogger(__name__)


def gather_full_weight_module(module):
    """Return an object exposing a full (non-DTensor) ``.weight`` tensor.

    The expert-parallel / FSDP-sharded DeepSeek V4 / GLM-5.2 targets store
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
    # (strategy, ep_size, moe, ...). DeepseekV4Config.from_pretrained is used
    # because the custom V4 model_type is not registered with stock AutoConfig.
    distributed_setup = create_distributed_setup_from_config(cfg, world_size=world_size)
    # Pass name_or_path explicitly (as the V4 finetune recipe does) so from_config
    # resolves the base checkpoint to load and dequantize from.
    target_config = DeepseekV4Config.from_pretrained(target_path, name_or_path=target_path, num_nextn_predict_layers=0)
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
    TileLang build is available. ``target_experts`` defaults to ``torch_mm`` (like
    the V4 DSpark backend) because ``gmm`` needs the optional ``grouped_gemm``
    package, which the current AutoModel image does not ship.
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
    target_config = AutoConfig.from_pretrained(target_path, trust_remote_code=trust_remote_code)
    # The published config's head_dim=192 clobbers qk_rope_head_dim on load via the
    # HF attribute_map, breaking checkpoint shape validation (see the helper).
    raw_config_dict, _ = PretrainedConfig.get_config_dict(target_path, trust_remote_code=trust_remote_code)
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
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    return target_config, target_model, distributed_setup


__all__ = [
    "build_deepseek_v4_backend",
    "build_deepseek_v4_target",
    "build_glm_5_2_backend",
    "build_glm_5_2_target",
    "gather_full_weight_module",
    "repair_glm_5_2_qk_rope_head_dim",
    "resolve_reduced_target_layers",
]
