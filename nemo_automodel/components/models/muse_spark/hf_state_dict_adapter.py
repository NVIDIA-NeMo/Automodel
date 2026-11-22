# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint adapter for the nested Hugging Face Muse Spark Open release."""

from __future__ import annotations

import re

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate

from nemo_automodel.components.models.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter
from nemo_automodel.components.moe import state_dict_utils

_TEXT_SUFFIXES = {
    "self_attn.pre_qkv_proj_norm.weight": "input_layernorm.weight",
    "self_attn.o_gate_proj.weight": "self_attn.gate_proj.weight",
    "feed_forward.experts.gate_and_up_projs": "mlp.experts.gate_up_proj",
    "feed_forward.experts.down_projs": "mlp.experts.down_proj",
    "feed_forward.post_expert_norm.weight": "mlp.experts.post_expert_norm.weight",
    "feed_forward.pre_expert_norm.weight": "mlp.pre_expert_norm.weight",
    "feed_forward.fc1_latent_proj.weight": "mlp.pre_expert_proj.weight",
    "feed_forward.fc2_latent_proj.weight": "mlp.post_expert_proj.weight",
    "feed_forward.gate.weight": "mlp.gate.weight",
    "feed_forward.gate.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "pre_ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffn_norm.weight": "post_feedforward_layernorm.weight",
    "add_attn.gate": "post_attention_residual_gate.gate",
    "add_ffn.gate": "post_feedforward_residual_gate.gate",
    **{f"self_attn.{proj}_proj.weight": f"self_attn.{proj}_proj.weight" for proj in ("q", "k", "v", "o")},
}
_VISION_SUFFIXES = {
    **{
        f"attn.{proj}_proj.{part}": f"attn.{proj}_proj.{part}"
        for proj in ("q", "k", "v")
        for part in ("weight", "bias")
    },
    **{
        f"{native}.{part}": f"{hf}.{part}"
        for native, hf in (
            ("attn.out_proj", "attn.proj"),
            ("ln_1", "norm1"),
            ("ln_2", "norm2"),
            ("mlp.c_fc", "mlp.fc1"),
            ("mlp.c_proj", "mlp.fc2"),
        )
        for part in ("weight", "bias")
    },
}
_EXACT_KEYS = {
    "lm_head.weight": "lm_head.weight",
    "model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
    "model.norm.weight": "model.language_model.norm.weight",
    "vision_model.conv1_linear.weight": "model.vision_tower.patch_embedder.patch_embedding.weight",
    "vision_model.positional_embedding_vlm": "model.vision_tower.patch_embedder.position_embedding_table.weight",
    "vision_model.adapter.c_fc.weight": "model.vision_adapter.fc1.weight",
    "vision_model.adapter.c_proj.weight": "model.vision_adapter.fc2.weight",
    "vision_projection.weight": "model.vision_projection.weight",
    **{
        f"vision_model.{norm}.{part}": f"model.vision_tower.{norm}.{part}"
        for norm in ("ln_pre", "ln_post")
        for part in ("weight", "bias")
    },
    "audio_encoder.norm.weight": "model.audio_tower.norm.weight",
    "audio_encoder.dual_norm.weight": "model.audio_tower.post_proj_norm.weight",
    **{f"audio_encoder.o_proj.{part}": f"model.audio_tower.proj.{part}" for part in ("weight", "bias")},
}
_AUDIO_BLOCK_SUFFIXES = {
    **{
        f"{native}.{part}": f"{hf}.{part}"
        for native, hf in (("conv.conv", "conv"), ("linear1", "mlp.fc1"), ("linear2", "mlp.fc2"))
        for part in ("weight", "bias")
    },
    "norm.weight": "norm.weight",
}
_HF_TEXT_SUFFIXES = {value: key for key, value in _TEXT_SUFFIXES.items()}
_HF_VISION_SUFFIXES = {value: key for key, value in _VISION_SUFFIXES.items()}
_HF_EXACT_KEYS = {value: key for key, value in _EXACT_KEYS.items()}
_HF_AUDIO_BLOCK_SUFFIXES = {value: key for key, value in _AUDIO_BLOCK_SUFFIXES.items()}


def _native_key_to_hf(key: str) -> str | None:
    """Map one native text, vision, or audio state name to the HF release."""
    # Adapter factors are native-format checkpoint state. In particular Q/K
    # LoRA B rows must stay in native RoPE order when saving a native adapter.
    if ".lora_" in key:
        return key
    if key == "model.rotary_emb.inv_freq" or key.endswith("_extra_state"):
        return None
    if key in _EXACT_KEYS:
        return _EXACT_KEYS[key]
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", key)
    if match and match[2] in _TEXT_SUFFIXES:
        return f"model.language_model.layers.{match[1]}." + _TEXT_SUFFIXES[match[2]]
    match = re.fullmatch(r"vision_model\.encoder\.layers\.(\d+)\.(.+)", key)
    if match and match[2] in _VISION_SUFFIXES:
        return f"model.vision_tower.layers.{match[1]}." + _VISION_SUFFIXES[match[2]]
    match = re.fullmatch(r"audio_encoder\.stages\.(\d+)\.(\d+)\.(.+)", key)
    if match:
        stage, block, suffix = match.groups()
        if block == "0" and suffix in ("linear.weight", "linear.bias", "norm.weight"):
            return f"model.audio_tower.layers.{stage}.patch_embedder." + suffix.replace("linear.", "proj.")
        if int(block) > 0 and suffix in _AUDIO_BLOCK_SUFFIXES:
            return f"model.audio_tower.layers.{stage}.conv_blocks.{int(block) - 1}." + _AUDIO_BLOCK_SUFFIXES[suffix]
    raise ValueError(f"Unmapped native Muse Spark Open checkpoint tensor: {key}")


def _hf_key_to_native(key: str) -> str:
    """Map one HF release name to the native model hierarchy."""
    if ".lora_" in key:
        return key
    if key in _HF_EXACT_KEYS:
        return _HF_EXACT_KEYS[key]
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", key)
    if match and match[2] in _HF_TEXT_SUFFIXES:
        return f"model.layers.{match[1]}." + _HF_TEXT_SUFFIXES[match[2]]
    match = re.fullmatch(r"model\.vision_tower\.layers\.(\d+)\.(.+)", key)
    if match and match[2] in _HF_VISION_SUFFIXES:
        return f"vision_model.encoder.layers.{match[1]}." + _HF_VISION_SUFFIXES[match[2]]
    match = re.fullmatch(r"model\.audio_tower\.layers\.(\d+)\.patch_embedder\.(proj\.(?:weight|bias)|norm.weight)", key)
    if match:
        return f"audio_encoder.stages.{match[1]}.0." + match[2].replace("proj.", "linear.")
    match = re.fullmatch(r"model\.audio_tower\.layers\.(\d+)\.conv_blocks\.(\d+)\.(.+)", key)
    if match and match[3] in _HF_AUDIO_BLOCK_SUFFIXES:
        return f"audio_encoder.stages.{match[1]}.{int(match[2]) + 1}." + _HF_AUDIO_BLOCK_SUFFIXES[match[3]]
    raise ValueError(f"Unmapped HF Muse Spark Open checkpoint tensor: {key}")


def _permute_rope_rows(tensor: torch.Tensor, head_dim: int, *, to_hf: bool) -> torch.Tensor:
    """Convert adjacent-pair and split-half RoPE projection rows.

    Args:
        tensor: Projection [heads * head_dim, hidden] or bias [heads * head_dim].
            A DTensor retains its global shape and arbitrary Shard/Replicate
            placements. Rank-local row fragments need not span complete heads.
        head_dim: Width of a complete attention head.
        to_hf: Convert adjacent pairs to split halves when true; invert otherwise.

    Returns:
        Independent contiguous tensor with the input shape, dtype and device.
        DTensor output preserves the original mesh and placements. When a head
        crosses ranks, a single projection is gathered temporarily, permuted,
        and resharded; expert tensors never use this path.
    """
    if head_dim % 2 or tensor.shape[0] % head_dim:
        raise ValueError("Muse Spark Open Q/K rows must contain complete even-width heads")
    if tensor.is_meta:
        return torch.empty_like(tensor)
    original = tensor
    if isinstance(tensor, DTensor):
        if any(isinstance(placement, Partial) for placement in tensor.placements):
            raise ValueError("Muse Spark Open checkpoint parameters cannot have Partial placements")
        local = tensor.to_local()
        # Decide from global metadata so empty or uneven shards still enter
        # the same collectives on every rank.
        rows = tensor.shape[0]
        fragments_head = False
        for axis, placement in enumerate(tensor.placements):
            if placement.is_shard(0):
                size = tensor.device_mesh.size(axis)
                rows = (rows + size - 1) // size
                fragments_head |= rows % head_dim != 0
        if fragments_head:
            tensor = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim)
            local = tensor.to_local()
    else:
        local = tensor
    shape = (-1, head_dim // 2, 2, *local.shape[1:]) if to_hf else (-1, 2, head_dim // 2, *local.shape[1:])
    converted = local.reshape(shape).transpose(1, 2).reshape_as(local).contiguous()
    if isinstance(tensor, DTensor):
        converted = DTensor.from_local(
            converted,
            tensor.device_mesh,
            tensor.placements,
            shape=tensor.shape,
            stride=tensor.stride(),
            run_check=False,
        )
        if tensor.placements != original.placements:
            converted = converted.redistribute(placements=original.placements)
    return converted


class MuseSparkHFStateDictAdapter(MuseSparkStateDictAdapter):
    """Load the HF release through rank-local, predominantly direct DCP storage.

    Expert transposes are views into final native storage, including without EP.
    Only Q/K row permutations allocate, bounded by the Q/K projections owned by
    this stage and at most one gathered projection during conversion. No full
    expert tensor is rebuilt after a direct load.
    """

    @property
    def supports_low_memory_dcp_load(self) -> bool:
        """Enable direct destinations only for the supported dense expert layouts."""
        return self.backend.experts in ("torch_mm", "gmm")

    def _rope_head_dim(self, native_key: str) -> int | None:
        """Return the head width for text/vision Q/K weights and vision Q/K biases."""
        if native_key.startswith("model.layers.") and native_key.endswith(
            ("self_attn.q_proj.weight", "self_attn.k_proj.weight")
        ):
            return self.config.head_dim
        if native_key.startswith("vision_model.encoder.layers.") and native_key.endswith(
            ("attn.q_proj.weight", "attn.k_proj.weight", "attn.q_proj.bias", "attn.k_proj.bias")
        ):
            vision = self.config.vision_navit_config
            return int(vision["hidden_size"]) // int(vision["num_attention_heads"])
        return None

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: torch.Tensor, **kwargs: str | bool | None
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert one native tensor into an HF checkpoint tensor.

        Args:
            fqn: Native state name.
            tensor: Expert [experts, in, out], router [experts, hidden],
                projection [out, in], bias/norm [out], or embedding [vocab, hidden].
                DTensors retain their global shapes and parameter placements.
            **kwargs: Checkpointer's exclude_key_regex and for_checkpoint_load.

        Returns:
            Zero or one (HF name, tensor) pairs. Expert [experts, out, in]
            outputs alias input storage. Unchanged tensors alias the input.
            Q/K outputs own bounded temporary storage; load destinations are
            empty and must be filled before from_hf restores native row order.
        """
        key = _native_key_to_hf(fqn)
        exclude = kwargs.get("exclude_key_regex")
        if key is None or (exclude is not None and re.search(exclude, key)):
            return []
        head_dim = self._rope_head_dim(fqn)
        if head_dim is not None:
            tensor = (
                torch.empty_like(tensor)
                if kwargs.get("for_checkpoint_load", False)
                else _permute_rope_rows(tensor, head_dim, to_hf=True)
            )
        elif fqn.endswith((".experts.gate_and_up_projs", ".experts.down_projs")):
            tensor = tensor.transpose(-1, -2)
        return [(key, tensor)]

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        **kwargs: str | bool | None,
    ) -> dict[str, torch.Tensor]:
        """Convert HF text, vision and audio tensors to the native checkpoint.

        Args:
            hf_state_dict: HF names mapped to experts [experts, out, in],
                routers [experts, hidden], projections [out, in], vectors [out],
                and embeddings [vocab, hidden]. DCP destinations may be strided
                model-storage views or DTensors with global shapes.
            device_mesh: Optional EP mesh used to slice unsharded expert input.
                Already-sharded DTensor experts preserve their existing mesh.
            **kwargs: Unused shared checkpoint adapter arguments.

        Returns:
            Native names and tensor layouts. Expert transposes and unchanged
            values alias their inputs when dtype conversion is unnecessary.
            Q/K permutations and regenerated FP32 rotary frequencies own storage.
        """
        del kwargs
        start, end, rank = 0, self.moe_config.n_routed_experts, None
        if device_mesh is not None:
            start, end = state_dict_utils.get_expert_range_for_rank_from_mesh(
                device_mesh, self.moe_config.n_routed_experts
            )
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if ("ep" in device_mesh.mesh_dim_names)
                else device_mesh.get_rank()
            )
        converted = {}
        for key, value in hf_state_dict.items():
            layer = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
            if layer and int(layer[1]) >= self.config.num_hidden_layers:
                continue
            native = _hf_key_to_native(key)
            head_dim = self._rope_head_dim(native)
            if head_dim is not None:
                value = _permute_rope_rows(value, head_dim, to_hf=False)
            elif native.endswith((".experts.gate_and_up_projs", ".experts.down_projs")):
                if isinstance(value, DTensor):
                    value = value.transpose(-1, -2)
                else:
                    value = value[start:end].transpose(-1, -2).to(self.dtype)
                    if device_mesh is not None:
                        value = state_dict_utils.create_dtensor_from_local(value, device_mesh, rank)
            elif native.endswith((".feed_forward.gate.weight", ".feed_forward.gate.e_score_correction_bias")):
                value = value.to(torch.float32)
            converted[native] = value
        positions = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32)
        converted["model.rotary_emb.inv_freq"] = 1.0 / (self.config.rope_theta ** (positions / self.config.head_dim))
        return converted
