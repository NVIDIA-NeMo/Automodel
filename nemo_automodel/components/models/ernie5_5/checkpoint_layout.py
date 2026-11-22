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


"""Name tables and eager conversions between the released ERNIE 5.5 checkpoint and the native modules.

Every conversion translates complete, unsharded tensors between the released Hugging Face layout and the native
module layout, one component at a time: the model-level tensors (token embedding, final norm, output head,
MultiMax and hyper-head), one hyper-connection (mHC) unit, one HCA attention layer, one DSA attention layer, one
MoE feed-forward, one decoder layer and the whole decoder-only model. The name tables are shared with the
distributed destinations and export views of ``checkpoint_io``.

Memory. A conversion to the released layout returns views: the per-expert matrices and the hyper-head matrix
are transposed views of the native storage, and the only tensor it allocates is the fused K/V projection of each
DSA layer. A conversion to the native layout packs every routed expert of a layer into new grouped tensors.

Limitations:
    * The conversions accept only complete, plain tensors. ``map_model_from_hf`` holds a second copy of every
      routed expert, so it suits small checkpoints and shape-only tensors on the meta device, not the released
      checkpoint of several hundred GB. The conversions do not shard, stream from disk or export incrementally.
    * MTP weights are not mapped. The whole-model conversions raise for a config with MTP layers and reject any
      key outside the decoder layers and the model-level tensors.
    * The decoder-layer conversions name the feed-forward ``feed_forward.*`` for both kinds, while the model
      registers it as ``mlp.*`` (dense) or ``moe.*`` (MoE); ``Ernie5_5StateDictAdapter`` renames it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config, Ernie5_5LayerPlan, Ernie5_5ModelPlan

# Name prefixes: selecting the entries of one component and putting converted entries back.


def _pop_prefix(remaining: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Remove every entry whose name starts with ``prefix`` and return them without the prefix.

    Args:
        remaining: Names mapped to tensors of any shape. Matching entries are removed in place.
        prefix: Name prefix to select and strip.

    Returns:
        The removed entries keyed by the rest of their names, in their original order. Tensors are the
        input objects.
    """
    keys = [key for key in remaining if key.startswith(prefix)]
    return {key[len(prefix) :]: remaining.pop(key) for key in keys}


def _put_prefix(result: dict[str, torch.Tensor], prefix: str, state: Mapping[str, torch.Tensor]) -> None:
    """Add every entry of ``state`` to ``result`` under ``prefix + name``.

    Args:
        result: Names mapped to tensors; updated in place.
        prefix: Prefix prepended to every name of ``state``.
        state: Names mapped to tensors of any shape. Tensors are stored as given.
    """
    result.update({prefix + key: value for key, value in state.items()})


# Model-level tensors: token embedding, final norm, output head, MultiMax and hyper-head.


def common_tensor_specs(*, vocab: int, hidden: int, streams: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return the eight model-level tensors as ``native name -> (Hugging Face name, native shape)``.

    The hyper-head matrix is the only one whose released layout differs from the native layout: the
    checkpoint stores ``hc_head_fn`` as [streams * hidden, streams], the transpose of ``hc_fn``.

    Args:
        vocab: Vocabulary size.
        hidden: Hidden size.
        streams: Number of residual streams.

    Returns:
        Native names, relative to the inner model, mapped to the Hugging Face name and the native shape.
        The order is the order in which destinations and converted tensors are produced.
    """
    return {
        "embed_tokens.weight": ("model.embed_tokens.weight", (vocab, hidden)),
        "lm_head.weight": ("lm_head.weight", (vocab, hidden)),
        "norm.weight": ("model.norm.weight", (hidden,)),
        "multimax.multimax_ranges": ("model.lm_head.multimax_ranges", (4,)),
        "multimax.multimax_ts": ("model.lm_head.multimax_ts", (4,)),
        "hyper_head.hc_fn": ("model.mhc_contract.hc_head_fn", (streams, streams * hidden)),
        "hyper_head.hc_base": ("model.mhc_contract.hc_head_base", (streams,)),
        "hyper_head.hc_scale": ("model.mhc_contract.hc_head_scale", (1,)),
    }


# The names do not depend on the dimensions; unit dimensions are used only to read the names.
_COMMON_NAMES = {native: hf for native, (hf, _shape) in common_tensor_specs(vocab=1, hidden=1, streams=1).items()}
COMMON_KEYS = frozenset(_COMMON_NAMES)
_HEAD_PREFIX = "model.mhc_contract."
_DIRECT = {hf: native for native, hf in _COMMON_NAMES.items() if not native.startswith("hyper_head.")}
_HEAD_KEYS = {hf.removeprefix(_HEAD_PREFIX) for native, hf in _COMMON_NAMES.items() if native.startswith("hyper_head.")}


def _require_common_keys(state: Mapping[str, torch.Tensor], expected: set[str], description: str) -> None:
    """Require exactly the names of the eight model-level tensors.

    Args:
        state: Names mapped to tensors of any shape.
        expected: The eight names that must be present, with no others.
        description: What the names are, for the message.

    Raises:
        ValueError: If a name is missing or unexpected. The message lists the missing names and the first five
            unexpected ones; the keys of an MTP block, which a decoder-only conversion does not map, end here.
    """
    missing, unexpected = sorted(expected - state.keys()), sorted(state.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"Expected exactly the {description} keys: missing={missing}, "
            f"unexpected={unexpected[:5]} ({len(unexpected)} in all)"
        )


def _validate_direct(state: Mapping[str, torch.Tensor], *, vocab: int, hidden: int) -> None:
    """Check the shapes of the five model-level tensors that keep their released layout.

    Args:
        state: Mapping that contains the Hugging Face names ``model.embed_tokens.weight`` [vocab, hidden],
            ``lm_head.weight`` [vocab, hidden], ``model.norm.weight`` [hidden],
            ``model.lm_head.multimax_ranges`` [4] and ``model.lm_head.multimax_ts`` [4]. Other entries are
            ignored.
        vocab: Vocabulary size.
        hidden: Hidden size.

    Raises:
        ValueError: If ``vocab`` or ``hidden`` is not positive or one of the tensors has another shape.
    """
    if vocab < 1 or hidden < 1:
        raise ValueError(f"Vocabulary and hidden dimensions must be positive; got {vocab} and {hidden}")
    # The five shapes checked here do not depend on the stream count.
    specs = common_tensor_specs(vocab=vocab, hidden=hidden, streams=1)
    expected_shapes = {hf: shape for hf, shape in specs.values() if hf in _DIRECT}
    for name, shape in expected_shapes.items():
        if state[name].shape != shape:
            raise ValueError(f"{name} has shape {tuple(state[name].shape)}; expected {shape}")


def map_common_from_hf(
    state: Mapping[str, torch.Tensor], *, vocab: int, hidden: int, streams: int
) -> dict[str, torch.Tensor]:
    """Map the eight model-level Hugging Face tensors to native names.

    Args:
        state: Exactly the eight released tensors ``model.embed_tokens.weight`` [vocab, hidden],
            ``lm_head.weight`` [vocab, hidden], ``model.norm.weight`` [hidden],
            ``model.lm_head.multimax_ranges`` [4], ``model.lm_head.multimax_ts`` [4],
            ``model.mhc_contract.hc_head_fn`` [streams * hidden, streams],
            ``model.mhc_contract.hc_head_base`` [streams] and ``model.mhc_contract.hc_head_scale`` [1].
            The three hyper-head tensors must be FP32; the dtype of the others is not checked.
        vocab: Vocabulary size.
        hidden: Hidden size.
        streams: Number of residual streams.

    Returns:
        Native tensors named relative to the inner model: ``embed_tokens.weight`` [vocab, hidden],
        ``lm_head.weight`` [vocab, hidden], ``norm.weight`` [hidden], ``multimax.multimax_ranges`` [4],
        ``multimax.multimax_ts`` [4], ``hyper_head.hc_fn`` [streams, streams * hidden],
        ``hyper_head.hc_base`` [streams] and ``hyper_head.hc_scale`` [1]. ``hyper_head.hc_fn`` is the
        transpose of the released matrix, made contiguous; every other tensor is the input object.

    Raises:
        ValueError: If the key set differs from the eight names, a dimension is not positive, a shape does
            not match or a hyper-head tensor is not FP32.
    """
    expected = set(_DIRECT) | {_HEAD_PREFIX + key for key in _HEAD_KEYS}
    _require_common_keys(state, expected, "common embedding/norm/head")
    _validate_direct(state, vocab=vocab, hidden=hidden)
    head = map_hyper_head_from_hf(
        {name: state[_HEAD_PREFIX + name] for name in _HEAD_KEYS}, streams=streams, hidden=hidden
    )
    return {**{native: state[hf] for hf, native in _DIRECT.items()}, **{f"hyper_head.{k}": v for k, v in head.items()}}


def map_common_to_hf(
    state: Mapping[str, torch.Tensor], *, vocab: int, hidden: int, streams: int
) -> dict[str, torch.Tensor]:
    """Restore the eight model-level Hugging Face tensors from their native names.

    Args:
        state: Exactly the eight native tensors, named relative to the inner model:
            ``embed_tokens.weight`` [vocab, hidden], ``lm_head.weight`` [vocab, hidden], ``norm.weight``
            [hidden], ``multimax.multimax_ranges`` [4], ``multimax.multimax_ts`` [4], ``hyper_head.hc_fn``
            [streams, streams * hidden], ``hyper_head.hc_base`` [streams] and ``hyper_head.hc_scale`` [1].
            The three hyper-head tensors must be FP32; the dtype of the others is not checked.
        vocab: Vocabulary size.
        hidden: Hidden size.
        streams: Number of residual streams.

    Returns:
        The released tensors in the layouts listed for ``map_common_from_hf``.
        ``model.mhc_contract.hc_head_fn`` [streams * hidden, streams] is a transposed view of the native
        matrix; every other tensor is the input object. Writing a result writes the input.

    Raises:
        ValueError: If the key set differs from the eight names, a dimension is not positive, a shape does
            not match or a hyper-head tensor is not FP32.
    """
    head_keys = {"hc_fn", "hc_base", "hc_scale"}
    expected = set(_DIRECT.values()) | {"hyper_head." + key for key in head_keys}
    _require_common_keys(state, expected, "native common embedding/norm/head")
    head = map_hyper_head_to_hf(
        {name: state["hyper_head." + name] for name in head_keys}, streams=streams, hidden=hidden
    )
    result = {**{hf: state[native] for hf, native in _DIRECT.items()}, **{_HEAD_PREFIX + k: v for k, v in head.items()}}
    _validate_direct(result, vocab=vocab, hidden=hidden)
    return result


# Hyper-head: the contraction of the residual streams before the final norm.


def map_hyper_head_from_hf(state: Mapping[str, torch.Tensor], *, streams: int, hidden: int) -> dict[str, torch.Tensor]:
    """Map the released hyper-head tensors to the native stream-contraction module.

    Args:
        state: Exactly ``hc_head_fn`` [streams * hidden, streams], ``hc_head_base`` [streams] and
            ``hc_head_scale`` [1], all FP32, named relative to ``model.mhc_contract``.
        streams: Number of residual streams.
        hidden: Hidden size.

    Returns:
        ``hc_fn`` [streams, streams * hidden], the transpose of ``hc_head_fn`` made contiguous, and
        ``hc_base`` [streams] and ``hc_scale`` [1], which are the input tensors.

    Raises:
        ValueError: If the key set differs, a dimension is not positive, a shape does not match or a
            tensor is not FP32.
    """
    expected = {"hc_head_fn", "hc_head_base", "hc_head_scale"}
    if state.keys() != expected:
        raise ValueError(f"Expected HF hyper-head keys {sorted(expected)}")
    if streams < 1 or hidden < 1:
        raise ValueError("streams and hidden must be positive")
    if state["hc_head_fn"].shape != (streams * hidden, streams):
        raise ValueError("Unexpected HF hyper-head matrix shape")
    if state["hc_head_base"].shape != (streams,) or state["hc_head_scale"].shape != (1,):
        raise ValueError("Unexpected HF hyper-head base or scale shape")
    if any(tensor.dtype != torch.float32 for tensor in state.values()):
        raise ValueError("HF hyper-head parameters must be FP32")
    return {
        "hc_fn": state["hc_head_fn"].T.contiguous(),
        "hc_base": state["hc_head_base"],
        "hc_scale": state["hc_head_scale"],
    }


def map_hyper_head_to_hf(state: Mapping[str, torch.Tensor], *, streams: int, hidden: int) -> dict[str, torch.Tensor]:
    """Restore the released hyper-head tensors from the native stream-contraction module.

    Args:
        state: Exactly ``hc_fn`` [streams, streams * hidden], ``hc_base`` [streams] and ``hc_scale`` [1],
            all FP32.
        streams: Number of residual streams.
        hidden: Hidden size.

    Returns:
        ``hc_head_fn`` [streams * hidden, streams], a transposed view of ``hc_fn``, and ``hc_head_base``
        [streams] and ``hc_head_scale`` [1], which are the input tensors. Names are relative to
        ``model.mhc_contract``. Writing a result writes the input.

    Raises:
        ValueError: If the key set differs, a dimension is not positive, a shape does not match or a
            tensor is not FP32.
    """
    expected = {"hc_fn", "hc_base", "hc_scale"}
    if state.keys() != expected:
        raise ValueError(f"Expected native hyper-head keys {sorted(expected)}")
    if streams < 1 or hidden < 1 or state["hc_fn"].shape != (streams, streams * hidden):
        raise ValueError("Unexpected native hyper-head matrix shape")
    result = {
        "hc_head_fn": state["hc_fn"].T,
        "hc_head_base": state["hc_base"],
        "hc_head_scale": state["hc_scale"],
    }
    map_hyper_head_from_hf(result, streams=streams, hidden=hidden)
    return result


# Hyper-connection (mHC): one unit before the attention and one before the feed-forward of every layer.

_HF_KEYS = {"mapping_proj.weight", "bias", "alpha_pre", "alpha_post", "alpha_res"}
_NATIVE_KEYS = {"fn", "base", "scale"}


def _validate_keys(state: Mapping[str, torch.Tensor], expected: set[str]) -> None:
    """Require exactly the expected names of one mHC unit and FP32 tensors.

    Args:
        state: Relative mHC names mapped to tensors of any shape.
        expected: Names that must be present, with no others.

    Raises:
        ValueError: If a name is missing or unexpected, or a tensor is not FP32.
    """
    missing, extra = expected - state.keys(), state.keys() - expected
    if missing or extra:
        raise ValueError(f"Invalid mHC keys: {sorted(missing)=}, {sorted(extra)=}")
    if any(tensor.dtype != torch.float32 for tensor in state.values()):
        raise ValueError("mHC tensors must be FP32")


def map_mhc_from_hf(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert the five released tensors of one mHC unit to the native layout.

    The checkpoint stores the projection as [inputs, outputs] with three one-element alphas; the module
    stores it as [outputs, inputs] with one scale vector. In the model ``inputs`` is ``streams * hidden``
    and ``outputs`` is ``(2 + streams) * streams``; this function only checks that projection and bias agree.

    Args:
        state: Exactly ``mapping_proj.weight`` [inputs, outputs], ``bias`` [outputs], ``alpha_pre`` [1],
            ``alpha_post`` [1] and ``alpha_res`` [1], all FP32, named relative to one hyper-connection.

    Returns:
        ``fn`` [outputs, inputs], a transposed view of the projection; ``base`` [outputs], the input bias;
        and ``scale`` [3], a new tensor holding the pre, post and residual alphas in that order.

    Raises:
        ValueError: If the key set differs, a tensor is not FP32, the projection is not a matrix, the bias
            does not match the projection or an alpha does not have shape [1].
    """
    _validate_keys(state, _HF_KEYS)
    projection, bias = state["mapping_proj.weight"], state["bias"]
    alpha = [state[name] for name in ("alpha_pre", "alpha_post", "alpha_res")]
    if projection.ndim != 2 or bias.shape != (projection.shape[1],) or any(t.shape != (1,) for t in alpha):
        raise ValueError(
            "Expected an mHC projection [inputs, outputs], a bias [outputs] and three alphas [1]; got shapes "
            f"{tuple(projection.shape)}, {tuple(bias.shape)} and {[tuple(t.shape) for t in alpha]}"
        )
    return {"fn": projection.T, "base": bias, "scale": torch.cat(alpha)}


def map_mhc_to_hf(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert one native mHC unit back to the five released tensors.

    Args:
        state: Exactly ``fn`` [outputs, inputs], ``base`` [outputs] and ``scale`` [3], all FP32.

    Returns:
        ``mapping_proj.weight`` [inputs, outputs], a transposed view of ``fn``; ``bias`` [outputs], the
        input ``base``; and ``alpha_pre``, ``alpha_post`` and ``alpha_res`` [1], views of elements 0, 1 and
        2 of ``scale``. Names are relative to one hyper-connection.

    Raises:
        ValueError: If the key set differs, a tensor is not FP32, ``fn`` is not a matrix, ``base`` does not
            match ``fn`` or ``scale`` does not have shape [3].
    """
    _validate_keys(state, _NATIVE_KEYS)
    projection, bias, scale = state["fn"], state["base"], state["scale"]
    if projection.ndim != 2 or bias.shape != (projection.shape[0],) or scale.shape != (3,):
        raise ValueError(
            "Expected a native mHC projection [outputs, inputs], a base [outputs] and a scale [3]; got shapes "
            f"{tuple(projection.shape)}, {tuple(bias.shape)} and {tuple(scale.shape)}"
        )
    return {
        "mapping_proj.weight": projection.T,
        "bias": bias,
        "alpha_pre": scale[0:1],
        "alpha_post": scale[1:2],
        "alpha_res": scale[2:3],
    }


# HCA attention. Released names are relative to ``self_attn`` and follow the released layer-0 BF16 checkpoint.

HCA_HF_TO_NATIVE = {
    "linear_q_down_proj.weight": "wq_a.weight",
    "linear_q_up_proj.weight": "wq_b.weight",
    "q_layernorm.weight": "q_norm.weight",
    "linear_kv_proj.weight": "wkv.weight",
    "kv_layernorm.weight": "kv_norm.weight",
    "linear_o_group_proj": "wo_a.weight",
    "o_proj.weight": "wo_b.weight",
    "gate_proj.weight": "gate_proj.weight",
    "vha_postmix_U": "vha.vha_postmix_U",
    "vha_postmix_V": "vha.vha_postmix_V",
    "core_attention.attn_sink": "sinks_param.weight",
    "core_attention.compressor.ape": "compressor.ape_param.weight",
    "core_attention.compressor.linear_wgate.weight": "compressor.wgate.weight",
    "core_attention.compressor.linear_wkv.weight": "compressor.wkv.weight",
    "core_attention.compressor.norm.weight": "compressor.kv_norm.weight",
}


def map_hca_from_hf(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a complete single HCA layer's relative keys without copying tensors.

    Args:
        state: Exactly the fifteen released suffixes relative to a single ``self_attn``, mapped to tensors
            of any shape and dtype. Shapes are not checked and no layout is changed.

    Returns:
        The same tensor objects under the relative names expected by ``Ernie5_5HCAAttention``.

    Raises:
        ValueError: If any expected key is missing or an unexpected key is present.
    """
    missing = HCA_HF_TO_NATIVE.keys() - state.keys()
    unexpected = state.keys() - HCA_HF_TO_NATIVE.keys()
    if missing or unexpected:
        raise ValueError(f"Invalid HCA checkpoint keys: {sorted(missing)=}, {sorted(unexpected)=}")
    return {HCA_HF_TO_NATIVE[key]: value for key, value in state.items()}


def map_hca_to_hf(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Restore a complete HCA component's relative released key names.

    Args:
        state: Exactly the fifteen native names of one ``Ernie5_5HCAAttention``, mapped to tensors of any
            shape and dtype. Shapes are not checked and no layout is changed.

    Returns:
        The same tensor objects under the released suffixes relative to a single ``self_attn``.

    Raises:
        ValueError: If any expected key is missing or an unexpected key is present.
    """
    expected = set(HCA_HF_TO_NATIVE.values())
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(f"Invalid native HCA checkpoint keys: {sorted(missing)=}, {sorted(unexpected)=}")
    return {hf: state[native] for hf, native in HCA_HF_TO_NATIVE.items()}


# DSA attention. Released names are relative to ``self_attn``; the fused K/V projection is handled apart.

DSA_HF_TO_NATIVE = {
    "q_a_proj.weight": "q_a_proj.weight",
    "q_a_layernorm.weight": "q_a_layernorm.weight",
    "q_b_proj.weight": "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "kv_a_layernorm.weight",
    "gate_proj.weight": "gate_proj.weight",
    "o_proj.weight": "o_proj.weight",
    "vha_postmix_U": "fp32_params.vha_postmix_U",
    "vha_postmix_V": "fp32_params.vha_postmix_V",
    "core_attention.softmax_offset": "fp32_params.softmax_offset",
    "core_attention.indexer.wq_b.weight": "indexer.wq_b.weight",
    "core_attention.indexer.wk.weight": "indexer.wk.weight",
    "core_attention.indexer.k_norm.weight": "indexer.k_norm.weight",
    "core_attention.indexer.k_norm.bias": "indexer.k_norm.bias",
    "core_attention.indexer.weights_proj.weight": "indexer.weights_proj.weight",
}


@dataclass(frozen=True)
class DSAKVLayout:
    """Per-head dimensions of the released fused K/V projection of a DSA layer.

    Attributes:
        heads: Number of attention heads.
        latent: Latent width of the K/V projection (the columns of the fused matrix).
        key_channels: Key rows per head in the fused matrix.
        value_channels: Value rows per head in the fused matrix.

    Raises:
        ValueError: If a dimension is not positive.
    """

    heads: int
    latent: int
    key_channels: int
    value_channels: int

    def __post_init__(self) -> None:
        """Reject a layout with a dimension that is not positive."""
        if min(self.heads, self.latent, self.key_channels, self.value_channels) <= 0:
            raise ValueError("All KV dimensions must be positive")


def split_fused_kv(
    combined: torch.Tensor, *, heads: int, latent: int, key_channels: int, value_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the released fused K/V projection into the separate native projections.

    The fused matrix stores, for each head in turn, ``key_channels`` key rows followed by
    ``value_channels`` value rows, every row of length ``latent``.

    Args:
        combined: Tensor of shape [heads * (key_channels + value_channels), latent]. It may be the global
            matrix or one rank's shard of whole heads, with ``heads`` counting the heads it holds.
        heads: Number of attention heads held by ``combined``.
        latent: Latent width of the K/V projection.
        key_channels: Key rows per head.
        value_channels: Value rows per head.

    Returns:
        Keys of shape [heads * latent, key_channels], each head's key block transposed, and values of
        shape [heads * value_channels, latent], the value rows in head order. A result is a view of
        ``combined`` when the reordering needs no copy.
    """
    per_head = combined.reshape(heads, key_channels + value_channels, latent)
    keys = per_head[:, :key_channels].transpose(1, 2).reshape(heads * latent, key_channels)
    values = per_head[:, key_channels:].reshape(heads * value_channels, latent)
    return keys, values


def dsa_kv_layout(config: Ernie5_5Config) -> DSAKVLayout:
    """Return the per-head dimensions of the fused K/V projection of a configuration's DSA layers.

    Args:
        config: Model configuration; supplies ``hybrid_mla_num_attention_heads``,
            ``hybrid_mla_kv_lora_rank``, ``hybrid_mla_qk_nope_head_dim`` and ``hybrid_mla_v_head_dim``.

    Returns:
        The head count, the latent width and the key and value rows per head.

    Raises:
        ValueError: If a dimension is not positive.
    """
    return DSAKVLayout(
        config.hybrid_mla_num_attention_heads,
        config.hybrid_mla_kv_lora_rank,
        config.hybrid_mla_qk_nope_head_dim,
        config.hybrid_mla_v_head_dim,
    )


def _split_fused_kv_of(combined: torch.Tensor, layout: DSAKVLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a complete released fused K/V projection after checking its shape.

    Args:
        combined: Tensor of shape [heads * (key_channels + value_channels), latent].
        layout: Per-head dimensions of the fused K/V projection.

    Returns:
        Keys of shape [heads * latent, key_channels] and values of shape [heads * value_channels, latent], as
        described for ``split_fused_kv``.

    Raises:
        ValueError: If the fused matrix does not have the layout's shape.
    """
    expected_shape = (layout.heads * (layout.key_channels + layout.value_channels), layout.latent)
    if combined.shape != expected_shape:
        raise ValueError(f"Expected combined KV shape {expected_shape}, got {tuple(combined.shape)}")
    return split_fused_kv(
        combined,
        heads=layout.heads,
        latent=layout.latent,
        key_channels=layout.key_channels,
        value_channels=layout.value_channels,
    )


@torch.no_grad()
def install_fused_kv(combined: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layout: DSAKVLayout) -> None:
    """Split a released fused K/V projection and write the two parts into the native projections in place.

    This completes a checkpoint read into an unsharded model: the fused projection is the one released tensor
    of a DSA layer that cannot be a view of model storage.

    Args:
        combined: Tensor of shape [heads * (key_channels + value_channels), latent] in the released layout.
        keys: Native key projection of shape [heads * latent, key_channels]; overwritten in place, with a
            cast to its dtype.
        values: Native value projection of shape [heads * value_channels, latent]; overwritten in place, with
            a cast to its dtype.
        layout: Per-head dimensions of the fused K/V projection.

    Raises:
        ValueError: If the fused matrix does not have the layout's shape.
    """
    split_keys, split_values = _split_fused_kv_of(combined, layout)
    keys.copy_(split_keys)
    values.copy_(split_values)


def map_dsa_from_hf(state: Mapping[str, torch.Tensor], layout: DSAKVLayout) -> dict[str, torch.Tensor]:
    """Convert all relative attention keys of one Hugging Face DSA layer to native keys.

    Args:
        state: Exactly the fifteen renamed suffixes relative to a single ``self_attn``, mapped to tensors
            of any shape and dtype (not checked, not copied), plus ``kv_b_proj.weight`` of shape
            [heads * (key_channels + value_channels), latent].
        layout: Per-head dimensions of the fused K/V projection.

    Returns:
        The renamed tensors as given, plus ``k_b_proj`` [heads * latent, key_channels] and ``v_b_proj``
        [heads * value_channels, latent], both contiguous and of the dtype of the fused matrix.

    Raises:
        ValueError: If a key is missing or unexpected, or the fused matrix does not have the layout's shape.
    """
    expected = {*DSA_HF_TO_NATIVE, "kv_b_proj.weight"}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(f"Invalid DSA layer HF keys: {sorted(missing)=}, {sorted(unexpected)=}")
    keys, values = _split_fused_kv_of(state["kv_b_proj.weight"], layout)
    result = {DSA_HF_TO_NATIVE[key]: value for key, value in state.items() if key != "kv_b_proj.weight"}
    result.update(k_b_proj=keys.contiguous(), v_b_proj=values.contiguous())
    return result


def map_dsa_to_hf(state: Mapping[str, torch.Tensor], layout: DSAKVLayout) -> dict[str, torch.Tensor]:
    """Restore relative HF attention keys, retaining input tensor dtypes.

    Args:
        state: Exactly the fifteen renamed native names of one DSA attention, mapped to tensors of any
            shape and dtype (not checked, not copied), plus ``k_b_proj`` [heads * latent, key_channels] and
            ``v_b_proj`` [heads * value_channels, latent] with one common dtype.
        layout: Per-head dimensions of the fused K/V projection.

    Returns:
        The renamed tensors as given, plus a new ``kv_b_proj.weight`` of shape
        [heads * (key_channels + value_channels), latent] that stores, for each head in turn, its
        transposed key block followed by its value rows.

    Raises:
        ValueError: If a key is missing or unexpected, the K/V projections do not have the layout's shapes
            or their dtypes differ.
    """
    expected = {*DSA_HF_TO_NATIVE.values(), "k_b_proj", "v_b_proj"}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(f"Invalid DSA layer native keys: {sorted(missing)=}, {sorted(unexpected)=}")
    keys, values = state["k_b_proj"], state["v_b_proj"]
    if keys.shape != (layout.heads * layout.latent, layout.key_channels) or values.shape != (
        layout.heads * layout.value_channels,
        layout.latent,
    ):
        raise ValueError("Unexpected separated K/V projection shapes")
    if keys.dtype != values.dtype:
        raise ValueError("Separated K/V projections must have the same dtype")
    keys = keys.reshape(layout.heads, layout.latent, layout.key_channels).transpose(1, 2)
    values = values.reshape(layout.heads, layout.value_channels, layout.latent)
    combined = torch.cat((keys, values), dim=1).reshape(
        layout.heads * (layout.key_channels + layout.value_channels), layout.latent
    )
    result = {hf: state[native] for hf, native in DSA_HF_TO_NATIVE.items()}
    result["kv_b_proj.weight"] = combined
    return result


# MoE feed-forward. Released names are relative to ``block_sparse_moe``.

MOE_HF_TO_NATIVE = {
    "gate.weight": "gate.weight",
    "gate.weight_1": "gate.weight_1",
    "gate.routed_scaling_factor_param": "gate.routed_scaling_factor_param",
    "e_score_correction_bias": "gate.e_score_correction_bias",
    "fc1_latent_proj.weight": "fc1_latent_proj.weight",
    "fc2_latent_proj.weight": "fc2_latent_proj.weight",
    "latent_norm.weight": "latent_norm.weight",
    "shared_experts.w1.weight": "shared_experts.gate_proj.weight",
    "shared_experts.w2.weight": "shared_experts.down_proj.weight",
    "shared_experts.w3.weight": "shared_experts.up_proj.weight",
}


def _checked_ids(expert_ids: Sequence[int]) -> tuple[int, ...]:
    """Return the expert indices as a tuple after validating them.

    Args:
        expert_ids: Indices of routed experts.

    Returns:
        The indices in the given order.

    Raises:
        ValueError: If the sequence is empty, an index is not a non-negative integer or an index repeats.
    """
    result = tuple(expert_ids)
    if not result or any(isinstance(expert, bool) or not isinstance(expert, int) or expert < 0 for expert in result):
        raise ValueError("expert_ids must contain non-negative integers")
    if len(set(result)) != len(result):
        raise ValueError("expert_ids must be unique")
    return result


def pack_experts(state: Mapping[str, torch.Tensor], expert_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack selected HF experts in the supplied order into grouped matrices.

    Only the selected experts are allocated.

    Args:
        state: Mapping that contains, for every selected expert, ``experts.{id}.w1.weight`` (gate
            projection) [intermediate, latent], ``experts.{id}.w2.weight`` (down projection)
            [latent, intermediate] and ``experts.{id}.w3.weight`` (up projection) [intermediate, latent].
            The selected tensors must share one dtype and one device. Other entries are ignored.
        expert_ids: Unique non-negative expert indices. Their order is the order along the expert axis of
            the results.

    Returns:
        Two new tensors with the dtype and device of the inputs: gate/up of shape
        [experts, latent, 2 * intermediate], whose last axis holds the transposed gate projection followed
        by the transposed up projection, and down of shape [experts, intermediate, latent], the transposed
        down projections.

    Raises:
        ValueError: If the indices are invalid, a selected key is missing, the first gate projection is
            not a matrix, the shapes of an expert do not match, or dtypes or devices differ.
    """
    ids = _checked_ids(expert_ids)
    expected = {f"experts.{expert}.w{number}.weight" for expert in ids for number in (1, 2, 3)}
    if expected - state.keys():
        raise ValueError(f"Missing selected expert keys: {sorted(expected - state.keys())}")
    first = state[f"experts.{ids[0]}.w1.weight"]
    if first.ndim != 2:
        raise ValueError("Expert gate projection must be a matrix")
    intermediate, latent = first.shape
    gate_up = first.new_empty((len(ids), latent, 2 * intermediate))
    down_out = first.new_empty((len(ids), intermediate, latent))
    for local, expert in enumerate(ids):
        gate, down, up = (state[f"experts.{expert}.w{number}.weight"] for number in (1, 2, 3))
        if gate.shape != (intermediate, latent) or up.shape != gate.shape or down.shape != (latent, intermediate):
            raise ValueError(f"Invalid projection shapes for expert {expert}")
        if any(tensor.dtype != first.dtype or tensor.device != first.device for tensor in (gate, up, down)):
            raise ValueError("Expert tensors must share a dtype and device")
        gate_up[local, :, :intermediate].copy_(gate.T)
        gate_up[local, :, intermediate:].copy_(up.T)
        down_out[local].copy_(down.T)
    return gate_up, down_out


def unpack_experts(gate_up: torch.Tensor, down: torch.Tensor, expert_ids: Sequence[int]) -> dict[str, torch.Tensor]:
    """Restore selected grouped experts to their HF per-expert matrices.

    Args:
        gate_up: Tensor of shape [experts, latent, 2 * intermediate]. The last axis holds the transposed
            gate projection followed by the transposed up projection.
        down: Tensor of shape [experts, intermediate, latent] with the dtype and device of ``gate_up``.
        expert_ids: Unique non-negative expert indices, one per entry of the expert axis and in its order.

    Returns:
        For every index, ``experts.{id}.w1.weight`` [intermediate, latent], ``experts.{id}.w2.weight``
        [latent, intermediate] and ``experts.{id}.w3.weight`` [intermediate, latent]. Each is a transposed
        view of the grouped storage, not a copy: no expert is allocated a second time, the views are not
        contiguous, and writing a view writes ``gate_up`` or ``down``.

    Raises:
        ValueError: If the indices are invalid, ``gate_up`` does not have three axes with an even last
            axis and one entry per index, or ``down`` does not match it in shape, dtype or device.
    """
    ids = _checked_ids(expert_ids)
    if gate_up.ndim != 3 or gate_up.shape[0] != len(ids) or gate_up.shape[-1] % 2:
        raise ValueError("Expected gate/up [experts,latent,2*intermediate]")
    _, latent, twice_intermediate = gate_up.shape
    intermediate = twice_intermediate // 2
    if down.shape != (len(ids), intermediate, latent) or down.dtype != gate_up.dtype or down.device != gate_up.device:
        raise ValueError("Expected matching grouped down projections")
    result = {}
    for local, expert in enumerate(ids):
        result[f"experts.{expert}.w1.weight"] = gate_up[local, :, :intermediate].T
        result[f"experts.{expert}.w2.weight"] = down[local].T
        result[f"experts.{expert}.w3.weight"] = gate_up[local, :, intermediate:].T
    return result


def map_moe_from_hf(state: Mapping[str, torch.Tensor], expert_ids: Sequence[int]) -> dict[str, torch.Tensor]:
    """Map non-expert keys and selected experts of one HF MoE layer.

    Args:
        state: Exactly the ten released non-expert tensors of one ``block_sparse_moe`` (any shape and
            dtype; not checked, not copied) and, for every selected expert, ``experts.{id}.w1.weight``
            [intermediate, latent], ``experts.{id}.w2.weight`` [latent, intermediate] and
            ``experts.{id}.w3.weight`` [intermediate, latent].
        expert_ids: Unique non-negative expert indices, in the order of the native expert axis.

    Returns:
        The non-expert tensors as given under their native names, plus ``experts.gate_and_up_projs``
        [experts, latent, 2 * intermediate] and ``experts.down_projs`` [experts, intermediate, latent],
        packed as described for ``pack_experts``.

    Raises:
        ValueError: If the indices are invalid, a key is missing or unexpected, or the experts cannot be
            packed.
    """
    ids = _checked_ids(expert_ids)
    expected = set(MOE_HF_TO_NATIVE) | {f"experts.{expert}.w{number}.weight" for expert in ids for number in (1, 2, 3)}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(f"Invalid MoE HF keys: {sorted(missing)=}, {sorted(unexpected)=}")
    gate_up, down = pack_experts(state, ids)
    result = {native: state[hf] for hf, native in MOE_HF_TO_NATIVE.items()}
    result.update({"experts.gate_and_up_projs": gate_up, "experts.down_projs": down})
    return result


def map_moe_to_hf(state: Mapping[str, torch.Tensor], expert_ids: Sequence[int]) -> dict[str, torch.Tensor]:
    """Restore a local MoE's grouped experts and non-expert keys to HF names.

    Args:
        state: Exactly the ten native non-expert tensors of one MoE (any shape and dtype; not checked,
            not copied), ``experts.gate_and_up_projs`` [experts, latent, 2 * intermediate] and
            ``experts.down_projs`` [experts, intermediate, latent].
        expert_ids: Unique non-negative expert indices, one per entry of the expert axis and in its order.

    Returns:
        The non-expert tensors as given under their released names, plus the three per-expert matrices
        of every index, laid out as described for ``unpack_experts``.

    Raises:
        ValueError: If the indices are invalid, a key is missing or unexpected, or the grouped tensors do
            not match the indices.
    """
    ids = _checked_ids(expert_ids)
    expected = set(MOE_HF_TO_NATIVE.values()) | {"experts.gate_and_up_projs", "experts.down_projs"}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(f"Invalid MoE native keys: {sorted(missing)=}, {sorted(unexpected)=}")
    result = {hf: state[native] for hf, native in MOE_HF_TO_NATIVE.items()}
    result.update(unpack_experts(state["experts.gate_and_up_projs"], state["experts.down_projs"], ids))
    return result


# One decoder layer: two norms, two mHC units, one attention and one feed-forward.

# The two norm weights open every converted layer, in this order. A tuple is iterated rather than a set so
# that all processes produce one key order whatever their string hash seed is.
_NORM_KEYS = ("post_attention_layernorm.weight", "input_layernorm.weight")
_NORMS = set(_NORM_KEYS)
_DENSE = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}


def _validate_kinds(attention: str, ffn: str, layout: DSAKVLayout | None, expert_ids: Sequence[int] | None) -> None:
    """Check that the layer kinds are known and carry exactly the arguments they need.

    Args:
        attention: ``"hca"`` or ``"dsa"``.
        ffn: ``"dense"`` or ``"moe"``.
        layout: K/V layout; required for DSA and not accepted for HCA.
        expert_ids: Routed experts; required for MoE and not accepted for a dense feed-forward.

    Raises:
        ValueError: If a kind is unknown or an argument is missing or superfluous.
    """
    if attention not in ("hca", "dsa") or ffn not in ("dense", "moe"):
        raise ValueError("Expected HCA/DSA attention and dense/MoE FFN")
    if (attention == "dsa") != (layout is not None):
        raise ValueError("DSA requires its KV layout; HCA does not take one")
    if (ffn == "moe") != (expert_ids is not None):
        raise ValueError("MoE requires selected expert IDs; dense does not take them")


def map_decoder_from_hf(
    state: Mapping[str, torch.Tensor],
    *,
    attention: Literal["hca", "dsa"],
    ffn: Literal["dense", "moe"],
    layout: DSAKVLayout | None = None,
    expert_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Map relative HF keys for one decoder layer to native block keys.

    Selected experts are packed eagerly.

    Args:
        state: Complete tensors of one layer, named relative to ``model.layers.{index}``:
            ``input_layernorm.weight`` and ``post_attention_layernorm.weight`` (any shape, passed
            through); ``self_attention_hyper_connection.*`` and ``mlp_hyper_connection.*`` in the layout
            of ``map_mhc_from_hf``; ``self_attn.*`` in the layout of ``map_hca_from_hf`` or
            ``map_dsa_from_hf``; and either ``mlp.gate_proj.weight``, ``mlp.up_proj.weight`` and
            ``mlp.down_proj.weight`` (any shape, passed through) or ``block_sparse_moe.*`` in the layout
            of ``map_moe_from_hf``.
        attention: Attention kind of the layer.
        ffn: Feed-forward kind of the layer.
        layout: K/V layout; required for DSA and not accepted for HCA.
        expert_ids: Routed experts to pack, in the order of the native expert axis; required for MoE and
            not accepted for a dense feed-forward.

    Returns:
        Native tensors named relative to one decoder block: the two norm weights as given, ``attn_hc.*``
        and ``ffn_hc.*`` in the layout of ``map_mhc_from_hf``, ``attention.*`` and ``feed_forward.*``. The
        feed-forward prefix is ``feed_forward.`` for both kinds. The norm weights come first, in the order
        of ``_NORM_KEYS``, then the groups in the order listed here; the order never depends on the string
        hash seed of the process.

    Raises:
        ValueError: If the kinds and their arguments are inconsistent, a key is missing or unexpected, or
            a component conversion rejects its tensors.
    """
    _validate_kinds(attention, ffn, layout, expert_ids)
    remaining = dict(state)
    if not _NORMS <= remaining.keys():
        raise ValueError("Missing decoder norm keys")
    result = {name: remaining.pop(name) for name in _NORM_KEYS}
    for hf, native in (("self_attention_hyper_connection.", "attn_hc."), ("mlp_hyper_connection.", "ffn_hc.")):
        _put_prefix(result, native, map_mhc_from_hf(_pop_prefix(remaining, hf)))
    attn_state = _pop_prefix(remaining, "self_attn.")
    converted_attn = map_hca_from_hf(attn_state) if attention == "hca" else map_dsa_from_hf(attn_state, layout)
    _put_prefix(result, "attention.", converted_attn)
    ffn_state = _pop_prefix(remaining, "mlp." if ffn == "dense" else "block_sparse_moe.")
    if ffn == "dense":
        if ffn_state.keys() != _DENSE:
            raise ValueError("Invalid dense FFN keys")
        converted_ffn = ffn_state
    else:
        converted_ffn = map_moe_from_hf(ffn_state, expert_ids)
    _put_prefix(result, "feed_forward.", converted_ffn)
    if remaining:
        raise ValueError(f"Unexpected decoder keys: {sorted(remaining)}")
    return result


def map_decoder_to_hf(
    state: Mapping[str, torch.Tensor],
    *,
    attention: Literal["hca", "dsa"],
    ffn: Literal["dense", "moe"],
    layout: DSAKVLayout | None = None,
    expert_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Restore one native decoder block to relative HF decoder-layer keys.

    Args:
        state: Complete tensors of one decoder block, named relative to it: ``input_layernorm.weight``
            and ``post_attention_layernorm.weight`` (any shape, passed through); ``attn_hc.*`` and
            ``ffn_hc.*`` in the layout of ``map_mhc_to_hf``; ``attention.*`` in the layout of
            ``map_hca_to_hf`` or ``map_dsa_to_hf``; and ``feed_forward.*``, either the three dense
            projection weights (any shape, passed through) or the layout of ``map_moe_to_hf``.
        attention: Attention kind of the layer.
        ffn: Feed-forward kind of the layer.
        layout: K/V layout; required for DSA and not accepted for HCA.
        expert_ids: Routed experts, one per entry of the native expert axis and in its order; required
            for MoE and not accepted for a dense feed-forward.

    Returns:
        Released tensors named relative to ``model.layers.{index}``, in the layouts accepted by
        ``map_decoder_from_hf``. The norm weights come first, in the order of ``_NORM_KEYS``, then the two
        hyper-connections, the attention and the feed-forward; the order never depends on the string hash
        seed of the process.

    Raises:
        ValueError: If the kinds and their arguments are inconsistent, a key is missing or unexpected, or
            a component conversion rejects its tensors.
    """
    _validate_kinds(attention, ffn, layout, expert_ids)
    remaining = dict(state)
    if not _NORMS <= remaining.keys():
        raise ValueError("Missing decoder norm keys")
    result = {name: remaining.pop(name) for name in _NORM_KEYS}
    for native, hf in (("attn_hc.", "self_attention_hyper_connection."), ("ffn_hc.", "mlp_hyper_connection.")):
        _put_prefix(result, hf, map_mhc_to_hf(_pop_prefix(remaining, native)))
    attn_state = _pop_prefix(remaining, "attention.")
    converted_attn = map_hca_to_hf(attn_state) if attention == "hca" else map_dsa_to_hf(attn_state, layout)
    _put_prefix(result, "self_attn.", converted_attn)
    ffn_state = _pop_prefix(remaining, "feed_forward.")
    if ffn == "dense":
        if ffn_state.keys() != _DENSE:
            raise ValueError("Invalid dense FFN keys")
        converted_ffn = ffn_state
    else:
        converted_ffn = map_moe_to_hf(ffn_state, expert_ids)
    _put_prefix(result, "mlp." if ffn == "dense" else "block_sparse_moe.", converted_ffn)
    if remaining:
        raise ValueError(f"Unexpected native decoder keys: {sorted(remaining)}")
    return result


# The whole decoder-only model.


def _layer_layout_and_experts(
    config: Ernie5_5Config, layer: Ernie5_5LayerPlan
) -> tuple[DSAKVLayout | None, tuple[int, ...] | None]:
    """Return what the decoder-layer conversions need besides the kinds of one planned layer.

    Args:
        config: Model configuration; supplies the DSA K/V dimensions and the routed expert count.
        layer: Planned attention and feed-forward kinds of the layer.

    Returns:
        The K/V layout of a DSA layer (``None`` for an HCA layer) and the indices of all routed experts in
        order for an MoE layer (``None`` for a dense feed-forward).
    """
    layout = dsa_kv_layout(config) if layer.attention == "dsa" else None
    expert_ids = tuple(range(config.n_routed_experts)) if layer.feed_forward == "moe" else None
    return layout, expert_ids


def map_model_from_hf(state: Mapping[str, torch.Tensor], config: Ernie5_5Config) -> dict[str, torch.Tensor]:
    """Map exactly a decoder-only HF state dict to the native names of the inner model.

    All routed experts of every MoE layer are packed in memory, so this is meant for small checkpoints
    and for shape-only tensors on the meta device.

    Args:
        state: Complete, unsharded released tensors: ``model.layers.{index}.*`` for every decoder layer in
            the layouts of ``map_decoder_from_hf``, and the eight model-level tensors in the layouts of
            ``map_common_from_hf``.
        config: Model configuration without MTP layers.

    Returns:
        Native tensors named relative to the inner model: ``layers.{index}.*`` in the layouts returned by
        ``map_decoder_from_hf`` and the eight model-level tensors returned by ``map_common_from_hf``.

    Raises:
        NotImplementedError: If the configuration has MTP layers.
        ValueError: If a key is missing or unexpected (MTP and unknown keys are rejected, not dropped) or
            a component conversion rejects its tensors.
    """
    plan = Ernie5_5ModelPlan.from_config(config)
    if plan.mtp_attention:
        raise NotImplementedError("Decoder-only mapping cannot preserve MTP weights")
    remaining = dict(state)
    result = {}
    for layer in plan.decoder_layers:
        hf_prefix, native_prefix = f"model.layers.{layer.layer_index}.", f"layers.{layer.layer_index}."
        layout, expert_ids = _layer_layout_and_experts(config, layer)
        converted = map_decoder_from_hf(
            _pop_prefix(remaining, hf_prefix),
            attention=layer.attention,
            ffn=layer.feed_forward,
            layout=layout,
            expert_ids=expert_ids,
        )
        result.update({native_prefix + key: value for key, value in converted.items()})
    # MTP and unknown keys must not be silently dropped by a decoder-only map.
    result.update(
        map_common_from_hf(remaining, vocab=config.vocab_size, hidden=config.hidden_size, streams=config.hc_mult)
    )
    return result


def map_model_to_hf(state: Mapping[str, torch.Tensor], config: Ernie5_5Config) -> dict[str, torch.Tensor]:
    """Map exactly a decoder-only native state dict of the inner model back to HF names.

    Args:
        state: Complete, unsharded native tensors named relative to the inner model: ``layers.{index}.*``
            for every decoder layer in the layouts of ``map_decoder_to_hf``, and the eight model-level
            tensors in the layouts of ``map_common_to_hf``.
        config: Model configuration without MTP layers.

    Returns:
        Released tensors: ``model.layers.{index}.*`` in the layouts returned by ``map_decoder_to_hf`` and
        the eight model-level tensors returned by ``map_common_to_hf``.

    Raises:
        NotImplementedError: If the configuration has MTP layers.
        ValueError: If a key is missing or unexpected or a component conversion rejects its tensors.
    """
    plan = Ernie5_5ModelPlan.from_config(config)
    if plan.mtp_attention:
        raise NotImplementedError("Decoder-only mapping cannot preserve MTP weights")
    remaining = dict(state)
    result = {}
    for layer in plan.decoder_layers:
        native_prefix, hf_prefix = f"layers.{layer.layer_index}.", f"model.layers.{layer.layer_index}."
        layout, expert_ids = _layer_layout_and_experts(config, layer)
        converted = map_decoder_to_hf(
            _pop_prefix(remaining, native_prefix),
            attention=layer.attention,
            ffn=layer.feed_forward,
            layout=layout,
            expert_ids=expert_ids,
        )
        result.update({hf_prefix + key: value for key, value in converted.items()})
    result.update(
        map_common_to_hf(remaining, vocab=config.vocab_size, hidden=config.hidden_size, streams=config.hc_mult)
    )
    return result
