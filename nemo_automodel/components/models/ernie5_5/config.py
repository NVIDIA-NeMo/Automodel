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

"""Configuration and layer schedule of ERNIE 5.5 causal language models.

``Ernie5_5Config`` reads a released ``config.json`` without remote code. The functions of this module derive the
layer schedule from it and hold the checks that the model applies before it allocates parameters:

* ``Ernie5_5ModelPlan.from_config``: the attention and feed-forward kind of every decoder layer, and the attention
  schedule of the MTP block that the released checkpoint carries.
* ``decoder_only_config``: a copy of a configuration without the MTP block. The model is built from this form only.
* ``validate_architecture``: rejects the settings that select an architecture the model does not implement.
* ``main_rope_theta`` and ``hca_rope_scaling``: the two rotary settings in the form the attention layers take.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from transformers import PretrainedConfig

HFConfigValue: TypeAlias = (
    str | int | float | bool | torch.dtype | list["HFConfigValue"] | dict[str, "HFConfigValue"] | None
)

_HCA_COMPRESS_RATIO = 128
_DSA_COMPRESS_RATIO = -2
_HCA_YARN_KEYS = ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow")


class Ernie5_5Config(PretrainedConfig):
    """ERNIE 5.5 configuration with the values of the released ERNIE-5.5-Flash checkpoint as defaults.

    Five fields do not default to the released value:

    * ``compress_ratios`` has no default. It lists one entry per decoder layer, 128 for an HCA layer and -2 for a
      DSA layer; the released file adds one entry for its MTP layer.
    * ``num_nextn_predict_layers`` defaults to 0 (released: 1). The model does not build the MTP block, so a
      configuration written by hand needs no override. The released ``config.json`` carries the explicit value 1
      and is rejected until the recipe overrides it with 0: see ``decoder_only_config``.
    * ``rope_interleave`` defaults to None (released: true). A configuration with a DSA layer must declare the
      rotary pairing of the checkpoint itself, here or in ``mqa_latent_rope_adjacent_pairing``.
    * ``rotary_base`` defaults to None (released: 8000000). The released file repeats the main rotary base in this
      key. When it is set it must equal ``rope_theta``; with a default, overriding ``rope_theta`` alone would be a
      conflict for every configuration.
    * ``pad_token_id`` defaults to None (released: 0). The model rejects input that contains the padding token, so
      a configuration must name it explicitly.

    Two fields default to None, which selects the released value: ``rope_theta`` (8000000, unless
    ``rope_parameters`` carries a base) and ``rope_scaling`` (the YaRN settings of the HCA layers). The attribute
    ``rope_theta`` reads and writes ``rope_parameters["rope_theta"]``.

    A configuration may carry the rotary settings twice, in ``rope_parameters`` and in the released keys
    ``rope_theta`` and ``rope_scaling``; the two forms must agree. Fields of the released ``config.json`` that are
    not listed here are kept by ``PretrainedConfig``, so loading and saving a checkpoint does not drop them.
    """

    model_type = "ernie5_5_moe"

    def __init__(
        self,
        vocab_size: int = 201216,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 43,
        max_position_embeddings: int = 1048576,
        # Layer schedule
        compress_ratios: list[int] | None = None,
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        num_nextn_predict_layers: int = 0,
        mtp_shared_last_layer: bool = True,
        # Heavily compressed attention (HCA) layers
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1024,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        sliding_window: int = 128,
        compress_rope_theta: float = 160000.0,
        hca_rope_type: str = "yarn",
        use_qk_norm: bool = True,
        sigmoid_gate_fusion: bool = False,
        # Attention variant of the released checkpoint, shared by both kinds of layer
        qk_norm_type: str = "per_head",
        use_gated_attn: bool = True,
        gated_attn_act: str = "sigmoid",
        gated_attn_use_q_lora: bool = True,
        use_vha_attention: bool = True,
        add_full_attention_sink_bias: bool = True,
        attention_bias: bool = False,
        # Dynamic sparse attention (DSA) layers
        hybrid_mla_attention: str = "mqa_dsa",
        mqa_split_kv_b_proj: bool = True,
        use_fast_hadamard: bool = True,
        hybrid_mla_num_attention_heads: int = 64,
        hybrid_mla_q_lora_rank: int = 1024,
        hybrid_mla_kv_lora_rank: int = 512,
        hybrid_mla_qk_nope_head_dim: int = 192,
        hybrid_mla_qk_rope_head_dim: int = 64,
        hybrid_mla_v_head_dim: int = 256,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        rope_interleave: bool | None = None,
        mqa_latent_rope_adjacent_pairing: bool | None = None,
        # Rotary position embedding
        rope_theta: float | None = None,
        rope_scaling: dict[str, HFConfigValue] | None = None,
        rope_parameters: dict[str, HFConfigValue] | None = None,
        rotary_base: float | None = None,
        # Hyper-connections and normalization
        enable_hyper_connections: bool = True,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        # Feed-forward and mixture of experts
        hidden_act: str = "situ",
        activation_situ_beta: float = 4.0,
        activation_situ_linear_beta: float = 25.0,
        n_routed_experts: int = 512,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 10,
        moe_intermediate_size: int = 2048,
        moe_latent_size: int = 2048,
        routed_scaling_factor: float = 3.0,
        routed_scaling_factor_learnable: bool = True,
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        moe_split_feature_routing: bool = True,
        latent_moe_use_norm: bool = True,
        # Output head
        multimax: str = "lm_head",
        # Initialization, tokens, weight tying and the released inference default
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        **kwargs: HFConfigValue,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.compress_ratios = compress_ratios
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_shared_last_layer = mtp_shared_last_layer
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.sliding_window = sliding_window
        self.compress_rope_theta = compress_rope_theta
        self.hca_rope_type = hca_rope_type
        self.use_qk_norm = use_qk_norm
        self.sigmoid_gate_fusion = sigmoid_gate_fusion
        self.qk_norm_type = qk_norm_type
        self.use_gated_attn = use_gated_attn
        self.gated_attn_act = gated_attn_act
        self.gated_attn_use_q_lora = gated_attn_use_q_lora
        self.use_vha_attention = use_vha_attention
        self.add_full_attention_sink_bias = add_full_attention_sink_bias
        self.attention_bias = attention_bias
        self.hybrid_mla_attention = hybrid_mla_attention
        self.mqa_split_kv_b_proj = mqa_split_kv_b_proj
        self.use_fast_hadamard = use_fast_hadamard
        self.hybrid_mla_num_attention_heads = hybrid_mla_num_attention_heads
        self.hybrid_mla_q_lora_rank = hybrid_mla_q_lora_rank
        self.hybrid_mla_kv_lora_rank = hybrid_mla_kv_lora_rank
        self.hybrid_mla_qk_nope_head_dim = hybrid_mla_qk_nope_head_dim
        self.hybrid_mla_qk_rope_head_dim = hybrid_mla_qk_rope_head_dim
        self.hybrid_mla_v_head_dim = hybrid_mla_v_head_dim
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.rope_interleave = rope_interleave
        self.mqa_latent_rope_adjacent_pairing = mqa_latent_rope_adjacent_pairing
        self.rotary_base = rotary_base
        self.enable_hyper_connections = enable_hyper_connections
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.routed_scaling_factor = routed_scaling_factor
        self.routed_scaling_factor_learnable = routed_scaling_factor_learnable
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_split_feature_routing = moe_split_feature_routing
        self.latent_moe_use_norm = latent_moe_use_norm
        self.multimax = multimax
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        # Not read by the model, which keeps no KV cache. PretrainedConfig drops the key unless it is an attribute,
        # and readers of the released format expect it in a saved config.json.
        self.use_cache = use_cache
        # The released config.json stores the main rotary base as ``rope_theta`` and the YaRN settings of the HCA
        # layers as ``rope_scaling``; a file saved by transformers also carries both in ``rope_parameters``. All
        # of them end up in ``rope_parameters``, which must exist before the PretrainedConfig initializer runs.
        if rope_parameters is not None and rope_scaling is not None:
            conflicts = sorted(
                name
                for name, value in rope_scaling.items()
                if name in rope_parameters and rope_parameters[name] != value
            )
            if conflicts:
                raise ValueError(
                    f"Conflicting RoPE settings: rope_scaling and rope_parameters differ in {conflicts}; "
                    f"rope_scaling={rope_scaling!r}, rope_parameters={rope_parameters!r}"
                )
        if rope_parameters is None:
            rope_parameters = (
                rope_scaling
                if rope_scaling is not None
                else {
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 65536,
                    "beta_fast": 32,
                    "beta_slow": 1,
                }
            )
        parameters = dict(rope_parameters)
        stored_theta = parameters.get("rope_theta")
        if rope_theta is not None and stored_theta is not None and rope_theta != stored_theta:
            raise ValueError(
                f"Conflicting RoPE base values: rope_theta={rope_theta!r}, "
                f"rope_parameters['rope_theta']={stored_theta!r}"
            )
        if stored_theta is None:
            parameters["rope_theta"] = 8_000_000 if rope_theta is None else rope_theta
        self.rope_parameters = parameters
        kwargs.setdefault("architectures", ["Ernie5_5_MoeForCausalLM"])
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def rope_theta(self) -> float | None:
        """Main rotary base, read from ``rope_parameters["rope_theta"]``.

        Assigning the attribute writes the same entry, so a configuration override of ``rope_theta`` reaches the
        value that ``main_rope_theta`` returns.
        """
        return (self.rope_parameters or {}).get("rope_theta")

    @rope_theta.setter
    def rope_theta(self, value: float) -> None:
        self.rope_parameters = {**(self.rope_parameters or {}), "rope_theta": value}

    def to_dict(self) -> dict[str, HFConfigValue]:
        """Serialize the configuration with the rotary and dtype keys of the released ``config.json``.

        Transformers keeps the rotary settings in ``rope_parameters`` and the dtype in ``dtype``. Readers of the
        released format look up ``rope_theta``, ``rope_scaling`` and ``torch_dtype``, so these keys are written
        next to the standardized ones, with the same values.

        Returns:
            The serialized configuration.
        """
        output = super().to_dict()
        parameters = output.get("rope_parameters") or {}
        output["rope_theta"] = parameters.get("rope_theta")
        output["rope_scaling"] = {
            name: value for name, value in parameters.items() if name not in ("rope_theta", "rope_type")
        }
        output["torch_dtype"] = output.get("dtype")
        return output


@dataclass(frozen=True)
class Ernie5_5LayerPlan:
    """Attention and feed-forward kind of one decoder layer."""

    layer_index: int
    attention: Literal["hca", "dsa"]
    feed_forward: Literal["dense", "moe"]


@dataclass(frozen=True)
class Ernie5_5ModelPlan:
    """Decoder layer schedule, and the attention schedule of the MTP block."""

    decoder_layers: tuple[Ernie5_5LayerPlan, ...]
    mtp_attention: tuple[Literal["hca", "dsa"], ...]
    mtp_shared_last_layer: bool

    @classmethod
    def from_config(cls, config: Ernie5_5Config) -> Ernie5_5ModelPlan:
        """Derive the layer schedule, keeping the MTP entries of ``compress_ratios`` apart from the decoder layers.

        Args:
            config: Model configuration. ``compress_ratios`` holds one entry per decoder layer followed by one
                entry per MTP layer.

        Returns:
            The attention and feed-forward kind of every decoder layer, the attention kind of every MTP layer and
            the ``mtp_shared_last_layer`` flag.

        Raises:
            ValueError: If a layer count is not an integer in its valid range, ``moe_layer_freq`` is not 1,
                ``compress_ratios`` does not have one entry per decoder and MTP layer, an entry is neither 128
                (HCA) nor -2 (DSA), or ``mtp_shared_last_layer`` is not a boolean.
        """
        layers = config.num_hidden_layers
        mtp_layers = config.num_nextn_predict_layers
        dense_layers = config.first_k_dense_replace
        if not _is_integer(layers) or layers < 1:
            raise ValueError("num_hidden_layers must be a positive integer")
        if not _is_integer(mtp_layers) or mtp_layers < 0:
            raise ValueError("num_nextn_predict_layers must be a non-negative integer")
        if not _is_integer(dense_layers) or not 0 <= dense_layers <= layers:
            raise ValueError("first_k_dense_replace must be between zero and num_hidden_layers")
        if not _is_integer(config.moe_layer_freq) or config.moe_layer_freq != 1:
            raise ValueError("Only moe_layer_freq=1 is currently supported by the plan")
        ratios = config.compress_ratios
        if not isinstance(ratios, (list, tuple)) or len(ratios) != layers + mtp_layers:
            raise ValueError("compress_ratios must contain one entry per decoder and MTP layer")
        if any(not _is_integer(value) or value not in (_HCA_COMPRESS_RATIO, _DSA_COMPRESS_RATIO) for value in ratios):
            raise ValueError("Only HCA ratio 128 and DSA ratio -2 are currently supported")
        names = tuple("hca" if ratio == _HCA_COMPRESS_RATIO else "dsa" for ratio in ratios)
        decoder = tuple(Ernie5_5LayerPlan(i, names[i], "dense" if i < dense_layers else "moe") for i in range(layers))
        if not isinstance(config.mtp_shared_last_layer, bool):
            raise ValueError("mtp_shared_last_layer must be boolean")
        return cls(decoder, names[layers:], config.mtp_shared_last_layer)


def _is_integer(value: HFConfigValue) -> bool:
    """Return whether ``value`` is an integer and not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def decoder_only_config(config: Ernie5_5Config) -> Ernie5_5Config:
    """Return a copy of ``config`` that describes the decoder without the MTP block.

    The model does not build or train the MTP block of the released checkpoint, so it accepts a configuration only
    when ``num_nextn_predict_layers`` is 0. The released ``compress_ratios`` then still ends with the entry of the
    MTP layer, which shares the attention kind of the last decoder layer; the copy drops that entry and clears
    ``mtp_shared_last_layer``, so that a configuration saved from the model matches the weights saved with it.

    Args:
        config: Model configuration. It is not modified.

    Returns:
        A deep copy with ``num_nextn_predict_layers == 0``, one ``compress_ratios`` entry per decoder layer and
        ``mtp_shared_last_layer`` set to False.

    Raises:
        ValueError: If ``num_nextn_predict_layers`` is not 0, ``compress_ratios`` has neither one entry per decoder
            layer nor one additional entry of a shared MTP layer, or ``Ernie5_5ModelPlan.from_config`` rejects
            the result.
    """
    result = copy.deepcopy(config)
    layers, depth, ratios = result.num_hidden_layers, result.num_nextn_predict_layers, result.compress_ratios
    if not _is_integer(layers) or layers < 1:
        raise ValueError("num_hidden_layers must be a positive integer")
    if not _is_integer(depth) or depth != 0:
        raise ValueError(
            f"ERNIE 5.5 is trained without its MTP block and requires MTP depth 0, but the configuration has "
            f"num_nextn_predict_layers={depth!r}. Set `model.num_nextn_predict_layers: 0` in the recipe; the MTP "
            f"weights of the checkpoint are then not loaded."
        )
    if not isinstance(ratios, (list, tuple)):
        raise ValueError("compress_ratios must list one entry per decoder layer")
    if len(ratios) == layers + 1:
        if result.mtp_shared_last_layer is not True or ratios[-1] != ratios[-2]:
            raise ValueError(
                "Extra schedule entry is not a compatible shared MTP layer: compress_ratios has one entry more "
                "than num_hidden_layers, which is accepted only when mtp_shared_last_layer is true and the entry "
                "repeats the last decoder layer"
            )
        result.compress_ratios = list(ratios[:-1])
    elif len(ratios) != layers:
        raise ValueError(
            f"Unexpected decoder schedule length: compress_ratios has {len(ratios)} entries for "
            f"num_hidden_layers={layers}"
        )
    result.mtp_shared_last_layer = False
    Ernie5_5ModelPlan.from_config(result)
    return result


def validate_dsa_rope_layout(config: Ernie5_5Config) -> None:
    """Reject DSA checkpoints whose rotary channels are not stored as adjacent pairs.

    ``mqa_latent_rope_adjacent_pairing`` takes precedence over ``rope_interleave`` when it is set.

    Args:
        config: Model configuration.

    Raises:
        ValueError: If neither field holds a boolean.
        NotImplementedError: If the checkpoint declares the non-adjacent layout, which would need a permutation of
            the rotary rows that is not implemented.
    """
    adjacent = config.mqa_latent_rope_adjacent_pairing
    if adjacent is None:
        adjacent = config.rope_interleave
    if not isinstance(adjacent, bool):
        raise ValueError("DSA checkpoint must explicitly declare a boolean RoPE pairing")
    if not adjacent:
        raise NotImplementedError("Non-adjacent DSA RoPE checkpoint rows require an unsupported permutation")


def hca_rope_scaling(config: Ernie5_5Config) -> dict[str, HFConfigValue]:
    """Return the YaRN settings of the HCA compress rotary embedding.

    Args:
        config: Model configuration.

    Returns:
        Mapping with ``type`` set to ``"yarn"`` and the keys ``factor``, ``original_max_position_embeddings``,
        ``beta_fast`` and ``beta_slow`` of ``config.rope_parameters``.

    Raises:
        ValueError: If ``config.rope_parameters`` lacks one of the four keys.
    """
    parameters = config.rope_parameters or {}
    missing = [name for name in _HCA_YARN_KEYS if name not in parameters]
    if missing:
        raise ValueError(f"Missing released HCA YaRN settings in rope_parameters: {missing}")
    return {"type": "yarn", **{name: parameters[name] for name in _HCA_YARN_KEYS}}


def validate_hca_options(config: Ernie5_5Config) -> None:
    """Reject the HCA configuration branches that are not implemented.

    Args:
        config: Model configuration.

    Raises:
        NotImplementedError: If Q/K normalization is disabled, the HCA rotary type is not YaRN, or the fused
            sigmoid gate is enabled.
        ValueError: If the YaRN settings are incomplete.
    """
    if config.use_qk_norm is not True:
        raise NotImplementedError("Only HCA with Q/K normalization is supported")
    if config.hca_rope_type != "yarn":
        raise NotImplementedError("Only the validated YaRN HCA RoPE configuration is supported")
    if config.sigmoid_gate_fusion is not False:
        raise NotImplementedError("Fused sigmoid gate configuration is not validated")
    hca_rope_scaling(config)


def main_rope_theta(config: Ernie5_5Config) -> float:
    """Return the rotary base of the DSA layers.

    Args:
        config: Model configuration. ``rope_parameters["rope_theta"]`` holds the base; ``rotary_base``, when set,
            must repeat it.

    Returns:
        The positive rotary base.

    Raises:
        ValueError: If the base is missing or not positive, or ``rotary_base`` differs from it.
    """
    theta = (config.rope_parameters or {}).get("rope_theta")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)) or theta <= 0:
        raise ValueError(f"A positive main RoPE base is required; got rope_theta={theta!r}")
    if config.rotary_base is not None and config.rotary_base != theta:
        raise ValueError(f"Conflicting RoPE base values: rope_theta={theta!r}, rotary_base={config.rotary_base!r}")
    return theta


def validate_architecture(config: Ernie5_5Config) -> None:
    """Reject the settings that select an architecture the ERNIE 5.5 model does not implement.

    The model implements one variant of the attention, feed-forward and output options of the released
    checkpoint. The options it reads are checked individually; the released flags it hard-codes are compared with
    the implemented value.

    Args:
        config: Model configuration.

    Raises:
        ValueError: If the layer schedule is invalid, the activation is not SiTU, the MoE does not have exactly
            one shared expert and one router group with split-feature routing and a latent norm, the mHC and
            RMSNorm epsilons differ, or a rotary setting is missing or inconsistent.
        NotImplementedError: If a DSA layer uses the non-adjacent rotary layout, an HCA layer uses an option
            that is not implemented, or a released flag differs from the implemented variant.
    """
    attention_kinds = {layer.attention for layer in Ernie5_5ModelPlan.from_config(config).decoder_layers}
    if "dsa" in attention_kinds:
        validate_dsa_rope_layout(config)
    if "hca" in attention_kinds:
        validate_hca_options(config)
    if config.hidden_act != "situ" or config.n_shared_experts != 1:
        raise ValueError(
            f"ERNIE 5.5 requires hidden_act='situ' and one shared expert; got hidden_act={config.hidden_act!r}, "
            f"n_shared_experts={config.n_shared_experts!r}"
        )
    if config.n_group != 1 or config.topk_group != 1:
        raise ValueError(
            f"ERNIE 5.5 supports one router group; got n_group={config.n_group!r}, topk_group={config.topk_group!r}"
        )
    if not config.moe_split_feature_routing or not config.latent_moe_use_norm:
        raise ValueError(
            "ERNIE 5.5 requires split-feature routing and latent normalization; got "
            f"moe_split_feature_routing={config.moe_split_feature_routing!r}, "
            f"latent_moe_use_norm={config.latent_moe_use_norm!r}"
        )
    if config.hc_eps != config.rms_norm_eps:
        raise ValueError(
            f"ERNIE 5.5 requires equal mHC and RMSNorm epsilons; got hc_eps={config.hc_eps!r}, "
            f"rms_norm_eps={config.rms_norm_eps!r}"
        )
    main_rope_theta(config)
    # Released flags whose single implemented value is built into the layers.
    implemented = (
        ("qk_norm_type", config.qk_norm_type, "per_head"),
        ("use_gated_attn", config.use_gated_attn, True),
        ("gated_attn_act", config.gated_attn_act, "sigmoid"),
        ("gated_attn_use_q_lora", config.gated_attn_use_q_lora, True),
        ("use_vha_attention", config.use_vha_attention, True),
        ("add_full_attention_sink_bias", config.add_full_attention_sink_bias, True),
        ("attention_bias", config.attention_bias, False),
        ("hybrid_mla_attention", config.hybrid_mla_attention, "mqa_dsa"),
        ("mqa_split_kv_b_proj", config.mqa_split_kv_b_proj, True),
        ("use_fast_hadamard", config.use_fast_hadamard, True),
        ("enable_hyper_connections", config.enable_hyper_connections, True),
        ("scoring_func", config.scoring_func, "sigmoid"),
        ("routed_scaling_factor_learnable", config.routed_scaling_factor_learnable, True),
        ("multimax", config.multimax, "lm_head"),
    )
    different = [
        f"{name}={value!r} (implemented: {expected!r})"
        for name, value, expected in implemented
        # 1 and 0 are not accepted in place of true and false.
        if value != expected or isinstance(value, bool) != isinstance(expected, bool)
    ]
    if different:
        raise NotImplementedError(f"ERNIE 5.5 implements one variant of these settings; got {', '.join(different)}")
