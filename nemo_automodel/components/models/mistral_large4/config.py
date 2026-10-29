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

"""Configuration for Mistral Large 4."""

import copy
from typing import TypeAlias

from transformers import PretrainedConfig

from nemo_automodel.components.models.mistral_large4.flops import model_flops

# JSON/YAML values accepted at the checkpoint/configuration boundary.
ConfigValue: TypeAlias = bool | int | float | str | list["ConfigValue"] | dict[str, "ConfigValue"] | None


class MistralLarge4Config(PretrainedConfig):
    """Native Large 4 contract read from the released Transformers config.

    Vision settings are consumed by the conditional-generation class.
    The causal class remains text-only; draft execution is not enabled.
    """

    model_type = "mistral_large4"
    flops_formula = staticmethod(model_flops)

    def __init__(
        self,
        hidden_size: int = 6144,
        intermediate_size: int = 24576,
        num_hidden_layers: int = 72,
        num_attention_heads: int = 64,
        vocab_size: int = 204800,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 1048576,
        yarn_factor: float = 2048.0,
        yarn_original_max_position_embeddings: int = 512,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        lerope: bool = True,
        ssmax_scaling: bool = True,
        final_logit_softcapping: float = 30.0,
        nope_layer_flags: list[bool] | None = None,
        kv_source_layers: list[int] | None = None,
        index_sharing: list[str] | None = None,
        indexer_nope_layer_flags: list[bool] | None = None,
        index_topk: int = 2048,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        indexer_norm_eps: float = 1e-6,
        indexer_ssmax_scaling: bool = True,
        fp8_indexer: bool = True,
        fp8_kv_cache: bool = True,
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        n_routed_experts: int = 128,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 4,
        moe_intermediate_size: int = 6144,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        tie_word_embeddings: bool = False,
        vision_config: dict[str, ConfigValue] | None = None,
        **kwargs: ConfigValue,
    ) -> None:
        kwargs.setdefault("architectures", ["MistralLarge4ForCausalLM"])
        if tie_word_embeddings:
            raise ValueError("Mistral Large 4 requires untied embeddings")
        for name, expected in (
            ("q_lora_rank", None),
            ("hidden_act", "silu"),
            ("scoring_func", "softmax"),
            ("num_key_value_heads", num_attention_heads),
            ("head_dim", qk_nope_head_dim + qk_rope_head_dim),
        ):
            if name in kwargs and kwargs[name] != expected:
                raise ValueError(f"Mistral Large 4 requires {name}={expected!r}")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.yarn_factor = yarn_factor
        self.yarn_original_max_position_embeddings = yarn_original_max_position_embeddings
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.lerope = lerope
        self.ssmax_scaling = ssmax_scaling
        self.final_logit_softcapping = final_logit_softcapping
        self.nope_layer_flags = (
            list(nope_layer_flags) if nope_layer_flags is not None else [i % 4 == 3 for i in range(num_hidden_layers)]
        )
        self.kv_source_layers = (
            list(kv_source_layers)
            if kv_source_layers is not None
            else [35 if i >= 39 and i % 4 == 3 else i for i in range(num_hidden_layers)]
        )
        self.index_sharing = (
            list(index_sharing)
            if index_sharing is not None
            else ["shared" if i >= 17 and i % 4 in (1, 2) else "full" for i in range(num_hidden_layers)]
        )
        self.indexer_nope_layer_flags = (
            list(indexer_nope_layer_flags) if indexer_nope_layer_flags is not None else list(self.nope_layer_flags)
        )
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.indexer_norm_eps = indexer_norm_eps
        self.indexer_ssmax_scaling = indexer_ssmax_scaling
        self.fp8_indexer = fp8_indexer
        self.fp8_kv_cache = fp8_kv_cache
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.vision_config = copy.deepcopy(vision_config)
        self.q_lora_rank = None
        self.scoring_func = "softmax"
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.hidden_act = "silu"
        self.validate_architecture()

    def validate_architecture(self) -> None:
        """Reject settings that would silently change the checkpoint's semantics."""
        if (
            min(
                self.hidden_size,
                self.intermediate_size,
                self.vocab_size,
                self.moe_intermediate_size,
                self.v_head_dim,
                self.qk_nope_head_dim,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.index_topk,
                self.index_n_heads,
                self.kv_lora_rank,
                self.moe_layer_freq,
            )
            <= 0
        ):
            raise ValueError("Model dimensions, depth, MoE frequency and index_topk must be positive")
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2 or self.index_head_dim < self.qk_rope_head_dim:
            raise ValueError("RoPE dimensions must be positive, even and fit the indexer head")
        if not self.norm_topk_prob or self.n_shared_experts < 0 or self.first_k_dense_replace < 0:
            raise ValueError("Large 4 requires normalized router probabilities and nonnegative expert/layer counts")
        if self.final_logit_softcapping <= 0:
            raise ValueError("Large 4 requires a positive final logit softcap")
        if self.yarn_factor < 1 or self.yarn_original_max_position_embeddings <= 0:
            raise ValueError("YaRN requires factor >= 1 and a positive original context length")
        if not 0 < self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")
        for name, values in (
            ("nope_layer_flags", self.nope_layer_flags),
            ("kv_source_layers", self.kv_source_layers),
            ("index_sharing", self.index_sharing),
            ("indexer_nope_layer_flags", self.indexer_nope_layer_flags),
        ):
            if len(values) != self.num_hidden_layers:
                raise ValueError(f"{name} must contain exactly num_hidden_layers entries")
        if self.index_sharing[0] != "full" or set(self.index_sharing) - {"full", "shared"}:
            raise ValueError("IndexShare requires an initial full indexer and only full/shared entries")
        for layer, source in enumerate(self.kv_source_layers):
            if not 0 <= source <= layer or self.kv_source_layers[source] != source:
                raise ValueError(f"Layer {layer} must share KV with an earlier owning layer")
            if self.nope_layer_flags[layer] != self.nope_layer_flags[source]:
                raise ValueError(f"Layer {layer} and its KV source must use the same rotary layout")

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, ConfigValue], **kwargs: ConfigValue
    ) -> "MistralLarge4Config | tuple[MistralLarge4Config, dict[str, ConfigValue]]":
        """Read either the released Transformers config or a saved native config.

        Args:
            config_dict: ``config.json`` contents. The released checkpoint nests
                text and vision settings under ``text_config``/``vision_config``;
                configs saved by this class use the flat native fields.
            **kwargs: Existing ``PretrainedConfig.from_dict`` overrides.

        Returns:
            The configuration, plus unused kwargs when requested by the caller.
        """
        if "text_config" in config_dict:
            config_dict = _config_from_hf(config_dict)
        return super().from_dict(config_dict, **kwargs)


def _section(config: dict[str, ConfigValue], name: str) -> dict[str, ConfigValue]:
    """Resolve a nested JSON object at the checkpoint-format boundary."""
    value = config.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config.{name} must be an object")
    return value


_TEXT_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "vocab_size",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "rms_norm_eps",
    "max_position_embeddings",
    "final_logit_softcapping",
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    "first_k_dense_replace",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "routed_scaling_factor",
    "tie_word_embeddings",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
)
_VISION_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_channels",
    "patch_size",
    "image_size",
)


def _config_from_hf(config: dict[str, ConfigValue]) -> dict[str, ConfigValue]:
    """Map the released Transformers composite config onto the native fields."""
    text, vision = _section(config, "text_config"), _section(config, "vision_config")
    rope = _section(text, "rope_parameters")
    depth = int(text["num_hidden_layers"])
    dense = int(text.get("first_k_dense_replace", 0))
    if rope.get("rope_type") != "yarn" or rope.get("attention_factor", 1.0) != 1.0:
        raise ValueError("Mistral Large 4 uses YaRN frequencies without magnitude scaling")
    if text.get("n_group", 1) != 1 or text.get("topk_group", 1) != 1:
        raise ValueError("Mistral Large 4 uses ungrouped softmax top-k routing")
    if text.get("attention_bias", False):
        raise ValueError("Mistral Large 4 attention projections have no bias")
    if set(text.get("layer_types", ["indexed_attention"])) != {"indexed_attention"}:
        raise ValueError("Mistral Large 4 uses indexed sparse attention in every layer")
    if text.get("mlp_layer_types", ["dense"] * dense + ["sparse"] * (depth - dense)) != (
        ["dense"] * dense + ["sparse"] * (depth - dense)
    ):
        raise ValueError("mlp_layer_types must be dense for exactly the first first_k_dense_replace layers")
    rope_layers = text["no_rope_layers"]
    if not isinstance(rope_layers, list) or set(rope_layers) - {0, 1}:
        raise ValueError("no_rope_layers must list 1 for RoPE layers and 0 for NoPE layers")
    values = {name: text[name] for name in _TEXT_FIELDS if name in text}
    values.update(
        rope_theta=rope["rope_theta"],
        yarn_factor=rope["factor"],
        yarn_original_max_position_embeddings=rope["original_max_position_embeddings"],
        yarn_beta_fast=rope["beta_fast"],
        yarn_beta_slow=rope["beta_slow"],
        nope_layer_flags=[flag == 0 for flag in rope_layers],
        kv_source_layers=text["kv_source_layers"],
        index_sharing=text["indexer_types"],
    )
    for name in ("architectures", "dtype", "quantization_config", "transformers_weights"):
        if name in config:
            values[name] = config[name]
    if vision:
        settings = {name: vision[name] for name in _VISION_FIELDS if name in vision}
        settings.update(
            max_image_size=vision["image_size"],
            rope_theta=_section(vision, "rope_parameters")["rope_theta"],
            spatial_merge_size=config["spatial_merge_size"],
            adapter_bias=config.get("multimodal_projector_bias", False),
            image_token_id=config["image_token_index"],
        )
        values["vision_config"] = settings
    return values
