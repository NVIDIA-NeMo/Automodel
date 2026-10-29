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

"""Configuration for the Mistral Large 4 Mistral Large 4 text backbone."""

import copy
import json
from pathlib import Path
from typing import TypeAlias

from transformers import PretrainedConfig

from nemo_automodel.components.models.mistral_large4.flops import model_flops

# JSON/YAML values accepted at the checkpoint/configuration boundary.
ConfigValue: TypeAlias = bool | int | float | str | list["ConfigValue"] | dict[str, "ConfigValue"] | None


class MistralLarge4Config(PretrainedConfig):
    """Native Large 4 contract with explicit source precision metadata.

    Native vision metadata is consumed by the conditional-generation class.
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
        source_quantization: dict[str, ConfigValue] | None = None,
        vision_config: dict[str, ConfigValue] | None = None,
        mtp_config: dict[str, ConfigValue] | None = None,
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
        self.source_quantization = copy.deepcopy(source_quantization)
        self.vision_config = copy.deepcopy(vision_config)
        self.mtp_config = copy.deepcopy(mtp_config)
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
    def from_native(cls, path: str | Path) -> "MistralLarge4Config":
        """Read params.json without modifying the source checkpoint."""
        source = Path(path)
        if source.is_dir():
            source = source / "params.json"
        return _config_from_params(json.loads(source.read_text()))


def _section(params: dict[str, ConfigValue], name: str) -> dict[str, ConfigValue]:
    """Resolve a native JSON object at the checkpoint-format boundary."""
    value = params.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"params.{name} must be an object")
    return value


def _pattern(value: ConfigValue, depth: int, default: str) -> list[str]:
    """Expand native comma-separated patterns exactly as the vendor converter."""
    if value is None:
        return [default] * depth
    if not isinstance(value, str) or not value or any(not x for x in value.split(",")):
        raise ValueError("Per-layer patterns must be nonempty comma-separated strings")
    parts = value.split(",")
    return (parts * ((depth + len(parts) - 1) // len(parts)))[:depth]


def _config_from_params(params: dict[str, ConfigValue]) -> MistralLarge4Config:
    """Convert the released native configuration with explicit unsupported-setting checks."""
    depth = int(params["n_layers"])
    if depth <= 0:
        raise ValueError("n_layers must be positive")
    if (
        params.get("q_lora_rank") is not None
        or params.get("ragged_attention") is not None
        or params.get("llama_4_scaling") is not None
    ):
        raise ValueError("Mistral Large 4 backbone requires direct queries and full causal MLA")
    if params.get("n_kv_heads", params["n_heads"]) != params["n_heads"]:
        raise ValueError("Mistral Large 4 MLA expects equal query and expanded KV head counts")
    dsa, moe, yarn = (_section(params, key) for key in ("dsa", "moe", "yarn"))
    quant = _section(params, "quantization")
    if quant and (
        quant.get("qscheme_act") != "BLOCK"
        or not quant.get("qdynamic_act")
        or not quant.get("use_ue8m0_for_qscales")
        or quant.get("qformat_weight") != "fp8_e4m3"
        or quant.get("block_size_out") != 128
        or quant.get("block_size_in") != 128
        or quant.get("qscheme_weight") != "BLOCK"
        or quant.get("quantize_kv_a", False)
    ):
        raise ValueError("Only E4M3 128x128 block weights with unquantized KV-A are supported")
    if yarn.get("apply_scale", False):
        raise ValueError("Mistral Large 4 does not support YaRN magnitude scaling")
    if moe.get("num_expert_groups", 1) != 1 or moe.get("num_expert_groups_per_tok", 1) != 1:
        raise ValueError("Mistral Large 4 uses ungrouped softmax top-k routing")
    rope_dim = int(params["qk_rope_head_dim"])
    if dsa.get("rope_head_dim", rope_dim) not in (rope_dim, -1):
        raise ValueError("Indexer rotary dimension must match MLA's rotary dimension")
    if dsa.get("nope", False):
        raise ValueError("Use dsa.match_nope for per-layer indexer NoPE")
    nope = _pattern(params.get("nope"), depth, "rope")
    if set(nope) - {"rope", "nope"}:
        raise ValueError("nope pattern accepts only rope/nope")
    sources = _section(params, "kv_sharing").get("source_layers_str")
    if sources is not None and not isinstance(sources, str):
        raise ValueError("kv_sharing.source_layers_str must be comma-separated integers")
    mapping = {
        "dim": "hidden_size",
        "hidden_dim": "intermediate_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "softmax_tanh_gating": "final_logit_softcapping",
        "tied_embeddings": "tie_word_embeddings",
    }
    values = {
        mapping.get(k, k): params[k]
        for k in (
            "dim",
            "hidden_dim",
            "n_layers",
            "n_heads",
            "norm_eps",
            "vocab_size",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "rope_theta",
            "max_position_embeddings",
            "lerope",
            "ssmax_scaling",
            "softmax_tanh_gating",
            "tied_embeddings",
        )
        if k in params
    }
    values.update(
        nope_layer_flags=[x == "nope" for x in nope],
        kv_source_layers=list(map(int, sources.split(","))) if sources else list(range(depth)),
        index_sharing=_pattern(dsa.get("sharing"), depth, "full"),
        indexer_nope_layer_flags=[x == "nope" for x in nope] if dsa.get("match_nope", False) else [False] * depth,
        index_topk=dsa["topk"],
        index_n_heads=dsa["n_heads"],
        index_head_dim=dsa["head_dim"],
        indexer_norm_eps=dsa.get("norm_eps", 1e-6),
        indexer_ssmax_scaling=dsa.get("ssmax_scaling", False),
        fp8_indexer=dsa.get("fp8_indexer", True),
        fp8_kv_cache=dsa.get("fp8_kv_cache", False),
        first_k_dense_replace=moe.get("first_k_dense_replace", 0),
        moe_layer_freq=moe.get("route_every_n", 1),
        n_routed_experts=moe["num_experts"],
        n_shared_experts=moe.get("num_shared_experts", 0),
        num_experts_per_tok=moe["num_experts_per_tok"],
        moe_intermediate_size=moe["expert_hidden_dim"],
        routed_scaling_factor=moe.get("routed_scale", 1.0),
        norm_topk_prob=True,
        yarn_factor=yarn.get("factor", 1.0),
        yarn_original_max_position_embeddings=yarn.get(
            "original_max_position_embeddings", params["max_position_embeddings"]
        ),
        yarn_beta_fast=yarn.get("beta", 32.0),
        yarn_beta_slow=yarn.get("alpha", 1.0),
        dtype="bfloat16",
    )
    if quant:
        values["source_quantization"] = quant
        # Existing initialization detection selects bounded adapter load parts.
        # The shared loader already bypasses this metadata on native resume.
        values["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    values["vision_config"] = _section(params, "vision_encoder")
    values["mtp_config"] = _section(params, "mtp")
    return MistralLarge4Config(**values)
