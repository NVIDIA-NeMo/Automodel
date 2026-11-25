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

"""Hugging Face checkpoint configuration for tencent/Hy4-flash (``model_type`` ``hy_v4_flash``).

Field names follow the released ``config.json`` (revision ``eccd3123eec73f2040f4c9ed0de1883593470510``).
The transformers reference (PR 50903) renames a few of them; the aliases it accepts are accepted here too
so either spelling of the same checkpoint loads.
"""

from __future__ import annotations

import math
from typing import Literal, NotRequired, TypeAlias, TypedDict

from transformers import PretrainedConfig

AttentionType: TypeAlias = Literal["deepseek_sparse_attention", "sliding_attention"]
MlpType: TypeAlias = Literal["dense", "sparse"]

# Layer-type spellings seen in the released config and in the transformers reference.
_LAYER_TYPE_ALIASES: dict[str, AttentionType] = {
    "deepseek_sparse_attention": "deepseek_sparse_attention",
    "indexed_attention": "deepseek_sparse_attention",
    "sliding_attention": "sliding_attention",
}


class HY4FlashRopeParameters(TypedDict):
    """Rotary parameters of one attention type, as stored in ``rope_parameters``."""

    rope_type: Literal["default"]
    rope_theta: float
    mrope_section: list[int]
    mrope_interleaved: NotRequired[bool]


# ``engram_layer_head_vocab_sizes[str(one_based_layer_id)][ngram_order - 2][head]`` is one hash modulus.
EngramHeadVocabSizes: TypeAlias = dict[str, list[list[int]]]


class HY4FlashTextConfig(PretrainedConfig):
    """Text backbone: hyper-connected gated MLA with SWA / compressed-indexer layers, latent MoE and Engram.

    Layer indices in ``engram_layer_ids`` are **one-based**: the reference builds the Engram block of decoder
    layer ``i`` (zero-based) when ``i + 1 in engram_layer_ids``, so the released ``[2, 12, 22, 32]`` means
    zero-based layers 1, 11, 21 and 31, matching the model card.
    """

    model_type = "hy_v4_flash_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    @staticmethod
    def flops_formula(config: HY4FlashTextConfig, gbs: int = 1, seq_len: int | None = None) -> float:
        """Model FLOPs of one step (see ``hy_v4_flash.flops``); picked up by ``get_flops_formula_for_hf_config``."""
        from nemo_automodel.components.models.hy_v4_flash.flops import hy_v4_flash_flops

        return hy_v4_flash_flops(config, gbs, seq_len)

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 3072,
        intermediate_size: int = 13824,
        moe_intermediate_size: int = 1536,
        routed_expert_hidden_size: int = 1536,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 32,
        swa_num_attention_heads: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        n_routed_experts: int = 512,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 16,
        routed_scaling_factor: float = 3.998,
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        swiglu_limit: float = 10.0,
        q_lora_rank: int = 1024,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        swa_qk_nope_head_dim: int = 192,
        swa_qk_rope_head_dim: int = 64,
        swa_v_head_dim: int = 256,
        sliding_window: int = 512,
        learnable_sink_init: float = 0.0,
        mlp_layer_types: list[MlpType] | None = None,
        layer_types: list[str] | None = None,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 16,
        compressed_indexer_compress_ratio: int = 4,
        hc_mult: int = 4,
        hc_magnitude: float = 2.0,
        hc_eps: float = 1e-6,
        rope_parameters: dict[str, HY4FlashRopeParameters] | None = None,
        layer_rope_parameters: dict[str, HY4FlashRopeParameters] | None = None,
        engram_layer_ids: list[int] | None = None,
        engram_max_ngram_size: int = 3,
        engram_n_head_per_ngram: int = 8,
        engram_n_embed_per_ngram: int = 768,
        engram_layer_head_vocab_sizes: EngramHeadVocabSizes | None = None,
        engram_pad_id: int = 120002,
        engram_fp8_embedding: bool = False,
        engram_seed: int = 0,
        engram_lookup_table_size: int = 120818,
        dtype: str = "bfloat16",
        pad_token_id: int | None = 120002,
        bos_token_id: int | None = 120000,
        eos_token_id: int | list[int] | None = 120025,
        tie_word_embeddings: bool = False,
        **kwargs: object,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.swa_num_attention_heads = swa_num_attention_heads
        # MLA expands one K/V per query head, so keys are never grouped.
        kwargs.pop("num_key_value_heads", None)
        self.num_key_value_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.swiglu_limit = swiglu_limit
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.swa_qk_nope_head_dim = swa_qk_nope_head_dim
        self.swa_qk_rope_head_dim = swa_qk_rope_head_dim
        self.swa_v_head_dim = swa_v_head_dim
        self.sliding_window = sliding_window
        self.learnable_sink_init = learnable_sink_init
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        # The released config names the pooling ratio ``compressed_indexer_compress_ratio``; the reference ``index_kpool``.
        self.index_kpool = kwargs.pop("index_kpool", compressed_indexer_compress_ratio)
        self.hc_mult = hc_mult
        self.hc_magnitude = hc_magnitude
        self.hc_eps = hc_eps
        self.engram_layer_ids = None if engram_layer_ids is None else list(engram_layer_ids)
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_n_head_per_ngram = engram_n_head_per_ngram
        self.engram_n_embed_per_ngram = engram_n_embed_per_ngram
        self.engram_layer_head_vocab_sizes = engram_layer_head_vocab_sizes
        self.engram_pad_id = engram_pad_id
        self.engram_fp8_embedding = engram_fp8_embedding
        self.engram_seed = engram_seed
        self.engram_lookup_table_size = engram_lookup_table_size

        first_k_dense = int(kwargs.pop("first_k_dense_replace", 1))
        self.mlp_layer_types = (
            ["dense"] * min(first_k_dense, num_hidden_layers) + ["sparse"] * max(num_hidden_layers - first_k_dense, 0)
            if mlp_layer_types is None
            else list(mlp_layer_types)
        )
        if layer_types is None:
            # Two sparse layers, then three sliding layers for every sparse one.
            layer_types = [
                "deepseek_sparse_attention" if i < 2 or (i - 1) % 4 == 0 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        try:
            self.layer_types = [_LAYER_TYPE_ALIASES[t] for t in layer_types]
        except KeyError as exc:
            raise ValueError(f"unsupported layer type {exc.args[0]!r}; expected one of {sorted(_LAYER_TYPE_ALIASES)}") from exc

        # The base class rewrites ``rope_parameters`` with a flat default alongside the per-type entries, so the
        # per-type table is kept under its own name; a saved config carries both and the own name wins.
        source = layer_rope_parameters if layer_rope_parameters is not None else rope_parameters
        rope = {
            _LAYER_TYPE_ALIASES[k]: dict(v) for k, v in (source or {}).items() if k in _LAYER_TYPE_ALIASES
        }
        default_theta = {"deepseek_sparse_attention": 1.0e7, "sliding_attention": 1.0e4}
        for layer_type, theta in default_theta.items():
            params = rope.setdefault(layer_type, {"rope_type": "default"})
            params.setdefault("rope_theta", theta)
            params.setdefault("mrope_section", [12, 10, 10])
        self.layer_rope_parameters = rope

        # Fields of the released config that the model does not consume are retained by the base class so a
        # saved config round-trips; they are listed so a future semantic change is a conscious decision.
        self._validate()
        super().__init__(
            dtype=kwargs.pop("torch_dtype", dtype),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ------------------------------------------------------------------ derived shapes
    def attention_heads(self, attention_type: AttentionType) -> int:
        """Number of query heads of a layer of the given attention type."""
        return self.num_attention_heads if attention_type == "deepseek_sparse_attention" else self.swa_num_attention_heads

    def attention_dims(self, attention_type: AttentionType) -> tuple[int, int, int]:
        """Return ``(qk_nope_head_dim, qk_rope_head_dim, v_head_dim)`` of the given attention type."""
        if attention_type == "deepseek_sparse_attention":
            return self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        return self.swa_qk_nope_head_dim, self.swa_qk_rope_head_dim, self.swa_v_head_dim

    @property
    def engram_layer_indices(self) -> list[int]:
        """Zero-based indices of the decoder layers that carry an Engram block."""
        return [layer_id - 1 for layer_id in (self.engram_layer_ids or [])]

    # ------------------------------------------------------------------ validation
    def _validate(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "routed_expert_hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "swa_num_attention_heads",
            "n_routed_experts",
            "num_experts_per_tok",
            "q_lora_rank",
            "kv_lora_rank",
            "sliding_window",
            "index_topk",
            "index_head_dim",
            "index_n_heads",
            "index_kpool",
            "hc_mult",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if len(self.layer_types) != self.num_hidden_layers or len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types and mlp_layer_types must each have num_hidden_layers entries")
        if any(t not in ("dense", "sparse") for t in self.mlp_layer_types):
            raise ValueError(f"mlp_layer_types entries must be 'dense' or 'sparse', got {self.mlp_layer_types}")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok must not exceed n_routed_experts")
        if self.n_group != 1 or self.topk_group != 1:
            raise ValueError("Hy4-flash selects experts globally; only n_group=1, topk_group=1 is supported")
        if self.n_shared_experts != 1:
            raise ValueError("Hy4-flash has exactly one shared expert")
        if self.index_topk % self.index_kpool:
            raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool})")
        for name in ("rms_norm_eps", "hc_eps", "routed_scaling_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        for layer_type in set(self.layer_types):
            _, rope_dim, _ = self.attention_dims(layer_type)
            section = self.layer_rope_parameters[layer_type]["mrope_section"]
            if len(section) != 3 or sum(section) != rope_dim // 2:
                raise ValueError(
                    f"mrope_section must have three entries summing to {rope_dim // 2} for {layer_type!r}, got {section}"
                )
        if self.index_head_dim < self.qk_rope_head_dim:
            raise ValueError("index_head_dim must be at least qk_rope_head_dim")
        if self.engram_layer_ids:
            if self.engram_max_ngram_size < 2:
                raise ValueError("engram_max_ngram_size must be at least 2 when Engram is enabled")
            if self.engram_n_embed_per_ngram % self.engram_n_head_per_ngram:
                raise ValueError("engram_n_embed_per_ngram must be divisible by engram_n_head_per_ngram")
            if not 0 <= self.engram_pad_id < self.engram_lookup_table_size:
                raise ValueError("engram_pad_id must lie inside the lookup table")
            orders = self.engram_max_ngram_size - 1
            for layer_id in self.engram_layer_ids:
                if not 1 <= layer_id <= self.num_hidden_layers:
                    raise ValueError(f"engram layer id {layer_id} is outside the one-based range 1..{self.num_hidden_layers}")
                sizes = (self.engram_layer_head_vocab_sizes or {}).get(str(layer_id))
                if (
                    not isinstance(sizes, list)
                    or len(sizes) != orders
                    or any(len(order) != self.engram_n_head_per_ngram for order in sizes)
                ):
                    raise ValueError(
                        f"engram_layer_head_vocab_sizes[{layer_id!r}] must hold {orders} lists of "
                        f"{self.engram_n_head_per_ngram} moduli, got {sizes!r}"
                    )


class HY4FlashVisionConfig(PretrainedConfig):
    """Vision tower hyper-parameters (a 27-layer ViT plus the ``rmsnorm_mlp_adapter`` projector)."""

    model_type = "hy_v4_flash_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 3072,
        rms_norm_eps: float = 1e-5,
        rope_2d_theta: float = 10000.0,
        hidden_act: str = "gelu",
        max_image_size: int = 2048,
        interpolate_mode: str = "bilinear",
        **kwargs: object,
    ) -> None:
        if hidden_act != "gelu":
            raise ValueError(f"the vision tower implements the released gelu activation only, got {hidden_act!r}")
        self.max_image_size = max_image_size
        self.interpolate_mode = interpolate_mode
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_2d_theta = rope_2d_theta
        self.hidden_act = hidden_act
        # The released vision block carries the parent's model_type, which would re-enter this class.
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)


class HY4FlashConfig(PretrainedConfig):
    """Composite image-text configuration (``HYV4FlashForConditionalGeneration``)."""

    model_type = "hy_v4_flash"
    sub_configs = {"text_config": HY4FlashTextConfig, "vision_config": HY4FlashVisionConfig}

    @staticmethod
    def flops_formula(config: HY4FlashConfig, gbs: int = 1, seq_len: int | None = None) -> float:
        """Model FLOPs of one step of the text backbone; the vision tower is not counted."""
        from nemo_automodel.components.models.hy_v4_flash.flops import hy_v4_flash_flops

        return hy_v4_flash_flops(config, gbs, seq_len)

    def __init__(
        self,
        text_config: dict[str, object] | HY4FlashTextConfig | None = None,
        vision_config: dict[str, object] | HY4FlashVisionConfig | None = None,
        image_token_id: int = 120120,
        video_token_id: int = 120124,
        image_start_token_id: int = 120118,
        image_end_token_id: int = 120119,
        image_newline_token_id: int = 120121,
        tie_word_embeddings: bool = False,
        **kwargs: object,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = HY4FlashTextConfig(**{k: v for k, v in text_config.items() if k != "model_type"})
        self.text_config = text_config if text_config is not None else HY4FlashTextConfig()
        if isinstance(vision_config, dict):
            vision_config = HY4FlashVisionConfig(**vision_config)
        self.vision_config = vision_config if vision_config is not None else HY4FlashVisionConfig()
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_newline_token_id = image_newline_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
