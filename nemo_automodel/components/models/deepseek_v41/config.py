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

"""Configuration for DeepSeek V4.1 (``deepseek-ai/DeepSeek-V4.1-Flash``).

The released ``config.json`` nests the language backbone under ``text_config``
and the vision tower under ``vision_config``. This config preserves the flat
text-field API used by the decoder and materializes vision metadata as a typed
configuration. Defaults match the released text and vision configuration;
checkpoint-free tiny models can explicitly override their schedules and disable
the vision tower with ``num_hidden_layers=0`` in ``vision_config``.

Architecture summary (see ``DeepSeek_V41_Tech_Report.pdf``):

* Causal Encoder-Decoder (CED): the decoder's global KV is projected from the
  last encoder hidden state.  In the checkpoint this is expressed purely by
  ``kv_source_layer_ids``: the first decoder layer (``20``) is a KV source that
  reads its own input, i.e. the encoder output, and every later decoder layer
  reuses that cache.
* Compressed Sparse Attention 2 (CSA2): every attention layer is statically
  one of Full (``kv_source`` + ``index_source``), Reindex (``index_source``
  only) or Reuse (neither).  ``compress_ratios[i]`` is ``0`` for pure
  sliding-window layers, ``r`` for layers that attend over the shared
  ``r``-to-1 compressed KV on top of the local window.
* Hierarchical Sparse Indexer: ``candidate_source_layer_id`` builds a block
  candidate pool that later Reindex layers search inside.
* Single-pass mHC: each block's attention input mixing uses the coefficients
  produced by the previous block's FFN site.
* Engram conditional memory at ``engram_layer_ids``.
* DSpark draft layers live under ``mtp.*`` in the checkpoint and are not part
  of the trainable backbone here.
"""

from __future__ import annotations

import math
from typing import Any

from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerFast


class DeepseekV41VisionConfig(PretrainedConfig):
    """Configuration of the released 2D-RoPE vision encoder and image sizing."""

    model_type = "deepseek_v41_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        num_hidden_layers: int = 32,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        patch_size: int = 14,
        rope_theta: float = 10000.0,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 295936,
        max_wh_ratio: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio
        for name, value in (
            ("hidden_size", hidden_size),
            ("num_attention_heads", num_attention_heads),
            ("intermediate_size", intermediate_size),
            ("patch_size", patch_size),
            ("downsample_ratio", downsample_ratio),
            ("max_image_tokens", max_image_tokens),
            ("min_pixels", min_pixels),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"vision {name} must be a positive integer, got {value!r}")
        if type(num_hidden_layers) is not int or num_hidden_layers < 0:
            raise ValueError("vision num_hidden_layers must be a non-negative integer")
        if hidden_size % num_attention_heads or hidden_size // num_attention_heads % 4:
            raise ValueError("vision hidden_size must yield an integer head dimension divisible by 4 for 2D RoPE")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("vision rope_theta must be finite and positive")
        if max_wh_ratio is not None and (not math.isfinite(max_wh_ratio) or max_wh_ratio < 1):
            raise ValueError("vision max_wh_ratio must be None or finite and at least 1")


class DeepseekV41Config(PretrainedConfig):
    """Flat text configuration with released defaults and a typed vision tower.

    Reducing num_hidden_layers retains the full released sharing and hashing
    schedules. Explicit smaller schedules support independent tiny models.
    """

    model_type = "deepseek_v41"
    sub_configs = {"vision_config": DeepseekV41VisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        moe_intermediate_size: int = 2304,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        rms_norm_eps: float = 1e-20,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        # MoE
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        # Sliding window + CSA2
        sliding_window: int = 128,
        compress_ratios: list[int] | None = None,
        compress_rope_theta: float = 160000.0,
        kv_source_layer_ids: list[int] | None = None,
        index_source_layer_ids: list[int] | None = None,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 512,
        # Hierarchical sparse indexer (``candidate_source_layer_id < 0`` disables it)
        candidate_source_layer_id: int = 20,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        # Hyper-connections (mHC)
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # Engram conditional memory
        engram_layer_ids: list[int] | None = None,
        engram_num_embeddings: list[int] | None = None,
        engram_max_ngram_size: int = 4,
        engram_vocab_size: int = 16000000,
        engram_n_heads: int = 8,
        engram_head_dim: int = 256,
        engram_pad_token_id: int = 2,
        engram_compressed_vocab_size: int = 99092,
        # DSpark draft head (checkpoint metadata only; not trained here)
        num_nextn_predict_layers: int = 3,
        dspark_block_size: int = 5,
        dspark_noise_token_id: int = 128799,
        dspark_target_layer_ids: list[int] | None = None,
        dspark_markov_rank: int = 256,
        dspark_n_routed_experts: int = 128,
        dspark_num_experts_per_tok: int = 3,
        # Multimodal bridge metadata; zero vision layers disable the tower
        image_token_id: int = 129264,
        vision_config: dict[str, Any] | DeepseekV41VisionConfig | None = None,
        # AutoModel training knobs
        engram_enabled: bool = True,
        engram_trainable: bool = False,
        kv_cache_fake_quant: bool = True,
        # Standard options
        pad_token_id: int | None = 2,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pretraining_tp: int = 1,
        torch_dtype: str | None = None,
        **kwargs: Any,
    ) -> None:
        dtype = kwargs.pop("dtype", None)
        resolved_dtype = dtype if dtype is not None else torch_dtype
        if resolved_dtype is None:
            resolved_dtype = "bfloat16"
        # Preserve the released storage metadata: NeMo's initial Checkpointer
        # uses it to request packed weights and scales for adapter dequantization.

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = (
            {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            }
            if rope_scaling is None
            else dict(rope_scaling)
        )
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        # The released schedule includes three SWA-only DSpark layers.
        self.compress_ratios = (
            [0, 0] + [2] * 18 + [1] * 20 + [0] * 3 if compress_ratios is None else list(compress_ratios)
        )
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = [2, 8, 14, 20] if kv_source_layer_ids is None else list(kv_source_layer_ids)
        self.index_source_layer_ids = (
            [2, 8, 14, 20, 24, 28, 32, 36] if index_source_layer_ids is None else list(index_source_layer_ids)
        )
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.engram_layer_ids = [1, 14] if engram_layer_ids is None else list(engram_layer_ids)
        self.engram_num_embeddings = (
            ([384006168, 384016682] if self.engram_layer_ids else [])
            if engram_num_embeddings is None
            else list(engram_num_embeddings)
        )
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = (
            [37, 38, 39] if dspark_target_layer_ids is None else list(dspark_target_layer_ids)
        )
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok
        self.image_token_id = image_token_id
        if vision_config is None:
            vision_config = DeepseekV41VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = DeepseekV41VisionConfig(**vision_config)
        elif not isinstance(vision_config, DeepseekV41VisionConfig):
            raise TypeError("vision_config must be a DeepseekV41VisionConfig, dictionary, or None")
        self.vision_config = vision_config
        self.engram_enabled = engram_enabled
        self.engram_trainable = engram_trainable
        self.kv_cache_fake_quant = kv_cache_fake_quant
        self.pretraining_tp = pretraining_tp
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError(
                "engram_layer_ids and engram_num_embeddings must have the same length, got "
                f"{self.engram_layer_ids} and {self.engram_num_embeddings}"
            )

        self._validate_dimensions()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            dtype=resolved_dtype,
            **kwargs,
        )

    def _validate_dimensions(self) -> None:
        """Reject dimensions and numeric settings that cannot define the backbone."""
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("head_dim", self.head_dim),
            ("q_lora_rank", self.q_lora_rank),
            ("o_lora_rank", self.o_lora_rank),
            ("o_groups", self.o_groups),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("sliding_window", self.sliding_window),
            ("index_n_heads", self.index_n_heads),
            ("index_head_dim", self.index_head_dim),
            ("index_topk", self.index_topk),
            ("hc_mult", self.hc_mult),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters),
            ("max_position_embeddings", self.max_position_embeddings),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_key_value_heads != 1:
            raise ValueError("DeepSeek-V4.1 requires num_key_value_heads=1 for shared latent KV")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if (
            type(self.qk_rope_head_dim) is not int
            or self.qk_rope_head_dim <= 0
            or self.qk_rope_head_dim % 2
            or self.qk_rope_head_dim > min(self.head_dim, self.index_head_dim)
        ):
            raise ValueError("qk_rope_head_dim must be positive, even, and no larger than head_dim or index_head_dim")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok must not exceed n_routed_experts")
        if self.n_shared_experts != 1:
            raise ValueError("DeepSeek-V4.1 requires n_shared_experts=1")
        for name, value in (
            ("rms_norm_eps", self.rms_norm_eps),
            ("hc_eps", self.hc_eps),
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
            ("routed_scaling_factor", self.routed_scaling_factor),
            ("swiglu_limit", self.swiglu_limit),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must lie in [0, 1)")
        if (
            type(self.num_nextn_predict_layers) is not int
            or type(self.dspark_block_size) is not int
            or self.num_nextn_predict_layers < 0
            or self.dspark_block_size < 0
        ):
            raise ValueError("num_nextn_predict_layers and dspark_block_size must be non-negative")
        if self.engram_enabled and self.engram_layer_ids:
            for name, value in (
                ("engram_n_heads", self.engram_n_heads),
                ("engram_head_dim", self.engram_head_dim),
                ("engram_vocab_size", self.engram_vocab_size),
            ):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer, got {value!r}")
            if type(self.engram_compressed_vocab_size) is not int or self.engram_compressed_vocab_size < 0:
                raise ValueError("engram_compressed_vocab_size must be non-negative")
            if type(self.engram_max_ngram_size) is not int or self.engram_max_ngram_size < 2:
                raise ValueError("engram_max_ngram_size must be at least 2 when Engram is enabled")
            if type(self.engram_pad_token_id) is not int or not 0 <= self.engram_pad_token_id < self.vocab_size:
                raise ValueError("engram_pad_token_id must be within the token vocabulary")

        if not math.isfinite(self.initializer_range) or self.initializer_range < 0:
            raise ValueError("initializer_range must be finite and non-negative")
        if self.rope_scaling is not None:
            if not isinstance(self.rope_scaling, dict):
                raise TypeError("rope_scaling must be a dictionary or None")
            for name, default in (("factor", 1), ("beta_fast", 32), ("beta_slow", 1)):
                value = self.rope_scaling.get(name, default)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"rope_scaling.{name} must be finite and positive")
            original_length = self.rope_scaling.get("original_max_position_embeddings", 0)
            if type(original_length) is not int or original_length < 0:
                raise ValueError("rope_scaling.original_max_position_embeddings must be a non-negative integer")

    def build_tokenizer(self) -> PreTrainedTokenizerFast:
        """Load the pinned fast tokenizer used to build Engram's compressed IDs.

        The tokenizer is a runtime construction input and is never stored on
        this configuration. Models constructed without a checkpoint source
        must receive an explicit tokenizer in their constructor.

        Returns:
            The checkpoint's fast tokenizer, using the resolved config commit.

        Raises:
            ValueError: No checkpoint source is configured.
            TypeError: The checkpoint tokenizer is not a fast tokenizer.
        """
        if not self._name_or_path:
            raise ValueError("Engram tokenizer construction requires a checkpoint source or explicit tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            self._name_or_path,
            revision=self._commit_hash,
            trust_remote_code=False,
            use_fast=True,
        )
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError("DeepSeek V4.1 Engram requires the checkpoint's original fast tokenizer")
        return tokenizer

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "DeepseekV41Config":
        """Build the config from a checkpoint-style dict, hoisting the nested ``text_config``.

        The released ``config.json`` nests the backbone fields under ``text_config``;
        they are flattened onto the top level here so ``from_pretrained`` and the
        flat constructor share one field protocol.
        """
        text_config = config_dict.get("text_config")
        if isinstance(text_config, dict):
            config_dict = {**config_dict, **{k: v for k, v in text_config.items() if k != "model_type"}}
            config_dict.pop("text_config")
        return super().from_dict(config_dict, **kwargs)

    # ------------------------------------------------------------------
    # Derived per-layer helpers
    # ------------------------------------------------------------------

    def compress_ratio(self, layer_idx: int) -> int:
        """Compression ratio of ``layer_idx`` (``0`` for pure sliding-window layers)."""
        if layer_idx < len(self.compress_ratios):
            return int(self.compress_ratios[layer_idx])
        return 0

    def is_kv_source(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` computes the shared compressed KV (CSA2 Full mode)."""
        return layer_idx in self.kv_source_layer_ids

    def is_index_source(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` runs its own indexer (CSA2 Full or Reindex mode)."""
        return layer_idx in self.index_source_layer_ids

    def csa2_mode(self, layer_idx: int) -> str:
        """Return ``"swa"``, ``"full"``, ``"reindex"`` or ``"reuse"`` for ``layer_idx``."""
        if self.compress_ratio(layer_idx) == 0:
            return "swa"
        if self.is_kv_source(layer_idx):
            return "full"
        if self.is_index_source(layer_idx):
            return "reindex"
        return "reuse"

    def validate_layer_layout(self) -> None:
        """Check that the CSA2 layer assignment is well formed.

        Every layer with a compression ratio must be preceded (or equal) by a KV
        source layer with the same ratio, and every Reuse layer must be preceded
        by an index source computed against that same KV source.
        """
        self._validate_dimensions()
        if any(type(ratio) is not int or ratio < 0 for ratio in self.compress_ratios):
            raise ValueError("compress_ratios must contain non-negative integers")
        for name, layer_ids in (
            ("kv_source_layer_ids", self.kv_source_layer_ids),
            ("index_source_layer_ids", self.index_source_layer_ids),
            ("engram_layer_ids", self.engram_layer_ids),
        ):
            if any(type(layer_id) is not int or layer_id < 0 for layer_id in layer_ids):
                raise ValueError(f"{name} must contain non-negative integer layer IDs")
            if layer_ids != sorted(set(layer_ids)):
                raise ValueError(f"{name} must contain strictly increasing layer IDs")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("engram_layer_ids and engram_num_embeddings must have the same length")
        if any(type(rows) is not int or rows <= 0 for rows in self.engram_num_embeddings):
            raise ValueError("engram_num_embeddings must contain positive integer row counts")
        # Retain all global IDs for a reduced prefix. Future sources must still
        # describe explicit compressed layers; omitted schedule tails mean SWA.
        for layer_id in self.kv_source_layer_ids + self.index_source_layer_ids:
            if layer_id >= len(self.compress_ratios):
                raise ValueError("KV/index source IDs must refer to layers covered by compress_ratios")
        if type(self.candidate_source_layer_id) is not int or self.candidate_source_layer_id < -1:
            raise ValueError("candidate_source_layer_id must be -1 or a non-negative layer ID")
        for name, value in (
            ("candidate_block_size", self.candidate_block_size),
            ("candidate_topk_blocks", self.candidate_topk_blocks),
        ):
            if type(value) is not int or value < 0 or (self.candidate_source_layer_id >= 0 and value == 0):
                raise ValueError(f"{name} must be non-negative and positive when the candidate source is enabled")
        last_kv_source: int | None = None
        last_index_source: int | None = None
        for layer_idx in range(self.num_hidden_layers):
            ratio = self.compress_ratio(layer_idx)
            if ratio == 0:
                if self.is_kv_source(layer_idx) or self.is_index_source(layer_idx):
                    raise ValueError(f"layer {layer_idx} has compress_ratio 0 but is listed as a CSA2 source")
                continue
            if self.is_kv_source(layer_idx):
                last_kv_source = layer_idx
                if not self.is_index_source(layer_idx):
                    raise ValueError(f"KV source layer {layer_idx} must also be an index source")
            if last_kv_source is None:
                raise ValueError(f"layer {layer_idx} attends over compressed KV but no KV source precedes it")
            if self.compress_ratio(last_kv_source) != ratio:
                raise ValueError(
                    f"layer {layer_idx} has compress_ratio {ratio} but reuses KV from layer "
                    f"{last_kv_source} with ratio {self.compress_ratio(last_kv_source)}"
                )
            if self.is_index_source(layer_idx):
                last_index_source = layer_idx
            elif last_index_source is None or last_index_source < last_kv_source:
                raise ValueError(f"Reuse layer {layer_idx} has no index source computed against its KV source")
        if self.candidate_source_layer_id >= 0 and not self.is_index_source(self.candidate_source_layer_id):
            raise ValueError("candidate_source_layer_id must be an index source layer")
