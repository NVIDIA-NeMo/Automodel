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
and the vision tower under ``vision_config``.  AutoModel trains the text
backbone only, so this config flattens ``text_config`` onto the top level
(the field names below are exactly the ``text_config`` keys) and keeps
``vision_config`` as an opaque dict for round-tripping.

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

import inspect
from typing import Any

from transformers import PretrainedConfig


class DeepseekV41Config(PretrainedConfig):
    """Configuration class for the DeepSeek V4.1 text backbone."""

    model_type = "deepseek_v41"
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
        rope_scaling: dict | None = None,
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
        candidate_source_layer_id: int = -1,
        candidate_topk_blocks: int = 0,
        candidate_block_size: int = 0,
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
        engram_compressed_vocab_size: int = 0,
        # DSpark draft head (checkpoint metadata only; not trained here)
        num_nextn_predict_layers: int = 0,
        dspark_block_size: int = 0,
        dspark_noise_token_id: int = 0,
        dspark_target_layer_ids: list[int] | None = None,
        dspark_markov_rank: int = 256,
        dspark_n_routed_experts: int = 0,
        dspark_num_experts_per_tok: int = 0,
        # Multimodal bridge metadata (text-only training ignores it)
        image_token_id: int = 129264,
        vision_config: dict[str, Any] | None = None,
        # AutoModel training knobs
        engram_enabled: bool = True,
        engram_trainable: bool = False,
        kv_cache_fake_quant: bool = True,
        # Standard options
        pad_token_id: int | None = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pretraining_tp: int = 1,
        torch_dtype: str | None = None,
        **kwargs,
    ):
        # The released config nests the backbone under ``text_config``; hoist its
        # entries over the explicit arguments and re-enter with one flat call so
        # nested checkpoints and flat YAML configs share the same code path.
        text_config = kwargs.pop("text_config", None)
        if isinstance(text_config, dict):
            explicit = {name: value for name, value in locals().items() if name in _INIT_PARAMS}
            merged = {**explicit, **{k: v for k, v in text_config.items() if k != "model_type"}, **kwargs}
            self.__init__(**merged)
            return

        dtype = kwargs.pop("dtype", None)
        resolved_dtype = dtype if dtype is not None else torch_dtype
        if resolved_dtype is None:
            resolved_dtype = "bfloat16"
        # The released checkpoint carries a quantization block that only describes
        # the on-disk layout; the state-dict adapter dequantizes on load.
        kwargs.pop("quantization_config", None)

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
        self.rope_scaling = rope_scaling
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        self.compress_ratios = [int(r) for r in (compress_ratios or [])]
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = [int(i) for i in (kv_source_layer_ids or [])]
        self.index_source_layer_ids = [int(i) for i in (index_source_layer_ids or [])]
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.engram_layer_ids = [int(i) for i in (engram_layer_ids or [])]
        self.engram_num_embeddings = [int(n) for n in (engram_num_embeddings or [])]
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = [int(i) for i in (dspark_target_layer_ids or [])]
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok
        self.image_token_id = image_token_id
        self.vision_config = vision_config
        self.engram_enabled = engram_enabled
        self.engram_trainable = engram_trainable
        self.kv_cache_fake_quant = kv_cache_fake_quant
        self.pretraining_tp = pretraining_tp
        # DeepSeek V4.1 has no hash-routed layers; the shared DSV4 helpers read this.
        self.num_hash_layers = 0

        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError(
                "engram_layer_ids and engram_num_embeddings must have the same length, got "
                f"{self.engram_layer_ids} and {self.engram_num_embeddings}"
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            dtype=resolved_dtype,
            **kwargs,
        )

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


_INIT_PARAMS = frozenset(
    name for name in inspect.signature(DeepseekV41Config.__init__).parameters if name not in ("self", "kwargs")
)
