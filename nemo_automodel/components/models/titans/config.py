# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Configuration for the linear-memory Titans model.

Titans (Behrouz et al., 2024) augments a Gated DeltaNet (GDN) recurrence with a
*data-dependent momentum* over the per-token surprise and a (forget) decay gate.
With momentum disabled the recurrence reduces exactly to Gated DeltaNet, so this
config exposes ``momentum`` / ``forget`` toggles to recover the GDN baseline.

This is the ``mem_depth=1`` (linear) memory case. The deep-memory case
(``mem_depth>=2``, an MLP memory updated by test-time gradient descent) is a
separate effort; see ``NOTES.md`` for the shared ``NeuralMemory`` API contract.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class TitansConfig(PretrainedConfig):
    """Configuration class for the linear-memory Titans model.

    Args:
        vocab_size: Size of the token vocabulary.
        hidden_size: Model (residual stream) dimension.
        num_hidden_layers: Number of Titans decoder blocks.
        num_attention_heads: Number of neural-memory heads.
        head_dim: Per-head key/value (memory) dimension -- the ``mem_dim`` of the
            ``NeuralMemory`` module. ``num_attention_heads * head_dim`` is the inner
            memory width.
        intermediate_size: SwiGLU MLP inner dimension.
        hidden_act: MLP activation (``"silu"`` for SwiGLU).
        max_position_embeddings: Maximum sequence length (memory is recurrent, so
            this only bounds the dataloader / positional bookkeeping).
        rms_norm_eps: Epsilon for RMSNorm layers.
        momentum: If ``True`` (Titans), maintain a data-dependent momentum over the
            surprise term. If ``False``, the recurrence is exactly Gated DeltaNet.
        forget: If ``True``, apply the data-dependent decay (forget) gate
            ``alpha_t = exp(g_t)`` from ``A_log`` / ``dt_bias``. If ``False``, the
            decay gate is fixed to 1 (pure delta rule, no forgetting).
        mem_depth: Memory depth. ``1`` is the linear (matrix) memory; ``>=2`` is a
            deep MLP memory updated by chunkwise test-time gradient descent. Both
            share the :class:`NeuralMemory` API (see ``NOTES.md``).
        memory_expansion_factor: Hidden-width multiplier for intermediate layers
            of a deep memory MLP.
        memory_residual_norm: Wrap the deep memory in the paper's residual RMSNorm.
            The norm weight is part of the test-time-updated fast weights.
        qkv_conv_kernel_size: Width of the causal depthwise convolution applied
            after each Q/K/V projection. ``0`` disables the convolutions.
        chunk_size: Chunk size hint for the (future) chunked momentum kernel and
            for the fla GDN kernel reduction path.
        deep_memory_backend: Deep-memory update semantics. ``"reference"`` uses
            per-token updates within each chunk; ``"titans_pytorch"`` uses the
            public implementation's chunk-aggregated vectorized update.
        memory_batch_size: Number of tokens sharing one deep-memory gradient
            anchor for the ``"titans_pytorch"`` backend. Must be divisible by
            ``chunk_size``. ``None`` uses the full input sequence.
        num_persistent_memory_tokens: Number of learned, input-independent
            vectors prepended to every sequence before the decoder stack.
        tie_word_embeddings: Whether ``lm_head`` shares weights with ``embed_tokens``.
        initializer_range: Stddev for truncated-normal weight init.
        torch_dtype: Default compute dtype (``A_log`` / ``dt_bias`` always stay fp32).
    """

    model_type = "titans"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        head_dim: int = 64,
        intermediate_size: int = 1376,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        momentum: bool = True,
        forget: bool = True,
        mem_depth: int = 1,
        memory_expansion_factor: float = 4.0,
        memory_residual_norm: bool = True,
        qkv_conv_kernel_size: int = 4,
        chunk_size: int = 16,
        deep_memory_backend: str = "reference",
        memory_batch_size: int | None = None,
        num_persistent_memory_tokens: int = 0,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        torch_dtype: str = "bfloat16",
        pad_token_id: int | None = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        use_cache: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.momentum = momentum
        self.forget = forget
        self.mem_depth = mem_depth
        self.memory_expansion_factor = memory_expansion_factor
        self.memory_residual_norm = memory_residual_norm
        self.qkv_conv_kernel_size = qkv_conv_kernel_size
        self.chunk_size = chunk_size
        self.deep_memory_backend = deep_memory_backend
        self.memory_batch_size = memory_batch_size
        self.num_persistent_memory_tokens = num_persistent_memory_tokens
        self.initializer_range = initializer_range
        self.torch_dtype = torch_dtype

        if mem_depth < 1:
            raise ValueError(f"TitansConfig: mem_depth must be >= 1 (1 = linear, >=2 = deep MLP); got {mem_depth}.")
        if memory_expansion_factor <= 0:
            raise ValueError(
                f"TitansConfig: memory_expansion_factor must be greater than zero; got {memory_expansion_factor}."
            )
        if qkv_conv_kernel_size < 0:
            raise ValueError(f"TitansConfig: qkv_conv_kernel_size must be non-negative; got {qkv_conv_kernel_size}.")
        if chunk_size <= 0:
            raise ValueError(f"TitansConfig: chunk_size must be positive; got {chunk_size}.")
        if deep_memory_backend not in {"reference", "titans_pytorch"}:
            raise ValueError(
                "TitansConfig: deep_memory_backend must be 'reference' or 'titans_pytorch'; "
                f"got {deep_memory_backend!r}."
            )
        if memory_batch_size is not None:
            if memory_batch_size <= 0:
                raise ValueError("TitansConfig: memory_batch_size must be positive when provided.")
            if memory_batch_size % chunk_size != 0:
                raise ValueError(
                    "TitansConfig: memory_batch_size must be divisible by chunk_size; "
                    f"got {memory_batch_size} and {chunk_size}."
                )
        if num_persistent_memory_tokens < 0:
            raise ValueError(
                "TitansConfig: num_persistent_memory_tokens must be non-negative; "
                f"got {num_persistent_memory_tokens}."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
