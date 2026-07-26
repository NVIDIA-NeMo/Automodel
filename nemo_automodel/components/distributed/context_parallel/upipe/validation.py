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

"""Preflight checks for the UPipe attention backend.

UPipe has a hand-written backward and a permuted head layout, so an
unsupported configuration tends to surface as quietly wrong logits rather than
a crash. Everything that cannot be supported is rejected up front instead.
"""

from __future__ import annotations

# FlashAttention's rotary kernel requires an even head_dim it can halve.
UPIPE_MIN_ROTARY_HEAD_DIM = 2

# FlashAttention caps the head dimension it will run.
UPIPE_MAX_HEAD_DIM = 256


def flash_attn_available() -> bool:
    """Report whether the ``flash_attn`` package can be imported, without importing it."""
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None


def validate_upipe_attention(
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    cp_size: int,
    tp_size: int = 1,
    rope_backend: str = "torch",
    rope_fusion: bool = False,
    compile_attn: bool = False,
    require_flash_attn: bool = True,
) -> None:
    """Reject UPipe configurations that would be wrong or unsupported.

    Args:
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Attention head dimension.
        cp_size: Context-parallel (Ulysses) degree.
        tp_size: Tensor-parallel degree; must be 1.
        rope_backend: ``backend.rope``; must be ``"torch"``.
        rope_fusion: ``backend.rope_fusion``; must be False.
        compile_attn: ``backend.compile_attn``; must be False.
        require_flash_attn: Check that ``flash_attn`` is importable. Off in unit
            tests that only exercise the shape and config rules.

    Raises:
        ValueError: If any requirement is unmet, with the offending value named.
    """
    if require_flash_attn and not flash_attn_available():
        raise ValueError(
            "attn='upipe' requires the 'flash-attn' package, which is not importable. "
            "Install nemo-automodel[cuda] or build flash-attn for this platform."
        )

    if cp_size < 1:
        raise ValueError(f"attn='upipe' needs cp_size >= 1, got {cp_size}.")

    if n_heads % cp_size != 0:
        raise ValueError(
            f"attn='upipe' requires num_attention_heads ({n_heads}) to be divisible by cp_size ({cp_size}); "
            "each pipeline stage projects exactly cp_size query heads."
        )

    if n_kv_heads % cp_size != 0:
        raise ValueError(
            f"attn='upipe' requires num_key_value_heads ({n_kv_heads}) to be divisible by cp_size ({cp_size}); "
            "KV-weight replication for cp_size > num_key_value_heads is not supported yet."
        )

    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"attn='upipe' requires num_attention_heads ({n_heads}) to be divisible by "
            f"num_key_value_heads ({n_kv_heads})."
        )

    # Every stage consumes one KV chunk, so there must be at least as many
    # stages as the GQA ratio for torch.chunk to produce a non-empty split.
    pipe_degree = n_heads // cp_size
    gqa_ratio = n_heads // n_kv_heads
    if pipe_degree < gqa_ratio or pipe_degree % gqa_ratio != 0:
        raise ValueError(
            f"attn='upipe' requires n_heads/cp_size ({pipe_degree}) to be a positive multiple of "
            f"n_heads/n_kv_heads ({gqa_ratio}); got cp_size={cp_size}, n_heads={n_heads}, n_kv_heads={n_kv_heads}."
        )

    if head_dim % 2 != 0 or head_dim < UPIPE_MIN_ROTARY_HEAD_DIM:
        raise ValueError(f"attn='upipe' requires an even head_dim >= {UPIPE_MIN_ROTARY_HEAD_DIM}, got {head_dim}.")

    if head_dim > UPIPE_MAX_HEAD_DIM:
        raise ValueError(f"attn='upipe' requires head_dim <= {UPIPE_MAX_HEAD_DIM}, got {head_dim}.")

    if tp_size != 1:
        raise ValueError(
            f"attn='upipe' does not support tensor parallelism (tp_size={tp_size}). UPipe reads q/k/v_proj.weight "
            "directly, which under TP is a sharded DTensor the fused op cannot consume."
        )

    if rope_backend != "torch":
        raise ValueError(f"attn='upipe' requires backend.rope='torch', got '{rope_backend}'.")

    if rope_fusion:
        raise ValueError("attn='upipe' requires backend.rope_fusion=False; UPipe applies RoPE inside its own kernel.")

    if compile_attn:
        raise ValueError(
            "attn='upipe' requires backend.compile_attn=False; the fused op is an opaque custom op that "
            "fullgraph compilation cannot trace."
        )


def has_non_trailing_padding(attention_mask) -> bool:
    """Report whether a 2-D attention mask pads anywhere other than the tail.

    A row is acceptable when its ones form a prefix, i.e. the mask is
    non-increasing along the sequence. Anything else -- left padding, or holes
    in the middle -- means a real token would attend to a pad.

    Args:
        attention_mask: ``[batch, seq]`` mask where 1 marks a real token, or None.

    Returns:
        True if any row pads somewhere other than its tail.
    """
    if attention_mask is None or attention_mask.dim() != 2 or attention_mask.shape[1] < 2:
        return False
    mask = attention_mask.bool()
    return bool((mask[:, 1:] & ~mask[:, :-1]).any())


def validate_upipe_runtime(*, has_peft: bool, is_packed: bool, has_non_trailing_pad: bool) -> None:
    """Reject runtime states UPipe cannot serve.

    The framework's derived 4-D causal mask is *not* a reason to reject: UPipe
    is unconditionally causal and reproduces it exactly. Nor is ordinary right
    padding, because under a causal mask a real token never attends to a pad
    that follows it, and the loss mask discards the pads' own outputs. Only
    padding that some real token would attend to is fatal.

    Args:
        has_peft: Whether PEFT/LoRA adapters are attached.
        is_packed: Whether the batch uses packed/THD sequences.
        has_non_trailing_pad: Whether any sequence is padded anywhere but its tail.

    Raises:
        ValueError: If any state is unsupported.
    """
    if has_peft:
        raise ValueError(
            "attn='upipe' is incompatible with PEFT/LoRA: the fused op reads q/k/v_proj.weight directly and "
            "would bypass the adapters."
        )

    if is_packed:
        raise ValueError("attn='upipe' does not support packed (THD) sequences yet.")

    if has_non_trailing_pad:
        raise ValueError(
            "attn='upipe' is causal-only and reads no padding mask, so it is safe with right padding but not "
            "with left or interior padding, where real tokens would attend to pads. Switch the collator to "
            "right padding, or use a different attention backend."
        )
