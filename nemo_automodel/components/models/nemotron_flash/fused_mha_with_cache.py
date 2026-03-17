# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch

dtype_int = torch.int32


def fused_mha_interface(
    query_states: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key_states: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    value_states: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    k_cache: torch.Tensor,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD] or [num_pages, page_size, n, d] for paged attn
    v_cache: torch.Tensor,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    position_ids: torch.Tensor = None,
    page_table: torch.Tensor = None,  # [b, max_num_pages_per_seq] # loc of the block page in the cache.
    max_seq_len=None,
) -> torch.Tensor:
    """
    Replacement for _flash_attention_forward(...) that uses
    Triton’s fused_mha_with_paged_cache under the hood.
    Returns: [batch, q_len, heads*head_dim]
    """
    # unpack shapes
    b, ql, n_heads, head_dim = query_states.shape
    _, kvl, n_kv_heads, _ = key_states.shape

    q = query_states.reshape(b, ql, n_heads * head_dim)
    k = key_states.reshape(b, kvl, n_kv_heads * head_dim)
    v = value_states.reshape(b, kvl, n_kv_heads * head_dim)

    if position_ids is not None:
        if ql == 1:  # Generate phase - single token
            input_pos = position_ids[:, -1]  # Use the last position for each sequence
        else:  # Context phase - multiple tokens
            input_pos = position_ids[:, 0]  # Use the starting position for each sequence
    else:
        # Fallback: assume starting from 0 for all sequences
        input_pos = torch.zeros(b, device=q.device, dtype=torch.int32)

    freqs_cis = None

    if page_table is None:
        y = torch.ops.attention.fused_mha_with_cache(
            q,
            k,
            v,
            input_pos,
            k_cache,
            v_cache,
            freqs_cis,
        )

    else:
        batch_size = b

        # cache_loc: identity mapping [0, 1, ..., b-1]
        cache_loc = torch.arange(batch_size, device=q.device, dtype=dtype_int)

        # input_positions: assume pure context (all start from 0)
        input_positions = torch.zeros(batch_size, device=q.device, dtype=dtype_int)

        # seq_len: each sequence length is kvl
        seq_len = torch.full((batch_size,), kvl, device=q.device, dtype=dtype_int)

        # seq_start: flattened starting index for each sequence
        seq_start = (seq_len.cumsum(0) - seq_len).to(dtype=dtype_int)

        assert max_seq_len is not None, "max_seq_len must be provided when using paged attention."

        y = torch.ops.attention.fused_mha_with_paged_cache(
            q,
            k,
            v,
            input_positions,
            cache_loc,
            seq_len,
            seq_start,
            page_table,
            max_seq_len,
            k_cache,
            v_cache,
            freqs_cis,
        )

    y = y.view(b, ql, n_heads, head_dim)

    return y


def main():
    # ––– Test hyperparameters –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    batch_size = 1
    q_len = 1
    kv_len = 1
    num_heads = 16
    n_kv_heads = 16  # noqa: F841
    head_dim = 128

    max_batch_size = 1
    max_seq_len = 1024

    page_size = 256  # noqa: F841

    device = "cuda"

    # ––– Random query, key, value tensors –––––––––––––––––––––––––––––––––––––––––––––––––––
    query_states = torch.randn(batch_size, q_len, num_heads, head_dim, device=device)
    key_states = torch.randn(batch_size, kv_len, num_heads, head_dim, device=device)
    value_states = torch.randn(batch_size, kv_len, num_heads, head_dim, device=device)

    k_cache = torch.randn(max_batch_size, max_seq_len, num_heads, head_dim, device=device)
    v_cache = torch.randn(max_batch_size, max_seq_len, num_heads, head_dim, device=device)

    attn_out = fused_mha_interface(
        query_states,
        key_states,
        value_states,
        k_cache=k_cache,
        v_cache=v_cache,
    )

    expected_shape = (batch_size, q_len, num_heads, head_dim)
    print(f"[test] output shape: {attn_out.shape} (expected {expected_shape})")

    if attn_out.shape == expected_shape:
        print("[test] ✅ Success: output tensor has correct shape.")
    else:
        print("[test] ❌ Failure: shape mismatch.")


if __name__ == "__main__":
    main()
