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

from typing import Callable, ClassVar

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)

# FlexAttention mask type. For each mask type, we initialize it at most once per
# batch. To record what it is initialized, FLEX_ATTN_MASK_T is used as the key to
# track the initialized mask.
FLEX_ATTN_MASK_T = tuple[str, int | None]


# Adapted from https://github.com/pytorch/torchtitan/pull/1559
# Reference: https://github.com/KhoomeiK/torchtitan/blob/d0a54fb7de5382a5683205b4826543bd4c227b08/torchtitan/models/attention.py
class FlexAttention(torch.nn.Module):
    """FlexAttention module that uses torch.nn.attention.flex_attention.

    This module is a wrapper around torch.nn.attention.flex_attention. This module
    implements certain common attention types, such as causal and block_causal.

    Args:
        attn_mask_type (str): The type of attention mask. Currently, we support
            "causal" and "block_causal". "causal" means the lower triangle of the
            attention matrix is masked. "block_causal" means the attention matrix
            is divided into blocks, where block boundary is defined by EOS token,
            and the lower triangle of each block is masked.
        fixed_block_size (int | None): The block size to be used to perform attention.
            If specified, each sequence will be further divided to blocks, where each
            block has the maximum size of ``fixed_block_size``. A query will only attend
            to the keys within the same block.
    """

    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    flex_attn: ClassVar[Callable] = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[FLEX_ATTN_MASK_T, BlockMask]] = {}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        sink_weights: torch.Tensor | None = None,
        sliding_window: int = 0,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if sink_weights is None:
            block_mask = FlexAttention.block_masks[self.mask_key]
            return FlexAttention.flex_attn(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa)

        B, H_q, S_q, D = q.shape
        _, H_kv, S_kv, _ = k.shape

        # regular (no-sink) mask + no extra KV col
        mask_key = (sliding_window, S_q, S_kv)
        if mask_key not in FlexAttention.block_masks:
            if sliding_window is not None and sliding_window > 0:
                mask_mod = FlexAttention._get_sliding_window_mask_mod(sliding_window)
            else:
                mask_mod = FlexAttention._get_causal_mask_mod()
            block_mask = create_block_mask(
                mask_mod,
                B,
                H_q,
                S_q,
                S_kv,
                _compile=False,  # NOTE: _compile=True leads to hangs during sampling, so setting it to False for now
                device=q.device,
            )
            FlexAttention.block_masks[mask_key] = block_mask

        block_mask = FlexAttention.block_masks[mask_key]

        # run fast flex_attn and return LSE
        out, lse = FlexAttention.flex_attn(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa, return_lse=True)

        # rescale by sigma(lse - w[h]) and broadcast over D
        if sink_weights is not None:
            w = sink_weights  # [H]
            scale = torch.sigmoid(lse - w.view(1, -1, 1)).unsqueeze(-1)  # [B,H,S,1]
            out = out * scale

        out = out.to(q.dtype)
        return out

    @staticmethod
    def _get_sliding_window_mask_mod(window: int):
        """
        Returns a mask_mod function that
        - only allows kv_idx ≤ q_idx (causal)
        - and only if (q_idx - kv_idx) ≤ window
        """

        def sliding_mod(b, h, q_idx, kv_idx):
            # causal within window
            keep = (kv_idx <= q_idx) & (q_idx - kv_idx < window)
            return keep

        return sliding_mod

    @staticmethod
    def _get_causal_mask_mod() -> _mask_mod_signature:
        def causal_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
            return q_idx >= kv_idx

        return causal_mask

    @staticmethod
    def _get_block_causal_mask_mod(batch: torch.Tensor, eos_id: int) -> _mask_mod_signature:
        # batch is [b, s, h, d] shape
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
        seq_idx[:, 1:] = acc_mask[:, :-1]

        def block_causal_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    @staticmethod
    def _fixed_block_mask_mod(mask_mod: _mask_mod_signature, fixed_block_size: int) -> _mask_mod_signature:
        """
        Given an arbitrary mask_mod, divide the input sequence to blocks
        and only allow attention within the same block.

        Args:
            mask_mod: The mask mod to apply to the documents
            fixed_block_size: The number of tokens in each block.
        """

        # Credit to @drisspg.
        def blocked_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
            # Get the block index of the query and key
            q_block = q_idx // fixed_block_size
            kv_block = kv_idx // fixed_block_size
            # Only allow attention within the same block
            same_block = q_block == kv_block
            # Apply the original mask mod
            inner_mask = mask_mod(b, h, q_idx % fixed_block_size, kv_idx % fixed_block_size)

            return same_block & inner_mask

        blocked_mask_mod.__name__ = f"blocked_mask_mod_{mask_mod.__name__}_fixed_block_size_{fixed_block_size}"

        return blocked_mask_mod
