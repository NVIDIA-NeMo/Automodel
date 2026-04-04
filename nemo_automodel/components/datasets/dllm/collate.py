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

"""Collate function for dLLM training.

Expects datasets that produce **unshifted** format (``input_ids`` +
``loss_mask``, via ``_package_tokenized_example(unshifted=True)``).
Goes directly from variable-length sample lists to block-aligned tensors
in a single pass.

Two-stage block-aligned padding layout::

    [real tokens][EOS block-pad, loss=1][PAD global-pad, loss=0]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch


class DLLMCollator:
    """Collator for dLLM (diffusion LLM) training.

    Goes directly from variable-length sample dicts to block-aligned
    tensors in a single pass — no intermediate pad-to-max step.

    Expects each sample to have ``input_ids``, ``loss_mask``, and
    ``attention_mask`` (as produced by
    ``_package_tokenized_example(unshifted=True)``).

    Args:
        pad_token_id: Token ID for global (stage-2) padding.
        eos_token_id: Token ID for block (stage-1) padding.  Only used
            when *block_size* is set.
        block_size: If set, apply two-stage block-aligned padding.
        pad_seq_len_divisible: Round final length to
            ``lcm(block_size, pad_seq_len_divisible)``.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        block_size: Optional[int] = None,
        pad_seq_len_divisible: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        supervise_padding: bool = False,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.block_size = block_size
        self.pad_seq_len_divisible = pad_seq_len_divisible
        self.max_seq_len = max_seq_len
        self.block_pad_token_id = eos_token_id if eos_token_id is not None else pad_token_id
        self.supervise_padding = supervise_padding

    def __call__(self, batch: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        for sample in batch:
            sample.pop("___PAD_TOKEN_IDS___", None)

        content_lengths = torch.tensor([len(s["input_ids"]) for s in batch], dtype=torch.long)
        target_len = self._compute_target_length(content_lengths)

        input_ids = self._pad_and_fill(
            [s["input_ids"] for s in batch],
            content_lengths,
            target_len,
            pad_value=self.pad_token_id,
            block_pad_value=self.block_pad_token_id,
        )
        loss_mask_pad_value = 1 if self.supervise_padding else 0
        loss_mask = self._pad_and_fill(
            [s["loss_mask"] for s in batch],
            content_lengths,
            target_len,
            pad_value=loss_mask_pad_value,
            block_pad_value=1,
        ).float()
        attention_mask = self._pad_and_fill(
            [s["attention_mask"] for s in batch],
            content_lengths,
            target_len,
            pad_value=0,
            block_pad_value=0,
            apply_block_fill=False,
        )

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "input_lengths": content_lengths,
        }

    def _compute_target_length(self, content_lengths: torch.Tensor) -> int:
        # Clamp individual content lengths to max_seq_len before alignment
        if self.max_seq_len is not None:
            content_lengths = content_lengths.clamp(max=self.max_seq_len)

        bs = self.block_size
        if bs is not None and bs > 1:
            block_aligned = ((content_lengths + bs - 1) // bs) * bs
            max_len = block_aligned.max().item()
        else:
            max_len = content_lengths.max().item()

        psd = self.pad_seq_len_divisible
        if psd is not None and psd > 1:
            alignment = math.lcm(bs or 1, psd)
            max_len = ((max_len + alignment - 1) // alignment) * alignment

        # Hard cap after alignment to prevent OOM
        if self.max_seq_len is not None:
            max_len = min(max_len, self.max_seq_len)

        return max_len

    def _pad_and_fill(
        self,
        samples: List[list],
        content_lengths: torch.Tensor,
        target_len: int,
        pad_value: int,
        block_pad_value: int,
        apply_block_fill: bool = True,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        """Pad variable-length lists to *target_len* with two-stage fill.

        For each sample:
          - ``[0, content_length)`` → original content
          - ``[content_length, block_aligned)`` → *block_pad_value*
          - ``[block_aligned, target_len)`` → *pad_value*
        """
        B = len(samples)
        out = torch.full((B, target_len), pad_value, dtype=dtype)
        bs = self.block_size

        for b in range(B):
            cl = content_lengths[b].item()
            seq = samples[b]
            copy_len = min(cl, target_len, len(seq))
            out[b, :copy_len] = torch.tensor(seq[:copy_len], dtype=dtype)

            if apply_block_fill and bs is not None and bs > 1:
                ba = min(((cl + bs - 1) // bs) * bs, target_len)
                if ba > cl:
                    out[b, cl:ba] = block_pad_value

        return out
