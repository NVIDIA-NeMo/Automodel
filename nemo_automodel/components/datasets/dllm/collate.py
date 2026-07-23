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
        response_window: gemma4 response-window mode. When ``True`` the EOS
            block-fill is RESPONSE-RELATIVE (aligned on the first supervised
            position, matching Google's ChunkResponseIntoCanvases) and the fill
            is marked **attended** (``attention_mask=1``), and a one-time
            single-turn guard rejects multi-turn ``loss_mask``. When ``False``
            (default; llada / nemotron full-sequence denoising) the fill is
            ABSOLUTE (block-aligned on the content length) and **not** attended,
            and no single-turn guard runs — the pre-response-window behavior.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        block_size: Optional[int] = None,
        pad_seq_len_divisible: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        supervise_padding: bool = False,
        response_window: bool = False,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.block_size = block_size
        self.pad_seq_len_divisible = pad_seq_len_divisible
        self.max_seq_len = max_seq_len
        self.block_pad_token_id = eos_token_id if eos_token_id is not None else pad_token_id
        self.supervise_padding = supervise_padding
        self.response_window = response_window

    def __call__(self, batch: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        for sample in batch:
            sample.pop("___PAD_TOKEN_IDS___", None)

        content_lengths = torch.tensor([len(s["input_ids"]) for s in batch], dtype=torch.long)
        if self.response_window:
            # gemma4 response-window mode (see __init__ docstring).
            # Single-turn guard (EVERY batch): the response-window recipe requires the
            # supervised response to be ONE contiguous run in the RAW loss_mask. A
            # multi-turn ChatDataset mask (one supervised run per assistant turn) would
            # make the window treat intervening user turns as response.
            # Checked here on the raw mask (before the EOS block-fill, which legitimately
            # appends a second run after any trailing unsupervised token). Use
            # ChatDataset(mask_history=True) to collapse multi-turn data.
            #
            # Run on EVERY batch, not once: this collator instance is shared across the
            # train and ALL val dataloaders, so a once-only check (the old `_runs_checked`
            # flag) let later batches — including any val batch — silently pass a multi-turn
            # mask, mis-windowing an intervening user turn into the response canvas with no
            # error. The check is O(B*L) over Python lists, negligible vs the tensor work below.
            for s in batch:
                lm = s["loss_mask"]
                runs = sum(1 for i, v in enumerate(lm) if v and (i == 0 or not lm[i - 1]))
                if runs > 1:
                    raise AssertionError(
                        f"DLLMCollator: loss_mask has {runs} supervised runs (multi-turn); the "
                        "block-diffusion response window needs a single contiguous response. "
                        "Set mask_history=True on the dataset to supervise only the final turn."
                    )
            # Response start per sample = first supervised (loss_mask==1) position. The
            # EOS block-fill completes the RESPONSE's last canvas in RESPONSE-RELATIVE
            # space (matching Google's ChunkResponseIntoCanvases). Aligning the fill on
            # the absolute (prompt+response) length instead shifts the canvas grid by
            # (prompt % block_size) and spills a spurious all-EOS canvas block, because
            # the decoder canvas grid (train_ft._build_response_window) is built
            # response-relative from prefix_lengths.
            prefix_lengths = torch.tensor(
                [self._first_supervised_index(s["loss_mask"], len(s["input_ids"])) for s in batch],
                dtype=torch.long,
            )
            # The EOS block-fill is real, supervised, ATTENDED canvas content.
            attn_block_pad_value = 1
        else:
            # Pre-response-window behavior (llada / nemotron full-sequence denoising):
            # ABSOLUTE block-align (prefix=0 makes _block_fill_ends round the content
            # length), and the EOS-fill is NOT attended (attention stays 0 over it).
            prefix_lengths = torch.zeros(len(batch), dtype=torch.long)
            attn_block_pad_value = 0
        fill_ends = self._block_fill_ends(content_lengths, prefix_lengths)
        target_len = self._compute_target_length(fill_ends)

        input_ids = self._pad_and_fill(
            [s["input_ids"] for s in batch],
            content_lengths,
            fill_ends,
            target_len,
            pad_value=self.pad_token_id,
            block_pad_value=self.block_pad_token_id,
        )
        loss_mask_pad_value = 1 if self.supervise_padding else 0
        loss_mask = self._pad_and_fill(
            [s["loss_mask"] for s in batch],
            content_lengths,
            fill_ends,
            target_len,
            pad_value=loss_mask_pad_value,
            block_pad_value=1,
        ).float()
        # In response-window mode the EOS block-fill is real, supervised, ATTENDED
        # canvas content: the model must learn to emit those EOS tokens to terminate
        # a block, exactly as Google marks the trailing canvas valid (canvas_mask=True
        # over the EOS-fill), so loss and attention agree over [content_length,
        # fill_end). In the pre-response-window path attn_block_pad_value=0 leaves the
        # fill unattended. Either way only the stage-2 global pad stays masked.
        attention_mask = self._pad_and_fill(
            [s["attention_mask"] for s in batch],
            content_lengths,
            fill_ends,
            target_len,
            pad_value=0,
            block_pad_value=attn_block_pad_value,
        )

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "input_lengths": content_lengths,
        }

    @staticmethod
    def _first_supervised_index(loss_mask: list, default: int) -> int:
        """First index where ``loss_mask`` is truthy (the response start), else
        ``default`` (no supervised token -> treat the whole sample as prefix)."""
        for i, v in enumerate(loss_mask):
            if v:
                return i
        return default

    def _block_fill_ends(self, content_lengths: torch.Tensor, prefix_lengths: torch.Tensor) -> torch.Tensor:
        """Per-sample end of the EOS block-fill, RESPONSE-RELATIVE.

        The fill rounds the response length (measured from ``prefix``, the response
        start) up to a ``block_size`` multiple, so ``fill_end - prefix`` is a whole
        number of canvas blocks. With ``prefix == 0`` (no prompt, e.g. plain MDLM)
        this reduces to the old absolute rounding, so non-SFT paths are unchanged.
        """
        cl = content_lengths
        if self.max_seq_len is not None:
            cl = cl.clamp(max=self.max_seq_len)
        bs = self.block_size
        if bs is None or bs <= 1:
            return cl.clone()
        prefix = prefix_lengths.clamp(max=cl)
        resp = (cl - prefix).clamp(min=0)
        resp_blocks = ((resp + bs - 1) // bs) * bs
        return prefix + resp_blocks

    def _compute_target_length(self, fill_ends: torch.Tensor) -> int:
        max_len = int(fill_ends.max().item()) if fill_ends.numel() else 0

        psd = self.pad_seq_len_divisible
        bs = self.block_size
        if psd is not None and psd > 1:
            alignment = math.lcm(bs or 1, psd)
            max_len = ((max_len + alignment - 1) // alignment) * alignment

        # Hard cap after alignment to prevent OOM
        if self.max_seq_len is not None:
            max_len = min(max_len, self.max_seq_len)

        return max(max_len, 1)

    def _pad_and_fill(
        self,
        samples: List[list],
        content_lengths: torch.Tensor,
        fill_ends: torch.Tensor,
        target_len: int,
        pad_value: int,
        block_pad_value: int,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        """Pad variable-length lists to *target_len* with two-stage fill.

        For each sample:
          - ``[0, content_length)``       → original content
          - ``[content_length, fill_end)``→ *block_pad_value* (EOS block-fill,
            response-relative; ``fill_end`` from :meth:`_block_fill_ends`)
          - ``[fill_end, target_len)``    → *pad_value* (stage-2 global pad)
        """
        B = len(samples)
        out = torch.full((B, target_len), pad_value, dtype=dtype)

        for b in range(B):
            cl = int(content_lengths[b].item())
            seq = samples[b]
            copy_len = min(cl, target_len, len(seq))
            out[b, :copy_len] = torch.tensor(seq[:copy_len], dtype=dtype)

            fe = min(int(fill_ends[b].item()), target_len)
            if fe > cl:
                out[b, cl:fe] = block_pad_value

        return out
