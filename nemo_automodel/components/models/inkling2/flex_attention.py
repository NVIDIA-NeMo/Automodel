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

"""Compiled learned-relative GQA, with differentiable per-query attention sinks."""

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from nemo_automodel.shared.import_utils import safe_import_from

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import AuxOutput

# Keep the eager model importable with older PyTorch installations.
_HAVE_AUX, AuxRequest = safe_import_from("torch.nn.attention.flex_attention", "AuxRequest")


class Inkling2FlexAttention(nn.Module):
    """Own compiled callables, with masks rebuilt from each forward's token validity.

    Relative profiles have shape [batch, heads, sequence, relative_extent]; no
    sequence-by-sequence score/bias tensor is constructed. The Triton backend
    supports gradients to captured profiles and returned log-sum-exp values.
    """

    def __init__(self, *, head_dim: int, window: int | None) -> None:
        super().__init__()
        if not _HAVE_AUX:
            raise ImportError(
                "Inkling2 flex attention requires PyTorch with differentiable FlexAttention auxiliary outputs"
            )
        self.scale = head_dim**-0.5
        self.window = window
        self._attention = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        self._make_mask = torch.compile(create_block_mask, fullgraph=True, dynamic=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        profiles: torch.Tensor,
        sink: torch.Tensor,
        valid: torch.Tensor,
        key_valid: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
        query_offset: int = 0,
    ) -> torch.Tensor:
        """Compute content plus learned relative attention, including its sink.

        Args:
            q: Query tensor of shape [batch, sequence, heads, head_dim].
            k: Key tensor of shape [batch, key_sequence, kv_heads, head_dim].
            v: Value tensor of shape [batch, key_sequence, kv_heads, head_dim].
            profiles: Relative logits of shape [batch, heads, sequence, relative_extent].
                Retains the projection's FP32/BF16 rounding boundary.
            sink: Sink logits of shape [batch, sequence, heads].
            valid: Boolean tensor of shape [batch, sequence]. Each batch row has
                a local slice of one global contiguous real-token span.
            key_valid: Boolean [batch, key_sequence] validity, global under CP;
                defaults to valid for CP1. Global real-token positions are consecutive.
            positions: Logical query positions [batch, sequence], reset per document.
            key_positions: Logical key positions [batch, key_sequence], global under CP.
            query_offset: First local query's global token column, zero without CP.

        Returns:
            FP32 tensor of shape [batch, sequence, heads, head_dim]. Inputs remain
            unchanged. CUDA is required; Q/K/V and score accumulation use FP32.
        """
        if q.device.type != "cuda":
            raise ValueError("Inkling2 flex attention requires CUDA; use attn='eager' on CPU")
        batch, sequence = valid.shape
        key_valid = valid if key_valid is None else key_valid
        key_sequence = key_valid.shape[1]
        if (positions is None) != (key_positions is None):
            raise ValueError("Packed attention requires both query and key positions")
        window, extent, scale = self.window, profiles.shape[-1], self.scale

        def mask(b: torch.Tensor, h: torch.Tensor, qi: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
            """Compute token validity.

            Args:
                b: Scalar batch index.
                h: Scalar head index, unused because masks are shared across heads.
                qi: Scalar query index, including kernel padding.
                ki: Scalar key index, including kernel padding.

            Returns:
                Scalar Boolean indicating an allowed query/key pair.
            """
            allowed = (qi < sequence) & (ki < key_sequence) & (ki <= qi + query_offset)
            allowed = allowed & valid[b, qi.clamp(max=sequence - 1)] & key_valid[b, ki.clamp(max=key_sequence - 1)]
            if positions is not None and key_positions is not None:
                query_origin = qi + query_offset - positions[b, qi.clamp(max=sequence - 1)]
                key_origin = ki - key_positions[b, ki.clamp(max=key_sequence - 1)]
                allowed = allowed & (query_origin == key_origin)
            return allowed if window is None else allowed & (qi + query_offset - ki < window)

        def score_mod(
            score: torch.Tensor, b: torch.Tensor, h: torch.Tensor, qi: torch.Tensor, ki: torch.Tensor
        ) -> torch.Tensor:
            """Add learned relative logits after content-score scaling.

            Args:
                score: Scalar FP32 unscaled content score.
                b: Scalar batch index.
                h: Scalar query-head index.
                qi: Scalar query position.
                ki: Scalar key position.

            Returns:
                Scalar FP32 content-plus-relative attention logit.
            """
            distance = qi + query_offset - ki
            relative = profiles[b, h, qi.clamp(max=sequence - 1), distance.clamp(0, extent - 1)].float()
            relative = torch.where((distance >= 0) & (distance < extent), relative, 0.0)
            return score * scale + relative

        # Compilation is essential here: eager create_block_mask would construct
        # a dense token mask while reducing it to block metadata.
        block_mask = self._make_mask(mask, batch, None, sequence, key_sequence, device=q.device)
        # Triton's dot product requires width >= 16. Pad tiny test heads without
        # changing the original scale or the externally visible head dimension.
        pad = max(16, 1 << (q.shape[-1] - 1).bit_length()) - q.shape[-1]
        result = self._attention(
            F.pad(q.transpose(1, 2).float(), (0, pad)),
            F.pad(k.transpose(1, 2).float(), (0, pad)),
            F.pad(v.transpose(1, 2).float(), (0, pad)),
            score_mod=score_mod,
            block_mask=block_mask,
            scale=1.0,
            enable_gqa=True,
            return_aux=AuxRequest(lse=True),
            kernel_options={"BACKEND": "TRITON"},
        )
        attended, auxiliary = cast("tuple[torch.Tensor, AuxOutput]", result)
        # Do not detach LSE: its gradient supplies the sink's normalization term
        # to Q/K and relative profiles as well as to the sink projection itself.
        assert auxiliary.lse is not None
        factor = torch.sigmoid(auxiliary.lse - sink.float().transpose(1, 2))
        return (attended[..., : q.shape[-1]] * factor.unsqueeze(-1)).transpose(1, 2)
