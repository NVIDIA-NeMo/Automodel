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

"""Model-owned PP2 partition and wire layouts for the CSA2 source boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import DeepseekV41Config


@dataclass(frozen=True)
class DeepseekV41BackboneOutput:
    """Backbone values before or after the stage-local final norm.

    Attributes:
        hidden_states: [batch, local_sequence, streams, hidden] on an intermediate
            stage, or normalized [batch, local_sequence, hidden] on the last stage.
        pre_mix: FP32 coefficients [batch, local_sequence, streams]. They remain
            differentiable across a pipeline boundary.
        residual_streams: Optional pre-block states, each
            [batch, local_sequence, streams, hidden], for parity diagnostics.
    """

    hidden_states: torch.Tensor
    pre_mix: torch.Tensor
    residual_streams: tuple[torch.Tensor, ...] | None


class DeepseekV41PipelineMixin:
    """PP2 at layer 14: no shared CSA2 KV survives this source-layer boundary."""

    config: DeepseekV41Config
    model: nn.Module
    lm_head: nn.Module | None
    _pp_keep_self_forward = True
    _pp_return_hidden_states_supported = True
    _pp_return_hidden_states = False

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep language 0–13 and vision on stage 0; language 14–last on stage 1."""
        text = self.config.text_config
        if len(module_names_per_stage) != 2:
            raise ValueError("DeepSeek V4.1 currently supports two pipeline stages, split before layer 14")
        if text.num_hidden_layers <= 14 or 14 not in text.kv_source_layer_ids:
            raise ValueError("DeepSeek V4.1 PP2 requires layer 14 to be a retained CSA2 KV source")
        first = [f"{layers_prefix}embed_tokens", *[f"{layers_prefix}layers.{i}" for i in range(14)]]
        last = [
            *[f"{layers_prefix}layers.{i}" for i in range(14, text.num_hidden_layers)],
            f"{layers_prefix}norm",
            "lm_head",
        ]
        for modules in (first, last):
            # The hash module has deterministic buffers, not a trainable table.
            # Both stages need original token history for their local Engram layer.
            if self.model.engram_hash is not None:
                modules.append(f"{layers_prefix}engram_hash")
        if self.model.vision is not None:
            first.extend(
                f"{layers_prefix}{name}" for name in ("vision", "aligner", "image_start", "image_end", "image_newline")
            )
        return [first, last]

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Describe local CP-shard tensors sent between pipeline stages.

        Returns:
            Input/output meta tensors. The intermediate payload consists of
            hidden streams [microbatch, local_sequence, streams, hidden] in the
            compute dtype, FP32 pre-mix [microbatch, local_sequence, streams],
            and int64 token metadata [microbatch, local_sequence, 5]. Metadata
            columns are token ID, RoPE position, validity, document ID, and
            vision-token type. A document-ID column of -1 preserves unpacked mode.
            Final output is hidden [microbatch, local_sequence,
            hidden] for a linear loss, or FP32 logits [microbatch, local_sequence, vocab].
            seq_len is already local; do not divide it by CP again.
        """
        text = self.config.text_config
        shape = (microbatch_size, seq_len)
        payload = (
            torch.empty(*shape, text.hc_mult, text.hidden_size, device="meta", dtype=dtype),
            torch.empty(*shape, text.hc_mult, device="meta", dtype=torch.float32),
            torch.empty(*shape, 5, device="meta", dtype=torch.long),
        )
        inputs = (torch.empty(*shape, device="meta", dtype=torch.long),) if is_first else payload
        if self.lm_head is None:
            outputs = payload
        else:
            width = text.hidden_size if self._pp_return_hidden_states else text.vocab_size
            output_dtype = dtype if self._pp_return_hidden_states else torch.float32
            outputs = (torch.empty(*shape, width, device="meta", dtype=output_dtype),)
        return inputs, outputs
