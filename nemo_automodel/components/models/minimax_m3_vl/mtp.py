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

"""MiniMax M3 multi-token prediction (MTP), DeepSeek-V3 style.

The checkpoint carries a single MTP module under ``model.mtp.layers.0``:
``enorm``/``hnorm`` (Gemma RMSNorm of the next-token embedding and the previous
hidden state), ``eh_proj`` (Linear ``2*hidden -> hidden`` over their
concatenation), a full MoE+sparse decoder ``transformer_layer``, and
``final_layernorm``. There is no separate output projection — the prediction
head is the **shared** main ``lm_head``.

sglang skips MTP at load (inference-only); the reference is the DeepSeek-V3 MTP
algorithm.
"""

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.mtp.mtp import roll_tensor
from nemo_automodel.components.models.minimax_m3_vl.layers import Block, MiniMaxM3RMSNorm
from nemo_automodel.components.moe.layers import MoEConfig


class MiniMaxM3MTPBlock(nn.Module):
    """One MTP depth: ``eh_proj(cat[enorm(emb), hnorm(h)]) -> Block -> final_layernorm``.

    ``transformer_layer`` is a full M3 decoder block; it is constructed at the
    last decoder index (always MoE + sparse-attention in M3) so the shared
    :class:`~...layers.Block` builds the routed MoE and the sparse-attention
    indexer automatically.
    """

    def __init__(self, config: Any, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        gemma = getattr(config, "use_gemma_norm", False)
        self.enorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)
        self.hnorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)
        self.eh_proj = initialize_linear_module(backend.linear, 2 * config.hidden_size, config.hidden_size, bias=False)
        self.transformer_layer = Block(config.num_hidden_layers - 1, config, moe_config, backend)
        self.final_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        x = self.eh_proj(torch.cat([self.enorm(embed_input), self.hnorm(hidden_states)], dim=-1))
        x = self.transformer_layer(
            x=x, freqs_cis=freqs_cis, attention_mask=attention_mask, padding_mask=padding_mask, **attn_kwargs
        )
        return self.final_layernorm(x)

    def init_weights(self, buffer_device: torch.device) -> None:
        self.enorm.reset_parameters()
        self.hnorm.reset_parameters()
        self.final_layernorm.reset_parameters()
        nn.init.trunc_normal_(self.eh_proj.weight, mean=0.0, std=0.02)
        self.transformer_layer.init_weights(buffer_device)


class MiniMaxM3MTP(nn.Module):
    """Stack of MTP depths (M3 ships a single depth)."""

    def __init__(self, config: Any, moe_config: MoEConfig, backend: BackendConfig, num_modules: int):
        super().__init__()
        self.layers = nn.ModuleList([MiniMaxM3MTPBlock(config, moe_config, backend) for _ in range(num_modules)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        embed_fn,
        lm_head: nn.Module,
        freqs_cis: torch.Tensor,
        **block_kwargs: Any,
    ) -> list[torch.Tensor]:
        """Return per-depth next-token-+k logits using the shared ``lm_head``."""
        per_depth_logits: list[torch.Tensor] = []
        cur_ids = input_ids
        hidden = hidden_states
        for block in self.layers:
            cur_ids = roll_tensor(cur_ids, shifts=-1, dim=-1)
            hidden = block(hidden, embed_input=embed_fn(cur_ids), freqs_cis=freqs_cis, **block_kwargs)
            per_depth_logits.append(lm_head(hidden))
        return per_depth_logits

    def init_weights(self, buffer_device: torch.device) -> None:
        for block in self.layers:
            block.init_weights(buffer_device)
