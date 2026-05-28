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

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1RotaryEmbedding

from nemo_automodel.components.models.common import (
    BackendConfig,
    HFCheckpointingMixin,
)
from nemo_automodel.components.models.falcon_h1.layers import FalconH1DecoderLayer


class FalconH1Model(nn.Module):
    """Falcon-H1 backbone: embeddings + stack of parallel-fuse decoder layers + final norm."""

    def __init__(self, config, backend: BackendConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        # Embedding table
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_multiplier = config.embedding_multiplier

        # Stack of decoder layers (each runs Mamba ‖ attention in parallel + MLP)
        self.layers = nn.ModuleList(
            [FalconH1DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Final RMSNorm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embedding owned by the backbone, computed once per forward
        self.rotary_emb = FalconH1RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # 1. Embed tokens (or accept pre-computed embeds)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        # 2. Falcon-H1: scale embeddings by the µP multiplier
        hidden_states = inputs_embeds * self.embedding_multiplier

        # 3. Build default position_ids if not provided
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # 4. Compute RoPE once at the top, pass to every layer
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 5. Run the decoder stack
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )

        # 6. Final norm
        hidden_states = self.norm(hidden_states)
        return hidden_states


class FalconH1ForCausalLM(HFCheckpointingMixin, GenerationMixin, nn.Module):
    """Falcon-H1 with language modeling head."""

    main_input_name: str = "input_ids"

    # NeMo AutoModel TP/PP plans (required class attributes)
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    @classmethod
    def from_config(cls, config, backend: BackendConfig | None = None, **kwargs):
        return cls(config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(self, config, backend: BackendConfig | None = None, **kwargs):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        # Backbone
        self.model = FalconH1Model(config, backend=self.backend)

        # LM head — independent of embed_tokens because tie_word_embeddings=False
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.lm_head_multiplier = config.lm_head_multiplier

        # Required by GenerationMixin
        self.generation_config = GenerationConfig()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        # 1. Backbone forward
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # 2. LM head — optionally slice to last few positions for generation efficiency
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 3. Falcon-H1: scale logits by the µP multiplier
        logits = logits * self.lm_head_multiplier

        # 4. Optional loss for training
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


ModelClass = FalconH1ForCausalLM
