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

"""Linear-memory Titans causal LM (Gated DeltaNet + data-dependent momentum).

The model is a standard pre-norm decoder stack whose token mixer is the
:class:`NeuralMemory` linear memory. It is a self-contained HuggingFace
``PreTrainedModel`` (so ``AutoModelForCausalLM.from_config`` works) that also
carries NeMo AutoModel's :class:`HFCheckpointingMixin` and a declared
``ModelCapabilities`` for first-class use through ``NeMoAutoModelForCausalLM``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.titans.config import TitansConfig
from nemo_automodel.components.models.titans.layers import TitansBlock, TitansRMSNorm
from nemo_automodel.components.models.titans.state_dict_adapter import TitansStateDictAdapter
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class TitansPreTrainedModel(PreTrainedModel):
    """Base class wiring config, no-split modules, and fp32 precision contract."""

    config_class = TitansConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TitansBlock"]
    # A_log / dt_bias are exponentiated in the decay gate; keep them fp32 under
    # any mixed-precision sharding (see layers.NeuralMemory and state_dict_adapter).
    _keep_in_fp32_modules = ["A_log", "dt_bias"]
    _keep_in_fp32_modules_strict = ["A_log", "dt_bias"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, TitansRMSNorm):
            module.reset_parameters()


class TitansModel(TitansPreTrainedModel):
    """Embedding + Titans decoder blocks + final norm."""

    def __init__(self, config: TitansConfig):
        super().__init__(config)
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList([TitansBlock(config, dtype=dtype) for _ in range(config.num_hidden_layers)])
        self.norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = self._gradient_checkpointing_func(layer.__call__, h)
            else:
                h = layer(h)
        return self.norm(h)


class TitansForCausalLM(HFCheckpointingMixin, TitansPreTrainedModel):
    """Linear-memory Titans model with a language-modeling head."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities (FSDP2/DDP only for Phase 1)."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: TitansConfig, *args, **kwargs):
        super().__init__(config)
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.model = TitansModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        self.state_dict_adapter = TitansStateDictAdapter(config)
        self.post_init()

    # --- HF embedding plumbing -------------------------------------------------
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)
        slice_idx = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        )
        logits = self.lm_head(hidden[:, slice_idx, :]).float()

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # --- NeMo AutoModel construction / init -----------------------------------
    @classmethod
    def from_config(cls, config: TitansConfig, *args, **kwargs):
        return cls(config, *args, **kwargs)

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Explicit weight init used by NeMo recipes (mirrors GatedDeltaNet init)."""
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        std = self.config.initializer_range
        with buffer_device:
            nn.init.trunc_normal_(self.model.embed_tokens.weight, mean=0.0, std=std)
            self.model.norm.reset_parameters()
            for layer in self.model.layers:
                layer.init_weights(std)
            if not self.config.tie_word_embeddings:
                final_std = self.config.hidden_size**-0.5
                nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=final_std)
        self.tie_weights()


ModelClass = TitansForCausalLM
