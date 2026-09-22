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
from nemo_automodel.components.models.titans.layers import (
    NeuralMemoryState,
    TitansBlock,
    TitansMACBlock,
    TitansMAGBlock,
    TitansMALBlock,
    TitansRMSNorm,
)
from nemo_automodel.components.models.titans.state_dict_adapter import TitansStateDictAdapter
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


@dataclass(frozen=True)
class TitansInferenceState:
    """Recurrent deep-memory state for aligned LMM or MAC inference."""

    memory_states: tuple[NeuralMemoryState, ...]
    tokens_seen: int

    def detach(self) -> "TitansInferenceState":
        """Detach all fast-weight tensors before carrying state across calls."""

        def detach_memory(state: NeuralMemoryState) -> NeuralMemoryState:
            return NeuralMemoryState(
                weights=tuple(tensor.detach() for tensor in state.weights),
                momentum=tuple(tensor.detach() for tensor in state.momentum),
                qkv_history=tuple(tensor.detach() for tensor in state.qkv_history),
            )

        return TitansInferenceState(
            memory_states=tuple(detach_memory(state) for state in self.memory_states),
            tokens_seen=self.tokens_seen,
        )


class TitansPreTrainedModel(PreTrainedModel):
    """Base class wiring config, no-split modules, and fp32 precision contract."""

    config_class = TitansConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TitansBlock", "TitansMACBlock", "TitansMAGBlock", "TitansMALBlock"]
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
        if config.architecture_variant != "mac" and config.num_persistent_memory_tokens:
            self.persistent_memory = nn.Parameter(
                torch.empty(config.num_persistent_memory_tokens, config.hidden_size, dtype=dtype)
            )
            nn.init.trunc_normal_(self.persistent_memory, mean=0.0, std=config.initializer_range)
        if config.architecture_variant == "mac":
            self.longterm_memory = nn.Parameter(
                torch.empty(config.num_longterm_memory_tokens, config.hidden_size, dtype=dtype)
            )
            nn.init.trunc_normal_(self.longterm_memory, mean=0.0, std=config.initializer_range)
            block_cls = TitansMACBlock
        else:
            block_cls = {
                "lmm": TitansBlock,
                "mag": TitansMAGBlock,
                "mal": TitansMALBlock,
            }[config.architecture_variant]
        self.layers = nn.ModuleList([block_cls(config, dtype=dtype) for _ in range(config.num_hidden_layers)])
        self.norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def _insert_longterm_memory(self, h: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Append learned long-term-memory tokens to each padded MAC segment."""
        batch, sequence, dim = h.shape
        segment = self.config.attention_segment_size
        padding = (-sequence) % segment
        if padding:
            h = nn.functional.pad(h, (0, 0, 0, padding))
        groups = h.shape[1] // segment
        h = h.view(batch, groups, segment, dim)
        memory = self.longterm_memory.view(1, 1, -1, dim).expand(batch, groups, -1, -1)
        return torch.cat((h, memory), dim=2).reshape(batch, -1, dim), padding

    def _remove_longterm_memory(self, h: torch.Tensor, padding: int) -> torch.Tensor:
        batch, _, dim = h.shape
        augmented_segment = self.config.attention_segment_size + self.config.num_longterm_memory_tokens
        groups = h.shape[1] // augmented_segment
        h = h.view(batch, groups, augmented_segment, dim)
        h = h[:, :, : self.config.attention_segment_size].reshape(batch, -1, dim)
        return h[:, : h.shape[1] - padding] if padding else h

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        inference_state: TitansInferenceState | None = None,
        return_inference_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, TitansInferenceState]:
        streaming = inference_state is not None or return_inference_state
        if streaming:
            if self.config.architecture_variant not in {"lmm", "mac"}:
                raise NotImplementedError(
                    "Stateful inference currently supports LMM and segment-aligned MAC; "
                    "MAG/MAL also require attention-cache state."
                )
            if self.config.mem_depth < 2:
                raise NotImplementedError("Stateful inference requires deep memory (mem_depth >= 2).")
            if inference_state is not None and len(inference_state.memory_states) != len(self.layers):
                raise ValueError(
                    "Inference state layer count does not match the model: "
                    f"{len(inference_state.memory_states)} != {len(self.layers)}."
                )

        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        input_length = h.shape[1]
        if (
            streaming
            and self.config.architecture_variant == "mac"
            and input_length % self.config.attention_segment_size
        ):
            raise ValueError(
                "Stateful MAC calls must contain complete ordinary-token segments: "
                f"received {input_length} tokens, expected a multiple of "
                f"{self.config.attention_segment_size}."
            )
        persistent_length = (
            self.config.num_persistent_memory_tokens
            if self.config.architecture_variant != "mac"
            else 0
        )
        if self.config.architecture_variant == "mac":
            h, mac_padding = self._insert_longterm_memory(h)
        elif persistent_length and inference_state is None:
            persistent = self.persistent_memory.unsqueeze(0).expand(h.shape[0], -1, -1)
            h = torch.cat((persistent, h), dim=1)
        if streaming:
            alignment = self.config.chunk_size
            if self.config.deep_memory_backend == "titans_pytorch" and self.config.memory_batch_size is not None:
                alignment = self.config.memory_batch_size
            if h.shape[1] % alignment:
                raise ValueError(
                    "Stateful inference calls must preserve memory re-anchoring boundaries: "
                    f"received {h.shape[1]} embedded tokens, expected a multiple of {alignment}."
                )
        next_memory_states = []
        for layer_index, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                if streaming:
                    raise ValueError("Stateful inference is incompatible with training-time gradient checkpointing.")
                h = self._gradient_checkpointing_func(layer.__call__, h)
            elif streaming:
                past_memory = (
                    inference_state.memory_states[layer_index]
                    if inference_state is not None
                    else None
                )
                h, next_memory = layer(h, past_state=past_memory, return_state=True)
                next_memory_states.append(next_memory)
            else:
                h = layer(h)
        h = self.norm(h)
        if self.config.architecture_variant == "mac":
            h = self._remove_longterm_memory(h, mac_padding)
        elif persistent_length and inference_state is None:
            h = h[:, persistent_length:]
        if return_inference_state:
            state = TitansInferenceState(
                memory_states=tuple(next_memory_states),
                tokens_seen=(inference_state.tokens_seen if inference_state is not None else 0) + input_length,
            )
            return h, state
        return h


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
        inference_state: TitansInferenceState | None = None,
        return_inference_state: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        model_result = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            inference_state=inference_state,
            return_inference_state=return_inference_state,
        )
        if return_inference_state:
            hidden, next_inference_state = model_result
        else:
            hidden = model_result
            next_inference_state = None
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_inference_state,
        )

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        inference_state: TitansInferenceState | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> CausalLMOutputWithPast:
        """Run one aligned LMM/MAC prefill and return detached recurrent state."""
        output = self.forward(
            input_ids=input_ids,
            logits_to_keep=logits_to_keep,
            inference_state=inference_state,
            return_inference_state=True,
        )
        output.past_key_values = output.past_key_values.detach()
        return output

    @torch.no_grad()
    def generate_full_prefix(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate by recomputing the complete prefix at every decoding step.

        This is the correctness-first path for every Titans architecture. It
        preserves the configured chunk/segment semantics without pretending
        that hybrid attention or a partially filled memory chunk has a valid
        cache. Stateful decoding can replace it only after prefix-equivalence
        tests pass for the relevant architecture.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence]; got {tuple(input_ids.shape)}.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided.")

        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        stop_id = self.config.eos_token_id if eos_token_id is None else eos_token_id

        for _ in range(max_new_tokens):
            next_logits = self(generated, logits_to_keep=1).logits[:, -1]
            if do_sample:
                next_logits = next_logits / temperature
                if top_k is not None:
                    k = min(top_k, next_logits.shape[-1])
                    threshold = torch.topk(next_logits, k, dim=-1).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
                next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            if stop_id is not None:
                stop_tokens = torch.full_like(next_token, stop_id)
                next_token = torch.where(finished[:, None], stop_tokens, next_token)
                finished = finished | next_token.squeeze(-1).eq(stop_id)
            generated = torch.cat((generated, next_token), dim=-1)
            if stop_id is not None and finished.all():
                break

        return generated

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
            if hasattr(self.model, "persistent_memory"):
                nn.init.trunc_normal_(self.model.persistent_memory, mean=0.0, std=std)
            if hasattr(self.model, "longterm_memory"):
                nn.init.trunc_normal_(self.model.longterm_memory, mean=0.0, std=std)
            self.model.norm.reset_parameters()
            for layer in self.model.layers:
                layer.init_weights(std)
            if not self.config.tie_word_embeddings:
                final_std = self.config.hidden_size**-0.5
                nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=final_std)
        self.tie_weights()


ModelClass = TitansForCausalLM
