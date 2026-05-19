# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Compatibility patches for HuggingFace Qwen3 models."""

import functools
import logging

logger = logging.getLogger(__name__)


def apply_dtensor_logits_to_keep_patch() -> None:
    """Patch HF Qwen3 forward to avoid a no-op slice that breaks DTensor TP."""
    try:  # pragma: no cover
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    except Exception:
        logger.debug("Qwen3ForCausalLM is unavailable; skipping DTensor logits_to_keep patch.", exc_info=True)
        return

    if getattr(Qwen3ForCausalLM.forward, "__nemo_dtensor_logits_to_keep_patched__", False):
        return

    _orig_forward = Qwen3ForCausalLM.forward

    @functools.wraps(_orig_forward)
    def _patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    _patched_forward.__nemo_dtensor_logits_to_keep_patched__ = True  # type: ignore[attr-defined]
    Qwen3ForCausalLM.forward = _patched_forward  # type: ignore[method-assign]


def apply_global_patches() -> None:
    """Apply process-wide compatibility patches for HuggingFace Qwen3 models."""
    apply_dtensor_logits_to_keep_patch()
