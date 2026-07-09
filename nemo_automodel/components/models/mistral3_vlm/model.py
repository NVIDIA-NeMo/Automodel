# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""FP8-native Mistral3 VLM (dawn-ridge / Mistral-3.5 128B).

Custom wrapper around HF's ``Mistral3ForConditionalGeneration`` that:

  * Inherits the full VLM architecture (vision_tower + multi_modal_projector
    + Ministral3 language_model) so image inputs flow through Pixtral.
  * Attaches ``Mistral3FP8StateDictAdapter.for_vlm_full()`` so FP8 dequant
    runs inside the standard DCP load path (avoids HF's FineGrainedFP8
    loader, which materializes the full BF16 model on every rank pre-PP-split
    and OOMs on 80 GB H100).
  * Attaches a one-shot forward pre-hook on every rotary submodule to
    recompute ``inv_freq`` on first call — needed because HF's Ministral3 /
    Pixtral rotaries compute ``inv_freq`` in ``__init__``, so meta-init +
    ``to_empty`` leaves the buffer at uninitialized memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers import PretrainedConfig
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3CausalLMOutputWithPast,
)
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration as _HFMistral3ForConditionalGeneration,
)

from nemo_automodel.components.checkpoint.utils import reject_unsupported_tied_word_embeddings
from nemo_automodel.components.models.common.utils import compute_lm_head_logits
from nemo_automodel.components.models.mistral3_vlm.state_dict_adapter import (
    Mistral3FP8StateDictAdapter,
)

logger = logging.getLogger(__name__)


def _rotary_reinit_self_hook(module, args, kwargs):
    """One-shot forward pre-hook that recomputes *this rotary module's* own
    ``inv_freq`` on first call.

    Attached per-rotary rather than on the outer VLM so it fires correctly
    under pipeline parallelism, where the outer model's ``forward`` is never
    called directly — the PP schedule dispatches each stage's sub-modules
    individually, and rotary modules run inside every attention layer.

    Background: HF's Ministral3 / Pixtral rotary classes initialise
    ``inv_freq`` (and related attributes) in their ``__init__``. Under
    ``accelerate.init_empty_weights`` that becomes a meta tensor, and the
    subsequent ``to_empty(device)`` call leaves it uninitialised device
    memory. Neither class exposes ``rope_init_fn`` as an attribute, so the
    generic ``_reinit_non_persistent_buffers`` helper doesn't match. We
    recover correctness by re-running the module's own ``__init__`` on the
    target device outside the init_empty_weights context — both Ministral3
    (YaRN) and Pixtral (2D patch positions) produce the right values this
    way, since we defer to the class's authoritative init logic.
    """
    if getattr(module, "_mistral3_fp8_rotary_reinit_done", False):
        return
    # Pick a device that has real storage. Prefer a buffer with non-meta
    # device (rotary modules typically have `inv_freq` buffer only).
    device = None
    for buf in module.buffers(recurse=False):
        if buf.device.type != "meta":
            device = buf.device
            break
    if device is None:
        # Fallback: whatever pytorch's current device default is.
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    try:
        type(module).__init__(module, module.config, device=device)
    except TypeError:
        try:
            type(module).__init__(module, module.config)
        except Exception:
            module._mistral3_fp8_rotary_reinit_done = True
            return
    except Exception:
        module._mistral3_fp8_rotary_reinit_done = True
        return
    # Ensure every tensor the re-init created ends up on the right device.
    # PixtralRotaryEmbedding precomputes per-patch position caches beyond
    # just `inv_freq`, so move the whole module.
    module.to(device=device)
    module._mistral3_fp8_rotary_reinit_done = True


class Mistral3FP8VLMForConditionalGeneration(_HFMistral3ForConditionalGeneration):
    """Full-VLM (vision + text) FP8 loader for Mistral3ForConditionalGeneration.

    Used when the user instantiates through
    ``NeMoAutoModelForImageTextToText.from_pretrained`` on an FP8-native
    Mistral3 VLM checkpoint (e.g. dawn-ridge-128B).
    """

    # See checkpointing.py:initialize_model_weights — gate on this attribute
    # to skip HF's ``initialize_weights()``. The upcoming adapter load will
    # populate every tensor, and skipping avoids a stage-divergent DTensor
    # collective inside HF's init on PP setups that otherwise hangs
    # indefinitely (empirically verified: without this attribute the 4-layer
    # smoke never reaches the adapter load stage within 300s).
    _skip_init_weights_on_load = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: PretrainedConfig):
        # The supported Mistral3 checkpoint (mistralai/Mistral-Medium-3.5-128B) is
        # untied (tie_word_embeddings=False), so reject tie_word_embeddings=True.
        reject_unsupported_tied_word_embeddings(config, type(self).__name__)
        # HF's Mistral3ForConditionalGeneration.__init__ consults
        # ``config.quantization_config`` and swaps nn.Linear → FP8Linear for
        # every language_model Linear. FP8Linear registers a 0-d
        # ``weight_scale_inv`` parameter — which FSDP2's ``fully_shard``
        # rejects with "doesn't support scalar parameters". We handle FP8
        # dequant ourselves via the state_dict adapter, so tell HF to skip
        # the linear swap by flipping ``dequantize=True`` on the config
        # (see transformers/integrations/finegrained_fp8.py:742, which early-
        # returns from replace_with_fp8_linear when dequantize is truthy).
        # ``apply_model_infrastructure`` still sees hasattr(config,
        # 'quantization_config'), so ``dequantize_base_checkpoint`` is set
        # to True and our adapter's ``to_hf(quantization=True)`` path runs.
        qc = getattr(config, "quantization_config", None)
        if qc is not None:
            if isinstance(qc, dict):
                qc["dequantize"] = True
            else:
                try:
                    qc.dequantize = True
                except AttributeError:
                    pass
        super().__init__(config)
        self.state_dict_adapter = Mistral3FP8StateDictAdapter.for_vlm_full(config)

        # Lazy non-persistent buffer reinit. HF's Ministral3RotaryEmbedding /
        # PixtralRotaryEmbedding compute `inv_freq` in their __init__. Under
        # the accelerate ``init_empty_weights`` context (used by the NeMo Auto
        # meta-init path), the computed buffer is converted to meta; and when
        # the checkpointer later calls ``to_empty(device)`` it becomes
        # uninitialized real memory — garbage rotary positional embeddings.
        # ``_reinit_non_persistent_buffers`` in checkpointing.py has Pattern-1
        # logic that expects a ``rope_init_fn`` attribute, which HF's rotary
        # classes do not expose. Rather than touch the generic checkpointer,
        # we attach a one-shot forward pre-hook to every rotary submodule so
        # that it recomputes itself the first time it is called. Per-rotary
        # registration (as opposed to a single outer hook) is required so the
        # reinit fires under PP, where the outer VLM's forward is never
        # invoked directly — the PP schedule dispatches stage sub-modules
        # individually and rotary runs inside each attention layer.
        for sub in self.modules():
            if "inv_freq" in getattr(sub, "_buffers", {}):
                sub._mistral3_fp8_rotary_reinit_done = False
                sub.register_forward_pre_hook(_rotary_reinit_self_hook, with_kwargs=True, prepend=True)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Mistral3CausalLMOutputWithPast:
        """Forward pass with memory-efficient fused cross-entropy (cut-CE) support.

        Overrides HF's ``Mistral3ForConditionalGeneration.forward`` so the
        ``train_ft`` recipe can enable ``FusedLinearCrossEntropy``. The recipe
        only does so when (a) ``forward`` exposes a ``logits_to_keep`` parameter
        and (b) calling the model returns an output that carries the FINAL hidden
        states (full sequence) while ``logits`` cover only the kept positions.

        HF's stock forward gates ``hidden_states`` on a per-call
        ``output_hidden_states`` kwarg (which the recipe does not pass) and emits
        the full per-layer tuple. Here we instead resolve ``output_hidden_states``
        from the text sub-config and surface the inner model's ``last_hidden_state``
        (the single ``[B, S, H]`` tensor fed to ``lm_head``) directly, which is
        what ``get_final_hidden_states`` consumes.

        Args:
            input_ids: Input token IDs ``[B, S]``.
            pixel_values: Optional image pixel values for the vision tower.
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.
            past_key_values: Optional cached key/values.
            inputs_embeds: Optional pre-computed embeddings.
            labels: Optional labels for loss computation.
            use_cache: Whether to use KV caching.
            logits_to_keep: Number of final logits to compute (0=all, N=last N tokens).
            image_sizes: Optional image sizes for the vision tower.
            output_hidden_states: Whether to surface the final hidden states on the
                output (defaults to the text sub-config's ``output_hidden_states``).
            **kwargs: Additional arguments forwarded to the base model.

        Returns:
            :class:`~transformers.models.mistral3.modeling_mistral3.Mistral3CausalLMOutputWithPast`
            with ``logits``, optional ``loss``, ``past_key_values``, and (when
            ``output_hidden_states`` is set) the final ``hidden_states`` tensor.
        """
        text_config = getattr(self.config, "text_config", self.config)
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(text_config, "output_hidden_states", False)
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_sizes=image_sizes,
            **kwargs,
        )

        # Final hidden states fed to lm_head (single [B, S, H] tensor).
        hidden_states = outputs[0]

        logits = compute_lm_head_logits(self.lm_head, hidden_states, logits_to_keep).logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=text_config.vocab_size, **kwargs)

        return Mistral3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    @classmethod
    def supports_config(cls, config: PretrainedConfig) -> bool:
        """Claim FP8-native Mistral3 VLM configs.

        Matches ``Mistral3Config`` (outer VLM) with a ministral3 text backbone
        and ``quantization_config.quant_method == 'fp8'``.
        """
        text_config = getattr(config, "text_config", None)
        if text_config is None or getattr(text_config, "model_type", None) != "ministral3":
            return False
        qc = getattr(config, "quantization_config", None)
        if qc is None:
            return False
        method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        return method == "fp8"
