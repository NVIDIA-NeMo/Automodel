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

"""Qwen2.5-Omni Thinker for ASR / multimodal text generation.

Qwen2.5-Omni is the dense predecessor of Qwen3-Omni-Moe. For NeMo
AutoModel we only train the Thinker (audio + image + video + text); the
talker and token2wav components are dropped from the loaded checkpoint by
:class:`Qwen2_5OmniStateDictAdapter`.

Compared with :mod:`nemo_automodel.components.models.qwen3_omni_moe.model`,
this module is intentionally minimal:

- inherits HF's ``Qwen2_5OmniThinkerForConditionalGeneration`` directly
  (the text backbone is a standard dense Qwen2 transformer with MRoPE, so
  no custom rewrite is needed);
- adds :class:`HFCheckpointingMixin` for NeMo-compatible save/load;
- attaches :class:`Qwen2_5OmniStateDictAdapter` for ``thinker.*`` prefix
  handling;
- does NOT inherit ``MoEFSDPSyncMixin`` (dense, no experts).
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration as HFQwen2_5OmniThinkerForConditionalGeneration,
)

from nemo_automodel.components.checkpoint.utils import TieSupport, reject_unsupported_tie_word_embeddings
from nemo_automodel.components.models.common import BackendConfig, compute_lm_head_logits
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.qwen2_5_omni.state_dict_adapter import Qwen2_5OmniStateDictAdapter
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _resolve_thinker_config(config: Qwen2_5OmniConfig | Qwen2_5OmniThinkerConfig) -> Qwen2_5OmniThinkerConfig:
    """Return the thinker sub-config regardless of whether a full Omni or
    Thinker-only config was passed in."""
    if hasattr(config, "thinker_config") and config.thinker_config is not None:
        return config.thinker_config
    return config


class Qwen2_5OmniThinkerForConditionalGeneration(
    HFCheckpointingMixin,
    HFQwen2_5OmniThinkerForConditionalGeneration,
):
    """Qwen2.5-Omni Thinker (audio + image + video + text → text)."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        config = Qwen2_5OmniConfig.from_pretrained(pretrained_model_name_or_path)
        thinker_config = _resolve_thinker_config(config)
        return cls(thinker_config, backend=backend, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Qwen2_5OmniConfig | Qwen2_5OmniThinkerConfig,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(_resolve_thinker_config(config), backend=backend, **kwargs)

    def __init__(
        self,
        config: Qwen2_5OmniConfig | Qwen2_5OmniThinkerConfig,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        # Check the controlling top-level flag on the original config before
        # resolving to thinker_config and building the HF parent.
        reject_unsupported_tie_word_embeddings(type(self), config)
        thinker_config = _resolve_thinker_config(config)
        super().__init__(thinker_config)

        # HF Qwen2.5-Omni declares ``audio_tower.audio_bos_eos_token`` as an
        # ``nn.Embedding(2, output_dim)`` (modeling_qwen2_5_omni.py:751) but it
        # is never indexed in any forward (audio BOS/EOS are routed through the
        # text tokenizer's ``embed_tokens`` instead). The unused parameter
        # still has ``requires_grad=True``, so it ends up in the AdamW
        # ``param_groups`` and gets no gradients during training -> its AdamW
        # state stays empty -> DCP save omits its ``step``/``exp_avg`` keys ->
        # DCP load template expects them -> ``RuntimeError("Missing key in
        # checkpoint state_dict: …audio_bos_eos_token.weight.step")`` which is
        # masked as ``TypeError: cannot pickle code objects`` by DCP's
        # exception-propagation tuple. Delete the dead embedding here so it
        # never enters the optimizer; ``Qwen2_5OmniStateDictAdapter.from_hf``
        # also strips the matching HF checkpoint key.
        if hasattr(self, "audio_tower") and hasattr(self.audio_tower, "audio_bos_eos_token"):
            del self.audio_tower.audio_bos_eos_token

        self.backend = backend or BackendConfig()
        text_config = thinker_config.text_config if hasattr(thinker_config, "text_config") else thinker_config
        torch_dtype = getattr(text_config, "torch_dtype", None) or getattr(thinker_config, "torch_dtype", None)
        dtype = get_dtype(torch_dtype, torch.bfloat16) if torch_dtype is not None else torch.bfloat16

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen2_5OmniStateDictAdapter(
                thinker_config,
                backend=self.backend,
                dtype=dtype,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        video_second_per_grid: torch.Tensor | None = None,
        use_audio_in_video: bool | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        rope_deltas: torch.LongTensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Multimodal forward that mirrors HF's Thinker but supports cut-CE.

        This re-implements the body of HF's
        ``Qwen2_5OmniThinkerForConditionalGeneration.forward`` (same
        audio/image/video embedding merge and MRoPE index computation) so we
        can (a) gate the ``lm_head`` projection on ``logits_to_keep`` and
        (b) surface the FINAL hidden states (the ``lm_head`` input) on the
        returned :class:`~transformers.modeling_outputs.CausalLMOutputWithPast`.
        Together these let the recipe enable
        :class:`FusedLinearCrossEntropy` (cut-CE): it checks ``logits_to_keep``
        is in the signature and that the output carries ``hidden_states``.

        Audio is mandatory for ASR; image / video paths are kept enabled so
        the same class supports the full Thinker modality set.

        Args:
            logits_to_keep: If ``0`` (default), project all positions (no slice
                — DTensor cannot slice a full range). Otherwise compute logits
                only for the last ``logits_to_keep`` positions before ``lm_head``.
            output_hidden_states: When set, the returned output carries the
                final hidden states spanning the full sequence.

        Returns:
            :class:`~transformers.modeling_outputs.CausalLMOutputWithPast` with
            ``loss`` (when ``labels`` is given), ``logits``, ``past_key_values``,
            and ``hidden_states`` (the final hidden states when
            ``output_hidden_states`` is set, else ``None``).
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text, audios, image and video
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                return_dict=True,
            ).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if past_key_values_length == 0 or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = (past_key_values_length + self.rope_deltas).to(input_ids.device)
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = compute_lm_head_logits(self.lm_head, hidden_states, logits_to_keep).logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
        )


ModelClass = Qwen2_5OmniThinkerForConditionalGeneration
