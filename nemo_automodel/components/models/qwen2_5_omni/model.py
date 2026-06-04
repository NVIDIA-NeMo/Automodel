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
from typing import Any

import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration as HFQwen2_5OmniThinkerForConditionalGeneration,
)

from nemo_automodel.components.models.common import BackendConfig
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
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ):
        """Delegate to HF Thinker forward, passing all multimodal inputs through.

        The forward signature mirrors HF's
        ``Qwen2_5OmniThinkerForConditionalGeneration.forward``; we override
        only to make the call site uniform with the rest of NeMo AutoModel.
        Audio is mandatory for ASR; image / video paths are kept enabled so
        the same class supports the full Thinker modality set.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )
        if labels is not None:
            return {"logits": outputs.logits, "loss": outputs.loss}
        return outputs.logits if hasattr(outputs, "logits") else outputs


ModelClass = Qwen2_5OmniThinkerForConditionalGeneration
