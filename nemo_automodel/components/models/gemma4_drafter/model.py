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

"""Thin NeMo wrapper around HuggingFace ``Gemma4AssistantForCausalLM``.

The HF implementation in ``transformers.models.gemma4_assistant`` is used as-is;
this wrapper only adds :class:`HFCheckpointingMixin` so the drafter participates
in NeMo's distributed checkpointing pipeline and gives us a stable native class
name for the model registry.
"""

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.shared.import_utils import UnavailableError, UnavailableMeta


def _make_missing(name: str):
    return UnavailableMeta(name, (), {"_msg": "transformers.models.gemma4_assistant is not available."})


try:
    from transformers.models.gemma4_assistant.configuration_gemma4_assistant import Gemma4AssistantConfig
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM as HFGemma4AssistantForCausalLM,
    )

    _GEMMA4_ASSISTANT_HF_AVAILABLE = True
except (ModuleNotFoundError, ImportError, AttributeError):
    _GEMMA4_ASSISTANT_HF_AVAILABLE = False
    Gemma4AssistantConfig = _make_missing("Gemma4AssistantConfig")
    HFGemma4AssistantForCausalLM = _make_missing("Gemma4AssistantForCausalLM")


if _GEMMA4_ASSISTANT_HF_AVAILABLE:

    class Gemma4DrafterForCausalLM(HFCheckpointingMixin, HFGemma4AssistantForCausalLM):
        """NeMo subclass of HuggingFace ``Gemma4AssistantForCausalLM``.

        Inherits the HF forward unchanged. The subclass exists so that:
            * the drafter participates in NeMo distributed checkpointing via
              :class:`HFCheckpointingMixin`;
            * the architecture can be registered in NeMo's ``MODEL_ARCH_MAPPING``
              under a stable native class name.
        """

        @classmethod
        def from_config(cls, config: Gemma4AssistantConfig, **kwargs):
            return cls(config, **kwargs)

    ModelClass = Gemma4DrafterForCausalLM
else:

    class Gemma4DrafterForCausalLM:
        """Placeholder raised when ``transformers.models.gemma4_assistant`` is unavailable."""

        def __init__(self, *args, **kwargs):
            raise UnavailableError(
                "transformers.models.gemma4_assistant is not available. "
                "Install transformers>=5.8.0.dev (e.g. from the cloned "
                "transformers TOT) to use the Gemma4 drafter."
            )

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise UnavailableError(
                "transformers.models.gemma4_assistant is not available. "
                "Install transformers>=5.8.0.dev (e.g. from the cloned "
                "transformers TOT) to use the Gemma4 drafter."
            )


__all__ = ["Gemma4DrafterForCausalLM"]
