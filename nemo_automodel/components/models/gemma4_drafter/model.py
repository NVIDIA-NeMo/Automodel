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

The HF implementation in ``transformers.models.gemma4_assistant`` is used as-is,
except that the ordered-embedding output head scores the full vocabulary (see
:class:`Gemma4DrafterFullVocabEmbedder`). The wrapper adds
:class:`HFCheckpointingMixin` so the drafter participates in NeMo's distributed
checkpointing pipeline and gives us a stable native class name for the model
registry.
"""

from dataclasses import dataclass

import torch

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.shared.import_utils import UnavailableError, UnavailableMeta


def _make_missing(name: str):
    return UnavailableMeta(name, (), {"_msg": "transformers.models.gemma4_assistant is not available."})


try:
    from transformers.models.gemma4_assistant.configuration_gemma4_assistant import Gemma4AssistantConfig
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM as HFGemma4AssistantForCausalLM,
    )
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantMaskedEmbedder as HFGemma4AssistantMaskedEmbedder,
    )

    _GEMMA4_ASSISTANT_HF_AVAILABLE = True
except (ModuleNotFoundError, ImportError, AttributeError):
    _GEMMA4_ASSISTANT_HF_AVAILABLE = False
    Gemma4AssistantConfig = _make_missing("Gemma4AssistantConfig")
    HFGemma4AssistantForCausalLM = _make_missing("Gemma4AssistantForCausalLM")


if _GEMMA4_ASSISTANT_HF_AVAILABLE:

    class Gemma4DrafterFullVocabEmbedder(HFGemma4AssistantMaskedEmbedder):
        """The ordered-embedding head, scoring the full vocabulary (training and validation).

        The ordered head scores only the tokens of the top-k centroid clusters and gives every other
        token a constant logit without gradient, so training through it diverges (issue #4022). This
        keeps ``centroids`` and ``token_ordering`` for checkpoints and export (inference still uses the
        ordered head) and returns full-vocabulary logits.
        """

        def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
            """Logits [batch, sequence, vocab] from hidden_states [batch, sequence, hidden] and lm_head_weight [vocab, hidden]."""
            return torch.nn.functional.linear(hidden_states, lm_head_weight)

    class Gemma4DrafterForCausalLM(HFCheckpointingMixin, HFGemma4AssistantForCausalLM):
        """NeMo subclass of HuggingFace ``Gemma4AssistantForCausalLM``.

        Inherits the HF forward. The subclass exists so that:
            * the drafter participates in NeMo distributed checkpointing via
              :class:`HFCheckpointingMixin`;
            * the architecture can be registered in NeMo's ``MODEL_ARCH_MAPPING``
              under a stable native class name;
            * a drafter with ``use_ordered_embeddings`` trains on full-vocabulary
              logits (:class:`Gemma4DrafterFullVocabEmbedder`).
        """

        # Only tied Gemma4 assistant checkpoints ship; untying is unsupported.
        tie_word_embeddings_support: TieSupport = TieSupport.TIED_ONLY
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

        def __init__(self, config: Gemma4AssistantConfig, *args, **kwargs):
            reject_unsupported_tie_word_embeddings(type(self), config)
            super().__init__(config, *args, **kwargs)
            if self.masked_embedding is not None:
                # Same parameters and buffers, already initialized by post_init; only the forward differs.
                self.masked_embedding.__class__ = Gemma4DrafterFullVocabEmbedder
            self.tie_weights()

        def tie_weights(self, *_args: object, **_kwargs: object) -> None:
            """Tie ``lm_head`` to the drafter token embedding."""
            self.lm_head.weight = self.model.embed_tokens.weight

        @dataclass(frozen=True)
        class ModelCapabilities:
            """Declared parallelism capabilities for this model class."""

            supports_tp: bool = False
            supports_cp: bool = False
            supports_pp: bool = False
            supports_ep: bool = False

        @classmethod
        def from_config(cls, config: Gemma4AssistantConfig, **kwargs):
            return cls(config, **kwargs)

    ModelClass = Gemma4DrafterForCausalLM
else:

    class Gemma4DrafterForCausalLM:
        """Placeholder raised when ``transformers.models.gemma4_assistant`` is unavailable."""

        @dataclass(frozen=True)
        class ModelCapabilities:
            """Declared parallelism capabilities for this model class."""

            supports_tp: bool = False
            supports_cp: bool = False
            supports_pp: bool = False
            supports_ep: bool = False

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
