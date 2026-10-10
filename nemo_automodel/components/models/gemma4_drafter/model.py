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
from nemo_automodel.shared.import_utils import UnavailableError, safe_import_from

_GEMMA4_ASSISTANT_MSG = "transformers.models.gemma4_assistant is not available."
_GEMMA4_ASSISTANT_MODELING = "transformers.models.gemma4_assistant.modeling_gemma4_assistant"

_HAS_CONFIG, Gemma4AssistantConfig = safe_import_from(
    "transformers.models.gemma4_assistant.configuration_gemma4_assistant",
    "Gemma4AssistantConfig",
    msg=_GEMMA4_ASSISTANT_MSG,
)
_HAS_CAUSAL_LM, HFGemma4AssistantForCausalLM = safe_import_from(
    _GEMMA4_ASSISTANT_MODELING, "Gemma4AssistantForCausalLM", msg=_GEMMA4_ASSISTANT_MSG
)
_HAS_MASKED_EMBEDDER, HFGemma4AssistantMaskedEmbedder = safe_import_from(
    _GEMMA4_ASSISTANT_MODELING, "Gemma4AssistantMaskedEmbedder", msg=_GEMMA4_ASSISTANT_MSG
)
_GEMMA4_ASSISTANT_HF_AVAILABLE = _HAS_CONFIG and _HAS_CAUSAL_LM and _HAS_MASKED_EMBEDDER


if _GEMMA4_ASSISTANT_HF_AVAILABLE:

    class Gemma4DrafterFullVocabEmbedder(HFGemma4AssistantMaskedEmbedder):
        """Ordered-embedding head that scores the full vocabulary.

        The HF ordered head scores only the tokens of the top-k centroid clusters and gives every
        other token a constant logit without gradient, so training through it diverges (issue
        #4022). This head keeps the ``centroids`` and ``token_ordering`` state of the HF head, so
        checkpoints and exports are unchanged, but always returns full-vocabulary logits, in both
        training and eval mode. Only an exported checkpoint reloaded by HF or vLLM uses the ordered
        head.
        """

        def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
            """Score every vocabulary token.

            Args:
                hidden_states: Drafter hidden states of shape ``[batch, seq_len, hidden_size]``.
                lm_head_weight: Tied output embedding of shape ``[vocab_size, hidden_size]``.

            Returns:
                Logits of shape ``[batch, seq_len, vocab_size]``, ``hidden_states @ lm_head_weight.T``.
            """
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
                # Reuse the initialized HF centroids and token ordering so the state dict is unchanged.
                full_vocab_head = Gemma4DrafterFullVocabEmbedder(config)
                full_vocab_head.centroids = self.masked_embedding.centroids
                full_vocab_head.token_ordering = self.masked_embedding.token_ordering
                self.masked_embedding = full_vocab_head
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
