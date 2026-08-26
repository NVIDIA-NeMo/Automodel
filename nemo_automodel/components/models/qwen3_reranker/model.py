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

"""
Qwen3 reranker model for cross-encoder reranking tasks.

Unlike the bidirectional encoders in this package (e.g. ``llama_bidirectional``),
the Qwen3 reranker keeps the standard **causal** attention of ``Qwen3ForCausalLM``
and adds **no new classification head**. Instead, it reuses the language-model head
and turns reranking into a binary "yes"/"no" next-token prediction, exactly as the
official ``Qwen/Qwen3-Reranker-*`` models are trained and used.

Score convention
~~~~~~~~~~~~~~~~
``forward`` returns a single **raw logit** per query-document pair at the final
(non-padding) token position::

    score(query, doc) = logit("yes") - logit("no")          # unbounded log-odds

This is the "raw logit difference" described in the official model card. Two
downstream consumers use it:

* **Inference** — apply a sigmoid to recover the exact probability produced by
  the official ``compute_logits`` reference implementation::

      p(yes) = sigmoid(score) = softmax([logit("no"), logit("yes")])[yes]

  (The serialized checkpoint is a plain ``Qwen3ForCausalLM``; running the
  official ``compute_logits`` on it reproduces this same ``p(yes)``.)

* **Training** — ``TrainCrossEncoderRecipe`` reshapes the raw scores with
  ``logits.view(-1, n_passages)``, divides by the recipe-level ``temperature``, and
  applies ``F.cross_entropy(..., labels=0)``: a single softmax over each query's
  candidate passages (a listwise contrastive loss). Temperature is a property of the
  training objective, not of the model, so it lives in the recipe config and never
  touches ``forward``; inference scores keep their full discriminative range.

The model is auto-discovered by ``ModelRegistry`` via the ``ModelClass`` export.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3RerankerConfig(Qwen3Config):
    """Configuration for :class:`Qwen3RerankerForCausalReranking`.

    Extends :class:`Qwen3Config` with the token ids used for the binary
    "yes"/"no" relevance decision.

    Checkpoints are serialized as plain ``Qwen3ForCausalLM`` (``model_type:
    "qwen3"``) so they load in vLLM and stock HF Transformers without custom
    code. ``yes_token_id`` / ``no_token_id`` are preserved as extra JSON fields;
    ``PretrainedConfig.__init__`` stores unknown keys as instance attributes so
    they survive the config round-trip.
    """

    def __init__(
        self,
        yes_token_id: Optional[int] = None,
        no_token_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        super().__init__(**kwargs)

    def to_dict(self) -> dict:
        """Serialize as a plain ``Qwen3ForCausalLM`` config.

        Routing does not depend on this: the class is resolved through
        ``ModelRegistry`` by architecture name. Rewriting the serialized identity
        is what lets saved checkpoints load in vLLM and plain HF Transformers
        without ``trust_remote_code``, as standard ``Qwen3ForCausalLM`` weights.
        """
        output = super().to_dict()
        # Rewrite identity to standard Qwen3ForCausalLM.
        output["model_type"] = "qwen3"
        output["architectures"] = ["Qwen3ForCausalLM"]
        output.pop("auto_map", None)
        # Convert rope_parameters dict → flat rope_theta + rope_scaling, which
        # is the format vLLM's Qwen3 backend and the stock HF config use.
        rope_params = output.pop("rope_parameters", None)
        if rope_params and isinstance(rope_params, dict):
            output.setdefault("rope_theta", rope_params.get("rope_theta", 1_000_000))
            # Carry the scaling across rather than dropping it with the dict it lived in.
            # rope_parameters nests the scaling config; popping the dict and then defaulting
            # rope_scaling to None silently discards a non-default RoPE scale, so the saved
            # checkpoint reloads in HF or vLLM with different position encoding than it was
            # trained with. Everything except rope_theta and the type discriminator is the
            # scaling payload.
            scaling = {k: v for k, v in rope_params.items()
                       if k not in ("rope_theta", "rope_type", "type")}
            rope_type = rope_params.get("rope_type") or rope_params.get("type")
            if scaling:
                if rope_type:
                    scaling["rope_type"] = rope_type
                output.setdefault("rope_scaling", scaling)
        output.setdefault("rope_scaling", None)
        # vLLM manages its own KV cache; ensure use_cache is True.
        output["use_cache"] = True
        return output


def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the index of the last non-padding token for each row.

    Padding-side agnostic (mirrors the ``"last"`` branch of
    :func:`nemo_automodel._transformers.retrieval.pool`). With left padding the
    last token is at ``-1`` for every row; with right padding it is at
    ``attention_mask.sum(dim=1) - 1``.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return torch.full(
            (attention_mask.shape[0],),
            attention_mask.shape[1] - 1,
            device=attention_mask.device,
            dtype=torch.long,
        )
    return attention_mask.sum(dim=1) - 1


class Qwen3RerankerForCausalReranking(Qwen3ForCausalLM):
    """Qwen3 causal LM repurposed as a pointwise reranker.

    Scores each query-document pair with the raw logit difference
    ``logit("yes") - logit("no")`` at the final non-padding position. Apply a
    sigmoid to recover ``p(yes)`` (the official ``compute_logits`` output);
    ``TrainCrossEncoderRecipe`` consumes the raw score directly. Attention remains
    causal and the pretrained ``lm_head`` is reused (no new parameters).
    """

    config_class = Qwen3RerankerConfig

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load weights and resolve the "yes"/"no" token ids if not already set.

        Explicit ``yes_token_id``/``no_token_id`` (e.g. from the recipe YAML or a
        saved config) take precedence; otherwise they are resolved from the
        tokenizer of ``pretrained_model_name_or_path``.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # Use getattr so this works whether model.config is Qwen3RerankerConfig
        # (fresh base model) or a plain Qwen3Config with yes/no ids stored as
        # extra attributes (checkpoint saved with the vLLM-compatible format).
        if getattr(model.config, "yes_token_id", None) is None or getattr(model.config, "no_token_id", None) is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            if getattr(model.config, "yes_token_id", None) is None:
                model.config.yes_token_id = tokenizer.convert_tokens_to_ids("yes")
            if getattr(model.config, "no_token_id", None) is None:
                model.config.no_token_id = tokenizer.convert_tokens_to_ids("no")
            logger.info(
                f"Resolved reranker tokens: yes_token_id={model.config.yes_token_id}, "
                f"no_token_id={model.config.no_token_id}"
            )
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, SequenceClassifierOutputWithPast]:
        if attention_mask is None:
            raise ValueError("attention_mask is required to locate the final token for yes/no scoring")
        if getattr(self.config, "yes_token_id", None) is None or getattr(self.config, "no_token_id", None) is None:
            raise ValueError(
                "yes_token_id/no_token_id are unset on the config. Build the model via "
                "from_pretrained (which resolves them) or set them explicitly."
            )

        # kwargs may carry logits_to_keep / use_cache etc.; drop the ones that
        # do not apply to the bare decoder forward.
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("num_logits_to_keep", None)

        # Resolve return_dict ONCE, before it is used. The signature defaults it to None, so a
        # caller that omits it would otherwise fall through `if not return_dict` further down
        # and get a bare tuple -- silently ignoring config.use_return_dict and denying callers
        # the SequenceClassifierOutputWithPast they expect. Same normalisation the qwen2/qwen3
        # models in this repo use.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # The decoder call below pins return_dict=True regardless: last_hidden_state is only
        # reachable on a ModelOutput. That is independent of what WE return, resolved above.
        # return_dict=True is required, not cosmetic: last_hidden_state is only reachable on a
        # ModelOutput, and this call does not forward the wrapper's return_dict. With
        # config.use_return_dict False the decoder hands back a plain tuple and the attribute
        # access below raises. Pin it here so the wrapper's own return_dict setting -- which
        # controls the shape WE return -- cannot change how we read the decoder. Popped from
        # kwargs first so an explicit caller value cannot override it.
        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # Gather the final non-padding hidden state per row, then project only
        # those positions through the LM head (avoids a full-sequence vocab matmul).
        batch_size = hidden_states.shape[0]
        last_idx = _last_token_indices(attention_mask)
        last_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), last_idx]

        last_logits = self.lm_head(last_hidden)  # [B, vocab]
        yes_no_logits = torch.stack(
            [last_logits[:, self.config.no_token_id], last_logits[:, self.config.yes_token_id]],
            dim=-1,
        )  # [B, 2] ordered as (no, yes)
        # Raw logit difference (log-odds). The HF docs describe this as "raw logit
        # differences"; apply sigmoid to get p(yes) = softmax([no, yes])[1], which
        # matches the official compute_logits inference function.
        # Temperature is applied by the training recipe before cross_entropy — here
        # we return raw logits so the recipe doesn't double-softmax.
        score = (yes_no_logits[:, 1] - yes_no_logits[:, 0]).to(last_logits.dtype)

        pooled_logits = score.unsqueeze(-1)  # [B, 1]

        loss = None
        if labels is not None:
            # Standalone pointwise binary yes/no objective. The recipe pops labels
            # and computes its own listwise loss, so this branch is for direct callers.
            labels = labels.to(yes_no_logits.device).view(-1)
            loss = F.cross_entropy(yes_no_logits.float(), labels)

        if not return_dict:
            output = (pooled_logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


# Export for ModelRegistry auto-discovery
ModelClass = [Qwen3RerankerForCausalReranking]


__all__ = [
    "Qwen3RerankerForCausalReranking",
    "Qwen3RerankerConfig",
    "ModelClass",
]
