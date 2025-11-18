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

"""
Llama Bidirectional Model for NeMo AutoModel.

This module provides a bidirectional attention variant of Llama that is useful
for embedding and retrieval tasks. Unlike the standard causal Llama model,
this version can attend to all tokens bidirectionally.
"""

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaModel,
)
from transformers.utils import auto_docstring, can_return_tuple, logging

logger = logging.get_logger(__name__)


def contrastive_scores_and_labels(
    query: torch.Tensor, key: torch.Tensor, current_train_n_passages: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute contrastive scores and labels without in-batch negatives.

    Args:
        query: Query embeddings [batch_size, hidden_dim]
        key: Key/passage embeddings [batch_size * n_passages, hidden_dim]
        current_train_n_passages: Number of passages per query

    Returns:
        Tuple of (scores, labels) where scores is [batch_size, n_passages]
        and labels is [batch_size] of zeros (positive is first passage)
    """
    assert key.shape[0] % query.shape[0] == 0, "{} % {} > 0".format(key.shape[0], query.shape[0])
    query_shape = query.shape
    repeated_query = query.repeat(1, 1, current_train_n_passages).reshape(
        query_shape[0] * current_train_n_passages, query_shape[1]
    )
    qk = torch.sum(repeated_query * key, dim=-1).reshape(query_shape[0], current_train_n_passages)
    labels = torch.zeros(query_shape[0], dtype=torch.long, device=query.device)
    return qk, labels


def pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str) -> torch.Tensor:
    """
    Pool hidden states using the specified pooling method.

    Args:
        last_hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len]
        pool_type: Type of pooling to apply

    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weighted_avg":
        emb = last_hidden.sum(dim=1)
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    elif pool_type == "cls_last":
        emb = last_hidden[:, 0]
    elif pool_type == "colbert":
        emb = last_hidden
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


class LlamaBidirectionalConfig(LlamaConfig):
    """
    Configuration class for LlamaBidirectionalModel.

    Extends LlamaConfig with additional parameters for bidirectional attention
    and pooling configurations.
    """

    model_type = "llama_bidirec"

    def __init__(
        self,
        pooling: str = "avg",
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Initialize LlamaBidirectionalConfig.

        Args:
            pooling: Pooling strategy ('avg', 'cls', 'last', etc.)
            temperature: Temperature for scaling logits
            **kwargs: Additional arguments passed to LlamaConfig
        """
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(**kwargs)


class LlamaBidirectionalModel(LlamaModel):
    """
    Llama Model with bidirectional attention.

    This model removes causal masking from all attention layers, allowing tokens
    to attend to all other tokens in the sequence. This is useful for embedding
    and retrieval tasks where bidirectional context is beneficial.
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaConfig):
        """
        Initialize LlamaBidirectionalModel.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        # Disable causal attention for all layers
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
    ):
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask=attention_mask)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaBidirectionalForSequenceClassification(LlamaForSequenceClassification):
    """
    Llama Bidirectional Model with a sequence classification/regression head.

    This model adds a classification head on top of the bidirectional Llama model
    and includes configurable pooling strategies.
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config):
        """
        Initialize LlamaBidirectionalForSequenceClassification.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        # Release the parameters of LlamaModel
        # created by parent LlamaForSequenceClassification
        del self.model

        self.model = LlamaBidirectionalModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        Forward pass for sequence classification.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for generation
            inputs_embeds: Input embeddings (alternative to input_ids)
            labels: Labels for computing loss
            use_cache: Whether to use cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict

        Returns:
            SequenceClassifierOutputWithPast with loss, logits, and optional outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Pool hidden states using configured pooling strategy
        pooled_hidden_states = pool(
            last_hidden_states=hidden_states,
            attention_mask=attention_mask,
            pool_type=self.config.pooling,
        )

        # Apply classification head and temperature scaling
        pooled_logits = self.score(pooled_hidden_states)
        pooled_logits = pooled_logits / self.config.temperature

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@dataclass
class BiencoderOutput(ModelOutput):
    """Output dataclass for biencoder model."""

    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiencoderModel(nn.Module):
    """
    Biencoder Model with essential functions for training.

    This model encodes queries and passages separately and computes contrastive loss.
    """

    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        linear_pooler: nn.Module = None,
        train_n_passages: int = 1,
        eval_negative_size: int = 0,
        pooling: str = "avg",
        l2_normalize: bool = True,
        t: float = 1.0,
        share_encoder: bool = True,
        add_linear_pooler: bool = False,
    ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.train_n_passages = train_n_passages
        self.eval_negative_size = eval_negative_size
        self.pooling = pooling
        self.l2_normalize = l2_normalize
        self.t = t
        self.share_encoder = share_encoder
        self.add_linear_pooler = add_linear_pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.linear_pooler = linear_pooler if linear_pooler is not None else nn.Identity()
        self.config = self.lm_q.config
        self.trainer = None

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        """Forward pass for training."""

        # Get current number of passages per query
        if self.training:
            current_train_n_passages = self.train_n_passages
        else:
            current_train_n_passages = self.eval_negative_size + 1

        # Compute scores (encoding happens inside _compute_scores)
        scores, labels, q_reps, p_reps = self._compute_scores(
            query=query,
            passage=passage,
            current_train_n_passages=current_train_n_passages,
        )
        loss = self.cross_entropy(scores, labels)

        # Adding Dummy Gradients for vlm-based models
        if hasattr(self.lm_q, "module") and hasattr(self.lm_q.module, "post_loss"):
            loss = self.lm_q.module.post_loss(loss, passage)
        elif hasattr(self.lm_q, "post_loss"):
            # Not tested this branch
            loss = self.lm_q.post_loss(loss, passage)

        return BiencoderOutput(
            loss=loss,
            q_reps=q_reps,
            p_reps=p_reps,
            labels=labels.contiguous(),
            scores=scores,
        )

    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        """Encode input using the encoder."""
        if not input_dict:
            return None

        import inspect

        # Remove token_type_ids if encoder doesn't support it
        if (
            "token_type_ids" not in inspect.getfullargspec(encoder.forward).args
            and "token_type_ids" in input_dict.keys()
        ):
            input_dict = {k: v for k, v in input_dict.items() if k != "token_type_ids"}

        # Get encoder outputs
        outputs = encoder(
            **{k: v for k, v in input_dict.items() if k not in ["kd_labels"]},
            return_dict=True,
            output_hidden_states=True,
        )

        # Extract hidden states
        if hasattr(outputs, "last_hidden_state"):
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[-1]

        # Pool the representations
        embeds = pool(
            last_hidden_states=hidden_state,
            attention_mask=input_dict["attention_mask"],
            pool_type=self.pooling,
        )

        # Apply linear pooler
        embeds = self.linear_pooler(embeds)

        # L2 normalize if required
        if self.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)

        return embeds.contiguous()

    def _compute_scores(
        self,
        current_train_n_passages: int,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute similarity scores and labels."""

        # Encode query and passage
        q_reps = self._encode(self.lm_q, query)
        p_reps = self._encode(self.lm_p, passage)

        # Compute similarity scores using contrastive_scores_and_labels
        scores, labels = contrastive_scores_and_labels(
            query=q_reps,
            key=p_reps,
            current_train_n_passages=current_train_n_passages,
        )

        if self.l2_normalize:
            scores = scores / self.t

        return scores, labels, q_reps, p_reps

    @classmethod
    def build(
        cls,
        model_name_or_path: str,
        share_encoder: bool = True,
        add_linear_pooler: bool = False,
        out_dimension: int = 768,
        do_gradient_checkpointing: bool = False,
        train_n_passages: int = 1,
        eval_negative_size: int = 0,
        pooling: str = "avg",
        l2_normalize: bool = True,
        t: float = 1.0,
        **hf_kwargs,
    ):
        """
        Build biencoder model from pretrained.

        Args:
            model_name_or_path: Path to pretrained model or model identifier
            share_encoder: Whether to share encoder weights between query and passage
            add_linear_pooler: Whether to add a linear pooler layer
            out_dimension: Output dimension for linear pooler
            do_gradient_checkpointing: Whether to enable gradient checkpointing
            train_n_passages: Number of passages per query during training
            eval_negative_size: Number of negative samples during evaluation
            pooling: Pooling strategy ('avg', 'cls', 'last', etc.)
            l2_normalize: Whether to L2 normalize embeddings
            t: Temperature for scaling similarity scores
            **hf_kwargs: Additional arguments passed to model loading
        """

        logger.info(f"Building BiencoderModel from {model_name_or_path}")

        # Infer model class from model_name_or_path
        # Check config.json if it exists
        config_path = os.path.join(model_name_or_path, "config.json") if os.path.isdir(model_name_or_path) else None

        if config_path and os.path.exists(config_path):
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
                model_type = config.get("model_type", "")
        else:
            # If no config, infer from model name
            model_type = ""

        # Select model class based on model type
        if model_type == "llama" or "llama" in model_name_or_path.lower():
            ModelClass = LlamaBidirectionalModel
            logger.info("Using LlamaBidirectionalModel")
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. Cannot infer model class from {model_name_or_path}"
            )

        # Load model locally or from hub using selected model class
        if os.path.isdir(model_name_or_path):
            if share_encoder:
                lm_q = ModelClass.from_pretrained(model_name_or_path, trust_remote_code=True, **hf_kwargs)
                lm_p = lm_q
            else:
                _qry_model_path = os.path.join(model_name_or_path, "query_model")
                _psg_model_path = os.path.join(model_name_or_path, "passage_model")

                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_name_or_path
                    _psg_model_path = model_name_or_path

                lm_q = ModelClass.from_pretrained(_qry_model_path, trust_remote_code=True, **hf_kwargs)
                lm_p = ModelClass.from_pretrained(_psg_model_path, trust_remote_code=True, **hf_kwargs)
        else:
            # Load from hub
            lm_q = ModelClass.from_pretrained(model_name_or_path, **hf_kwargs)

            if share_encoder:
                lm_p = lm_q
            else:
                lm_p = copy.deepcopy(lm_q)

        # Enable gradient checkpointing if requested
        if do_gradient_checkpointing:
            lm_q.gradient_checkpointing_enable()
            if lm_p is not lm_q:
                lm_p.gradient_checkpointing_enable()

        # Create linear pooler if needed
        if add_linear_pooler:
            linear_pooler = nn.Linear(lm_q.config.hidden_size, out_dimension)

            pooler_path = os.path.join(model_name_or_path, "pooler.pt")
            if os.path.exists(pooler_path):
                logger.info("Loading pooler weights from local files")
                state_dict = torch.load(pooler_path, map_location="cpu")
                linear_pooler.load_state_dict(state_dict)
        else:
            linear_pooler = nn.Identity()

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            linear_pooler=linear_pooler,
            train_n_passages=train_n_passages,
            eval_negative_size=eval_negative_size,
            pooling=pooling,
            l2_normalize=l2_normalize,
            t=t,
            share_encoder=share_encoder,
            add_linear_pooler=add_linear_pooler,
        )
        return model

    def save(self, output_dir: str):
        """Save model to output directory."""

        logger.info(f"Saving BiencoderModel to {output_dir}")

        # Save the model
        if self.share_encoder:
            self.lm_q.save_pretrained(output_dir)
        else:
            os.makedirs(os.path.join(output_dir, "query_model"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "passage_model"), exist_ok=True)
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))

        # Save linear pooler if exists
        if self.add_linear_pooler:
            pooler_path = os.path.join(output_dir, "pooler.pt")
            logger.info(f"Saving linear pooler to {pooler_path}")
            torch.save(self.linear_pooler.state_dict(), pooler_path)
