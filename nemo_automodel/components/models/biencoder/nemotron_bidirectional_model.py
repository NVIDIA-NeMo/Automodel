# coding=utf-8
# Copyright 2024 HuggingFace Inc. team.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
Nemotron Bidirectional Model for NeMo AutoModel.

This module provides a bidirectional attention variant of Nemotron that is useful
for embedding and retrieval tasks. Unlike the standard causal Nemotron model,
this version can attend to all tokens bidirectionally.
"""

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pdb

from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from nemo_automodel.components.models.nemotron.configuration_nemotron_h import NemotronHConfig
from nemo_automodel.components.models.nemotron.modeling_nemotron_h import (
    HybridMambaAttentionDynamicCache,
    NemotronHModel,
    NemotronHOutput,
)

try:
    from nemo_automodel.components.models.biencoder.state_dict_adapter import BiencoderStateDictAdapter
except ImportError:
    BiencoderStateDictAdapter = object

from nemo_automodel.shared.import_utils import get_check_model_inputs_decorator

logger = logging.get_logger(__name__)
check_model_inputs = get_check_model_inputs_decorator()




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





class NemotronBidirectionalConfig(NemotronHConfig):
    """
    Configuration class for NemotronBidirectionalModel.
    
    Extends NemotronHConfig with additional parameters for bidirectional attention
    and pooling configurations.
    """
    
    model_type = "nemotron_bidirectional"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling = kwargs.get("pooling", "last")
        self.temperature = kwargs.get("temperature", 1.0)
        self.use_cache = kwargs.get("use_cache", False)
        logger.info(f"NemotronBidirectionalConfig initialized with pooling: {self.pooling} and temperature: {self.temperature}")
        logger.info(f"NemotronBidirectionalConfig initialized with kwargs: {kwargs}")



class NemotronBidirectionalModel(NemotronHModel):
    """
    Nemotron Bidirectional Model.
    
    This model is a bidirectional Nemotron model for embedding tasks.
    """
    config_class = NemotronBidirectionalConfig
    main_input_name = "input_ids"

    def __init__(self, config: NemotronBidirectionalConfig):
        super().__init__(config)
        self.config = config
        self.model = None
        self.tokenizer = None


    # @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # From zamba_modeling.py
        if use_cache and cache_params is None:
            logger.warning_once(
                "NemotronH requires an initialized `NemotronHHybridDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )

        hidden_states = inputs_embeds
        
        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # Until HERE
        
        for mixer_block in self.layers:
            # Depending on the layer type we opt for 2D base attention mask (Mamba) or 4D causal mask (Attention)
            if mixer_block.block_type == "mamba":
                layer_mask = mamba_mask
            elif mixer_block.block_type == "attention":
                layer_mask = causal_mask
            elif mixer_block.block_type == "mlp":
                layer_mask = None
            else:
                raise ValueError(f"Invalid block_type: {self.block_type}")

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, layer_mask
                )
            else:
            
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )

  
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return NemotronHOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
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
        pooling: str = "last",
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

        # For HuggingFace consolidated checkpoint compatibility
        self.name_or_path = os.path.abspath(__file__)
        self.state_dict_adapter = BiencoderStateDictAdapter()
        self.config.architectures = ["NemotronBidirectionalModel"]
        self.config.auto_map = {
            "AutoModel": "nemotron_bidirectional_model.NemotronBidirectionalModel",
            "AutoConfig": "nemotron_bidirectional_model.NemotronBidirectionalConfig",
        }

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
      
        outputs = encoder(input_ids=input_dict["input_ids"],
                          attention_mask=input_dict["attention_mask"],
                          output_hidden_states=True,
                          return_dict=True,)

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
        if model_type == "nemotron" or "nemotron" in model_name_or_path.lower():
            ModelClass = NemotronBidirectionalModel
            logger.info("Using NemotronBidirectionalModel")
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
