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
Nemotron Bidirectional Embedding Model

Uses Nemotron models to calculate text embeddings for retrieval tasks.
"""

import copy
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
    SequenceClassifierOutputWithPast,
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


@dataclass
class BiencoderOutput(ModelOutput):
    """Output for biencoder models."""
    loss: Optional[Tensor] = None
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class NemotronConfig(PretrainedConfig):
    """Base configuration class for Nemotron models."""
    
    model_type = "nemotron"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        super().__init__(**kwargs)


class NemotronBidirectionalConfig(NemotronConfig):
    """
    Configuration class for NemotronBidirectionalModel.
    
    Extends NemotronConfig with additional parameters for bidirectional attention
    and pooling configurations.
    """
    
    model_type = "nemotron_bidirectional"
    
    def __init__(
        self,
        pooling: str = "eos",
        temperature: float = 1.0,
        use_cache: bool = False,
        **kwargs,
    ):
        self.pooling = pooling
        self.temperature = temperature
        self.use_cache = use_cache
        super().__init__(**kwargs)

        logger.info(f"NemotronBidirectionalConfig initialized with pooling: {pooling} and temperature: {temperature}")
        logger.info(f"NemotronBidirectionalConfig initialized with kwargs: {kwargs}")


class NemotronBidirectionalModel(PreTrainedModel):
    """
    Nemotron Bidirectional Model.
    
    This model is a bidirectional Nemotron model for embedding tasks.
    """
    config_class = NemotronBidirectionalConfig
    base_model_prefix = "model"
    
    def __init__(self, config: NemotronBidirectionalConfig):
        super().__init__(config)
        self.config = config
        self.model = None
        self.tokenizer = None

    def _load_model(self, model_path: str):
        """Load the model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            output_hidden_states=True,
        )
     
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        use_cache: bool = False,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = True,
        **kwargs
    ):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        last_hidden_state = outputs.hidden_states[-1]

        past_key_values = None
        if self.config.use_cache and hasattr(outputs, 'past_key_values'):
            past_key_values = outputs.past_key_values

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
        )



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
        pooling: str = "eos",
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
            "AutoModel": "nemotron_bidirectional_embedding.NemotronBidirectionalModel",
            "AutoConfig": "nemotron_bidirectional_embedding.NemotronBidirectionalConfig",
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
        embeds = self._pool_embeddings(
            last_hidden_state=hidden_state,
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["attention_mask"],
        )

        # L2 normalize if required
        if self.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)

        return embeds.contiguous()


    def _get_eos_positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Find the position of the EOS token for each sequence in the batch.
        If no EOS token, use the last non-padded position.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            eos_positions: Position of EOS/last token for each sequence [batch_size]
        """
        batch_size = input_ids.shape[0]
        eos_token_id = self.tokenizer.eos_token_id
        
        eos_positions = []
        for i in range(batch_size):
            # Find EOS token position
            eos_mask = (input_ids[i] == eos_token_id)
            if eos_mask.any():
                # Use the first EOS token position
                eos_pos = eos_mask.nonzero(as_tuple=True)[0][0].item()
            else:
                # Use last non-padded position
                eos_pos = attention_mask[i].sum().item() - 1
            eos_positions.append(eos_pos)
        
        return torch.tensor(eos_positions, device=input_ids.device)


    def _pool_embeddings(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token-level hidden states to get sequence embeddings.
        
        Args:
            last_hidden_state: Hidden states from last layer [batch_size, seq_len, hidden_dim]
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            embeddings: Pooled sequence embeddings [batch_size, hidden_dim]
        """
        if self.pooling == "eos":
            # Use hidden state at EOS token position
            eos_positions = self._get_eos_positions(input_ids, attention_mask)
            batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            embeddings = last_hidden_state[batch_indices, eos_positions, :]
            
        elif self.pooling == "last":
            # Use hidden state at last non-padded position
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            embeddings = last_hidden_state[batch_indices, seq_lengths, :]
            
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return embeddings


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
        out_dimension_pooler: int = 768,
        do_gradient_checkpointing: bool = False,
        train_n_passages: int = 1,
        eval_negative_size: int = 0,
        pooling: str = "eos",
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
            out_dimension_pooler: Output dimension for linear pooler
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
            linear_pooler = nn.Linear(lm_q.config.hidden_size, out_dimension_pooler)

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

        
