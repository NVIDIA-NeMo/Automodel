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

import copy
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
from .llama_bidirectional_model import pool, LlamaBidirectionalModel

logger = logging.getLogger(__name__)


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
        args,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        linear_pooler: nn.Module = None,
    ):
        super().__init__()
        self.args = args
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.linear_pooler = linear_pooler if linear_pooler is not None else nn.Identity()
        self.config = self.lm_q.config
        self.trainer = None
        
    def forward(
        self, 
        query: Dict[str, Tensor] = None, 
        passage: Dict[str, Tensor] = None
    ):
        """Forward pass for training."""
        
        # Get current number of passages per query
        if self.training:
            current_train_n_passages = self.args.train_n_passages
        else:
            current_train_n_passages = self.args.eval_negative_size + 1
            
        # Encode query and passage
        q_reps = self._encode(self.lm_q, query)
        p_reps = self._encode(self.lm_p, passage)
        
        # Compute scores and loss
        scores, labels = self._compute_scores(q_reps, p_reps, current_train_n_passages)
        loss = self.cross_entropy(scores, labels)
        
        return BiencoderOutput(
            loss=loss,
            q_reps=q_reps,
            p_reps=p_reps,
            labels=labels,
            scores=scores,
        )
    
    def _encode(
        self, 
        encoder: PreTrainedModel, 
        input_dict: dict
    ) -> Optional[torch.Tensor]:
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
            pool_type=self.args.pooling,
        )
        import pdb; pdb.set_trace()
        
        # Apply linear pooler
        embeds = self.linear_pooler(embeds)
        
        # L2 normalize if required
        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
            
        return embeds.contiguous()
    
    def _compute_scores(
        self,
        q_reps: torch.Tensor,
        p_reps: torch.Tensor,
        current_train_n_passages: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity scores and labels."""
        
        # Reshape passage representations
        # p_reps: (batch_size * n_passages, hidden_dim)
        # q_reps: (batch_size, hidden_dim)
        batch_size = q_reps.shape[0]
        
        # Compute similarity scores
        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        
        # Create labels (first passage for each query is positive)
        labels = torch.arange(
            start=0,
            end=batch_size * current_train_n_passages,
            step=current_train_n_passages,
            dtype=torch.long,
            device=scores.device
        )
        
        # Apply temperature scaling if l2_normalize is enabled
        if self.args.l2_normalize:
            scores = scores / self.args.t
            
        return scores, labels
    
    @classmethod
    def build(cls, args, **hf_kwargs):
        """Build biencoder model from pretrained."""
        
        logger.info(f"Building BiencoderModel from {args.model_name_or_path}")
        
        # Select model class based on base_model
        if args.base_model == "bidirc_llama3":
            ModelClass = LlamaBidirectionalModel
            logger.info("Using LlamaBidirectionalModel for bidirc_llama3")
        else:
            ModelClass = NeMoAutoModelForCausalLM
            logger.info(f"Using NeMoAutoModelForCausalLM for {args.base_model}")
        
        # Load model locally or from hub using selected model class
        if os.path.isdir(args.model_name_or_path):
            if args.share_encoder:
                lm_q = ModelClass.from_pretrained(
                    args.model_name_or_path, 
                    trust_remote_code=True, 
                    **hf_kwargs
                )
                lm_p = lm_q
            else:
                _qry_model_path = os.path.join(args.model_name_or_path, "query_model")
                _psg_model_path = os.path.join(args.model_name_or_path, "passage_model")
                
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = args.model_name_or_path
                    _psg_model_path = args.model_name_or_path
                    
                lm_q = ModelClass.from_pretrained(_qry_model_path, trust_remote_code=True, **hf_kwargs)
                lm_p = ModelClass.from_pretrained(_psg_model_path, trust_remote_code=True, **hf_kwargs)
        else:
            # Load from hub
            lm_q = ModelClass.from_pretrained(args.model_name_or_path, **hf_kwargs)
            
            if args.share_encoder:
                lm_p = lm_q
            else:
                lm_p = copy.deepcopy(lm_q)
        
        if ModelClass == NeMoAutoModelForCausalLM:
            lm_q = lm_q.model
            lm_p = lm_p.model
            
        # Enable gradient checkpointing if requested
        if args.do_gradient_checkpointing:
            lm_q.gradient_checkpointing_enable()
            if lm_p is not lm_q:
                lm_p.gradient_checkpointing_enable()
        
        # Create linear pooler if needed
        if args.add_linear_pooler:
            linear_pooler = nn.Linear(lm_q.config.hidden_size, args.out_dimension)
            
            pooler_path = os.path.join(args.model_name_or_path, "pooler.pt")
            if os.path.exists(pooler_path):
                logger.info("Loading pooler weights from local files")
                state_dict = torch.load(pooler_path, map_location="cpu")
                linear_pooler.load_state_dict(state_dict)
        else:
            linear_pooler = nn.Identity()
            
        model = cls(args=args, lm_q=lm_q, lm_p=lm_p, linear_pooler=linear_pooler)
        return model
    
    def save(self, output_dir: str):
        """Save model to output directory."""
        
        logger.info(f"Saving BiencoderModel to {output_dir}")
        
        # Save the model
        if self.args.share_encoder:
            self.lm_q.save_pretrained(output_dir)
        else:
            os.makedirs(os.path.join(output_dir, "query_model"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "passage_model"), exist_ok=True)
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
        
        # Save linear pooler if exists
        if self.args.add_linear_pooler:
            pooler_path = os.path.join(output_dir, "pooler.pt")
            logger.info(f"Saving linear pooler to {pooler_path}")
            torch.save(self.linear_pooler.state_dict(), pooler_path)

