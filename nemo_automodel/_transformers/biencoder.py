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
Generic Biencoder Model for embedding and retrieval tasks.

This module provides the BiencoderModel class that works with any bidirectional
backbone registered in the ModelRegistry. The backbone models are loaded
dynamically based on the model type.
"""

import copy
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PreTrainedModel
from transformers.utils import logging

from nemo_automodel._transformers.registry import ModelRegistry

try:
    from nemo_automodel.components.models.common.bidirectional import BiencoderStateDictAdapter
except ImportError:
    BiencoderStateDictAdapter = object


logger = logging.get_logger(__name__)


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


# Mapping from HuggingFace model_type to bidirectional architecture name
# These architecture names must match the class names exported via ModelClass
# in the corresponding bidirectional model module (e.g., llama_bidirectional/model.py)
SUPPORTED_BACKBONES = {
    "llama": "LlamaBidirectionalModel",
    # Add more backbones as needed:
}


class BiencoderModel(nn.Module):
    """
    Biencoder model for embedding and retrieval.

    Encodes queries and passages separately using bidirectional backbone models.
    The backbone is loaded dynamically from the ModelRegistry based on model type.
    Training logic (loss, optimizer, etc.) lives in the recipe layer.
    """

    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        linear_pooler: nn.Module = None,
        pooling: str = "avg",
        l2_normalize: bool = True,
        share_encoder: bool = True,
        add_linear_pooler: bool = False,
    ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooling = pooling
        self.l2_normalize = l2_normalize
        self.share_encoder = share_encoder
        self.add_linear_pooler = add_linear_pooler
        self.linear_pooler = linear_pooler if linear_pooler is not None else nn.Identity()
        self.config = self.lm_q.config

        # HuggingFace consolidated checkpoint compatibility
        self.name_or_path = os.path.abspath(__file__)
        self.state_dict_adapter = BiencoderStateDictAdapter()
        encoder_class_name = self.lm_q.__class__.__name__
        self.config.architectures = [encoder_class_name]
        self.config.auto_map = {
            "AutoModel": f"model.{encoder_class_name}",
            "AutoConfig": f"model.{encoder_class_name.replace('Model', 'Config')}",
        }

    def encode(self, input_dict: dict, encoder: str = "query") -> Optional[torch.Tensor]:
        """Encode inputs using the query or passage encoder.

        Args:
            input_dict: Tokenized inputs (input_ids, attention_mask, etc.)
            encoder: "query" or "passage"

        Returns:
            Embeddings [batch_size, hidden_dim], or None if input_dict is empty.
        """
        model = self.lm_q if encoder == "query" else self.lm_p
        return self._encode(model, input_dict)

    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        """Encode input using the encoder."""
        if not input_dict:
            return None

        import inspect

        if (
            "token_type_ids" not in inspect.getfullargspec(encoder.forward).args
            and "token_type_ids" in input_dict.keys()
        ):
            input_dict = {k: v for k, v in input_dict.items() if k != "token_type_ids"}

        outputs = encoder(
            **{k: v for k, v in input_dict.items() if k not in ["kd_labels"]},
            return_dict=True,
            output_hidden_states=True,
        )

        if hasattr(outputs, "last_hidden_state"):
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[-1]

        embeds = pool(
            last_hidden_states=hidden_state,
            attention_mask=input_dict["attention_mask"],
            pool_type=self.pooling,
        )
        embeds = self.linear_pooler(embeds)

        if self.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)

        return embeds.contiguous()

    @classmethod
    def build(
        cls,
        model_name_or_path: str,
        share_encoder: bool = True,
        add_linear_pooler: bool = False,
        out_dimension: int = 768,
        do_gradient_checkpointing: bool = False,
        pooling: str = "avg",
        l2_normalize: bool = True,
        trust_remote_code: bool = False,
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
            pooling: Pooling strategy ('avg', 'cls', 'last', etc.)
            l2_normalize: Whether to L2 normalize embeddings
            trust_remote_code: Whether to trust remote code
            **hf_kwargs: Additional arguments passed to model loading
        """

        logger.info(f"Building BiencoderModel from {model_name_or_path}")

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", "")

        arch_name = SUPPORTED_BACKBONES.get(model_type)
        if arch_name is None:
            supported = ", ".join(SUPPORTED_BACKBONES.keys())
            raise ValueError(
                f"Unsupported model type '{model_type}' for biencoder. "
                f"Supported types: {supported}. "
                f"To add support for a new backbone, update SUPPORTED_BACKBONES and "
                "create a bidirectional model class with ModelClass export."
            )

        if arch_name not in ModelRegistry.model_arch_name_to_cls:
            raise ValueError(
                f"Bidirectional model class '{arch_name}' not found in ModelRegistry. "
                f"Ensure the model is exported via ModelClass in the corresponding module."
            )
        BidirectionalModelClass = ModelRegistry.model_arch_name_to_cls[arch_name]
        logger.info(f"Using {arch_name} from registry")

        if os.path.isdir(model_name_or_path):
            if share_encoder:
                lm_q = BidirectionalModelClass.from_pretrained(
                    model_name_or_path, trust_remote_code=trust_remote_code, **hf_kwargs
                )
                lm_p = lm_q
            else:
                _qry_model_path = os.path.join(model_name_or_path, "query_model")
                _psg_model_path = os.path.join(model_name_or_path, "passage_model")

                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_name_or_path
                    _psg_model_path = model_name_or_path

                lm_q = BidirectionalModelClass.from_pretrained(
                    _qry_model_path, trust_remote_code=trust_remote_code, **hf_kwargs
                )
                lm_p = BidirectionalModelClass.from_pretrained(
                    _psg_model_path, trust_remote_code=trust_remote_code, **hf_kwargs
                )
        else:
            # Load from hub
            lm_q = BidirectionalModelClass.from_pretrained(model_name_or_path, **hf_kwargs)

            if share_encoder:
                lm_p = lm_q
            else:
                lm_p = copy.deepcopy(lm_q)

        if do_gradient_checkpointing:
            lm_q.gradient_checkpointing_enable()
            if lm_p is not lm_q:
                lm_p.gradient_checkpointing_enable()

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
            pooling=pooling,
            l2_normalize=l2_normalize,
            share_encoder=share_encoder,
            add_linear_pooler=add_linear_pooler,
        )
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model to output directory.

        When a ``checkpointer`` is supplied (the normal training path), saving
        is delegated to :py:meth:`Checkpointer.save_model` so that:

        * The ``state_dict_adapter`` (to_hf / from_hf) conversions are applied,
          keeping the on-disk key format consistent with ``load_model``.
        * Distributed / FSDP state dicts are handled correctly.
        * Async and consolidated-safetensors paths are honoured.

        Without a checkpointer the method falls back to HuggingFace-native
        ``save_pretrained`` on the underlying encoder(s), which is useful for
        standalone / non-distributed export.

        Args:
            save_directory: Output path for the checkpoint.
            **kwargs: Forwarded from the recipe; expected keys include
                ``checkpointer``, ``tokenizer``, and ``peft_config``.
        """
        checkpointer = kwargs.get("checkpointer", None)
        if checkpointer is not None:
            # Delegate to Checkpointer.save_model() which handles:
            # - ModelState.state_dict()
            # - _maybe_adapt_state_dict_to_hf() via state_dict_adapter
            # - Distributed/sharded saving via DCP
            # - Consolidated HF safetensors output
            # - Async checkpointing if enabled
            checkpointer.save_model(
                model=self,
                weights_path=save_directory,
                peft_config=kwargs.get("peft_config", None),
                tokenizer=kwargs.get("tokenizer", None),
            )
            return

        # Fallback: HF-native save (no checkpointer available)
        logger.info(f"Saving BiencoderModel to {save_directory}")

        if self.share_encoder:
            self.lm_q.save_pretrained(save_directory)
        else:
            os.makedirs(os.path.join(save_directory, "query_model"), exist_ok=True)
            os.makedirs(os.path.join(save_directory, "passage_model"), exist_ok=True)
            self.lm_q.save_pretrained(os.path.join(save_directory, "query_model"))
            self.lm_p.save_pretrained(os.path.join(save_directory, "passage_model"))

        if self.add_linear_pooler:
            pooler_path = os.path.join(save_directory, "pooler.pt")
            logger.info(f"Saving linear pooler to {pooler_path}")
            torch.save(self.linear_pooler.state_dict(), pooler_path)
