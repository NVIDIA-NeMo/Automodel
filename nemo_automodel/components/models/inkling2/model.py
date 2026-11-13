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

# The output head (FP32 logits, softcap, unpadded vocabulary) follows the HuggingFace Inkling 2
# implementation (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Source: https://github.com/huggingface/transformers/pull/50682
# Revision: 109a7d19bf667168837f32929e00b791e195d14d
# Original file: src/transformers/models/inkling2/modeling_inkling2.py

"""Native NeMo AutoModel implementation of Inkling 2 (``Inkling2ForConditionalGeneration``).

The output head follows the Transformers ``inkling2`` reference (huggingface/transformers
commit 109a7d19bf667168837f32929e00b791e195d14d, Apache-2.0): logits are computed by the
model-dtype ``lm_head``, cast to FP32, softcapped with ``tanh(x / c) * c`` and sliced to the
unpadded vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .configuration import Inkling2Config
from .layers import inkling2_moe_config
from .multimodal import Inkling2Model
from .parallelization import PARALLELIZER
from .quantization import checkpoint_expert_formats
from .state_dict_adapter import Inkling2StateDictAdapter


@dataclass
class Inkling2CausalLMOutput(ModelOutput):
    """Inkling 2 language-model output.

    Attributes:
        logits: FP32 tensor of shape ``[batch, kept_sequence, unpadded_vocab]``.
        image_hidden_states: Optional tensor of shape ``[image_tokens, hidden]``.
    """

    logits: torch.Tensor | None = None
    image_hidden_states: torch.Tensor | None = None


class Inkling2ForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Native Inkling 2 vision-language MoE model for training."""

    # ``embed`` and ``unembed`` are separate tensors in the released checkpoints.
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    # The adapter covers every checkpoint tensor; skip initializing a model that is loaded next.
    _skip_init_weights_on_load: bool = True

    # FSDP sidecar: the text model's ``final_input`` query gets its own FP32 unit.
    parallelizer = PARALLELIZER

    # The Transformers reference keeps these modules in FP32 (``_keep_in_fp32_modules_strict``).
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "mlp.gate.weight",
        "post_attention_convolution",
        "post_mlp_convolution",
        "k_sconv.conv1d",
        "v_sconv.conv1d",
        "attention_input",
        "mlp_input",
        "final_input",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        # Decoder layers also carry ``block_states``, which single-tensor pipeline stage IO cannot pass.
        supports_pp: bool = False
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: Inkling2Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "Inkling2ForConditionalGeneration":
        """Construct the model from its local AutoModel config."""
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "Inkling2ForConditionalGeneration":
        """Construct the native module tree for checkpoint loading.

        The NeMo AutoModel bridge and checkpointer own weight loading; this method only
        resolves the config and builds the modules.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = Inkling2Config.from_pretrained(pretrained_model_name_or_path)
        torch_dtype = kwargs.pop("torch_dtype", getattr(config, "torch_dtype", torch.bfloat16))
        if isinstance(torch_dtype, str):
            torch_dtype = get_dtype(torch_dtype, torch.bfloat16)
        config.torch_dtype = torch_dtype
        config._name_or_path = pretrained_model_name_or_path
        model = cls.from_config(config, *model_args, **kwargs)
        model.name_or_path = pretrained_model_name_or_path
        return model

    def __init__(
        self,
        config: Inkling2Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the model.

        Args:
            config: Inkling 2 configuration; ``config.torch_dtype`` selects the storage dtype.
            moe_config: Optional routed-expert configuration; derived from the text config by default.
            backend: Backend configuration. ``attn`` must be ``"sdpa"`` or ``"eager"``; defaults to
                ``BackendConfig(attn="sdpa")``.
            **kwargs: Extra bridge arguments; ignored.
        """
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__()
        del kwargs
        self.config = config
        # The attention supports eager and SDPA only, so the default backend picks SDPA.
        self.backend = backend or BackendConfig(attn="sdpa")
        text_config = config.text_config
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.moe_config = moe_config or inkling2_moe_config(text_config, model_dtype)
        self.model = Inkling2Model(config, self.backend, self.moe_config, dtype=model_dtype)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False, dtype=model_dtype)
        if self.backend.enable_hf_state_dict_adapter:
            # quantization_config exists only when the checkpoint ships hf_quant_config.json (see Inkling2Config).
            self.state_dict_adapter = Inkling2StateDictAdapter(
                text_config,
                dtype=model_dtype,
                expert_formats=checkpoint_expert_formats(getattr(config, "quantization_config", None)),
            )
        cast_model_to_dtype(self, model_dtype)

    def get_input_embeddings(self) -> nn.Module:
        """Return the normalized token embedding."""
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        """Return the language-model output projection."""
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> Inkling2CausalLMOutput:
        """Compute softcapped FP32 logits.

        Args:
            input_ids: Optional long tensor of shape ``[batch, sequence]``.
            pixel_values: Optional tensor of shape ``[image_tokens, patch * merge, patch * merge, channels]``.
            image_grid_thw: Optional long tensor of shape ``[images, 3]``; required with ``pixel_values``.
            attention_mask: Optional padding tensor of shape ``[batch, sequence]``, nonzero for valid tokens.
            position_ids: Optional long tensor of shape ``[batch or 1, sequence]``.
            inputs_embeds: Optional normalized embeddings of shape ``[batch, sequence, hidden]``.
            logits_to_keep: Number of trailing positions to keep (``0`` keeps all), or a long
                tensor of sequence indices.

        Returns:
            Output with FP32 ``logits`` of shape ``[batch, kept_sequence, unpadded_vocab]``.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()
        softcap = self.config.text_config.final_logit_softcapping
        logits = torch.tanh(logits / softcap) * softcap
        logits = logits[..., : self.config.text_config.unpadded_vocab_size]
        return Inkling2CausalLMOutput(logits=logits, image_hidden_states=outputs.image_hidden_states)

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the text model and the output head for checkpoint-free construction."""
        del dtype
        buffer_device = buffer_device or next(self.parameters()).device
        self.model.init_weights(buffer_device)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.text_config.initializer_range)

    def update_moe_gate_bias(self) -> None:
        """Keep the released router correction bias fixed; Inkling 2 has no bias-update rule."""
        return


ModelClass = Inkling2ForConditionalGeneration

__all__ = ["Inkling2CausalLMOutput", "Inkling2ForConditionalGeneration", "ModelClass"]
