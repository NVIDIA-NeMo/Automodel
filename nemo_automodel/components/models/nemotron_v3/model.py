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

"""NemotronV3 (Nemotron-Nano) model support for NeMo Automodel.

This module provides state dict adapter support for the NVIDIA-Nemotron-3-Nano model,
which uses a hybrid Mamba2 + Attention + MoE architecture with relu2 activation.

Key characteristics:
- Hybrid architecture with pattern: MEMEM*EMEMEM*EMEMEM*... (M=Mamba2, *=Attention, E=MoE)
- 128 routed experts, top-6 routing, sigmoid with e_score_correction_bias
- relu2 activation (squared ReLU, not swiglu)
- Shared experts with different intermediate size
- Router/gate computation in float32 (matches HF behavior)

Usage:
    The model is loaded via HuggingFace with trust_remote_code=True.
    The state dict adapter handles conversion between HF and native formats.

Note:
    Gate precision defaults to float32 to match HuggingFace behavior.
    The HF NemotronH model computes router logits in float32 for numerical stability.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def create_moe_config_from_hf_config(hf_config) -> MoEConfig:
    """Create MoEConfig from HuggingFace NemotronHConfig.

    Args:
        hf_config: HuggingFace NemotronHConfig object

    Returns:
        MoEConfig: Configuration for MoE layers
    """
    return MoEConfig(
        dim=hf_config.hidden_size,
        inter_dim=hf_config.intermediate_size,
        moe_inter_dim=hf_config.moe_intermediate_size,
        n_routed_experts=hf_config.n_routed_experts,
        n_shared_experts=hf_config.n_shared_experts,
        n_activated_experts=hf_config.num_experts_per_tok,
        n_expert_groups=hf_config.n_group,
        n_limited_groups=hf_config.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.001,
        score_func="sigmoid",
        route_scale=hf_config.routed_scaling_factor,
        aux_loss_coeff=0,
        norm_topk_prob=hf_config.norm_topk_prob,
        expert_activation="relu2",  # NemotronV3 uses relu2
        shared_expert_inter_dim=hf_config.moe_shared_expert_intermediate_size,
    )


class NemotronV3StateDictAdapterMixin:
    """Mixin to add state dict adapter to HuggingFace NemotronH models.

    This mixin attaches a NemotronV3StateDictAdapter to the model for handling
    checkpoint conversion between HuggingFace and native formats.
    """

    def _setup_state_dict_adapter(
        self,
        config,
        backend: Optional[BackendConfig] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Setup the state dict adapter for checkpoint conversion.

        Args:
            config: HuggingFace NemotronHConfig
            backend: Backend configuration (defaults to BackendConfig())
            dtype: Model dtype (defaults to bfloat16)
        """
        backend = backend or BackendConfig()

        # NemotronV3/HF uses float32 for router computation - set default if not specified
        if backend.gate_precision is None:
            backend.gate_precision = torch.float32

        moe_config = create_moe_config_from_hf_config(config)

        if backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = NemotronV3StateDictAdapter(
                config, moe_config, backend, dtype=dtype
            )


class NemotronV3ForCausalLMWrapper(nn.Module, MoEFSDPSyncMixin, NemotronV3StateDictAdapterMixin):
    """Wrapper for HuggingFace NemotronHForCausalLM with native state dict adapter.

    This wrapper loads the HuggingFace model and attaches a state dict adapter
    for checkpoint conversion. The model is loaded with trust_remote_code=True.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> # Wrap with state dict adapter
        >>> wrapper = NemotronV3ForCausalLMWrapper(model)
    """

    def __init__(
        self,
        hf_model: nn.Module,
        backend: Optional[BackendConfig] = None,
    ):
        """Initialize the wrapper.

        Args:
            hf_model: HuggingFace NemotronHForCausalLM model
            backend: Backend configuration
        """
        super().__init__()
        self.model = hf_model
        self.config = hf_model.config

        # Get dtype from config
        dtype = get_dtype(getattr(self.config, "torch_dtype", "bfloat16"), torch.bfloat16)

        # Setup state dict adapter
        self._setup_state_dict_adapter(self.config, backend, dtype)

    def forward(self, *args, **kwargs):
        """Forward pass delegates to HF model."""
        return self.model(*args, **kwargs)

    def update_moe_gate_bias(self) -> None:
        """Update MoE gate bias for load balancing."""
        # Access the internal MoE layers and update their gate bias
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for layer in self.model.model.layers.values():
                if hasattr(layer, "mixer") and hasattr(layer.mixer, "gate"):
                    if hasattr(layer.mixer.gate, "e_score_correction_bias"):
                        # The HF model manages its own bias updates
                        pass


# For model registry
ModelClass = NemotronV3ForCausalLMWrapper
