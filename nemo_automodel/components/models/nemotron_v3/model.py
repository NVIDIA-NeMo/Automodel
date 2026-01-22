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

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class NemotronV3Model(nn.Module):
    """NemotronV3 base model (without LM head).

    This is a hybrid architecture with Mamba2, Attention, MLP, and MoE layers.
    """

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        *,
        moe_config: MoEConfig | None = None,
    ):
        """Initialize NemotronV3Model.

        Args:
            config: NemotronH config with model parameters
            backend: Backend configuration for MoE and other components
            moe_config: MoE configuration (optional, will create default if None)
        """
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.moe_config = moe_config or MoEConfig(
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=1,  # NemotronV3 has 1 shared expert
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=False,  # Router weights are trained but not using bias updates
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,  # No aux loss for NemotronV3
            score_func="sigmoid",  # NemotronV3 uses sigmoid scoring
            route_scale=config.routed_scaling_factor,
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,  # For shared expert
            moe_inter_dim=config.moe_intermediate_size,  # For routed experts
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=config.mlp_bias,
            expert_activation="relu2",  # NemotronV3 uses ReLU² activation
            dtype=config.torch_dtype,
            shared_expert_gate=False,
            shared_expert_inter_dim=config.moe_shared_expert_intermediate_size,
            shared_expert_activation="relu2",  # Use ReLU² for shared experts
        )

        # Embeddings
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        # Transformer layers (hybrid: mamba, attention, mlp, moe)
        self.layers = nn.ModuleDict()
        for idx in range(config.num_hidden_layers):
            self.layers[str(idx)] = NemotronV3Block(config, layer_idx=idx, moe_config=self.moe_config, backend=self.backend)

        # Final norm
        self.norm = initialize_rms_norm_module(
            self.backend.rms_norm,
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )


    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: torch.Tensor | None = None,
        causal_mask_mapping: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: 2D padding mask [batch_size, seq_len] (1=real, 0=padding)
            causal_mask_mapping: Dict with precomputed 4D causal masks for attention layers
            **kwargs: Additional arguments (ignored)

        Returns:
            Hidden states tensor [batch_size, seq_len, hidden_size]
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # TODO: attention mask currently does not work. A default causal mask is applied.

        # Get 4D causal mask for attention layers (from precomputed masks)
        causal_mask = causal_mask_mapping.get("full_attention") if causal_mask_mapping is not None else None

        # Apply transformer layers
        for layer in self.layers.values():
            # Pass appropriate mask based on layer type
            if layer.block_type == "attention":
                # Attention layers use 4D causal mask
                mask = causal_mask
            elif layer.block_type == "mamba":
                # Mamba layers use 2D padding mask
                mask = attention_mask
            else:
                # MLP/MoE layers don't use mask
                mask = None

            hidden_states = layer(hidden_states, attention_mask=mask)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _to_local(self, tensor):
        """Get local tensor from DTensor or return tensor as-is."""
        if DTensor is not None and isinstance(tensor, DTensor):
            return tensor.to_local()
        return tensor

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights according to NemotronV3 spec.

        Args:
            buffer_device: Device to use for buffer initialization (unused for NemotronV3)
        """
        # Embedding weights: normal initialization
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.initializer_range)

        # Initialize all layers
        for layer_idx, block in enumerate(self.layers.values()):
            # Initialize norm
            nn.init.ones_(block.norm.weight)

            # Initialize mixer based on type
            if block.block_type == "mamba":
                self._init_mamba_weights(block.mixer)
            elif block.block_type == "attention":
                self._init_attention_weights(block.mixer)
            elif block.block_type == "mlp":
                self._init_mlp_weights(block.mixer)
            elif block.block_type == "moe":
                self._init_moe_weights(block.mixer)

        # Final norm
        nn.init.ones_(self.norm.weight)

    def _init_mamba_weights(self, mamba_module):
        """Initialize Mamba2Mixer weights."""
        # Mark A_log and D for no weight decay
        mamba_module.A_log._no_weight_decay = True
        mamba_module.D._no_weight_decay = True

        # Special dt_bias initialization (inverse softplus)
        # Use _to_local to handle DTensor case after model parallelization
        dt_bias_local = self._to_local(mamba_module.dt_bias)
        local_num_heads = dt_bias_local.shape[0]  # May be sharded
        dt = torch.exp(
            torch.rand(local_num_heads, device=dt_bias_local.device)
            * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
            + math.log(self.config.time_step_min)
        ).clamp(min=self.config.time_step_floor)

        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_bias_local.copy_(inv_dt)
        mamba_module.dt_bias._no_reinit = True

        # Linear layers
        if mamba_module.in_proj.bias is not None:
            nn.init.zeros_(mamba_module.in_proj.bias)
        if mamba_module.out_proj.bias is not None:
            nn.init.zeros_(mamba_module.out_proj.bias)

        # Rescale out_proj if enabled
        if getattr(self.config, "rescale_prenorm_residual", True):
            with torch.no_grad():
                mamba_module.out_proj.weight /= math.sqrt(self.config.num_hidden_layers)

    def _init_attention_weights(self, attn_module):
        """Initialize attention weights."""
        # Zero biases if present
        for proj in [attn_module.q_proj, attn_module.k_proj, attn_module.v_proj, attn_module.o_proj]:
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # Rescale o_proj if enabled
        if getattr(self.config, "rescale_prenorm_residual", True):
            with torch.no_grad():
                attn_module.o_proj.weight /= math.sqrt(self.config.num_hidden_layers)

    def _init_mlp_weights(self, mlp_module):
        """Initialize MLP weights."""
        # Zero biases if present
        for proj in [mlp_module.up_proj, mlp_module.down_proj]:
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # Rescale down_proj if enabled
        if getattr(self.config, "rescale_prenorm_residual", True):
            with torch.no_grad():
                mlp_module.down_proj.weight /= math.sqrt(self.config.num_hidden_layers)

    def _init_moe_weights(self, moe_module):
        """Initialize MoE weights."""
        # Router: initialize gate weights (they are created with torch.empty)
        nn.init.normal_(moe_module.gate.weight, mean=0.0, std=self.config.initializer_range)
        if moe_module.gate.bias is not None:
            nn.init.zeros_(moe_module.gate.bias)

        # Experts: zero biases if present
        if hasattr(moe_module.experts, "gate_up_proj_bias") and moe_module.experts.gate_up_proj_bias is not None:
            nn.init.zeros_(moe_module.experts.gate_up_proj_bias)
        if hasattr(moe_module.experts, "down_proj_bias") and moe_module.experts.down_proj_bias is not None:
            nn.init.zeros_(moe_module.experts.down_proj_bias)

        # Shared expert
        if moe_module.shared_experts.up_proj.bias is not None:
            nn.init.zeros_(moe_module.shared_experts.up_proj.bias)
        if moe_module.shared_experts.down_proj.bias is not None:
            nn.init.zeros_(moe_module.shared_experts.down_proj.bias)

        # Rescale output projections if enabled
        if getattr(self.config, "rescale_prenorm_residual", True):
            with torch.no_grad():
                # Rescale expert down_projs
                moe_module.experts.down_projs /= math.sqrt(self.config.num_hidden_layers)
                # Rescale shared expert down_proj
                moe_module.shared_experts.down_proj.weight /= math.sqrt(self.config.num_hidden_layers)


class NemotronHForCausalLM(nn.Module, MoEFSDPSyncMixin):
    """NemotronV3 model with language modeling head."""

    @classmethod
    def from_config(
        cls,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Create model from config.

        Args:
            config: NemotronH config
            backend: Backend configuration
            **kwargs: Additional arguments

        Returns:
            NemotronHForCausalLM instance
        """
        return cls(config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """Load pretrained model.

        Args:
            pretrained_model_name_or_path: Path or name of pretrained model
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            NemotronHForCausalLM instance
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Initialize NemotronV3ForCausalLM.

        Args:
            config: NemotronH config
            backend: Backend configuration
            **kwargs: Additional arguments
        """
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        # Base model
        self.model = NemotronV3Model(config, backend=self.backend)

        # LM head
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )

        # Create state_dict_adapter if enabled (needed to convert HF checkpoints)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = NemotronV3StateDictAdapter(
                config=config,
                moe_config=self.model.moe_config,
                backend=self.backend,
                dtype=dtype,
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: torch.Tensor | None = None,
        causal_mask_mapping: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: 2D padding mask [batch_size, seq_len]
            causal_mask_mapping: Dict with precomputed 4D causal masks
            **kwargs: Additional arguments

        Returns:
            logits tensor [batch_size, seq_len, vocab_size]
        """
        # Forward through base model
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            causal_mask_mapping=causal_mask_mapping,
            **kwargs,
        )

        # Compute logits (in float32 for numerical stability)
        logits = self.lm_head(hidden_states).float()

        return logits

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize model weights.

        Args:
            buffer_device: Device to use for buffer initialization
            dtype: Target dtype for model weights
        """
        # Initialize base model
        self.model.initialize_weights(buffer_device=buffer_device)

        # Initialize LM head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)

        # Convert to target dtype
        self.to(dtype)


# Alias for consistency with other models
ModelClass = NemotronHForCausalLM
