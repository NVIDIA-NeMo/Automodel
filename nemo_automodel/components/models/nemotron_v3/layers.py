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

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.moe.utils import BackendConfig, initialize_rms_norm_module

try:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
except ImportError:
    print("torch.distributed.tensor is not available.")

try:
    from grouped_gemm import ops
except ImportError:
    print("grouped_gemm is not available.")


class NemotronV3Attention(nn.Module):
    """Multi-headed attention for NemotronV3 (Nano-v3).

    This is a standard GQA attention module following the NemotronH architecture.
    Uses PyTorch's scaled_dot_product_attention (SDPA) for the attention computation.
    Note: RoPE is not applied in this module, matching the HF NemotronHAttention implementation.
    """

    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_bias = getattr(config, "attention_bias", False)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # Q, K, V, O projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=self.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.size()

        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (B, H, S, D) for SDPA
        q = q.view(bsz, seqlen, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Run attention with SDPA
        is_causal = attention_mask is None and seqlen > 1
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=self.num_key_value_heads != self.num_attention_heads,
        )

        # Reshape back to (B, S, H * D)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, seqlen, self.num_attention_heads * self.head_dim)

        # Output projection
        output = self.o_proj(output)

        return output

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02):
        for linear in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)


class NemotronV3MambaRMSNormGated(nn.Module):
    """Gated RMSNorm for Mamba layers.

    Uses the fused triton kernel from mamba_ssm for efficiency.
    """

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

        return rmsnorm_fn(
            x=hidden_states,
            weight=self.weight,
            bias=None,
            z=gate,
            eps=self.variance_epsilon,
            group_size=self.group_size,
            norm_before_gate=False,
        )


class NemotronV3Mamba2Mixer(nn.Module):
    """Mamba2 mixer for NemotronV3 (training-only, uses CUDA kernels).

    This implementation uses the fused mamba_split_conv1d_scan_combined kernel
    for maximum training efficiency. Does not support inference caching.

    Requires mamba_ssm and causal_conv1d packages.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Model dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.ssm_state_size = config.ssm_state_size
        self.n_groups = config.n_groups
        self.chunk_size = config.chunk_size

        # Derived dimensions
        self.intermediate_size = self.num_heads * self.head_dim
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # Conv1d config
        self.conv_kernel_size = config.conv_kernel
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act

        # Time step limits
        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        # Layers
        # Input projection: projects to [gate, x, B, C, dt]
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)

        # Conv1d for sequence mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        # Gated RMSNorm
        self.norm = NemotronV3MambaRMSNormGated(
            self.intermediate_size,
            eps=config.layer_norm_epsilon,
            group_size=self.intermediate_size // self.n_groups,
        )

        # Output projection
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using fused CUDA kernels (training only).

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask (applied to padding)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

        # Apply mask to padding states if provided
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        # Input projection
        projected_states = self.in_proj(hidden_states)

        # Compute A from A_log
        A = -torch.exp(self.A_log.float())

        # Time step limit kwargs
        dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

        # Fused kernel: conv1d + SSM scan + gated norm + output projection
        out = mamba_split_conv1d_scan_combined(
            projected_states,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=self.D,
            chunk_size=self.chunk_size,
            seq_idx=None,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.variance_epsilon,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.head_dim,
            ngroups=self.n_groups,
            norm_before_gate=False,
            return_final_states=False,
            **dt_limit_kwargs,
        )

        return out

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.in_proj.weight, mean=0.0, std=init_std)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=init_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)


class NemotronV3Block(nn.Module):
    """NemotronV3 decoder block (training-only, simplified).

    Pre-norm architecture: norm → mixer → residual add
    Supports hybrid layer types: Mamba, Attention, MLP, MoE
    """

    def __init__(self, config, layer_idx: int, moe_config=None, backend=None):
        """Initialize NemotronV3Block.

        Args:
            config: Model configuration with layers_block_type attribute
            layer_idx: Index of this layer in the model
            moe_config: MoE configuration (required for MoE layers)
            backend: Backend configuration (optional)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)

        # RMSNorm
        self.norm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        # Determine layer type from config
        # 'M' → mamba, '*' → attention, '-' → mlp, other → moe
        self.block_type = config.layers_block_type[layer_idx]

        # Create mixer based on block type
        if self.block_type == "mamba":
            self.mixer = NemotronV3Mamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronV3Attention(config)
        elif self.block_type == "mlp":
            from nemo_automodel.components.moe.layers import MLP
            from nemo_automodel.shared.utils import dtype_from_str

            dtype = dtype_from_str(config.torch_dtype, torch.bfloat16)
            self.mixer = MLP(
                dim=config.hidden_size,
                inter_dim=config.intermediate_size,
                backend=backend.linear,
                dtype=dtype,
                activation=getattr(config, "mlp_hidden_act", "relu2"),
                bias=getattr(config, "mlp_bias", False),
            )
        elif self.block_type == "moe":
            from nemo_automodel.components.moe.layers import MoE

            # Use float32 for gate computation (numerical stability)
            if backend.gate_precision is None:
                backend.gate_precision = torch.float32

            self.mixer = MoE(moe_config, backend)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

    @property
    def mlp(self):
        """Return mixer for MoE blocks for compatibility with parallelizer."""
        if self.block_type == "moe":
            return self.mixer
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Mask tensor - type depends on layer:
                - For attention: 4D causal mask [batch, 1, seq_len, seq_len]
                - For mamba: 2D padding mask [batch, seq_len]
                - For mlp/moe: None

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Save residual
        residual = hidden_states

        # Pre-norm
        hidden_states = self.norm(hidden_states)

        # Optional fp32 residuals for numerical stability
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        # Apply mixer based on block type
        if self.block_type == "mamba":
            # Mamba takes 2D attention_mask for padding
            hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
        elif self.block_type == "attention":
            # Attention takes 4D causal mask
            hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
        elif self.block_type in ["mlp", "moe"]:
            # MLP/MoE don't use masks
            hidden_states = self.mixer(hidden_states)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states
