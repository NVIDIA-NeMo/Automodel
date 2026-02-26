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

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import initialize_rms_norm_module


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
        past_key_values=None,
        layer_idx: int | None = None,
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

        # Update KV cache if present
        if past_key_values is not None and layer_idx is not None:
            k, v = past_key_values.update(k, v, layer_idx)

        # Run attention with SDPA
        # During cached decode (q has 1 token, k/v have many), use explicit mask
        # instead of is_causal since SDPA's causal mask requires matching seq lengths.
        is_causal = attention_mask is None and q.shape[2] > 1 and q.shape[2] == k.shape[2]
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
    def init_weights(
        self,
        num_hidden_layers: int,
        rescale_prenorm_residual: bool = True,
        buffer_device: torch.device | None = None,
    ) -> None:
        """Initialize attention weights following NemotronV3 spec."""
        with buffer_device:
            for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

            # Rescale o_proj for stable residual stream
            if rescale_prenorm_residual:
                self.o_proj.weight /= math.sqrt(num_hidden_layers)


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
        self.time_step_floor = config.time_step_floor

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
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with three code paths.

        Path A (training): past_key_values is None → fully-fused kernel.
        Path B (prefill): past_key_values present, seq_len > 1 → unfused scan + cache init.
        Path C (decode): past_key_values present, seq_len == 1, has_previous_state → single-step update.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask (applied to padding)
            past_key_values: Optional NemotronHybridCache instance.
            cache_position: Token positions for cache updates.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        use_cache = past_key_values is not None
        use_precomputed_states = (
            use_cache
            and past_key_values.has_previous_state
            and seq_len == 1
            and cache_position is not None
            and cache_position[0] > 0
        )

        # --- Path A: Training (no cache) → fused kernel ---
        if not use_cache:
            from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1)

            projected_states = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

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

        # --- Path C: Decode (single token, cached state) ---
        if use_precomputed_states:
            from causal_conv1d import causal_conv1d_update
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1)

            projected_states = self.in_proj(hidden_states).squeeze(1)  # (B, proj_size)

            gate, hidden_states_B_C, dt = projected_states.split(
                [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )

            # Conv update (single step)
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                past_key_values.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            groups_time_state_size = self.n_groups * self.ssm_state_size
            hidden_states_inner, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )

            # Reshape for selective_state_update
            A = -torch.exp(self.A_log.float())
            A = A[:, None, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None].expand(-1, self.head_dim)
            D = self.D[:, None].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, self.ssm_state_size)
            C = C.view(batch_size, self.n_groups, self.ssm_state_size)
            hidden_states_reshaped = hidden_states_inner.view(batch_size, self.num_heads, self.head_dim)

            hidden_states_inner = selective_state_update(
                past_key_values.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states_inner = hidden_states_inner.view(batch_size, self.num_heads * self.head_dim)

            # Gated RMSNorm + output projection
            out = self.norm(hidden_states_inner, gate)
            out = self.out_proj(out[:, None, :])  # unsqueeze seq dim back
            return out

        # --- Path B: Prefill (multi-token, cache init) ---
        from causal_conv1d import causal_conv1d_fn
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        projected_states = self.in_proj(hidden_states)

        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # Store conv state for future decode steps
        conv_states = F.pad(
            hidden_states_B_C.permute(0, 2, 1),
            (self.conv_kernel_size - hidden_states_B_C.shape[1], 0),
        )
        past_key_values.update_conv_state(self.layer_idx, conv_states, cache_position)

        # Full-sequence conv1d (contiguous required by causal_conv1d kernel)
        hidden_states_B_C = causal_conv1d_fn(
            x=hidden_states_B_C.transpose(1, 2).contiguous(),
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
        ).transpose(1, 2)[:, :seq_len]

        groups_time_state_size = self.n_groups * self.ssm_state_size
        hidden_states_inner, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )

        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

        # Chunk SSM scan
        scan_output, ssm_state = mamba_chunk_scan_combined(
            hidden_states_inner.view(batch_size, seq_len, -1, self.head_dim),
            nn.functional.softplus(dt + self.dt_bias),
            A,
            B.view(batch_size, seq_len, self.n_groups, -1),
            C.view(batch_size, seq_len, self.n_groups, -1),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=None,
            return_final_states=True,
            **dt_limit_kwargs,
        )

        # Store SSM state for future decode steps
        if ssm_state is not None:
            past_key_values.ssm_states[self.layer_idx].copy_(ssm_state)

        scan_output = scan_output.view(batch_size, seq_len, -1)

        # Gated RMSNorm + output projection
        out = self.norm(scan_output, gate)
        out = self.out_proj(out)
        return out

    @torch.no_grad()
    def init_weights(
        self,
        num_hidden_layers: int,
        rescale_prenorm_residual: bool = True,
        buffer_device: torch.device | None = None,
    ) -> None:
        """Initialize Mamba2Mixer weights following NemotronV3 spec."""

        def _to_local(tensor):
            """Get local tensor from DTensor or return as-is."""
            if DTensor is not None and isinstance(tensor, DTensor):
                return tensor.to_local()
            return tensor

        with buffer_device:
            # dt_bias: inverse softplus initialization
            # Check _no_reinit flag to avoid re-initializing if called multiple times
            if not getattr(self.dt_bias, "_no_reinit", False):
                dt_bias_local = _to_local(self.dt_bias)
                local_num_heads = dt_bias_local.shape[0]
                dt = torch.exp(
                    torch.rand(local_num_heads, device=dt_bias_local.device)
                    * (math.log(self.time_step_max) - math.log(self.time_step_min))
                    + math.log(self.time_step_min)
                ).clamp(min=self.time_step_floor)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                dt_bias_local.copy_(inv_dt)
                self.dt_bias._no_reinit = True

            # Mark A_log and D for no weight decay
            self.A_log._no_weight_decay = True
            self.D._no_weight_decay = True

            # Zero biases (don't reinitialize weights - they use default init)
            if self.in_proj.bias is not None:
                nn.init.zeros_(self.in_proj.bias)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

            # Rescale out_proj for stable residual stream
            if rescale_prenorm_residual:
                self.out_proj.weight /= math.sqrt(num_hidden_layers)


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
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Mask tensor - type depends on layer:
                - For attention: 4D causal mask [batch, 1, seq_len, seq_len]
                - For mamba: 2D padding mask [batch, seq_len]
                - For mlp/moe: None
            past_key_values: Optional NemotronHybridCache for KV/SSM caching.
            cache_position: Token position indices for cache updates.

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
            hidden_states = self.mixer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
        elif self.block_type == "attention":
            hidden_states = self.mixer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                layer_idx=self.layer_idx,
            )
        elif self.block_type in ["mlp", "moe"]:
            hidden_states = self.mixer(hidden_states)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize block weights following NemotronV3 spec.

        Args:
            buffer_device: Device for buffer initialization (used by MLP/MoE)
        """
        num_hidden_layers = self.config.num_hidden_layers
        rescale_prenorm_residual = getattr(self.config, "rescale_prenorm_residual", True)
        init_std = getattr(self.config, "initializer_range", 0.02)

        # Initialize norm
        self.norm.reset_parameters()

        # Initialize mixer based on block type
        if self.block_type == "mamba" or self.block_type == "attention":
            self.mixer.init_weights(
                num_hidden_layers=num_hidden_layers,
                rescale_prenorm_residual=rescale_prenorm_residual,
                buffer_device=buffer_device,
            )
        elif self.block_type == "mlp":
            # MLP uses existing init_weights, then apply rescaling
            self.mixer.init_weights(buffer_device=buffer_device, init_std=init_std)
            if rescale_prenorm_residual:
                self.mixer.down_proj.weight /= math.sqrt(num_hidden_layers)
        elif self.block_type == "moe":
            # MoE: use existing init_weights for base initialization
            self.mixer.init_weights(buffer_device=buffer_device, init_std=init_std)

            # Override gate weight with normal (not trunc_normal) for backward compat
            if hasattr(self.mixer.gate, "weight"):
                nn.init.normal_(self.mixer.gate.weight, mean=0.0, std=init_std)
            if hasattr(self.mixer.gate, "bias") and self.mixer.gate.bias is not None:
                nn.init.zeros_(self.mixer.gate.bias)

            # Zero expert biases
            if hasattr(self.mixer.experts, "gate_up_proj_bias") and self.mixer.experts.gate_up_proj_bias is not None:
                nn.init.zeros_(self.mixer.experts.gate_up_proj_bias)
            if hasattr(self.mixer.experts, "down_proj_bias") and self.mixer.experts.down_proj_bias is not None:
                nn.init.zeros_(self.mixer.experts.down_proj_bias)

            # Zero shared expert biases
            if self.mixer.shared_experts.up_proj.bias is not None:
                nn.init.zeros_(self.mixer.shared_experts.up_proj.bias)
            if self.mixer.shared_experts.down_proj.bias is not None:
                nn.init.zeros_(self.mixer.shared_experts.down_proj.bias)

            # Apply rescaling
            if rescale_prenorm_residual:
                self.mixer.experts.down_projs /= math.sqrt(num_hidden_layers)
                self.mixer.shared_experts.down_proj.weight /= math.sqrt(num_hidden_layers)
