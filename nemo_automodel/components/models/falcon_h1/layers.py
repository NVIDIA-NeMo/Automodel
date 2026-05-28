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


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.falcon_h1.modeling_falcon_h1 import apply_rotary_pos_emb


class FalconH1MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers

    def forward(self, x):
        y = self.up_proj(x) * self.act_fn(self.gate_proj(x) * self.gate_multiplier)
        y = self.down_proj(y) * self.down_multiplier
        return y


class FalconH1Attention(nn.Module):
    """Multi-headed attention with Falcon-H1's key_multiplier."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.key_multiplier = config.key_multiplier

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) * self.key_multiplier
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=attention_mask is None,
            scale=self.scaling,
            enable_gqa=self.num_key_value_groups > 1,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def _compute_mup_vector(config):
    """Build the zxbcdt µP vector applied element-wise to in_proj's output.

    The in_proj output is laid out as concatenation of five sections:
        [z (intermediate_size) | x (intermediate_size) | B (n_groups*d_state) | C (n_groups*d_state) | dt (n_heads)]
    Each section is scaled by config.ssm_multipliers[i].
    """
    intermediate_size = (
        config.mamba_d_ssm if config.mamba_d_ssm is not None else int(config.mamba_expand * config.hidden_size)
    )
    groups_time_state_size = config.mamba_n_groups * config.mamba_d_state
    num_heads = config.mamba_n_heads
    multipliers = config.ssm_multipliers

    vector_shape = 2 * intermediate_size + 2 * groups_time_state_size + num_heads
    v = torch.ones(1, 1, vector_shape)
    v[:, :, :intermediate_size] *= multipliers[0]  # z
    v[:, :, intermediate_size : 2 * intermediate_size] *= multipliers[1]  # x
    v[:, :, 2 * intermediate_size : 2 * intermediate_size + groups_time_state_size] *= multipliers[2]  # B
    v[:, :, 2 * intermediate_size + groups_time_state_size : 2 * intermediate_size + 2 * groups_time_state_size] *= (
        multipliers[3]
    )  # C
    v[:, :, 2 * intermediate_size + 2 * groups_time_state_size :] *= multipliers[4]  # dt
    return v


class FalconH1Mamba(nn.Module):
    """Mamba-2 SSM mixer for Falcon-H1 (training-path only)."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.ssm_state_size = config.mamba_d_state
        self.n_groups = config.mamba_n_groups
        self.chunk_size = config.mamba_chunk_size
        self.activation = config.hidden_act

        self.intermediate_size = (
            config.mamba_d_ssm if config.mamba_d_ssm is not None else int(config.mamba_expand * self.hidden_size)
        )
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.mamba_proj_bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=config.mamba_d_conv,
            groups=self.conv_dim,
            padding=config.mamba_d_conv - 1,
            bias=config.mamba_conv_bias,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.projectors_bias)

        self.ssm_in_multiplier = config.ssm_in_multiplier
        self.register_buffer("mup_vector", _compute_mup_vector(config), persistent=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

        hidden_states = hidden_states * self.ssm_in_multiplier
        projected_states = self.in_proj(hidden_states)
        projected_states = projected_states * self.mup_vector

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
            rmsnorm_weight=None,
            rmsnorm_eps=None,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.head_dim,
            ngroups=self.n_groups,
            norm_before_gate=False,
            return_final_states=False,
            **dt_limit_kwargs,
        )
        return out


# Add to imports at top of layers.py:
from transformers.modeling_layers import GradientCheckpointingLayer


class FalconH1DecoderLayer(GradientCheckpointingLayer):
    """Parallel-fuse decoder layer: Mamba and attention run in parallel off
    a shared input norm, sum into the residual, then a post-norm MLP path.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.mamba = FalconH1Mamba(config, layer_idx=layer_idx)
        self.self_attn = FalconH1Attention(config, layer_idx=layer_idx)
        self.feed_forward = FalconH1MLP(config)

        # Falcon-H1 µP multipliers that live at the decoder-layer level
        self.attention_in_multiplier = config.attention_in_multiplier
        self.attention_out_multiplier = config.attention_out_multiplier
        self.ssm_out_multiplier = config.ssm_out_multiplier

        # Two norms: one shared for the parallel branches, one before the MLP
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Parallel Mamba ‖ attention branch
        residual = hidden_states
        h = self.input_layernorm(hidden_states)

        mamba_out = self.mamba(h, **kwargs) * self.ssm_out_multiplier

        attn_out, _ = self.self_attn(
            h * self.attention_in_multiplier,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        attn_out = attn_out * self.attention_out_multiplier

        hidden_states = residual + mamba_out + attn_out

        # Post-norm MLP path
        residual = hidden_states
        h = self.pre_ff_layernorm(hidden_states)
        hidden_states = residual + self.feed_forward(h)

        return hidden_states
