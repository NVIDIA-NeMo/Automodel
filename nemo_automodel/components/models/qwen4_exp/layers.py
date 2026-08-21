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

"""Qwen4-Exp model-local layers.

The HyperConnection equations in this module follow the Qwen4-Exp reference
implementation. They are intentionally not shared with DeepSeek-V4: that model
uses a different Sinkhorn-based HyperConnection parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention, Qwen3NextRMSNorm
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Qwen4ExpRMSNormGated(nn.Module):
    """Qwen4 GDN output normalization with its checkpoint-selected gate.

    Transformers' Qwen3.5 GatedDeltaNet hard-codes a SiLU output gate.  Qwen4
    keeps the same projections and delta-rule core but sets
    ``output_gate_type='sigmoid'``.  Keeping this model-local module avoids
    changing the shared Qwen3.5 execution contract.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str = "sigmoid",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if activation not in ("sigmoid", "silu"):
            raise ValueError(f"Unsupported Qwen4 GDN output gate {activation!r}")
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize in fp32, then apply the configured gate in fp32.

        Args:
            hidden_states: Tensor of shape ``[..., hidden_size]`` containing
                GatedDeltaNet values.
            gate: Tensor of shape ``[..., hidden_size]`` containing the
                elementwise output-gate logits.

        Returns:
            Tensor of shape ``[..., hidden_size]`` in the input dtype.
        """
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        # SGLang's layernorm_gated kernel keeps xhat*weight and gate multiply
        # in fp32 and casts only at the output store.
        normalized = self.weight.float() * normalized
        gate_fp32 = gate.float()
        activated_gate = torch.sigmoid(gate_fp32) if self.activation == "sigmoid" else F.silu(gate_fp32)
        return (normalized * activated_gate).to(input_dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Match the multiplicative RMSNorm checkpoint convention."""
        self.weight.fill_(1.0)


class Qwen4ExpGatedDeltaNet(CPAwareGatedDeltaNet):
    """CP-aware GatedDeltaNet with Qwen4's sigmoid output gate."""

    def __init__(self, config: object, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        output_gate_type = str(getattr(config, "output_gate_type", "sigmoid"))
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim,
            eps=float(getattr(config, "rms_norm_eps")),
            activation=output_gate_type,
            dtype=get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16),
        )


class Qwen4ExpGroupedRMSNorm(nn.Module):
    """Gemma-style RMS normalization over fixed-width HyperConnection branches.

    Args:
        hidden_size: Flattened feature width.
        group_size: Number of features normalized together. A flattened input
            of shape ``[..., hidden_size]`` is viewed as
            ``[..., hidden_size // group_size, group_size]``.
        eps: Variance epsilon.

    Tensor layout:
        Input and output have shape ``[..., hidden_size]``. The learned weight
        has shape ``[hidden_size]`` and is applied as ``1 + weight``.
    """

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0 or group_size <= 0 or hidden_size % group_size != 0:
            raise ValueError(
                "Qwen4ExpGroupedRMSNorm requires positive, divisible widths; "
                f"got hidden_size={hidden_size}, group_size={group_size}"
            )
        self.hidden_size = hidden_size
        self.group_size = group_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize each branch independently.

        Args:
            hidden_states: Flattened branch states of shape
                ``[..., hidden_size]``.

        Returns:
            Normalized states of shape ``[..., hidden_size]``.
        """
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden width {self.hidden_size}, got {hidden_states.shape[-1]}")
        input_dtype = hidden_states.dtype
        grouped = hidden_states.float().unflatten(-1, (-1, self.group_size))
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset the additive Gemma-style scale to zero."""
        self.weight.zero_()


@dataclass(frozen=True)
class Qwen4ExpHyperConnectionResidual:
    """Residual tensors retained between a HyperConnection read and write.

    Attributes:
        hidden_states: Flattened HC streams of shape ``[..., hc_count * hidden_size]``.
        normalized_states: Branch-normalized streams with the same shape.
    """

    hidden_states: torch.Tensor
    normalized_states: torch.Tensor


class Qwen4ExpHyperConnection(nn.Module):
    """Qwen4-Exp gated HyperConnection read/write transform.

    A read normalizes ``hc_count`` streams independently, predicts a feature
    gate, and averages the gated streams to one block input. A write predicts
    one injection gate per stream and adds the block output back to every
    stream. The final decoder mixer uses only the read side.

    Args:
        hidden_size: Width of one HC stream.
        hc_count: Number of streams.
        lowrank_size: Bottleneck width used to predict read gates.
        rms_norm_eps: Variance epsilon for branch normalization.
        backend: Linear backend configuration.
        use_combine: Whether to instantiate the write-side injection weight.
        dtype: Parameter dtype override. If omitted, the backend model dtype is
            resolved by the caller and should be passed explicitly.

    Tensor layout:
        Reads ``[..., hc_count * hidden_size]`` and returns
        ``[..., hidden_size]``. Writes a block tensor ``[..., hidden_size]``
        back to a residual tensor ``[..., hc_count * hidden_size]``.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        lowrank_size: int,
        rms_norm_eps: float,
        backend: BackendConfig,
        *,
        use_combine: bool = True,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        if hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {hc_count}")
        if hidden_size <= 0 or lowrank_size <= 0:
            raise ValueError(f"hidden_size and lowrank_size must be positive, got {hidden_size}, {lowrank_size}")
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.flat_hidden_size = hidden_size * hc_count
        self.lowrank_size = lowrank_size
        self.use_combine = use_combine
        parameter_dtype = get_dtype(dtype, torch.bfloat16)

        self.hc_norm = Qwen4ExpGroupedRMSNorm(
            self.flat_hidden_size,
            group_size=hidden_size,
            eps=rms_norm_eps,
        )
        self.input_mix_weight_down = initialize_linear_module(
            backend.linear,
            self.flat_hidden_size,
            lowrank_size,
            bias=False,
            dtype=parameter_dtype,
        )
        self.input_mix_weight_up = initialize_linear_module(
            backend.linear,
            lowrank_size,
            self.flat_hidden_size,
            bias=False,
            dtype=parameter_dtype,
        )
        self.block_inject_weight = (
            initialize_linear_module(
                backend.linear,
                self.flat_hidden_size,
                hc_count,
                bias=False,
                dtype=parameter_dtype,
            )
            if use_combine
            else None
        )

    def mix(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, Qwen4ExpHyperConnectionResidual]:
        """Collapse HC streams to the input of an attention or MoE block.

        Args:
            hidden_states: Flattened HC streams of shape
                ``[..., hc_count * hidden_size]``.

        Returns:
            A pair containing the mixed block input of shape
            ``[..., hidden_size]`` and the residual tensors needed by
            :meth:`combine`.
        """
        if hidden_states.shape[-1] != self.flat_hidden_size:
            raise ValueError(f"Expected HC width {self.flat_hidden_size}, got {hidden_states.shape[-1]}")
        normalized = self.hc_norm(hidden_states)
        gates = F.silu(self.input_mix_weight_down(normalized) / self.hc_count)
        gates = torch.sigmoid(self.input_mix_weight_up(gates))
        mixed = (
            gates.unflatten(-1, (self.hc_count, self.hidden_size))
            * normalized.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        residual = Qwen4ExpHyperConnectionResidual(hidden_states, normalized)
        return mixed, residual

    def combine(
        self,
        block_output: torch.Tensor,
        residual: Qwen4ExpHyperConnectionResidual,
    ) -> torch.Tensor:
        """Inject one block output into every HC stream.

        Args:
            block_output: Attention or MoE output of shape
                ``[..., hidden_size]``.
            residual: Flattened pre-block streams and their normalized values,
                each shaped ``[..., hc_count * hidden_size]``.

        Returns:
            Updated flattened streams of shape
            ``[..., hc_count * hidden_size]``.
        """
        if self.block_inject_weight is None:
            raise RuntimeError("This final-mixer HyperConnection has no combine side")
        if block_output.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected block output width {self.hidden_size}, got {block_output.shape[-1]}")
        if residual.hidden_states.shape[-1] != self.flat_hidden_size:
            raise ValueError(f"Expected residual width {self.flat_hidden_size}, got {residual.hidden_states.shape[-1]}")
        injection_gate = 2.0 * torch.sigmoid(self.block_inject_weight(residual.normalized_states) / self.hc_count)
        streams = residual.hidden_states.unflatten(-1, (self.hc_count, self.hidden_size))
        injection = block_output.unsqueeze(-2) * injection_gate.unsqueeze(-1)
        return (streams + injection).flatten(-2)

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize HC weights for training from scratch.

        Args:
            init_std: Standard deviation for all HC linear weights.
        """
        self.hc_norm.reset_parameters()
        nn.init.trunc_normal_(self.input_mix_weight_down.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.input_mix_weight_up.weight, mean=0.0, std=init_std)
        if self.block_inject_weight is not None:
            nn.init.trunc_normal_(self.block_inject_weight.weight, mean=0.0, std=init_std)


class Qwen4ExpQSAIndexerParameters(nn.Module):
    """Own the QSA indexer weights used by long-context full-attention layers.

    QSA only chooses which keys a sparse attention kernel evaluates; it does
    not alter attention values. For the target short-sequence SFT, every causal
    key falls within ``indexer_budget`` and dense attention is exactly
    equivalent. This module therefore owns and loads the checkpoint parameters
    while :class:`Qwen4ExpDenseAttention` executes the equivalent dense path.

    Args:
        config: Qwen4-Exp text configuration.
        backend: Linear backend configuration.

    Tensor layout:
        ``index_qk_proj`` maps ``[..., hidden_size]`` to
        ``[..., (indexer_n_heads + indexer_kv_heads) * indexer_head_dim]``.
    """

    def __init__(self, config: object, backend: BackendConfig) -> None:
        super().__init__()
        hidden_size = int(getattr(config, "hidden_size"))
        n_query_heads = int(getattr(config, "indexer_n_heads"))
        n_key_heads = int(getattr(config, "indexer_kv_heads"))
        head_dim = int(getattr(config, "indexer_head_dim"))
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.index_qk_proj = initialize_linear_module(
            backend.linear,
            hidden_size,
            (n_query_heads + n_key_heads) * head_dim,
            bias=False,
            dtype=dtype,
        )
        self.q_layernorm = Qwen3NextRMSNorm(head_dim, eps=float(getattr(config, "rms_norm_eps")))
        self.k_layernorm = Qwen3NextRMSNorm(head_dim, eps=float(getattr(config, "rms_norm_eps")))

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize QSA indexer parameters for training from scratch.

        Args:
            init_std: Standard deviation for the fused query/key projection.
        """
        nn.init.trunc_normal_(self.index_qk_proj.weight, mean=0.0, std=init_std)
        self.q_layernorm.reset_parameters()
        self.k_layernorm.reset_parameters()


class Qwen4ExpDenseAttention(Qwen3NextAttention):
    """Qwen4-Exp gated attention with checkpoint-compatible QSA parameters.

    The attention projection and output-gate equations are inherited from
    Qwen3-Next/Qwen3.5. ``indexer`` is retained for state-dict compatibility;
    the forward path evaluates dense causal attention, which is the exact QSA
    result while the sequence length does not exceed the indexer token budget.
    """

    def __init__(self, config: object, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__(config, layer_idx, backend)
        self.indexer_budget = int(getattr(config, "indexer_budget"))
        self.indexer = Qwen4ExpQSAIndexerParameters(config, backend)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: object,
    ) -> torch.Tensor:
        """Evaluate exact short-sequence QSA as dense causal attention.

        Args:
            x: Block input of shape ``[batch, sequence, hidden_size]``.
            freqs_cis: Rotary values of shape
                ``[axes, batch, sequence, rotary_dim]`` whose final axis stores
                concatenated cosine and sine values.
            attention_mask: Optional token mask of shape ``[batch, sequence]``
                or backend-specific causal attention mask.
            **attn_kwargs: Backend attention metadata.

        Returns:
            Attention output with the same shape as ``x``.
        """
        sequence_length = x.shape[-2] if x.ndim >= 3 else x.shape[0]
        if sequence_length > self.indexer_budget:
            raise NotImplementedError(
                "Qwen4-Exp QSA training above the configured indexer budget is not yet supported; "
                f"got sequence_length={sequence_length}, indexer_budget={self.indexer_budget}"
            )
        return super().forward(
            x,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize attention and indexer weights.

        Args:
            buffer_device: Device retained for the common attention initializer contract.
            init_std: Projection initialization standard deviation.
        """
        super().init_weights(buffer_device, init_std=init_std)
        self.indexer.init_weights(init_std=init_std)


class Qwen4ExpDecoderLayer(nn.Module):
    """One Qwen4-Exp decoder layer with two learned HyperConnections.

    Args:
        layer_idx: Zero-based decoder index.
        config: Qwen4-Exp text configuration.
        moe_config: Native MoE configuration.
        backend: Attention, linear, and expert backend configuration.
        ple: Optional Engram-derived PLE module. The checkpoint installs it
            only on decoder index 1.

    Tensor layout:
        The first layer accepts token embeddings ``[batch, sequence, hidden]``
        and expands them to flattened HC state
        ``[batch, sequence, hc_count * hidden]``. Every layer returns that
        flattened HC layout.
    """

    def __init__(
        self,
        layer_idx: int,
        config: object,
        moe_config: object,
        backend: BackendConfig,
        *,
        ple: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            block_types = getattr(config, "layers_block_type")
            layer_types = ["full_attention" if value == "attention" else value for value in block_types]
        self.layer_type = str(layer_types[layer_idx])
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.hc_count = int(getattr(config, "hc_count"))
        self.ple = ple
        # PLE's owner-sharded lookup uses mutable distributed collectives.
        # PyTorch's selective TorchDispatch checkpoint context cannot safely
        # cache those side effects, so the MoE parallelizer leaves only this
        # decoder block eager while checkpointing every other Qwen4 block.
        self._nemo_disable_activation_checkpointing = ple is not None
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen4ExpDenseAttention(config, layer_idx, backend)
        else:
            raise ValueError(f"Unsupported Qwen4-Exp layer type {self.layer_type!r}")

        self.mlp = MoE(moe_config, backend)
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        hc_kwargs = {
            "hidden_size": self.hidden_size,
            "hc_count": self.hc_count,
            "lowrank_size": int(getattr(config, "hc_lowrank")),
            "rms_norm_eps": float(getattr(config, "rms_norm_eps")),
            "backend": backend,
            "dtype": dtype,
        }
        self.attn_hyper_connection = Qwen4ExpHyperConnection(**hc_kwargs)
        self.mlp_hyper_connection = Qwen4ExpHyperConnection(**hc_kwargs)

    def _expand_initial_streams(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expand a one-stream decoder input into the persistent HC layout.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden_size]``
                or ``[batch, sequence, hc_count * hidden_size]``.

        Returns:
            Tensor of shape
            ``[batch, sequence, hc_count * hidden_size]``.
        """
        if hidden_states.shape[-1] == self.hidden_size * self.hc_count:
            return hidden_states
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Qwen4-Exp decoder input must contain one stream or all HC streams; "
                f"got width {hidden_states.shape[-1]}"
            )
        return torch.cat([hidden_states] * self.hc_count, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **attn_kwargs: object,
    ) -> torch.Tensor:
        """Run PLE, attention/GDN, and top-10 MoE updates.

        Args:
            hidden_states: One-stream input ``[batch, sequence, hidden]`` on
                the first layer, otherwise flattened HC streams
                ``[batch, sequence, hc_count * hidden]``.
            input_ids: Raw tokenizer IDs of shape ``[batch, sequence]`` used by
                the PLE hash path.
            freqs_cis: Rotary values of shape
                ``[axes, batch, sequence, rotary_dim]`` whose final axis stores
                concatenated cosine and sine values.
            attention_mask: Optional token mask of shape ``[batch, sequence]``
                or backend-specific causal attention mask.
            padding_mask: Optional ``[batch, sequence]`` mask where ``True``
                marks padding for MoE dispatch.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            **attn_kwargs: Attention backend metadata.

        Returns:
            Flattened HC streams of shape
            ``[batch, sequence, hc_count * hidden]``.
        """
        if attention_mask is not None and padding_mask is None and attention_mask.ndim <= 2:
            padding_mask = attention_mask.bool().logical_not()

        hidden_states = self._expand_initial_streams(hidden_states)
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, input_ids)

        attn_input, attn_residual = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            attn_output = self.linear_attn(
                hidden_states=attn_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            attn_output = self.self_attn(
                x=attn_input,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                **attn_kwargs,
            )
        hidden_states = self.attn_hyper_connection.combine(attn_output, attn_residual)

        mlp_input, mlp_residual = self.mlp_hyper_connection.mix(hidden_states)
        mlp_module = unwrap_checkpoint_wrapper(self.mlp)
        if not isinstance(mlp_module, MoE):
            raise TypeError(f"Qwen4-Exp requires an MoE block, got {type(mlp_module).__name__}")
        mlp_output = self.mlp(mlp_input, padding_mask)
        return self.mlp_hyper_connection.combine(mlp_output, mlp_residual)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize this decoder layer for training from scratch.

        Args:
            buffer_device: Device used by attention/MoE initializers.
            init_std: Standard deviation for dense projection weights.
        """
        if self.layer_type == "full_attention":
            self.self_attn.init_weights(buffer_device, init_std=init_std)
        else:
            self.linear_attn.dt_bias.fill_(1.0)
            self.linear_attn.A_log.uniform_(0, 16).log_()
            for linear in (
                self.linear_attn.in_proj_qkv,
                self.linear_attn.in_proj_z,
                self.linear_attn.in_proj_b,
                self.linear_attn.in_proj_a,
                self.linear_attn.out_proj,
            ):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if hasattr(self.linear_attn.norm, "reset_parameters"):
                self.linear_attn.norm.reset_parameters()
            else:
                self.linear_attn.norm.weight.zero_()
        self.attn_hyper_connection.init_weights(init_std=init_std)
        self.mlp_hyper_connection.init_weights(init_std=init_std)
        self.mlp.init_weights(buffer_device)
