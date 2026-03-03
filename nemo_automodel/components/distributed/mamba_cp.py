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

"""Context parallelism for Mamba/SSM layers using a hidden-parallel strategy.

Instead of splitting the sequence across CP ranks (as attention CP does), this module
uses an all-to-all redistribution so that each CP rank processes the *full* sequence
but only a *subset* of heads (d_inner / cp_size).  The data flow is::

    [B, L_local, D]  -->  all-to-all  -->  [B, L_global, D/cp]
        -->  conv1d + SSM kernel  -->
    [B, L_global, D/cp]  -->  all-to-all  -->  [B, L_local, D]

This module is intentionally **not** a subclass of ``nn.Module`` because it owns
no trainable parameters.  It holds *references* to the Mamba mixer's parameters
and slices them in the forward path so that gradients flow back to the full
(unsliced) parameters.
"""

import torch
import torch.distributed
import torch.nn as nn

# ---------------------------------------------------------------------------
# Autograd-aware all-to-all primitive
# ---------------------------------------------------------------------------


class _AllToAll(torch.autograd.Function):
    """Autograd wrapper around ``torch.distributed.all_to_all_single``.

    For equal-sized splits the all-to-all operation is its own inverse,
    so the backward pass is simply another all-to-all on the same group.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        output = torch.empty_like(input_)
        torch.distributed.all_to_all_single(output, input_, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx.group
        grad_input = torch.empty_like(grad_output)
        torch.distributed.all_to_all_single(grad_input, grad_output, group=group)
        return grad_input, None


def _all_to_all(input_: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
    """Functional entry-point for the autograd-aware all-to-all."""
    return _AllToAll.apply(input_, group)


# ---------------------------------------------------------------------------
# Sequence-sharded <-> Hidden-sharded layout transformations (batch-first)
# ---------------------------------------------------------------------------


def _all_to_all_cp2hp(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    batch_size: int,
) -> torch.Tensor:
    """Transform from sequence-sharded to hidden-sharded layout (batch-first).

    Args:
        input_: Tensor of shape ``[B, L_local, H]`` where H is the full hidden
            dimension on this rank.
        cp_group: Context-parallel process group.
        batch_size: Batch size ``B`` (needed to recover dimensions after reshape).

    Returns:
        Tensor of shape ``[B, L_global, H / cp_size]``.
    """
    cp_size = cp_group.size()
    B, L_local, H = input_.shape
    H_local = H // cp_size

    # [B*L_local, cp, H_local] -> [cp, B*L_local, H_local] -> flatten for all-to-all
    send_tensor = (
        input_.reshape(B * L_local, cp_size, H_local)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(cp_size * B * L_local, H_local)
    )

    recv_tensor = _all_to_all(send_tensor, cp_group)

    # [cp, B, L_local, H_local] -> [B, cp*L_local, H_local]
    return (
        recv_tensor.reshape(cp_size, B, L_local, H_local)
        .permute(1, 0, 2, 3)
        .contiguous()
        .reshape(B, cp_size * L_local, H_local)
    )


def _all_to_all_hp2cp(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    batch_size: int,
) -> torch.Tensor:
    """Transform from hidden-sharded to sequence-sharded layout (batch-first).

    This is the inverse of :func:`_all_to_all_cp2hp`.

    Args:
        input_: Tensor of shape ``[B, L_global, H_local]`` where ``H_local = H / cp_size``.
        cp_group: Context-parallel process group.
        batch_size: Batch size ``B``.

    Returns:
        Tensor of shape ``[B, L_local, H]`` where ``L_local = L_global / cp_size``
        and ``H = H_local * cp_size``.
    """
    cp_size = cp_group.size()
    B, L_global, H_local = input_.shape
    L_local = L_global // cp_size

    # [B, cp, L_local, H_local] -> [cp, B, L_local, H_local] -> flatten for all-to-all
    send_tensor = (
        input_.reshape(B, cp_size, L_local, H_local)
        .permute(1, 0, 2, 3)
        .contiguous()
        .reshape(cp_size * B * L_local, H_local)
    )

    recv_tensor = _all_to_all(send_tensor, cp_group)

    # [cp, B*L_local, H_local] -> [B*L_local, cp, H_local] -> [B, L_local, H]
    return (
        recv_tensor.reshape(cp_size, B * L_local, H_local)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(B, L_local, cp_size * H_local)
    )


# ---------------------------------------------------------------------------
# MambaContextParallel – orchestrates CP for a single Mamba mixer layer
# ---------------------------------------------------------------------------


class MambaContextParallel:
    """Hidden-parallel context parallelism for a Mamba2 mixer layer.

    This class does **not** own trainable parameters.  It stores *references*
    to the mixer's parameters (conv1d, dt_bias, A_log, D) and slices them on
    the fly so that gradients propagate to the original (full) parameters.

    Args:
        cp_group: Context-parallel process group.
        num_heads: Total number of SSM heads (before any parallelism).
        head_dim: Dimension per head.
        n_groups: Number of SSM groups (for grouped B/C states).
        d_state: SSM state dimension.
        conv1d: Reference to the mixer's ``nn.Conv1d`` module.
        dt_bias: Reference to the mixer's ``dt_bias`` parameter.
        A_log: Reference to the mixer's ``A_log`` parameter.
        D: Reference to the mixer's ``D`` parameter.
    """

    def __init__(
        self,
        cp_group: torch.distributed.ProcessGroup,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        d_state: int,
        conv1d: nn.Conv1d,
        dt_bias: torch.Tensor,
        A_log: torch.Tensor,
        D: torch.Tensor,
    ) -> None:
        self.cp_group = cp_group
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.d_state = d_state
        self.conv1d = conv1d
        self.dt_bias = dt_bias
        self.A_log = A_log
        self.D = D

        self.cp_size = cp_group.size()
        self.cp_rank = cp_group.rank()

        self.d_inner = num_heads * head_dim

        # --- Validate and compute per-rank sizes ---

        # Each CP rank must get at least one head.
        assert num_heads % self.cp_size == 0, f"num_heads ({num_heads}) must be divisible by cp_size ({self.cp_size})"
        self.num_heads_local = num_heads // self.cp_size
        self.d_inner_local = self.num_heads_local * head_dim

        # Groups: when n_groups < cp_size we need to replicate B/C states.
        if n_groups < self.cp_size:
            assert self.cp_size % n_groups == 0, (
                f"cp_size ({self.cp_size}) must be divisible by n_groups ({n_groups}) when n_groups < cp_size"
            )
            self.group_repeat_count = self.cp_size // n_groups
            self.n_groups_local = 1
        else:
            assert n_groups % self.cp_size == 0, f"n_groups ({n_groups}) must be divisible by cp_size ({self.cp_size})"
            self.group_repeat_count = 1
            self.n_groups_local = n_groups // self.cp_size

    # ------------------------------------------------------------------ #
    #  Activation transforms (before / after conv+SSM)                    #
    # ------------------------------------------------------------------ #

    def pre_conv_ssm(self, projected_states: torch.Tensor) -> torch.Tensor:
        """Redistribute ``[B, L_local, proj_dim]`` to ``[B, L_global, proj_dim_local]``.

        Splits the packed projection into [z, x, B, C, dt], optionally replicates
        B/C states, performs cp2hp all-to-all on each, and re-concatenates.
        """
        if self.cp_size == 1:
            return projected_states

        B = projected_states.shape[0]
        groups_state_size = self.n_groups * self.d_state

        z, x, B_state, C_state, dt = torch.split(
            projected_states,
            [self.d_inner, self.d_inner, groups_state_size, groups_state_size, self.num_heads],
            dim=-1,
        )

        # Replicate B and C group states when n_groups < cp_size so that
        # replicas land on consecutive CP ranks with their associated heads.
        if self.group_repeat_count > 1:
            B_state = self._repeat_group_state(B_state)
            C_state = self._repeat_group_state(C_state)

        z = _all_to_all_cp2hp(z, self.cp_group, B)
        x = _all_to_all_cp2hp(x, self.cp_group, B)
        B_state = _all_to_all_cp2hp(B_state, self.cp_group, B)
        C_state = _all_to_all_cp2hp(C_state, self.cp_group, B)
        dt = _all_to_all_cp2hp(dt, self.cp_group, B)

        return torch.cat([z, x, B_state, C_state, dt], dim=-1)

    def post_conv_ssm(self, output: torch.Tensor) -> torch.Tensor:
        """Redistribute SSM output from hidden-sharded back to sequence-sharded layout.

        Args:
            output: ``[B, L_global, d_inner / cp_size]`` — the (already gated)
                SSM output on this rank.

        Returns:
            ``[B, L_local, d_inner]`` — sequence-sharded output.
        """
        if self.cp_size == 1:
            return output

        B = output.shape[0]
        return _all_to_all_hp2cp(output, self.cp_group, B)

    # ------------------------------------------------------------------ #
    #  Parameter slicing (returns views so grads flow to full params)      #
    # ------------------------------------------------------------------ #

    def get_conv1d_weight(self) -> torch.Tensor:
        """Slice ``conv1d.weight`` for the current CP rank.

        Weight shape: ``[conv_dim, 1, kernel_size]`` where
        ``conv_dim = d_inner + 2 * n_groups * d_state``.
        Returns ``[conv_dim_local, kernel_size]`` (squeezed for causal_conv1d kernel).
        """
        return self._slice_conv_param(self.conv1d.weight).squeeze(1)

    def get_conv1d_bias(self) -> torch.Tensor:
        """Slice ``conv1d.bias`` for the current CP rank.

        Bias shape: ``[conv_dim]``.  Returns ``[conv_dim_local]``.
        """
        if self.conv1d.bias is None:
            return None
        return self._slice_conv_param(self.conv1d.bias)

    def get_dt_bias(self) -> torch.Tensor:
        """Slice ``dt_bias`` for the current CP rank."""
        return self._slice_vector_param(self.dt_bias)

    def get_A_log(self) -> torch.Tensor:
        """Slice ``A_log`` for the current CP rank."""
        return self._slice_vector_param(self.A_log)

    def get_D(self) -> torch.Tensor:
        """Slice ``D`` for the current CP rank."""
        return self._slice_vector_param(self.D)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _repeat_group_state(self, state: torch.Tensor) -> torch.Tensor:
        """Repeat group states for CP ranks when n_groups < cp_size.

        ``[B, L, n_groups * d_state]`` -> ``[B, L, n_groups * repeat * d_state]``
        """
        return (
            state.reshape(*state.shape[:-1], self.n_groups, self.d_state)
            .unsqueeze(-2)
            .expand(-1, -1, -1, self.group_repeat_count, -1)
            .reshape(*state.shape[:-1], self.n_groups * self.group_repeat_count * self.d_state)
        )

    def _slice_vector_param(self, param: torch.Tensor) -> torch.Tensor:
        """Slice a per-head vector parameter for the current CP rank."""
        start = self.cp_rank * self.num_heads_local
        return param[start : start + self.num_heads_local]

    def _slice_conv_param(self, param: torch.Tensor) -> torch.Tensor:
        """Slice a conv1d parameter (weight or bias) along its channel dimension.

        Parameter slicing is done in the forward path so that gradients
        backpropagate to the original (full) parameters.
        """
        groups_state_size = self.n_groups * self.d_state
        x, B_param, C_param = torch.split(
            param,
            [self.d_inner, groups_state_size, groups_state_size],
            dim=0,
        )

        x_start = self.cp_rank * self.d_inner_local
        x_sliced = x[x_start : x_start + self.d_inner_local]

        bc_size = self.n_groups_local * self.d_state
        bc_start = (self.cp_rank // self.group_repeat_count) * bc_size
        B_sliced = B_param[bc_start : bc_start + bc_size]
        C_sliced = C_param[bc_start : bc_start + bc_size]

        return torch.cat([x_sliced, B_sliced, C_sliced], dim=0).contiguous()
