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

"""Automodel adapter for the ``moonlight_v4_rmsnorm_h2048`` Kernel Factory kernel.

This is the hand-written half of the integration: it owns the autograd, DTensor,
``torch.compile`` and checkpoint contracts, and it is not regenerated. The kernel
it calls lives alone in :mod:`nemo_automodel.components.models.common.kf_rms_norm_kernel`,
which is the only file a campaign winner replaces.

Scope of the campaign, and therefore of this adapter:

- **Forward** comes from the generated kernel.
- **Backward** is the shipped fp32 reference gradient, reused verbatim from
  ``Float32RMSNorm``'s custom op. The campaign definition is forward only, so
  there is no searched backward to call; the gradient is the exact gradient of
  the reference forward, which the promotion gates require the generated forward
  to match to within one bf16 ulp. A training-speed claim needs a second campaign
  with a backward definition, not this module.

The backend is deliberately narrow. ``kf_triton_h2048`` names the exact contract
the campaign searched over — bf16 in and out, ``hidden == 2048``, ``eps == 1e-6``
— and every other shape, dtype or epsilon is rejected rather than silently run on
a kernel that was never benchmarked for it.
"""

import torch
from torch import nn
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding

from nemo_automodel.components.models.common import kf_rms_norm_kernel

# These two are the reference fp32 gradient of ``nemo_automodel::float32_rms_norm``.
# They are reused rather than reimplemented so the kf backend and the torch_fp32
# baseline cannot drift apart in the backward while the campaign is forward only.
from nemo_automodel.components.models.common.utils import (
    _float32_rms_norm_backward,
    _float32_rms_norm_setup_context,
)

KF_RMS_NORM_HIDDEN = 2048
KF_RMS_NORM_EPS = 1e-6
KF_RMS_NORM_DTYPE = torch.bfloat16


def _reject_unsupported_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    """Reject anything outside the contract the campaign searched over.

    Args:
        x: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Tensor of shape [hidden].

    Raises:
        ValueError: If the hidden size, the dtypes, or the weight rank differ from
            the ``rmsnorm_bf16_h2048_eps1e6`` definition.
    """
    if x.ndim == 0 or x.shape[-1] != KF_RMS_NORM_HIDDEN:
        raise ValueError(
            f"rms_norm='kf_triton_h2048' is specialized for hidden={KF_RMS_NORM_HIDDEN}, "
            f"got input of shape {tuple(x.shape)}. Use rms_norm='torch_fp32' for other sizes."
        )
    if weight.ndim != 1 or weight.shape[0] != KF_RMS_NORM_HIDDEN:
        raise ValueError(
            f"rms_norm='kf_triton_h2048' expects a weight of shape [{KF_RMS_NORM_HIDDEN}], got {tuple(weight.shape)}."
        )
    if x.dtype != KF_RMS_NORM_DTYPE or weight.dtype != KF_RMS_NORM_DTYPE:
        raise ValueError(
            f"rms_norm='kf_triton_h2048' is a bf16 kernel, got input dtype {x.dtype} and "
            f"weight dtype {weight.dtype}. Use rms_norm='torch_fp32' for other dtypes."
        )


# Keep the forward opaque, as the torch_fp32 path does, so grad and no_grad
# compilation run the same kernel.
@torch.library.custom_op("nemo_automodel::kf_triton_rms_norm", mutates_args=())
def _kf_triton_rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Run the generated RMSNorm kernel over the flattened rows of ``x``.

    Args:
        x: Tensor of shape [..., 2048] in bf16, with arbitrary leading dimensions.
            Non-contiguous inputs are copied to row-major before the kernel runs.
        weight: Tensor of shape [2048] in bf16 on the same device as x.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., 2048] in bf16, freshly allocated and not aliasing
        either input.
    """
    _reject_unsupported_inputs(x, weight)
    rows = x.reshape(-1, KF_RMS_NORM_HIDDEN)
    # The kernel indexes rows by a single stride and assumes unit stride on hidden.
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    out = torch.empty_like(rows)
    kf_rms_norm_kernel.run(rows, weight.contiguous(), float(eps), out)
    return out.view(x.shape)


@_kf_triton_rms_norm_impl.register_fake
def _kf_triton_rms_norm_meta(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Describe the output metadata for inputs x [..., 2048] and weight [2048].

    Args:
        x: Tensor of shape [..., 2048] in bf16, with arbitrary leading dimensions.
        weight: Tensor of shape [2048] in bf16.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., 2048] with x's dtype and device.
    """
    _reject_unsupported_inputs(x, weight)
    return torch.empty_like(x)


@register_sharding(torch.ops.nemo_automodel.kf_triton_rms_norm.default)
def _kf_triton_rms_norm_sharding(x, weight, eps):
    """Keep the normalized axis complete while allowing leading-axis sharding.

    Args:
        x: Tensor metadata of global shape [..., 2048].
        weight: Tensor metadata of global shape [2048].
        eps: Epsilon added to the mean square.

    Returns:
        Supported output/input placements on each mesh axis. Leading dimensions
        may be sharded; weight and the hidden dimension must be replicated.
    """
    strategies = [([Replicate()], [Replicate(), Replicate(), None])]
    strategies.extend(([Shard(dim)], [Shard(dim), Replicate(), None]) for dim in range(x.ndim - 1))
    return strategies


_kf_triton_rms_norm_impl.register_autograd(
    _float32_rms_norm_backward,
    setup_context=_float32_rms_norm_setup_context,
)


@torch.compile(dynamic=True)
def _kf_triton_rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compiled wrapper around the opaque forward — standalone to minimize dynamo guards.

    This mirrors ``Float32RMSNorm``: the forward itself is opaque either way, but
    compiling here lets Inductor fuse the fp32 reference backward exactly as the
    baseline does. Without it the backward runs eager and costs both time and peak
    memory, which would show up in an A/B as a loss that has nothing to do with the
    forward kernel under test.

    Args:
        x: Tensor of shape [..., 2048] in bf16, with arbitrary leading dimensions.
        weight: Tensor of shape [2048] in bf16 on the same device as x.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., 2048] in bf16.
    """
    return torch.ops.nemo_automodel.kf_triton_rms_norm(x, weight, eps)


class KFTritonRMSNorm(nn.Module):
    """RMSNorm whose forward is the Kernel Factory kernel and whose backward is the fp32 reference.

    Drop-in replacement for :class:`~nemo_automodel.components.models.common.utils.Float32RMSNorm`
    at ``hidden == 2048`` and ``eps == 1e-6``. The parameter is named ``weight`` and
    has the same shape and dtype, so checkpoint keys and FSDP2 sharding are unchanged.
    """

    def __init__(
        self,
        dim: int,
        eps: float = KF_RMS_NORM_EPS,
        device: torch.device | str | None = None,
        dtype: torch.dtype = KF_RMS_NORM_DTYPE,
    ) -> None:
        """Build the norm, rejecting any contract the campaign did not search.

        Args:
            dim: Normalized dimension. Must be 2048.
            eps: Epsilon added to the mean square. Must be 1e-6.
            device: Device to create the weight on. None uses the PyTorch default.
            dtype: Parameter dtype. Must be bfloat16.

        Raises:
            ValueError: If dim, eps, or dtype fall outside the campaign contract.
        """
        super().__init__()
        if dim != KF_RMS_NORM_HIDDEN:
            raise ValueError(
                f"rms_norm='kf_triton_h2048' is specialized for hidden={KF_RMS_NORM_HIDDEN}, got dim={dim}. "
                "Automodel call sites that normalize another size (for example the DeepSeek-V4 q/kv, "
                "compressor and indexer norms) must stay on rms_norm='torch_fp32'."
            )
        if float(eps) != KF_RMS_NORM_EPS:
            raise ValueError(
                f"rms_norm='kf_triton_h2048' was searched and benchmarked at eps={KF_RMS_NORM_EPS}, got eps={eps}."
            )
        if dtype != KF_RMS_NORM_DTYPE:
            raise ValueError(f"rms_norm='kf_triton_h2048' is a bf16 kernel, got dtype={dtype}.")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        """Materialize the weight to ones, as the torch_fp32 path does."""
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and scale the hidden axis.

        Args:
            x: Tensor of shape [..., 2048] in bf16, with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., 2048] in bf16, not aliasing x.
        """
        return _kf_triton_rms_norm_fwd(x, self.weight, self.eps)
