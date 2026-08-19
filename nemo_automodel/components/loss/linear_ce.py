# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (C) 2024 Apple Inc. All Rights Reserved.

# IMPORTANT:  This Apple software is supplied to you by Apple
# Inc. ("Apple") in consideration of your agreement to the following
# terms, and your use, installation, modification or redistribution of
# this Apple software constitutes acceptance of these terms.  If you do
# not agree with these terms, please do not use, install, modify or
# redistribute this Apple software.

# In consideration of your agreement to abide by the following terms, and
# subject to these terms, Apple grants you a personal, non-exclusive
# license, under Apple's copyrights in this original Apple software (the
# "Apple Software"), to use, reproduce, modify and redistribute the Apple
# Software, with or without modifications, in source and/or binary forms;
# provided that if you redistribute the Apple Software in its entirety and
# without modifications, you must retain this notice and the following
# text and disclaimers in all such redistributions of the Apple Software.
# Neither the name, trademarks, service marks or logos of Apple Inc. may
# be used to endorse or promote products derived from the Apple Software
# without specific prior written permission from Apple.  Except as
# expressly stated in this notice, no other rights or licenses, express or
# implied, are granted by Apple herein, including but not limited to any
# patent rights that may be infringed by your derivative works or by other
# works in which the Apple Software may be incorporated.

# The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
# MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
# THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
# OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.

# IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
# MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
# AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
# STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# -------------------------------------------------------------------------------
# SOFTWARE DISTRIBUTED WITH CUT CROSS ENTROPY:

# The Cut Cross Entropy software includes a number of subcomponents with separate
# copyright notices and license terms - please see the file ACKNOWLEDGEMENTS.md.
# -------------------------------------------------------------------------------


from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from packaging.version import Version

from nemo_automodel.shared.import_utils import MISSING_CUT_CROSS_ENTROPY_MSG

try:
    import cut_cross_entropy.tl_utils as tl_utils
    from cut_cross_entropy import linear_cross_entropy

    HAVE_CUT_CROSS_ENTROPY = True
except ImportError:  # pragma: no cover
    HAVE_CUT_CROSS_ENTROPY = False  # pragma: no cover


def _get_triton_version():
    for package_name in ("pytorch-triton", "triton"):
        try:
            return metadata_version(package_name), package_name
        except PackageNotFoundError:
            continue

    return None, None


def new_is_triton_greater_or_equal(version_str):
    """
    Check if pytorch-triton/triton version is greater than or equal to the specified version.

    Args:
        version_str: Version string to check

    Returns:
        bool: True if pytorch-triton/triton version >= specified version
    """
    triton_version, package_name = _get_triton_version()
    if triton_version is None:
        print("pytorch-triton/triton not found")
        return False

    current = Version(triton_version)
    required = Version(version_str)
    print(f"Current {package_name} version: {triton_version}, Required triton version: {version_str}")
    return current >= required


def new_is_triton_greater_or_equal_3_2_0():
    """
    Check if pytorch-triton/triton version is greater than or equal to 3.1.0.

    Returns:
        bool: True if pytorch-triton/triton version >= 3.1.0
    """
    return new_is_triton_greater_or_equal("3.1.0")


if HAVE_CUT_CROSS_ENTROPY:
    # Apply the monkey patches
    tl_utils.is_triton_greater_or_equal = new_is_triton_greater_or_equal
    tl_utils.is_triton_greater_or_equal_3_2_0 = new_is_triton_greater_or_equal_3_2_0


class FusedLinearCrossEntropy(nn.Module):
    """Fused linear-projection and cross-entropy loss module."""

    def __init__(self, ignore_index: int = -100, logit_softcapping: float = 0, reduction: str = "sum"):
        """
        Fused linear cross entropy loss.

        Args:
            ignore_index (int): Target value that is ignored when computing the loss. Defaults to -100.
            logit_softcapping (float): Value for softcapping logits (0 means no capping). Defaults to 0.
            reduction (str): Type of reduction. Defaults to "sum".
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.logit_softcapping = logit_softcapping
        self.reduction = reduction

    @staticmethod
    def materialize_lm_weight(
        lm_weight: torch.Tensor,
        *,
        grad_reduce_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Materialize an LM-head DTensor with gradient-correct reduction semantics.

        Fused linear CE consumes the LM-head weight outside the owning FSDP
        module's forward. Each data/context-parallel rank therefore computes a
        rank-local full-weight gradient. A plain ``DTensor.full_tensor()`` marks
        that gradient as replicated, so backward only slices the local result
        into the owned shard instead of combining peer contributions.

        Args:
            lm_weight: LM-head weight with global shape ``[vocab, hidden]``. A
                regular tensor is returned unchanged. A DTensor may have any
                FSDP sharding placement over its device mesh and is gathered to
                a rank-local regular tensor with the global shape, device, and
                dtype.
            grad_reduce_group: Process group whose ranks contribute independent
                token losses. Its size must match the LM-head DTensor mesh.

        Returns:
            Regular tensor with shape ``[vocab, hidden]``. For a DTensor input,
            backward reduce-scatters the averaged peer gradients into the
            original local shard. The gathered result does not alias the local
            DTensor shard; a regular-tensor input is returned by identity.

        Raises:
            ValueError: If a trainable sharded weight has no matching reduction
                group. This fails closed instead of producing rank-local shards.
        """
        if not hasattr(lm_weight, "full_tensor"):
            return lm_weight

        # Evaluation has no weight gradient to combine, so preserve the ordinary
        # gather path and do not require a process group from inference callers.
        if not torch.is_grad_enabled() or not lm_weight.requires_grad:
            return lm_weight.full_tensor()

        mesh = lm_weight.device_mesh
        mesh_world_size = mesh.size()
        reduce_world_size = dist.get_world_size(grad_reduce_group) if grad_reduce_group is not None else 1
        if mesh_world_size != reduce_world_size:
            raise ValueError(
                "FusedLinearCrossEntropy requires grad_reduce_group to match the LM-head "
                f"DTensor mesh: mesh size={mesh_world_size}, reduction group size={reduce_world_size}. "
                "Tensor-parallel or hierarchical layouts need an explicit compatible loss path."
            )
        if mesh_world_size == 1:
            return lm_weight.full_tensor()

        from torch.distributed.tensor import Partial

        # ``Partial`` tells DTensor autograd to reduce-scatter the full gradient
        # directly into the parameter's original FSDP shard. Training recipes
        # scale the local loss by the reduction world size before backward to
        # cancel FSDP's averaged-gradient convention, so restore that average
        # before the reduce-scatter sum.
        full_weight = lm_weight.full_tensor(
            grad_placements=tuple(Partial() for _ in range(mesh.ndim)),
        )
        full_weight.register_hook(lambda grad: grad / reduce_world_size)
        return full_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        lm_weight: torch.Tensor,
        num_label_tokens: Optional[int] = None,
        grad_reduce_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Compute fused linear cross entropy matching PyTorch behavior.

        Args:
            hidden_states: Rank-local hidden states with shape
                ``[batch, sequence, hidden]``.
            labels: Rank-local target token IDs with shape ``[batch, sequence]``.
            lm_weight: LM-head weight with global shape ``[vocab, hidden]``.
                It may be a regular tensor or an FSDP-sharded DTensor.
            num_label_tokens: Global number of non-padding target tokens used
                to normalize a sum-reduced loss.
            grad_reduce_group: Group that contributes independent loss shards
                when ``lm_weight`` is a sharded DTensor.

        Returns:
            Scalar loss tensor on the same device as ``hidden_states``. The
            inputs are not mutated and the result does not alias an input.
        """
        if not HAVE_CUT_CROSS_ENTROPY:
            raise ImportError(MISSING_CUT_CROSS_ENTROPY_MSG)

        lm_weight = self.materialize_lm_weight(
            lm_weight,
            grad_reduce_group=grad_reduce_group,
        )

        # First compute loss with sum reduction to handle normalization ourselves
        if self.logit_softcapping == 0:
            self.logit_softcapping = None

        # cut_cross_entropy fuses e @ c.T in the input precision (with fp32 logsumexp
        # accumulation internally), so e (hidden_states) and c (lm_weight) must share a
        # dtype. Cast the classifier weight to the activation dtype to support models whose
        # lm_head is kept in fp32 while activations are bf16 (e.g. DeepSeek V4).
        if lm_weight.dtype != hidden_states.dtype:
            lm_weight = lm_weight.to(hidden_states.dtype)

        # Compute loss with shift=False to match PyTorch behavior
        # Set filter_eps=None to avoid any token filtering
        loss = linear_cross_entropy(
            hidden_states,
            lm_weight,
            targets=labels,
            ignore_index=self.ignore_index,
            softcap=self.logit_softcapping,
            reduction=self.reduction,  # Use sum reduction to handle normalization ourselves
            shift=False,  # Match PyTorch behavior
            filter_eps=None,  # No token filtering
        )
        if num_label_tokens is not None:
            assert self.reduction == "sum", "num_label_tokens is only supported when reduction is 'sum'"
            loss = loss / num_label_tokens
        return loss
