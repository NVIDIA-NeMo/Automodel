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

from typing import Optional

import torch

from nemo_automodel.components.loss.triton.te_cross_entropy import (
    HAVE_TRITON,
    cross_entropy_backward,
    cross_entropy_forward,
)
from nemo_automodel.shared.import_utils import MISSING_TRITON_MSG

HAVE_TE_PARALLEL_CE = HAVE_TRITON
MISSING_TE_PARALLEL_CE_MSG = MISSING_TRITON_MSG


"""Cross Entropy Loss API from NVIDIA's TransformerEngine, available under the Apache License 2.0:
https://github.com/NVIDIA/TransformerEngine"""


class CrossEntropyFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Cross Entropy loss. The input tensor can be in BF16/FP32, the
    loss and gradient calculation happens in FP32 only. The returned loss is always in FP32, the input gradients are upcasted
    to the dataype of the input.
    """

    @staticmethod
    def forward(
        ctx,
        _input,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
    ):
        """
        The forward pass of the Cross Entropy loss. If dist_process_group is passed for distributed loss calculation, the input to each
        distributed rank should be (*,V/world_size). Note that each of the ranks should get equal shards along the V dimension.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor of shape (B, SQ, V) or (SQ, B, V) where B is batch size, SQ is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (B,SQ) or (SQ, B) where each value is in [0, V-1].
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduce_loss (bool): If true, returns the averaged loss across the B*SQ dimension.
        dist_process_group (torch.dist.ProcessGroup): The distributed process group the loss computation is split across, None if on 1 device.
        ignore_idx (int): The index for which loss and gradients are made to zero

        Returns:
        tensor: The computed loss.
        """
        loss, _input = cross_entropy_forward(
            _input, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx
        )

        ctx.save_for_backward(_input.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass of the Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.

        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        (_input,) = ctx.saved_tensors
        _input = cross_entropy_backward(_input, grad_output)
        return (
            _input,
            None,
            None,
            None,
            None,
            None,  # Modified original TransformerEngine version to return None for ignore_idx argument
        )


parallel_cross_entropy = CrossEntropyFunction.apply


class TEParallelCrossEntropy:
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "sum",
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """
        Cross entropy loss module based on TransformerEngine's parallel cross entropy triton kernel.

        Args:
            ignore_index (int): Target value that is ignored when computing the loss. Defaults to -100.
            reduction (str): Type of reduction ('none', 'mean', 'sum'). Defaults to "mean".
            tp_group (Optional[torch.distributed.ProcessGroup]): Process group for tensor parallelism. Defaults to None.
        """
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.tp_group = tp_group

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute parallel cross entropy loss that matches PyTorch's cross_entropy behavior.

        Args:
            logits: Input logits. Shape: [B, T, V]
            labels: Target labels. Shape: [B, T]
            mask: Mask to apply to the loss. Shape: [B, T]

        Returns:
            Computed loss tensor
        """
        if not HAVE_TE_PARALLEL_CE:
            raise ImportError(MISSING_TE_PARALLEL_CE_MSG)

        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)
                labels.masked_fill_(mask == 0, self.ignore_index)
                del mask

        reduce_loss = self.reduction == "mean"

        # Compute TE parallel cross entropy
        te_loss = parallel_cross_entropy(logits, labels, 0.0, reduce_loss, self.tp_group, self.ignore_index)

        # Apply reduction
        if self.reduction == "none" or self.reduction == "mean":
            return te_loss
        elif self.reduction == "sum":
            return te_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}. Must be one of 'none', 'mean', 'sum'")
