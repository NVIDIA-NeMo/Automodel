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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from nemo_automodel.components.loss.te_parallel_ce import (
    HAVE_TE_PARALLEL_CE,
    parallel_cross_entropy,
)


class MaskedCrossEntropy(nn.Module):
    def __init__(self, fp32_upcast: bool = True, ignore_index: int = -100, reduction: str = "sum"):
        """
        Masked cross-entropy loss.

        Args:
            fp32_upcast (bool): if True it will cast logits to float32 before computing
                cross entropy. Default: True.
            ignore_index (int): label to ignore in CE calculation. Defaults to -100.
            reduction (str): type of reduction. Defaults to "sum".
        """
        super().__init__()
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction

    def _try_tp_parallel_ce(
        self,
        logits: DTensor,
        labels: torch.Tensor,
        num_label_tokens: Optional[int],
    ) -> Optional[torch.Tensor]:
        """Run TP-aware CE when logits are sharded on vocab dim."""
        enable_tp_parallel_ce = os.environ.get("NEMOTRONH_TP_PARALLEL_CE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not enable_tp_parallel_ce:
            return None

        # Find TP mesh dimension that shards the last (vocab) dimension.
        vocab_shard_mesh_dim = None
        for mesh_dim, placement in enumerate(logits.placements):
            if isinstance(placement, Shard) and placement.dim in (-1, logits.ndim - 1):
                vocab_shard_mesh_dim = mesh_dim
                break

        if vocab_shard_mesh_dim is None:
            return None

        tp_group = logits.device_mesh.get_group(vocab_shard_mesh_dim)
        impl = os.environ.get("NEMOTRONH_TP_PARALLEL_CE_IMPL", "native").strip().lower()
        if impl in ("te", "transformer_engine"):
            return self._tp_parallel_ce_te(logits, labels, num_label_tokens, tp_group)
        if impl in ("native", "auto"):
            # Native path is the default because it has been more stable for NemotronH+TP2.
            return self._tp_parallel_ce_native(logits, labels, num_label_tokens, tp_group)
        if impl == "off":
            return None
        raise ValueError(f"Unknown NEMOTRONH_TP_PARALLEL_CE_IMPL={impl!r}")

    def _tp_parallel_ce_te(
        self,
        logits: DTensor,
        labels: torch.Tensor,
        num_label_tokens: Optional[int],
        tp_group,
    ) -> Optional[torch.Tensor]:
        if not HAVE_TE_PARALLEL_CE:
            return None

        local_logits = logits.to_local()
        if self.fp32_upcast:
            local_logits = local_logits.float()

        if isinstance(labels, DTensor):
            labels = labels.full_tensor()

        reduce_loss = self.reduction == "mean"
        tp_loss = parallel_cross_entropy(local_logits, labels, 0.0, reduce_loss, tp_group, self.ignore_index)

        if self.reduction in ("none", "mean"):
            return tp_loss
        if self.reduction == "sum":
            loss = tp_loss.sum()
            if num_label_tokens is not None:
                loss = loss / num_label_tokens
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")

    def _tp_parallel_ce_native(
        self,
        logits: DTensor,
        labels: torch.Tensor,
        num_label_tokens: Optional[int],
        tp_group,
    ) -> Optional[torch.Tensor]:
        local_logits = logits.to_local()
        if self.fp32_upcast:
            local_logits = local_logits.float()

        if isinstance(labels, DTensor):
            labels = labels.full_tensor()
        if labels.ndim != local_logits.ndim - 1:
            labels = labels.view(local_logits.shape[:-1])

        labels_flat = labels.reshape(-1)
        logits_flat = local_logits.view(-1, local_logits.size(-1))

        valid = labels_flat != self.ignore_index
        safe_labels = torch.where(valid, labels_flat, torch.zeros_like(labels_flat))

        local_shape, global_offset = compute_local_shape_and_global_offset(
            logits.shape,
            logits.device_mesh,
            logits.placements,
        )
        vocab_local = local_shape[-1]
        vocab_start = global_offset[-1]
        vocab_end = vocab_start + vocab_local

        in_local_shard = (safe_labels >= vocab_start) & (safe_labels < vocab_end) & valid
        local_target_idx = (safe_labels - vocab_start).clamp(min=0, max=vocab_local - 1)

        local_target_logits = logits_flat.gather(1, local_target_idx.unsqueeze(1)).squeeze(1)
        neg_inf = torch.full_like(local_target_logits, float("-inf"))
        local_target_logits = torch.where(
            in_local_shard,
            local_target_logits,
            torch.where(valid, neg_inf, torch.zeros_like(local_target_logits)),
        )

        global_target_logits = dist_nn_f.all_reduce(local_target_logits, op=dist.ReduceOp.MAX, group=tp_group)
        global_target_logits = torch.where(valid, global_target_logits, torch.zeros_like(global_target_logits))

        local_max = logits_flat.max(dim=-1).values
        global_max = dist_nn_f.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
        local_exp_sum = torch.exp(logits_flat - global_max.unsqueeze(-1)).sum(dim=-1)
        global_exp_sum = dist_nn_f.all_reduce(local_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)

        token_loss = -(global_target_logits - global_max - torch.log(global_exp_sum))
        token_loss = torch.where(valid, token_loss, torch.zeros_like(token_loss))

        debug_enabled = os.environ.get("NEMOTRONH_TP_PARALLEL_CE_DEBUG", "0").strip() in ("1", "true", "yes", "on")
        if debug_enabled:
            tp_group_size = dist.get_world_size(tp_group)
            label_checksum = labels_flat.to(torch.int64).sum()
            checksum_list = [torch.zeros_like(label_checksum) for _ in range(tp_group_size)]
            dist.all_gather(checksum_list, label_checksum, group=tp_group)
            if dist.get_rank() == 0 and not hasattr(self, "_tp_parallel_ce_debug_printed"):
                self._tp_parallel_ce_debug_printed = True
                max_label = safe_labels.max().item() if safe_labels.numel() > 0 else -1
                local_cov = int(in_local_shard.sum().item())
                valid_cnt = int(valid.sum().item())
                local_nonfinite = int((~torch.isfinite(logits_flat)).sum().item())
                local_loss_nonfinite = int((~torch.isfinite(token_loss)).sum().item())
                local_max_abs = float(
                    torch.nan_to_num(logits_flat, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()
                )
                label_checksums = [int(x.item()) for x in checksum_list]
                print(
                    f"[tp-ce-native] vocab_local={vocab_local} vocab_start={vocab_start} "
                    f"max_label={max_label} local_covered={local_cov}/{valid_cnt} "
                    f"logits_nonfinite_local={local_nonfinite} "
                    f"loss_nonfinite_local={local_loss_nonfinite} "
                    f"local_max_abs={local_max_abs:.4f} "
                    f"tp_group_size={tp_group_size} "
                    f"tp_label_checksums={label_checksums} "
                    f"mesh={tuple(logits.device_mesh.shape)} placements={logits.placements}"
                )

        if self.reduction == "none":
            return token_loss.view_as(labels)
        if self.reduction == "mean":
            denom = (
                torch.as_tensor(num_label_tokens, device=token_loss.device, dtype=token_loss.dtype)
                if num_label_tokens is not None
                else valid.sum().to(token_loss.dtype).clamp_min(1.0)
            )
            return token_loss.sum() / denom
        if self.reduction == "sum":
            loss = token_loss.sum()
            if num_label_tokens is not None:
                loss = loss / torch.as_tensor(num_label_tokens, device=loss.device, dtype=loss.dtype)
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the masked cross-entropy loss between logits and targets.

        If a mask is provided, the loss is computed per element, multiplied by the mask,
        and then averaged. If no mask is provided, the standard cross-entropy loss is used.

        Args:
            logits (torch.Tensor): The predicted logits with shape [batch_size, seq_len, vocab_size] where C is the number of classes.
            labels (torch.Tensor): The ground truth class indices with shape [batch_size, seq_len].
            mask (torch.Tensor, optional): A tensor that masks the loss computation. Items marked with
                1 will be used to calculate loss, otherwise ignored. Must be broadcastable to the shape
                of the loss. Defaults to None.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        # this may happen with CPUOffloadPolicy
        if labels.device != logits.device:
            labels = labels.to(logits.device)  # pragma: no cover

        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)  # pragma: no cover
                labels.masked_fill_(mask == 0, self.ignore_index)
                del mask

        if isinstance(logits, DTensor):
            tp_loss = self._try_tp_parallel_ce(logits, labels, num_label_tokens)
            if tp_loss is not None:
                return tp_loss

        # reshape to (N, C) and (N,) respectively
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        if self.fp32_upcast:
            logits = logits.float()

        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        if isinstance(labels, DTensor):
            labels = labels.full_tensor()

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        if num_label_tokens is not None:
            assert self.reduction == "sum", "num_label_tokens is only supported when reduction is 'sum'"
            loss = loss / num_label_tokens
        return loss
