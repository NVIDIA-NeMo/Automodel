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

from numbers import Integral
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

_DATASET_IGNORE_INDEX = -100


def _get_loss_ignore_index(loss_fn: object) -> int:
    """Return the label sentinel consumed by ``loss_fn``."""
    ignore_index = getattr(loss_fn, "ignore_index", _DATASET_IGNORE_INDEX)
    return int(ignore_index) if isinstance(ignore_index, Integral) else _DATASET_IGNORE_INDEX


def _normalize_loss_labels(labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Map dataset padding to a loss's configured ignore index.

    Args:
        labels: Integer target tensor of any shape.
        ignore_index: Label sentinel consumed by the loss.

    Returns:
        Target tensor with the same shape, dtype, and device as ``labels``.
        The input is returned unchanged when ``ignore_index`` is ``-100``;
        otherwise a new tensor maps dataset padding from ``-100`` to the
        configured sentinel.
    """
    if ignore_index == _DATASET_IGNORE_INDEX:
        return labels
    return labels.masked_fill(labels == _DATASET_IGNORE_INDEX, ignore_index)


def _normalize_kd_labels(
    labels: torch.Tensor,
    *,
    loss_ignore_index: int,
    kd_ignore_index: int,
) -> torch.Tensor:
    """Align KD supervision with the main loss mask.

    Args:
        labels: Integer target tensor of any shape.
        loss_ignore_index: Label sentinel consumed by the main loss.
        kd_ignore_index: Label sentinel consumed by the KD loss.

    Returns:
        Target tensor with the same shape, dtype, and device as ``labels``.
        Positions ignored by the dataset or main loss contain
        ``kd_ignore_index``. Valid labels equal to ``kd_ignore_index`` are
        remapped to ``loss_ignore_index`` so they remain supervised.
    """
    labels = _normalize_loss_labels(labels, loss_ignore_index)
    valid_mask = labels != loss_ignore_index
    kd_labels = labels.masked_fill(~valid_mask, kd_ignore_index)
    if kd_ignore_index != loss_ignore_index:
        kd_labels = kd_labels.masked_fill(valid_mask & (kd_labels == kd_ignore_index), loss_ignore_index)
    return kd_labels


def _count_label_tokens(labels: torch.Tensor, ignore_index: int) -> int:
    """Count supervised entries in an integer target tensor of any shape.

    Both the dataset's ``-100`` padding and the configured ``ignore_index``
    are excluded. The return value is a Python integer.
    """
    valid = labels != _DATASET_IGNORE_INDEX
    if ignore_index != _DATASET_IGNORE_INDEX:
        valid = valid & (labels != ignore_index)
    return int(valid.sum().item())


def _get_lm_head_module(model: nn.Module) -> nn.Module | None:
    """Return the model's LM-head module, if one can be found.

    Local copy of ``components.utils.model_utils.get_lm_head_module`` to keep
    ``components/loss/`` import-independent from ``components/utils/`` (see the
    ``Components must not import each other`` import-linter contract).
    """
    if hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            return lm_head
    for name, module in model.named_modules():
        if (name == "lm_head" or name.endswith(".lm_head")) and hasattr(module, "weight"):
            return module
    return None


def _get_lm_head_weight(model: nn.Module) -> torch.Tensor:
    """Return the model's LM-head weight without changing its distributed layout."""
    lm_head = _get_lm_head_module(model)
    if lm_head is not None:
        return lm_head.weight
    for name, param in model.named_parameters(remove_duplicate=False):
        if "lm_head" in name and name.endswith(".weight"):
            return param
    raise ValueError("lm_head.weight not found in model")


def _get_final_hidden_states(model_output: Any) -> Any | None:
    """Return the final hidden-states tensor from an HF-like model output.

    Local copy of ``components.training.model_output_utils.get_final_hidden_states``
    to keep ``components/loss/`` import-independent from ``components/training/``.
    """
    if model_output is None:
        return None
    if isinstance(model_output, dict):
        hidden_states = model_output.get("hidden_states", None)
    else:
        hidden_states = getattr(model_output, "hidden_states", None)
    if hidden_states is None:
        return None
    if isinstance(hidden_states, (list, tuple)):
        for item in reversed(hidden_states):
            if item is not None:
                return item
        return None
    return hidden_states


def calculate_loss(loss_fn: nn.Module, **kwargs: Any) -> torch.Tensor:
    """Calculate a logit-based or fused linear cross-entropy loss.

    Args:
        loss_fn: Loss module. ``FusedLinearCrossEntropy`` consumes
            ``hidden_states`` with shape ``[batch, sequence, hidden]``, labels
            with shape ``[batch, sequence]``, and an LM-head weight with global
            shape ``[vocab, hidden]``. Other loss modules consume logits with
            shape ``[batch, sequence, vocab]`` and labels.
        **kwargs: Loss inputs. Rank-local tensors keep their existing layout;
            ``grad_reduce_group`` describes the ranks contributing independent
            fused-loss shards. The caller's mapping and tensors are not mutated.

    Returns:
        Scalar loss tensor that does not alias an input.
    """
    loss_fn_kwargs = {"num_label_tokens": kwargs.pop("num_label_tokens", None)}
    labels = _normalize_loss_labels(kwargs.pop("labels"), _get_loss_ignore_index(loss_fn))
    loss_weights = kwargs.pop("loss_weights", None)
    if loss_weights is not None:
        loss_fn_kwargs["loss_weights"] = loss_weights
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        # Reuse a caller-materialized LM head when provided so a single
        # full_tensor() all-gather is shared across the main loss and every MTP
        # depth (see calculate_mtp_loss). Re-gathering the (vocab x hidden) head
        # per call leaves a copy retained for backward each time; they accumulate
        # on-device and OOM large-vocab MoE (e.g. Nemotron-Ultra, 256k vocab).
        lm_head = kwargs.pop("lm_weight", None)
        if lm_head is None:
            lm_head = _get_lm_head_weight(model)
        loss_fn_kwargs.update(
            {
                "hidden_states": kwargs.pop("hidden_states"),
                "labels": labels,
                "lm_weight": lm_head,
                "grad_reduce_group": kwargs.pop("grad_reduce_group", None),
            }
        )
    else:
        kwargs.pop("lm_weight", None)  # logit-based losses do not need the LM head
        kwargs.pop("grad_reduce_group", None)
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": labels,
            }
        )

    return loss_fn(**loss_fn_kwargs)
