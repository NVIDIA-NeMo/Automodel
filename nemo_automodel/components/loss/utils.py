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

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy


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
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")
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
                "labels": kwargs.pop("labels"),
            }
        )

    return loss_fn(**loss_fn_kwargs)
