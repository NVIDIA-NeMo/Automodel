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

"""Cross-class loss dispatcher.

Builds the right per-loss-class kwarg shape (FusedLinearCrossEntropy needs
hidden_states + lm_weight; everything else takes logits + labels) and invokes
the loss function. Used by recipes and by the LLM MTP auxiliary path.
"""

from __future__ import annotations

import torch

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy


def calculate_loss(loss_fn, **kwargs) -> torch.Tensor:
    """Calculate the loss.

    Args:
        loss_fn: Loss function (e.g. FusedLinearCrossEntropy, MaskedCrossEntropy).
        **kwargs: Keyword arguments for the loss function. Required keys depend on
            the loss class:

            - FusedLinearCrossEntropy: ``model``, ``labels``, ``hidden_states``,
              optionally ``num_label_tokens``. The function locates the shared
              LM head weight on the model and unshards it if needed.
            - Other (e.g. MaskedCrossEntropy): ``logits``, ``labels``, optionally
              ``num_label_tokens``.

    Returns:
        Scalar loss tensor with autograd graph.
    """
    loss_fn_kwargs = {"num_label_tokens": kwargs.pop("num_label_tokens", None)}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")

        # find the lm_head in the model
        lm_head = None
        if hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings().weight
        else:
            for n, p in model.named_parameters(remove_duplicate=False):
                if "lm_head" in n and n.endswith(".weight"):
                    lm_head = p
                    break
        if lm_head is None:
            raise ValueError("lm_head.weight not found in model")

        # unshard the possibly sharded lm_head
        lm_head = lm_head.full_tensor() if hasattr(lm_head, "full_tensor") else lm_head
        loss_fn_kwargs.update(
            {
                "hidden_states": kwargs.pop("hidden_states"),
                "labels": labels,
                "lm_weight": lm_head,
            }
        )
    else:
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
            }
        )

    return loss_fn(**loss_fn_kwargs)


__all__ = ["calculate_loss"]
