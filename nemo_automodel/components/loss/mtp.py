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

import torch
import torch.nn as nn

from nemo_automodel.components.models.common.mtp import get_mtp_loss_scaling_factor
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states


class PipelineCausalLMLoss(nn.Module):
    """Pipeline schedule loss that can add MTP auxiliary CE on the last stage."""

    def __init__(self, loss_fn: nn.Module, model: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model

    def forward(self, output, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(output, tuple):
            logits = output[0]
            hidden_states = None
            mtp_per_depth_h = list(output[1:]) if len(output) > 1 else None
            scaling_factor = get_mtp_loss_scaling_factor(self.model)
        else:
            logits = getattr(output, "logits", output)
            hidden_states = get_final_hidden_states(output)
            mtp_per_depth_h = getattr(output, "mtp_per_depth_h", None)
            scaling_factor = getattr(output, "mtp_loss_scaling_factor", get_mtp_loss_scaling_factor(self.model))

        from nemo_automodel.recipes.llm.train_ft import calculate_loss, calculate_mtp_loss

        loss = calculate_loss(
            self.loss_fn,
            logits=logits,
            labels=labels,
            model=self.model,
            hidden_states=hidden_states,
        )
        if mtp_per_depth_h is not None and self.model.training:
            loss = loss + calculate_mtp_loss(
                self.loss_fn,
                mtp_per_depth_h=mtp_per_depth_h,
                labels=labels,
                model=self.model,
                scaling_factor=scaling_factor,
            )
        return loss
