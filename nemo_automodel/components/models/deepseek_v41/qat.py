# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Native V4.1 operand QAT; master parameters and checkpoint names stay unchanged."""

import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4GroupedLinear
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_native
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
from nemo_automodel.components.moe.layers import MLP


class NativeQATLinear(torch.nn.Linear):
    """Dense 32x32-weight / rowwise-activation QAT."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate a dense W8A8 projection with STE operands.

        Args:
            values: Tensor [..., in_features], in the compute dtype.

        Returns:
            Tensor [..., out_features], in the compute dtype.
        """
        activations = quantize_native(values.reshape(-1, values.shape[-1])).reshape_as(values)
        weight = quantize_native(self.weight.to(values.dtype), weight=True)
        return F.linear(activations, weight, self.bias)


class NativeQATGroupedLinear(DeepseekV4GroupedLinear):
    """Grouped attention projection with native operand QAT."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate the grouped attention output projection.

        Args:
            values: Tensor [..., groups, in_features_per_group].

        Returns:
            Tensor [..., groups, out_features_per_group], in the input dtype.
        """
        shape = values.shape
        activations = quantize_native(values.reshape(-1, shape[-1])).reshape(-1, self.n_groups, shape[-1])
        weight = quantize_native(self.weight.to(values.dtype), weight=True)
        weight = weight.reshape(self.n_groups, -1, shape[-1])
        output = torch.bmm(activations.transpose(0, 1), weight.transpose(-1, -2)).transpose(0, 1)
        return output.reshape(*shape[:-1], self.out_features // self.n_groups)


class NativeQATMLP(MLP):
    """Shared expert QAT preserving the released SwiGLU rounding."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate shared-expert W8A8 with the existing clamped SwiGLU order.

        Args:
            values: Tensor [..., hidden], in the compute dtype.

        Returns:
            Tensor [..., hidden], with STE gradients to all three weights.
        """
        activations = quantize_native(values.reshape(-1, values.shape[-1])).reshape_as(values)
        weight = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0).to(values.dtype)
        gate, up = F.linear(activations, quantize_native(weight, weight=True)).chunk(2, dim=-1)
        gate = gate.float().clamp(max=self.swiglu_limit)
        up = up.float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        hidden = (F.silu(gate).to(values.dtype).float() * up).to(values.dtype)
        hidden = quantize_native(hidden.reshape(-1, hidden.shape[-1])).reshape_as(hidden)
        return F.linear(hidden, quantize_native(self.down_proj.weight.to(values.dtype), weight=True))


class NativeQATExpertsDeepEP(GroupedExpertsDeepEP):
    """Routed expert W4A8 compute between flex-dispatcher (HybridEP) dispatch and combine."""

    def compute_grouped(
        self,
        hidden_states: torch.Tensor,
        gate_and_up_projs: torch.Tensor,
        down_projs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate routed W4A8 on the tokens this rank received.

        Args:
            hidden_states: Tensor [tokens, hidden], sorted by local expert.
            gate_and_up_projs: Tensor [local_experts, hidden, 2 * intermediate].
            down_projs: Tensor [local_experts, intermediate, hidden].
            tokens_per_expert: Integer tensor [local_experts].
            permuted_probs: Tensor [tokens, 1]; the dispatch supplies ones
                when routing weights are applied after the down projection.

        Returns:
            Tensor [tokens, hidden]. Quantization runs on this rank's complete
            expert matrices, after EP ownership selection.
        """
        offsets = tokens_per_expert.cumsum(0).to(torch.int32)
        # AM stores [input, output]; checkpoint MXFP4 groups INPUT channels.
        gate_up = quantize_native(gate_and_up_projs.transpose(-1, -2), fp4=True, weight=True).transpose(-1, -2)
        down = quantize_native(down_projs.transpose(-1, -2), fp4=True, weight=True).transpose(-1, -2)
        hidden = torch._grouped_mm(quantize_native(hidden_states), gate_up, offsets)
        hidden = self.expert_activation(hidden, permuted_probs)
        return torch._grouped_mm(quantize_native(hidden), down, offsets)
