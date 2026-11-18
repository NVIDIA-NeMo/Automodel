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

"""LM-head configuration and causal model output packaging."""

import sys
from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common.logits.logits_processor import LogitsProcessor
from nemo_automodel.components.models.common.logits.logits_processor import head_compute_dtype as head_compute_dtype

_model_processor = LogitsProcessor()
_fp32_processor = LogitsProcessor(out_dtype=torch.float32)


def compute_lm_head_logits(
    lm_head: nn.Module | torch.Tensor | None,
    hidden_states: torch.Tensor,
    logits_to_keep: int | torch.Tensor = 0,
    is_thd: bool = False,
    output_hidden_states: bool = False,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> CausalLMOutputWithPast:
    """Call the processor using the historical FP32 default for bare weights."""
    if out_dtype not in (None, torch.float32):
        raise ValueError(f"out_dtype must be None or torch.float32; got {out_dtype}.")
    processor = _fp32_processor if out_dtype is torch.float32 or isinstance(lm_head, torch.Tensor) else _model_processor
    return processor.compute_lm_head_logits(
        lm_head, hidden_states, logits_to_keep, is_thd, output_hidden_states, bias=bias
    )


class Float32LogitsLinear(nn.Linear):
    """Linear LM head with ordinary parameter storage and FP32 projection output."""

    fp32_logits = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project through the FP32 kernel after module hooks materialize the weight."""
        return _fp32_processor(self.weight, input, self.bias)


# Older serialized heads reference this forward by its original name.
fp32_linear_forward = Float32LogitsLinear.forward


def enable_fp32_logits(
    head: nn.Module, *, forward: Callable[[nn.Linear, torch.Tensor], torch.Tensor] = fp32_linear_forward
) -> nn.Module:
    """Specialize a vocabulary head in place; retain model-owned projections and tied parameters."""
    if hasattr(head, "fp32_logits"):
        head.fp32_logits = True
        return head
    if type(head) is nn.Linear:
        if "forward" in vars(head):
            raise ValueError("FP32 logits cannot replace an LM head with a custom forward.")
        # Match the repository's TPLinear conversion: preserve existing references and module hooks.
        head.__class__ = Float32LogitsLinear
    elif type(head) in (
        getattr(sys.modules.get("transformer_engine.pytorch.module.linear"), "Linear", None),
        getattr(sys.modules.get("quack.linear"), "Linear", None),
    ):
        linear = Float32LogitsLinear(head.in_features, head.out_features, bias=False, device="meta")
        linear.weight = head.weight
        if isinstance(head.bias, nn.Parameter):
            linear.bias = head.bias
        head = linear.train(head.training)
    else:
        raise ValueError(f"FP32 logits require a plain linear kernel backend; got {type(head).__name__}.")
    if forward is not fp32_linear_forward:
        head.forward = partial(forward, head)
    return head
