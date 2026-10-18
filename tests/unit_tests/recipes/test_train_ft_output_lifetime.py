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

import copy
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


@pytest.mark.parametrize("output_kind", ["tensor", "hf", "mtp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("recompute", [False, True])
def test_recipe_releases_unused_outputs_before_backward(output_kind, dtype, recompute):
    """Releasing CE logits preserves loss, accumulated gradients, norm and updates."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(19, 8, dtype=dtype)
            self.lm_head = torch.nn.Linear(8, 19, dtype=dtype)
            self.logit_refs = []

        def forward(self, input_ids):
            """Project tokens, with the same FP32 output upcast as Nemotron-H.

            Args:
                input_ids: Integer tensor of shape [batch, sequence].

            Returns:
                FP32 logits of shape [batch, sequence, vocab], directly or in
                an output object. MTP adds one logits tensor of the same shape.
            """
            hidden = self.embed(input_ids)
            logits = (
                checkpoint(self.lm_head, hidden, use_reentrant=False) if recompute else self.lm_head(hidden)
            ).float()
            self.logit_refs.append(weakref.ref(logits))
            if output_kind == "tensor":
                return logits
            if output_kind == "hf":
                return CausalLMOutput(logits=logits)
            auxiliary = self.lm_head(hidden * 0.5).float()
            self.logit_refs.append(weakref.ref(auxiliary))
            return SimpleNamespace(logits=logits, mtp_per_depth_logits=[auxiliary], mtp_loss_scaling_factor=0.3)

    torch.manual_seed(4180)
    model = Model()
    reference = copy.deepcopy(model)
    loss_fn = MaskedCrossEntropy()
    alive_at_backward = []

    def record_lifetime(module, inputs, loss):
        """Observe output lifetime at the beginning of each CE backward.

        Args:
            module: The loss module.
            inputs: Logits [batch, sequence, vocab] and labels [batch, sequence].
            loss: Scalar normalized cross entropy tensor.
        """

        def record(grad):
            """Observe live logits without retaining them.

            Args:
                grad: Scalar loss gradient.
            """
            alive_at_backward.append([ref() is not None for ref in model.logit_refs])

        loss.register_hook(record)

    handle = loss_fn.register_forward_hook(record_lifetime)
    recipe = SimpleNamespace(
        model_parts=[model],
        loss_fn=loss_fn,
        dist_env=SimpleNamespace(device=torch.device("cpu")),
        pp_enabled=False,
        device_mesh=None,
        tokenizer=None,
        domain_mixture=None,
        te_fp8=None,
        cfg=SimpleNamespace(mtp=SimpleNamespace(scaling_factor=None)),
        distributed_config=SimpleNamespace(defer_fsdp_grad_sync=True),
        _get_cp_group_size=lambda: 1,
        _get_dp_group_size=lambda **kw: 1,
    )
    try:
        for microbatch in range(2):
            input_ids = torch.randint(19, (2, 7))
            labels = torch.randint(19, (2, 7))
            labels[:, :2] = -100
            count = (labels != -100).sum()
            output = reference(input_ids)
            logits = output if output_kind == "tensor" else output.logits
            expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum") / count
            if output_kind == "mtp":
                shifted = labels.roll(-1, -1)
                shifted[:, -1] = -100
                expected = (
                    expected
                    + 0.3
                    * F.cross_entropy(output.mtp_per_depth_logits[0].flatten(0, 1), shifted.flatten(), reduction="sum")
                    / count
                )
            expected.backward()
            losses = []
            TrainFinetuneRecipeForNextTokenPrediction._forward_backward_step(
                recipe,
                microbatch,
                {"input_ids": input_ids, "labels": labels},
                loss_buffer=losses,
                num_label_tokens=count,
                num_batches=2,
            )
            torch.testing.assert_close(losses[0], expected, rtol=0, atol=0)
            for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)
    finally:
        handle.remove()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    torch.testing.assert_close(norm, ref_norm, rtol=0, atol=0)
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    torch.optim.SGD(reference.parameters(), lr=0.1).step()
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, ref_parameter, rtol=0, atol=0)
    assert alive_at_backward
    assert not any(any(alive) for alive in alive_at_backward), "Unused logits survived into backward"
