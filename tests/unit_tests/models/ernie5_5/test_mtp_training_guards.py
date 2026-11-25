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

"""Input guards and activation-checkpoint dispatcher behavior for ERNIE MTP."""

import pytest
import torch

from nemo_automodel._transformers.capabilities import ModelSupports
from nemo_automodel.components.distributed.activation_checkpointing import (
    unwrap_checkpoint_wrapper,
)
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.ernie5_5.test_mtp_training_local import batch, model

pytestmark = pytest.mark.cpu_only


def test_actual_apply_ac_skips_shared_body_and_preserves_gradients() -> None:
    reference, checkpointed = model(), model()
    checkpointed.load_state_dict(reference.state_dict())
    shared = checkpointed.model.layers[-1]
    apply_ac(checkpointed, ignore_router=True)
    assert unwrap_checkpoint_wrapper(checkpointed.model.layers[0]) is not checkpointed.model.layers[0]
    assert checkpointed.model.layers[-1] is shared
    assert unwrap_checkpoint_wrapper(shared) is shared
    tokens, _labels = batch()
    for candidate in [reference, checkpointed]:
        output = candidate(tokens)
        (output.logits.float().square().mean() + output.mtp_per_depth_logits[0].float().square().mean()).backward()
    expected = dict(reference.named_parameters())
    for raw_name, parameter in checkpointed.named_parameters():
        name = raw_name.replace("_checkpoint_wrapped_module.", "")
        gradient = expected[name].grad
        if gradient is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("argument", ["document_ids", "cp_group"])
def test_inner_callback_rejects_packing_and_context_parallelism(argument: str) -> None:
    candidate = model()
    tokens, _ = batch()
    metadata = torch.zeros_like(tokens) if argument == "document_ids" else object()
    with pytest.raises(ValueError, match="unpadded single document and CP1"):
        candidate.model(tokens, **{argument: metadata}, mtp_forward=lambda streams, inputs: streams)


def test_runtime_capabilities_reject_cp_and_packing() -> None:
    candidate = model()
    supports = ModelSupports(candidate)
    assert not supports.supports_cp
    assert not supports.supports_sequence_packing
    assert not supports.supports_thd
