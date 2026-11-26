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
from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.distributed.activation_checkpointing import (
    unwrap_checkpoint_wrapper,
)
from nemo_automodel.components.moe.parallelizer import apply_ac
from nemo_automodel.components.utils.model_utils import FreezeConfig
from tests.unit_tests.models.ernie5_5.test_lora import _install_lora, _recipe_targets
from tests.unit_tests.models.ernie5_5.test_mtp_training_local import OWNED_PARAMETERS, batch, model

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


def _two_rows() -> dict[str, torch.Tensor]:
    return {"input_ids": torch.arange(1, 17).reshape(2, 8)}


def _pad_token() -> dict[str, torch.Tensor]:
    tokens = batch()[0].clone()
    tokens[0, 3] = 0
    return {"input_ids": tokens}


def _masked_token() -> dict[str, torch.Tensor]:
    mask = torch.ones(1, 8, dtype=torch.int64)
    mask[0, -1] = 0
    return {"input_ids": batch()[0], "attention_mask": mask}


def _padding_mask() -> dict[str, torch.Tensor]:
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, -1] = True
    return {"input_ids": batch()[0], "padding_mask": mask}


def _restarted_positions() -> dict[str, torch.Tensor]:
    return {"input_ids": batch()[0], "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])}


@pytest.mark.parametrize(
    "arguments,message",
    [
        (_two_rows, r"Expected one nonempty document with input_ids \[1, sequence\]; got \(2, 8\)"),
        (_pad_token, "Padding is unsupported: input_ids contain pad_token_id=0"),
        (_masked_token, "Padding is unsupported: attention_mask must be one for every token"),
        (_padding_mask, "Padding is unsupported: padding_mask must be zero for every token"),
        (_restarted_positions, "Only contiguous zero-based position_ids are supported"),
    ],
    ids=["two_rows", "pad_token", "masked_token", "padding_mask", "restarted_positions"],
)
def test_mtp_forward_rejects_padded_and_multi_document_rows(arguments, message: str) -> None:
    """The MTP forward itself must refuse every row that is not one unpadded document.

    The recipe drops ``seq_lens`` and ``packed_seq_ids`` before the call, so this
    check is what keeps padded or packed rows out of the auxiliary path.
    """
    candidate = model()
    assert candidate.training
    with pytest.raises(ValueError, match=message):
        candidate(**arguments())


def test_training_forward_rejects_a_frozen_mtp_parameter_and_accepts_the_full_model() -> None:
    candidate = model()
    tokens, _ = batch()
    assert {name for name, parameter in candidate.named_parameters() if name.startswith("mtp.")} == OWNED_PARAMETERS
    assert candidate(tokens).mtp_per_depth_logits[0].shape == (1, 8, 32)
    _apply_trainability_policy(
        candidate,
        peft_enabled=False,
        freeze_config=FreezeConfig(freeze_parameters=("mtp.h_proj.weight",)),
        strict=True,
    )
    assert not candidate.mtp.h_proj.weight.requires_grad
    message = (
        r"full-parameter training only: LoRA \(peft\) and a freeze_config that freezes the MTP tensors "
        r"are not supported with Ernie5_5MTPForCausalLM; frozen: \['mtp\.h_proj\.weight'\]"
    )
    with pytest.raises(ValueError, match=message):
        candidate(tokens)
    # Evaluation computes the main logits only and is not restricted.
    assert candidate.eval()(tokens).mtp_per_depth_logits is None


def test_training_forward_rejects_lora_adapters() -> None:
    candidate = model()
    _install_lora(candidate, PeftConfig(target_modules=_recipe_targets(), dim=2, alpha=4))
    trainable = {name for name, parameter in candidate.named_parameters() if parameter.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    frozen = sorted(OWNED_PARAMETERS)
    with pytest.raises(ValueError, match=r"LoRA \(peft\).*frozen: ") as caught:
        candidate(batch()[0])
    assert all(repr(name) in str(caught.value) for name in frozen)


def test_trainable_mtp_guard_leaves_logits_and_gradients_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the check disabled, the same model gives bit-identical outputs and gradients."""
    guarded, unguarded = model(), model()
    unguarded.load_state_dict(guarded.state_dict())
    tokens, _ = batch()
    results = []
    for candidate, disable in ((guarded, False), (unguarded, True)):
        if disable:
            monkeypatch.setattr(type(candidate), "_require_trainable_mtp", lambda self: None)
        output = candidate(tokens)
        (output.logits.float().square().mean() + output.mtp_per_depth_logits[0].float().square().mean()).backward()
        results.append(output)
    torch.testing.assert_close(results[0].logits, results[1].logits, rtol=0, atol=0)
    torch.testing.assert_close(results[0].mtp_per_depth_logits[0], results[1].mtp_per_depth_logits[0], rtol=0, atol=0)
    expected = dict(unguarded.named_parameters())
    for name, parameter in guarded.named_parameters():
        if parameter.grad is None:
            assert expected[name].grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected[name].grad, rtol=0, atol=0, msg=name)
