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

"""Activation checkpointing of NemotronV3 MoE blocks: MoE-only mode (``mlp`` aliases ``mixer``) and the weight-tied MTP block."""

import copy

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.parallelizer import _get_moe_module, apply_ac
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_model import MockNemotronV3Config

pytestmark = pytest.mark.cpu_only


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def test_moe_only_wraps_the_registered_moe_child() -> None:
    config = MockNemotronV3Config(layers_block_type=["mamba", "attention", "moe"], num_hidden_layers=3)
    model = NemotronHForCausalLM(config, backend=_backend())

    apply_ac(model, moe_only=True, hidden_size=config.hidden_size, num_experts=config.n_routed_experts)

    mamba_block, attention_block, moe_block = model.model.layers.values()
    assert all(isinstance(block, NemotronV3Block) for block in (mamba_block, attention_block, moe_block))
    # ``mlp`` is a property that aliases ``mixer``: the wrapper has to replace the registered child.
    assert isinstance(moe_block.mixer, CheckpointWrapper)
    assert moe_block.mlp is moe_block.mixer
    assert isinstance(_get_moe_module(moe_block), MoE)
    assert not isinstance(mamba_block.mixer, CheckpointWrapper)
    assert not isinstance(attention_block.mixer, CheckpointWrapper)


def test_moe_only_keeps_loss_and_gradients() -> None:
    """One CPU forward/backward with and without the MoE-only checkpoint: loss and gradients are bitwise equal."""
    config = MockNemotronV3Config(layers_block_type=["attention", "moe"], torch_dtype="float32")
    torch.manual_seed(0)
    reference = NemotronHForCausalLM(config, backend=_backend()).to(torch.float32)
    # The constructor leaves the routed-expert weights at zero; without this the expert gradients would be zero too.
    reference.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    reference.train()
    checkpointed = copy.deepcopy(reference)
    apply_ac(checkpointed, moe_only=True, hidden_size=config.hidden_size, num_experts=config.n_routed_experts)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(3))

    def loss_and_grads(model: NemotronHForCausalLM) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1))
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ref_loss, ref_grads = loss_and_grads(reference)
    loss, grads = loss_and_grads(checkpointed)

    assert torch.equal(loss, ref_loss)
    # The wrapper adds one path component to the names of the MoE parameters.
    assert {name.replace("._checkpoint_wrapped_module", "") for name in grads} == set(ref_grads)
    assert all(grad.abs().sum() > 0 for name, grad in ref_grads.items() if ".experts." in name)
    for name, grad in grads.items():
        assert torch.equal(grad, ref_grads[name.replace("._checkpoint_wrapped_module", "")]), name


MTP_DEPTHS = 2


def _model() -> tuple[NemotronHForCausalLM, MockNemotronV3Config]:
    """Two backbone blocks and one physical ``[attention, moe]`` MTP depth that runs ``MTP_DEPTHS`` times."""
    config = MockNemotronV3Config(
        layers_block_type=["attention", "moe"],
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        torch_dtype="float32",
    )
    backend = _backend()
    torch.manual_seed(0)
    model = NemotronHForCausalLM(
        config,
        backend=backend,
        num_nextn_predict_layers=MTP_DEPTHS,
        mtp_use_repeated_layer=True,
    ).to(torch.float32)
    # The constructor leaves the routed-expert weights at zero; without this the expert gradients would be zero too.
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.train()
    return model, config


def _is_checkpointed(model: NemotronHForCausalLM) -> list[bool]:
    return [isinstance(block, CheckpointWrapper) for block in model.mtp.layers]


def test_the_moe_block_of_a_repeated_mtp_depth_is_checkpointed_only_on_request() -> None:
    default, config = _model()
    apply_ac(default, hidden_size=config.hidden_size, num_experts=config.n_routed_experts)
    assert _is_checkpointed(default) == [True, False]

    requested, config = _model()
    apply_ac(
        requested,
        hidden_size=config.hidden_size,
        num_experts=config.n_routed_experts,
        checkpoint_repeated_mtp_moe=True,
    )
    assert _is_checkpointed(requested) == [True, True]


def test_checkpointing_the_repeated_mtp_moe_block_keeps_loss_and_gradients() -> None:
    """One CPU forward/backward over the backbone and both MTP depths: loss and gradients are bitwise equal."""
    reference, config = _model()
    default = copy.deepcopy(reference)
    apply_ac(default, hidden_size=config.hidden_size, num_experts=config.n_routed_experts)
    requested = copy.deepcopy(reference)
    apply_ac(
        requested,
        hidden_size=config.hidden_size,
        num_experts=config.n_routed_experts,
        checkpoint_repeated_mtp_moe=True,
    )

    input_ids = torch.randint(0, config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(3))

    def loss_and_grads(model: NemotronHForCausalLM) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = model(input_ids)
        assert len(out.mtp_per_depth_h) == MTP_DEPTHS
        targets = input_ids.reshape(-1)
        loss = torch.nn.functional.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), targets)
        for depth_hidden in out.mtp_per_depth_h:
            depth_logits = model.lm_head(depth_hidden)
            loss = loss + 0.3 * torch.nn.functional.cross_entropy(
                depth_logits.reshape(-1, depth_logits.shape[-1]), targets
            )
        loss.backward()
        grads = {
            name.replace("._checkpoint_wrapped_module", ""): param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        return loss.detach(), grads

    ref_loss, ref_grads = loss_and_grads(reference)
    mtp_expert_grads = [grad for name, grad in ref_grads.items() if name.startswith("mtp.") and ".experts." in name]
    assert mtp_expert_grads and all(grad.abs().sum() > 0 for grad in mtp_expert_grads)

    for model in (default, requested):
        loss, grads = loss_and_grads(model)
        assert torch.equal(loss, ref_loss)
        assert set(grads) == set(ref_grads)
        for name, grad in grads.items():
            assert torch.equal(grad, ref_grads[name]), name
