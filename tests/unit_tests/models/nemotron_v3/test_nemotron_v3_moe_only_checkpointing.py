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

"""MoE-only activation checkpointing on NemotronV3 blocks, whose ``mlp`` is a property aliasing ``mixer``."""

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
