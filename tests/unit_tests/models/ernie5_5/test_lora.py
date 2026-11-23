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

"""Tests of LoRA on ERNIE 5.5 in one process.

The adapters are installed with the framework's own entry point and its trainability policy, in the order in which
the framework runs them before it shards a model. Two models are used:

* a small FP32 model with materialized, seeded weights (HCA with a dense MLP, HCA with an MoE, DSA with an MoE) for
  everything that needs a forward pass;
* the registered class on the meta device with three layers of the released shapes for the key plan and for the
  expert-parallel size that the parallelizer records.

Training with adapters over several ranks, and saving them, is covered by
``tests/functional_tests/moe/test_ernie5_5_lora_training.py``.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components.checkpoint.stateful_wrappers import _has_expert_parallelism
from nemo_automodel.components.distributed import model_parallelizer
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4GroupedLinear
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.moe import Ernie5_5MoE
from nemo_automodel.components.moe import layers as shared_moe_layers
from tests.unit_tests.models.ernie5_5.test_model import _tiny_model, _tokens
from tests.unit_tests.models.ernie5_5.test_moe import _moe_config, _situ
from tests.unit_tests.models.ernie5_5.test_parallelization import (
    _context,
    _fsdp_stage,
    _Mesh,
    _model,
    _record_sharding,
    _set_world_size,
)

pytestmark = pytest.mark.cpu_only

RECIPE = Path(__file__).resolve().parents[4] / "examples/llm_finetune/ernie5_5/ernie5_5_flash_squad_ep_lora.yaml"
# One compression window of the HCA layers and a few tokens more, so that the compressor takes part in the forward.
LENGTH = 140

HCA_PROJECTIONS = ("wq_a", "wq_b", "wkv", "wo_b")
DSA_PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# The supported target classes, as patterns of ``peft.target_modules``.
TARGET_CLASSES = {
    "hca_projections": [f"model.layers.*.attention.{name}" for name in HCA_PROJECTIONS],
    "attention_output_gate": ["model.layers.*.attention.gate_proj"],
    "dsa_projections": [f"model.layers.*.attention.{name}" for name in DSA_PROJECTIONS],
    "routed_experts": ["model.layers.*.moe.experts"],
    "dense_ffn": [f"model.layers.*.mlp.{name}" for name in FFN_PROJECTIONS],
    "shared_expert": [f"model.layers.*.moe.shared_experts.{name}" for name in FFN_PROJECTIONS],
}
# Modules of the three-layer models (HCA + dense MLP, HCA + MoE, DSA + MoE) that the recipe's patterns select.
WRAPPED_LINEARS = [
    *(f"model.layers.0.attention.{name}" for name in (*HCA_PROJECTIONS, "gate_proj")),
    *(f"model.layers.0.mlp.{name}" for name in FFN_PROJECTIONS),
    *(f"model.layers.1.attention.{name}" for name in (*HCA_PROJECTIONS, "gate_proj")),
    *(f"model.layers.1.moe.shared_experts.{name}" for name in FFN_PROJECTIONS),
    *(f"model.layers.2.attention.{name}" for name in (*DSA_PROJECTIONS, "gate_proj")),
    *(f"model.layers.2.moe.shared_experts.{name}" for name in FFN_PROJECTIONS),
]
WRAPPED_EXPERTS = ["model.layers.1.moe.experts", "model.layers.2.moe.experts"]


def _recipe_targets() -> list[str]:
    """Return the ``peft.target_modules`` patterns of the checked-in LoRA recipe."""
    return yaml.safe_load(RECIPE.read_text(encoding="utf-8"))["peft"]["target_modules"]


def _install_lora(model: nn.Module, peft: PeftConfig) -> None:
    """Install adapters and resolve trainability as the framework does before it shards a model."""
    apply_lora_to_linear_modules(model, peft, skip_freeze=True)
    _apply_trainability_policy(model, peft_enabled=True, freeze_config=None, strict=True)


def _adapter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if "lora_" in name]


def _randomize_second_factors(model: nn.Module) -> None:
    """Give the zero-initialized second factor of every adapter random values, so that the adapters act."""
    torch.manual_seed(5591)
    with torch.no_grad():
        for name in _adapter_names(model):
            if name.endswith(("lora_B.weight", "_B")):
                model.get_parameter(name).normal_(std=0.5)


@pytest.mark.parametrize("num_tokens", [0, 3])
def test_hybridep_lora_expert_preserves_situ_and_adapter_math(monkeypatch, num_tokens):
    """Compare dispatched LoRA experts with explicit SiTU, including an empty receive."""
    monkeypatch.setattr(shared_moe_layers, "get_world_size_safe", lambda: 2)
    model = Ernie5_5MoE(_moe_config(), BackendConfig(experts="torch_mm", dispatcher="hybridep", linear="torch"))
    torch.manual_seed(5593)
    with torch.no_grad():
        model.experts.gate_and_up_projs.normal_(std=0.5)
        model.experts.down_projs.normal_(std=0.5)
    _install_lora(model, PeftConfig(target_modules=["experts"], dim=2, alpha=4))
    experts = model.experts
    assert isinstance(experts, GroupedExpertsDeepEPLoRA)
    _randomize_second_factors(model)

    class Dispatcher:
        def token_permutation2(self, hidden_states, num_local_tokens, token_probs, token_indices):
            counts = [2, 1, 0] if num_local_tokens else [0, 0, 0]
            return hidden_states, torch.tensor(counts), token_probs[:, 0]

        def token_unpermutation(self, hidden_states):
            return hidden_states

    experts.token_dispatcher = Dispatcher()

    def grouped_mm(x, weights, *, offs):
        starts = [0, *offs.tolist()[:-1]]
        return torch.cat([x[start:end] @ weight for start, end, weight in zip(starts, offs.tolist(), weights)])

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm)
    x = torch.randn(num_tokens, experts.config.expert_dim, requires_grad=True)
    weights = torch.full((num_tokens, 1), 0.75)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long)
    output = experts(x, torch.ones(num_tokens, dtype=torch.bool), weights, indices)
    if num_tokens:
        assert experts.scale == 2.0
        outputs = []
        for expert_id, token_slice in enumerate((slice(0, 2), slice(2, 3))):
            gate_up = x[token_slice] @ (
                experts.gate_and_up_projs[expert_id]
                + 2.0 * experts.lora_gate_and_up_A[expert_id] @ experts.lora_gate_and_up_B[expert_id]
            )
            gate, up = gate_up.chunk(2, dim=-1)
            activated = _situ(gate, up) * weights[token_slice]
            outputs.append(
                activated
                @ (
                    experts.down_projs[expert_id]
                    + 2.0 * experts.lora_down_A[expert_id] @ experts.lora_down_B[expert_id]
                )
            )
        torch.testing.assert_close(output, torch.cat(outputs), rtol=1e-5, atol=1e-5)
        output.sum().backward()
        assert experts.lora_gate_and_up_B.grad is not None
        assert experts.lora_down_B.grad is not None
        assert x.grad is not None
        for expert_id in range(2):
            assert torch.count_nonzero(experts.lora_gate_and_up_B.grad[expert_id]) > 0
            assert torch.count_nonzero(experts.lora_down_B.grad[expert_id]) > 0
    else:
        assert output.shape == x.shape
        output.sum().backward()
        assert x.grad is not None
        assert experts.lora_gate_and_up_B.grad is not None
        assert experts.lora_down_B.grad is not None


def test_recipe_patterns_wrap_the_supported_modules_and_nothing_else():
    model = _tiny_model()
    _install_lora(model, PeftConfig(target_modules=_recipe_targets(), dim=2, alpha=4))
    wrapped = {
        name: module for name, module in model.named_modules() if isinstance(module, (LinearLoRA, GroupedExpertsLoRA))
    }
    assert set(wrapped) == {*WRAPPED_LINEARS, *WRAPPED_EXPERTS}
    assert all(type(wrapped[name]).__name__ == "PatchedLinearLoRA" for name in WRAPPED_LINEARS)
    assert all(type(wrapped[name]) is GroupedExpertsLoRA for name in WRAPPED_EXPERTS)
    # Two factors per linear projection, four grouped factors per expert module.
    adapters = _adapter_names(model)
    assert len(adapters) == 2 * len(WRAPPED_LINEARS) + 4 * len(WRAPPED_EXPERTS) == 56
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == set(adapters)
    # The grouped projection keeps its class and with it its grouped forward.
    assert type(model.model.layers[0].attention.wo_a) is DeepseekV4GroupedLinear


@pytest.mark.parametrize("memory_efficient", [True, False], ids=["memory_efficient", "plain"])
@pytest.mark.parametrize("target_class", [*TARGET_CLASSES, "recipe"])
def test_output_is_unchanged_at_lora_initialization(target_class, memory_efficient):
    model = _tiny_model()
    tokens = _tokens(LENGTH)
    before = model(tokens).logits.detach().clone()
    targets = _recipe_targets() if target_class == "recipe" else TARGET_CLASSES[target_class]
    _install_lora(model, PeftConfig(target_modules=targets, dim=2, alpha=4, use_memory_efficient_lora=memory_efficient))
    assert _adapter_names(model)
    # The second factor of every adapter starts at zero, so each wrapper adds an exact zero to the base output.
    torch.testing.assert_close(model(tokens).logits, before, rtol=0, atol=0)
    # Control: the wrapped modules are on the forward path, so the output moves once the adapters act.
    _randomize_second_factors(model)
    assert not torch.equal(model(tokens).logits, before)


@pytest.mark.parametrize("memory_efficient", [True, False], ids=["memory_efficient", "plain"])
def test_shared_expert_computes_situ_with_adapters_on_a_materialized_model(memory_efficient):
    """The fused LoRA MLP path replaces the forward of a SwiGLU module; the shared expert must keep its own."""
    model = _tiny_model()
    shared = model.model.layers[1].moe.shared_experts
    torch.manual_seed(5592)
    # Projections of order one and inputs of order three put the gate where SiTU and SwiGLU differ clearly.
    with torch.no_grad():
        for projection in (shared.gate_proj, shared.up_proj, shared.down_proj):
            projection.weight.normal_(std=0.5)
    inputs = 3 * torch.randn(5, model.config.hidden_size)
    before = shared(inputs).detach().clone()
    swiglu = shared.down_proj(F.silu(shared.gate_proj(inputs)) * shared.up_proj(inputs))
    assert (swiglu - before).abs().max() > 1.0

    _install_lora(
        model,
        PeftConfig(
            target_modules=TARGET_CLASSES["shared_expert"], dim=2, alpha=4, use_memory_efficient_lora=memory_efficient
        ),
    )
    torch.testing.assert_close(shared(inputs), before, rtol=0, atol=0)
    assert shared.activation == "situ"


@pytest.mark.parametrize("memory_efficient", [True, False], ids=["memory_efficient", "plain"])
def test_shared_expert_computes_situ_on_the_effective_weights_once_the_adapters_act(memory_efficient):
    """With non-zero factors the output is SiTU on W + scale * B A, computed here without the LoRA modules."""
    model = _tiny_model()
    shared = model.model.layers[1].moe.shared_experts
    torch.manual_seed(5592)
    with torch.no_grad():
        for projection in (shared.gate_proj, shared.up_proj, shared.down_proj):
            projection.weight.normal_(std=0.5)
    inputs = 3 * torch.randn(5, model.config.hidden_size)
    _install_lora(
        model,
        PeftConfig(
            target_modules=TARGET_CLASSES["shared_expert"], dim=2, alpha=4, use_memory_efficient_lora=memory_efficient
        ),
    )
    _randomize_second_factors(model)
    # The constants of the reference expression.
    assert (shared.beta, shared.linear_beta) == (4, 25)
    projections = (shared.gate_proj, shared.up_proj, shared.down_proj)

    def effective(projection: nn.Module) -> torch.Tensor:
        """Return the adapted weight W + scale * B A of a LoRA linear layer, of shape [out_features, in_features]."""
        return projection.weight + projection.scale * (projection.lora_B.weight @ projection.lora_A.weight)

    def situ_mlp(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        """Compute the shared expert on ``inputs`` of shape [tokens, hidden] from explicit weights.

        Args:
            gate: Gate projection weight of shape [intermediate, hidden].
            up: Up projection weight of shape [intermediate, hidden].
            down: Down projection weight of shape [hidden, intermediate].

        Returns:
            Output of shape [tokens, hidden].
        """
        return F.linear(_situ(F.linear(inputs, gate), F.linear(inputs, up)), down)

    with torch.no_grad():
        gate, up, down = (effective(projection) for projection in projections)
        situ = situ_mlp(gate, up, down)
        swiglu = F.linear(F.silu(F.linear(inputs, gate)) * F.linear(inputs, up), down)
        # The adapters move the output, and SwiGLU on the same effective weights is far from SiTU.
        assert (situ - situ_mlp(*(projection.weight for projection in projections))).abs().max() > 1.0
        assert (situ - swiglu).abs().max() > 1.0
        torch.testing.assert_close(shared(inputs), situ, rtol=1e-5, atol=1e-4)


def test_only_the_adapters_receive_gradients():
    model = _tiny_model()
    _install_lora(model, PeftConfig(target_modules=_recipe_targets(), dim=2, alpha=4))
    tokens = _tokens(LENGTH)
    logits = model(tokens).logits
    F.cross_entropy(logits[0, :-1].float(), tokens[0, 1:]).backward()
    adapters = set(_adapter_names(model))
    with_gradient = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert with_gradient == adapters
    # The first factor has a zero gradient while the second factor is zero; the second factor has a signal.
    second_factors = [name for name in adapters if name.endswith(("lora_B.weight", "_B"))]
    assert len(second_factors) == len(adapters) // 2
    for name in second_factors:
        gradient = model.get_parameter(name).grad
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient), name
    indexer = [name for name, _ in model.named_parameters() if ".attention.indexer." in name]
    assert len(indexer) == 5 and not any(model.get_parameter(name).requires_grad for name in indexer)


def test_checkpoint_adapter_plans_the_keys_of_the_recipe_adapters():
    """The framework lets the checkpoint adapter plan the keys of the unsharded model, adapters included."""
    model = _model()
    _install_lora(model, PeftConfig(target_modules=_recipe_targets()))
    adapters = _adapter_names(model)
    assert len(adapters) == 56
    planned = model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
    assert set(adapters) <= set(planned)


def test_training_forward_rejects_an_adapter_on_an_indexer():
    model = _tiny_model()
    _install_lora(model, PeftConfig(target_modules=["model.layers.*.attention.indexer.wq_b"], dim=2, alpha=4))
    tokens = _tokens(LENGTH)
    model.train()
    message = r"model\.layers\.2\.attention\.indexer has trainable parameters.*peft\.target_modules"
    with pytest.raises(ValueError, match=message):
        model(tokens)
    # The guard is a guard of training: an evaluation forward of the same model runs.
    model.eval()
    assert torch.isfinite(model(tokens).logits).all()


def test_failed_parallelization_leaves_no_expert_parallel_size_on_the_experts(monkeypatch):
    """The size is recorded once the shared flow has sharded the model through the sidecar, not before."""
    model = _model()
    _set_world_size(monkeypatch)
    monkeypatch.setattr(model_parallelizer, "_parallelize_moe", Mock(return_value=model))
    with pytest.raises(RuntimeError, match="did not shard ERNIE 5.5 through its sidecar"):
        parallelize_model(model, _context())
    assert not _has_expert_parallelism(model)


@pytest.mark.parametrize("with_lora", [False, True], ids=["full_fine_tuning", "lora"])
def test_parallelization_records_the_expert_parallel_size_on_the_routed_experts(monkeypatch, with_lora):
    """The checkpointer collects a PEFT adapter on every rank only for a model that it detects as expert parallel."""
    model, mesh = _model(), _Mesh()
    if with_lora:
        _install_lora(model, PeftConfig(target_modules=_recipe_targets()))
    assert not _has_expert_parallelism(model)
    _set_world_size(monkeypatch)
    _record_sharding(monkeypatch)
    monkeypatch.setattr(model_parallelizer, "_parallelize_moe", _fsdp_stage(mesh))

    assert parallelize_model(model, _context(ep_size=8)) is model

    assert isinstance(model, Ernie5_5_MoeForCausalLM)
    experts = [layer.moe.experts for layer in model.model.layers if layer.is_moe]
    assert len(experts) == 2 and [module.ep_size for module in experts] == [8, 8]
    assert all(isinstance(module, GroupedExpertsLoRA) for module in experts) == with_lora
    assert _has_expert_parallelism(model)
