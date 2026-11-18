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
* the registered class on the meta device with three layers of the released shapes for the admission tests, which
  call ``parallelize_model`` with every executor of the shared dispatcher replaced by a recorder, so that a rejection
  is shown to happen before the model is sharded.

Training with adapters over several ranks, and saving them, is covered by
``tests/functional_tests/moe/test_ernie5_5_lora_training.py``.
"""

import re
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
from nemo_automodel.components.models.common.utils import prepare_frozen_parameters_for_sharding
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4GroupedLinear
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.moe import Ernie5_5MoE
from nemo_automodel.components.models.ernie5_5.parallelization import PARALLELIZER
from nemo_automodel.components.moe import layers as shared_moe_layers
from tests.unit_tests.models.ernie5_5.conftest import checkpoint_style_config
from tests.unit_tests.models.ernie5_5.test_model import _tiny_model, _tokens
from tests.unit_tests.models.ernie5_5.test_moe import _moe_config, _situ
from tests.unit_tests.models.ernie5_5.test_parallelization import (
    _assert_untouched,
    _backend,
    _context,
    _fsdp_stage,
    _Mesh,
    _model,
    _record_executors,
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


def test_recipe_patterns_are_admitted_before_sharding(monkeypatch):
    model = _model()
    _install_lora(model, PeftConfig(target_modules=_recipe_targets()))
    _set_world_size(monkeypatch)
    assert PARALLELIZER._prepare_model(model, _context()) is None
    _assert_untouched(model)


@pytest.mark.parametrize(
    ("peft", "module", "reason"),
    [
        pytest.param(
            PeftConfig(target_modules=["model.layers.*.attention.wo_a"]),
            "model.layers.0.attention.wo_a (the first of 2 such modules)",
            "wo_a is a grouped projection",
            id="grouped_projection",
        ),
        pytest.param(
            PeftConfig(target_modules=["model.layers.*.moe.fc1_latent_proj"]),
            "model.layers.1.moe.fc1_latent_proj (the first of 2 such modules)",
            "latent projections of the MoE cast their FP32 weight",
            id="first_latent_projection",
        ),
        pytest.param(
            PeftConfig(target_modules=["model.layers.2.moe.fc2_latent_proj"]),
            "model.layers.2.moe.fc2_latent_proj:",
            "latent projections of the MoE cast their FP32 weight",
            id="second_latent_projection",
        ),
        pytest.param(
            PeftConfig(target_modules=["model.layers.*.attention.indexer.wq_b"]),
            "model.layers.2.attention.indexer.wq_b:",
            "the DSA indexer only selects keys and is not trained",
            id="indexer_query_projection",
        ),
        pytest.param(
            PeftConfig(target_modules=["*.indexer.wk", "*.indexer.weights_proj"]),
            "model.layers.2.attention.indexer.wk (the first of 2 such modules)",
            "the DSA indexer only selects keys and is not trained",
            id="indexer_key_and_weight_projections",
        ),
        pytest.param(
            PeftConfig(target_modules=["model.layers.*.attention.compressor.wkv"]),
            "model.layers.0.attention.compressor.wkv (the first of 2 such modules)",
            "LoRA has not been validated on this module",
            id="compressor_projection",
        ),
        pytest.param(
            PeftConfig(target_modules=["model.lm_head"]),
            "model.lm_head:",
            "LoRA has not been validated on this module",
            id="output_head",
        ),
    ],
)
def test_unsupported_target_is_rejected_before_any_executor(monkeypatch, peft, module, reason):
    model = _model()
    _install_lora(model, peft)
    _set_world_size(monkeypatch)
    executors = _record_executors(monkeypatch)
    message = f"ERNIE 5.5 does not support LoRA on {re.escape(module)}.*{re.escape(reason)}.*"
    # The message ends with the supported names of each kind of layer.
    supported = (
        r"\. Set peft\.target_modules to patterns that select supported modules only\. "
        r"Under model\.layers\.<index> these are attention\.<name> with <name> in "
        r"\(wq_a, wq_b, wkv, wo_b, gate_proj\) for an HCA layer and in "
        r"\(q_a_proj, q_b_proj, kv_a_proj_with_mqa, gate_proj, o_proj\) for a DSA layer, "
        r"moe\.experts, and moe\.shared_experts\.<name> and mlp\.<name> with <name> in "
        r"\(gate_proj, up_proj, down_proj\)$"
    )
    with pytest.raises(ValueError, match=message + supported):
        parallelize_model(model, _context())
    for executor in executors.values():
        executor.assert_not_called()
    _assert_untouched(model)


@pytest.mark.parametrize(
    ("peft", "first", "count"),
    [
        pytest.param(PeftConfig(), "model.layers.1.moe.fc1_latent_proj", 5, id="default_patterns"),
        pytest.param(PeftConfig(match_all_linear=True), "model.layers.0.attention.wo_a", 14, id="all_linear_layers"),
        pytest.param(
            PeftConfig(exclude_modules=["model.lm_head"]),
            "model.layers.0.attention.wo_a",
            13,
            id="all_but_the_output_head",
        ),
    ],
)
def test_patterns_that_cover_every_projection_are_rejected(monkeypatch, peft, first, count):
    """Without explicit targets the module matcher also selects the unsupported linear layers.

    The three-layer model has 14 linear layers outside the supported targets: ``wo_a`` and two compressor
    projections in each of the two HCA layers, two latent projections in each of the two MoE layers, three indexer
    projections in the DSA layer, and the output head. The default pattern selects the four latent projections and
    one indexer projection. The message counts them, so a module that the allow-list admitted by mistake would
    lower the count.
    """
    model = _model()
    _install_lora(model, peft)
    _set_world_size(monkeypatch)
    executors = _record_executors(monkeypatch)
    message = f"ERNIE 5.5 does not support LoRA on {first} (the first of {count} such modules):"
    with pytest.raises(ValueError, match=re.escape(message)):
        parallelize_model(model, _context())
    for executor in executors.values():
        executor.assert_not_called()
    _assert_untouched(model)


def test_supported_targets_next_to_an_unsupported_one_are_not_named(monkeypatch):
    model = _model()
    _install_lora(model, PeftConfig(target_modules=[*_recipe_targets(), "model.layers.1.attention.wo_a"]))
    _set_world_size(monkeypatch)
    with pytest.raises(ValueError, match=r"does not support LoRA on model\.layers\.1\.attention\.wo_a:"):
        PARALLELIZER._prepare_model(model, _context())


def test_dora_is_rejected_before_any_executor(monkeypatch):
    """DoRA on a supported linear projection adds a magnitude vector that the checkpoint adapter does not save."""
    model = _model()
    _install_lora(model, PeftConfig(target_modules=["model.layers.*.attention.wq_a"], use_dora=True))
    assert "model.layers.0.attention.wq_a.lora_magnitude" in _adapter_names(model)
    _set_world_size(monkeypatch)
    executors = _record_executors(monkeypatch)
    message = r"does not support DoRA: model\.layers\.0\.attention\.wq_a carries.*Disable peft\.use_dora$"
    with pytest.raises(ValueError, match=message):
        parallelize_model(model, _context())
    for executor in executors.values():
        executor.assert_not_called()
    _assert_untouched(model)


def test_adapter_tensor_other_than_the_lora_factors_is_rejected(monkeypatch):
    """The check admits the tensors that the checkpoint adapter saves, not every tensor named like an adapter."""
    model = _model()
    _install_lora(model, PeftConfig(target_modules=["model.layers.0.attention.wq_a"]))
    projection = model.model.layers[0].attention.wq_a
    projection.register_parameter(
        "lora_gain", nn.Parameter(torch.empty(1, device="meta", dtype=projection.weight.dtype))
    )
    _set_world_size(monkeypatch)
    message = r"does not support the adapter tensor model\.layers\.0\.attention\.wq_a\.lora_gain:"
    with pytest.raises(ValueError, match=message):
        PARALLELIZER._prepare_model(model, _context())


@pytest.mark.parametrize(
    ("targets", "tensor", "module"),
    [
        pytest.param(
            None, r"model\.layers\.0\.attention\.wq_a\.lora_A\.weight", "model.layers.0.attention.wq_a", id="recipe"
        ),
        pytest.param(
            TARGET_CLASSES["routed_experts"],
            r"model\.layers\.1\.moe\.experts\.lora_\w+",
            "model.layers.1.moe.experts",
            id="routed_experts",
        ),
    ],
)
def test_adapter_in_another_dtype_than_its_module_is_rejected_before_any_executor(monkeypatch, targets, tensor, module):
    """With ``peft.lora_dtype`` the factors no longer share the dtype of the weight they adapt.

    Under an FSDP policy without a compute dtype the shared flow then fails while it shards a decoder block, with
    a message about frozen weights. The check rejects the adapter first and names the option.
    """
    model = _model()
    _install_lora(
        model, PeftConfig(target_modules=_recipe_targets() if targets is None else targets, lora_dtype="float32")
    )
    _set_world_size(monkeypatch)
    executors = _record_executors(monkeypatch)
    message = rf"{tensor} is torch\.float32 and {re.escape(module)} is torch\.bfloat16\. Leave peft\.lora_dtype unset"
    with pytest.raises(ValueError, match=message):
        parallelize_model(model, _context())
    for executor in executors.values():
        executor.assert_not_called()
    _assert_untouched(model)


def test_fp32_adapter_on_weights_stored_in_the_compute_dtype_is_rejected_and_names_the_policy(monkeypatch):
    """An FP32 model under a BF16 compute policy has FP32 factors on BF16 weights without ``peft.lora_dtype``.

    Before it calls the parallelizer, the framework stores the frozen weights of a PEFT model in the compute dtype
    of the FSDP policy, which is BF16 by default. The message therefore names the policy next to the option.
    """
    values = checkpoint_style_config()
    values.update(num_hidden_layers=3, compress_ratios=[128, 128, -2], num_nextn_predict_layers=0)
    values.update(torch_dtype="float32")
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(Ernie5_5Config(**values), _backend())
    _install_lora(model, PeftConfig(target_modules=_recipe_targets()))
    prepare_frozen_parameters_for_sharding(model, torch.bfloat16)
    _set_world_size(monkeypatch)
    executors = _record_executors(monkeypatch)
    message = (
        r"model\.layers\.0\.attention\.wq_a\.lora_A\.weight is torch\.float32 and model\.layers\.0\.attention\.wq_a "
        r"is torch\.bfloat16\. Leave peft\.lora_dtype unset\. If it is unset.*distributed\.mp_policy\.param_dtype"
    )
    with pytest.raises(ValueError, match=message):
        parallelize_model(model, _context())
    for executor in executors.values():
        executor.assert_not_called()
    _assert_untouched(model)


def test_checkpoint_adapter_plans_the_keys_of_every_adapter_that_the_parallelizer_admits(monkeypatch):
    """The parallelizer and the checkpoint adapter agree on the adapter tensors of the recipe.

    The framework lets the checkpoint adapter plan the keys of the unsharded model before it calls the
    parallelizer. An adapter tensor that the parallelizer admits and the plan rejects could never be trained.
    """
    model = _model()
    _install_lora(model, PeftConfig(target_modules=_recipe_targets()))
    _set_world_size(monkeypatch)
    assert PARALLELIZER._prepare_model(model, _context()) is None
    adapters = _adapter_names(model)
    assert len(adapters) == 56
    planned = model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
    assert set(adapters) <= set(planned)


def test_checkpoint_adapter_rejects_dora_when_it_plans_the_keys():
    """A model built by the framework with DoRA ends in the key plan, which names DoRA and the option to change."""
    model = _model()
    _install_lora(model, PeftConfig(target_modules=["model.layers.*.attention.wq_a"], use_dora=True))
    with pytest.raises(ValueError) as rejection:
        model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
    assert "DoRA" in str(rejection.value) and "peft.use_dora" in str(rejection.value)


def test_training_forward_rejects_an_adapter_on_an_indexer_without_the_parallelizer():
    """A caller that trains the model in one process meets the forward guard instead of the parallelizer."""
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
