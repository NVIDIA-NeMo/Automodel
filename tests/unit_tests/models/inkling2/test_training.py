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

"""Loss semantics, shared training interfaces and eager-path regression checks."""

from collections.abc import Iterator

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.loss.utils import prepare_lm_weight
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM

from .test_reference import tiny_config

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_fused_loss_materializes_usable_bf16_head_rows(dtype: str) -> None:
    """The shared recipe hook hands the kernel the usable rows in BF16; gradients reach the full head."""
    torch.manual_seed(107)
    model = Inkling2ForCausalLM(tiny_config(dtype=dtype))
    loss_fn = Inkling2FusedLinearCrossEntropy(37)
    assert loss_fn.logit_softcapping == 30.0 and loss_fn.reduction == "sum" and loss_fn.impl == "cce"
    weight = prepare_lm_weight(loss_fn, model)
    assert weight.dtype == torch.bfloat16 and weight.shape == (37, 16)
    torch.testing.assert_close(weight, model.lm_head.weight[:37].to(torch.bfloat16), rtol=0, atol=0)
    upstream = torch.randn_like(weight)
    weight.backward(upstream)
    grad = model.lm_head.weight.grad
    assert grad.dtype == model.lm_head.weight.dtype
    torch.testing.assert_close(grad[:37], upstream.to(grad.dtype), rtol=0, atol=0)
    assert torch.count_nonzero(grad[37:]) == 0


def test_output_selection_and_loss_boundary_errors() -> None:
    model = Inkling2ForCausalLM(tiny_config())
    ids = torch.tensor([[1, 2, 3, 4]])
    full = model(ids)
    torch.testing.assert_close(model(ids, logits_to_keep=2), full[:, -2:])
    torch.testing.assert_close(model(ids, logits_to_keep=torch.tensor([0, 2])), full[:, [0, 2]])
    with pytest.raises(ValueError, match="nonnegative"):
        model(ids, logits_to_keep=-1)
    with pytest.raises(ValueError, match="int64"):
        model(ids, logits_to_keep=torch.tensor([0.5]))
    with pytest.raises(ValueError, match="unpadded_vocab_size must be positive"):
        Inkling2FusedLinearCrossEntropy(0)
    with pytest.raises(ValueError, match="logit_softcapping must be positive"):
        Inkling2FusedLinearCrossEntropy(37, logit_softcapping=0)
    with pytest.raises(ValueError, match="fewer than unpadded_vocab_size"):
        Inkling2FusedLinearCrossEntropy(41).materialize_lm_weight(model.lm_head.weight)
    backend = BackendConfig(attn="flex", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    with pytest.raises(ValueError, match="requires CUDA"):
        Inkling2ForCausalLM(tiny_config(), backend=backend)(ids)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_projection_runs_module_hooks_and_preserves_fp32(dtype: torch.dtype) -> None:
    """The head's callable boundary is necessary for FSDP weight-gather hooks."""
    model = Inkling2ForCausalLM(tiny_config(dtype=str(dtype).removeprefix("torch.")))
    calls = []

    def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        """Record projection inputs shaped [batch, sequence, hidden].

        Args:
            module: Output projection being invoked.
            args: Tuple whose first tensor has shape [batch, sequence, hidden].
        """
        calls.append(args[0])

    handle = model.lm_head.register_forward_pre_hook(capture)
    try:
        output = model(torch.tensor([[1, 2, 3]]), output_hidden_states=True)
    finally:
        handle.remove()
    assert len(calls) == 1
    expected = F.linear(output.hidden_states.float(), model.lm_head.weight.float())
    expected = 30 * torch.tanh(expected / 30)
    assert output.logits.dtype == torch.float32
    torch.testing.assert_close(output.logits[..., :37], expected[..., :37], rtol=0, atol=0)


@pytest.mark.parametrize(
    "strategy_dtype,moe_dtype",
    [(torch.float16, None), (torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)],
)
def test_parallelizer_rejects_unqualified_compute_before_sharding(
    strategy_dtype: torch.dtype, moe_dtype: torch.dtype | None
) -> None:
    """A policy must not silently bypass the model's FP32/BF16 compute contract."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
    from nemo_automodel.components.distributed.mesh import MeshContext

    model = Inkling2ForCausalLM(tiny_config())
    context = MeshContext(
        strategy_config=FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=strategy_dtype)),
        moe_parallel_config=MoEParallelizerConfig(mp_policy=MixedPrecisionPolicy(param_dtype=moe_dtype))
        if moe_dtype is not None
        else None,
    )
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        model.parallelizer.parallelize(model, context)
    assert all(isinstance(parameter, torch.nn.Parameter) for parameter in model.parameters())


def test_tiny_recipe_config_and_next_token_labels() -> None:
    """The recipe data/loss configs resolve and supply labels shifted exactly once."""
    from pathlib import Path

    import yaml

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes._typed_config import RecipeConfig

    path = Path(__file__).resolve().parents[3] / "functional_tests/models/inkling2/tiny_train.yaml"
    raw = yaml.safe_load(path.read_text())
    # The GPU functional test resolves the public model factory. Isolate the
    # data/loss boundary here so CPU environments do not need CUDA loader extras.
    config = RecipeConfig(ConfigNode({"loss_fn": raw["loss_fn"]}))
    loss_fn = config.loss_fn.build()
    assert isinstance(loss_fn, Inkling2FusedLinearCrossEntropy) and loss_fn.unpadded_vocab_size == 37
    dataset = ConfigNode(raw["dataset"]).instantiate().build()
    assert len(dataset) == 32
    for row in dataset:
        ids, valid, labels = map(torch.tensor, (row["input_ids"], row["attention_mask"], row["labels"]))
        supervised = labels[:-1] != -100
        torch.testing.assert_close(labels[:-1][supervised], ids[1:][supervised])
        assert torch.all(valid[:-1][supervised] & valid[1:][supervised])
        assert labels[-1] == -100
    assert set(dataset[3]["labels"]) == {-100}


def test_fsdp_units_keep_fp32_modules_separate_and_never_wrap_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks and the root shard their FP32 modules first; buffer-only containers never become units."""
    from unittest.mock import Mock

    from torch import nn
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from nemo_automodel.components.models.inkling2 import parallelization

    model = Inkling2ForCausalLM(tiny_config(dtype="bfloat16"))
    sharded: list[tuple[nn.Module, dict[str, object]]] = []
    monkeypatch.setattr(parallelization, "fully_shard", lambda module, **kwargs: sharded.append((module, kwargs)))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    parallelizer = parallelization.Inkling2Parallelizer()

    block = model.layers["0"]
    parallelizer._fully_shard_module(block, mesh=Mock(), mp_policy=policy)
    *island_calls, block_call = sharded
    islands = [module for module, _ in island_calls]
    island_names = [name for name, module in block.named_modules() if any(module is island for island in islands)]
    assert block_call[0] is block
    assert island_names and all(
        any(name == suffix or name.endswith("." + suffix) for suffix in parallelization.FP32_MODULES)
        for name in island_names
    )
    assert all({param.dtype for param in island.parameters()} == {torch.float32} for island in islands)
    assert all(kwargs["mp_policy"].param_dtype == torch.float32 for _, kwargs in island_calls)
    assert block_call[1]["mp_policy"].param_dtype == torch.bfloat16
    assert block_call[1]["ignored_params"] == {param for island in islands for param in island.parameters()}

    sharded.clear()
    layer_params = {param for layer in model.layers.values() for param in layer.parameters()}
    parallelizer._fully_shard_module(model, mesh=Mock(), mp_policy=policy, ignored_params=layer_params)
    assert [module for module, _ in sharded] == [model.output_query, model]
    assert sharded[0][1]["mp_policy"].param_dtype == torch.float32
    assert sharded[1][1]["ignored_params"] == layer_params | set(model.output_query.parameters())
    assert all(not isinstance(module, (nn.ModuleList, nn.ModuleDict)) for module, _ in sharded)


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    "recipe_path",
    [
        "examples/vlm_finetune/inkling2/inkling2_medpix_lora_ep32.yaml",
        "examples/vlm_finetune/inkling2/inkling2_medpix_sft_ep64_pp4.yaml",
        "examples/llm_finetune/inkling2/inkling2_tulu3_packed128k_ep32_pp2_cp32.yaml",
    ],
)
def test_recipe_hybridep_activation_and_lora_preserve_interleaved_caps(
    recipe_path: str, world_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recipe backends retain capped gate/up math and gradients through LoRA construction.

    Mock only world-size discovery to construct both shared expert classes on CPU;
    the real dispatch/collective lifecycle is covered by the two-GPU EP cases.
    """
    from pathlib import Path

    import yaml

    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.moe import layers as moe_layers
    from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
    from nemo_automodel.shared.peft_config import PeftConfig

    root = Path(__file__).resolve().parents[4]
    raw = yaml.safe_load((root / recipe_path).read_text())
    backend = ConfigNode(raw["model"]["backend"]).instantiate()
    assert backend.dispatcher == "hybridep" and backend.experts == "torch_mm"
    monkeypatch.setattr(moe_layers, "get_world_size_safe", lambda: world_size)
    cfg = tiny_config(dtype=raw["model"]["dtype"])
    cfg.num_hidden_layers = 1
    if world_size == 1:
        with pytest.warns(UserWarning, match="world size is 1"):
            model = Inkling2ForCausalLM(cfg, backend=backend)
    else:
        model = Inkling2ForCausalLM(cfg, backend=backend)
    if "peft" in raw:
        apply_lora_to_linear_modules(
            model,
            PeftConfig(target_modules=["*.mlp.experts"], dim=8, alpha=32, lora_dtype="bfloat16"),
        )
    experts = model.layers["0"].mlp.experts
    if world_size == 1:
        assert isinstance(experts, GroupedExpertsLoRA if "peft" in raw else GroupedExperts)
        activation = experts.expert_activation_grouped
    else:
        assert isinstance(experts, GroupedExpertsDeepEPLoRA if "peft" in raw else GroupedExpertsDeepEP)
        activation = experts.expert_activation
        assert experts.dispatcher_backend == "hybridep"

    # Values beyond both caps catch a silent fallback to ordinary split-half SwiGLU.
    values = torch.linspace(-60, 60, 4 * 2 * cfg.intermediate_size).reshape(4, -1)
    values = values.to(experts.gate_and_up_projs.dtype).requires_grad_()
    weights = torch.tensor([[0.1], [0.3], [0.7], [1.0]], requires_grad=True)
    actual = activation(values, weights)
    expected_values = values.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_()
    gate, up = expected_values.float().reshape(4, -1, 2).unbind(-1)
    expected = (
        cfg.mlp_gate_softcap
        * torch.tanh(gate / cfg.mlp_gate_softcap)
        * torch.sigmoid(gate)
        * (cfg.mlp_up_softcap * torch.tanh(up / cfg.mlp_up_softcap))
        * expected_weights
    ).to(values.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(values.grad, expected_values.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, expected_weights.grad, rtol=0, atol=0)
