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

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.loss.utils import calculate_loss, prepare_lm_weight
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.layers import Inkling2LMHead
from nemo_automodel.components.models.inkling2.loss import (
    Inkling2ChunkedLinearCrossEntropy,
    Inkling2FusedLinearCrossEntropy,
)
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


def _softcapped_token_losses(hidden: torch.Tensor, labels: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Per-token cross entropy [batch, sequence] from FP32 logits of the 37 usable BF16 head rows."""
    logits = F.linear(hidden.float(), head[:37].to(torch.bfloat16).float())
    capped = 30 * torch.tanh(logits / 30)
    return F.cross_entropy(capped.flatten(0, 1), labels.flatten(), reduction="none").reshape(labels.shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_tokens,scale", [(3, 4.0), (64, 4.0), (3, 0.25)])
def test_chunked_loss_matches_softcapped_cross_entropy(dtype: torch.dtype, chunk_tokens: int, scale: float) -> None:
    """Chunks of supervised tokens give the loss and both gradients of the unchunked FP32 computation.

    Large logits exercise the softcap; small ones make the padded vocabulary columns visible
    if they entered the softmax.
    """
    torch.manual_seed(211)
    head = torch.randn(40, 16, requires_grad=True)
    hidden = (scale * torch.randn(2, 9, 16)).to(dtype).requires_grad_(True)
    labels = torch.randint(0, 37, (2, 9))
    labels[0, 2] = labels[1, -1] = -100
    count = int((labels != -100).sum())
    loss_fn = Inkling2ChunkedLinearCrossEntropy(37, chunk_tokens=chunk_tokens)
    weight = loss_fn.materialize_lm_weight(head)
    assert weight.dtype == torch.bfloat16 and weight.shape == (37, 16)
    loss = loss_fn(hidden, labels, weight, num_label_tokens=count)
    loss.backward()

    expected_hidden = hidden.detach().clone().requires_grad_(True)
    expected_head = head.detach().clone().requires_grad_(True)
    supervised = labels != -100
    expected = _softcapped_token_losses(expected_hidden, labels.clamp_min(0), expected_head)[supervised].sum() / count
    expected.backward()
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
    # Returned gradients are rounded to the operand dtypes; BF16 keeps eight mantissa bits.
    torch.testing.assert_close(hidden.grad.float(), expected_hidden.grad.float(), rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(head.grad[:37], expected_head.grad[:37], rtol=1e-2, atol=1e-5)
    assert torch.count_nonzero(head.grad[37:]) == 0
    assert torch.count_nonzero(hidden.grad[~supervised]) == 0


def test_chunked_loss_reductions_weights_and_empty_supervision() -> None:
    """Per-token losses, token weights and an all-ignored batch follow the shared loss contract."""
    torch.manual_seed(223)
    head = torch.randn(40, 16)
    hidden = (4 * torch.randn(2, 5, 16)).to(torch.bfloat16)
    labels = torch.randint(0, 37, (2, 5))
    labels[1, 0] = -100
    supervised = labels != -100
    expected = _softcapped_token_losses(hidden, labels.clamp_min(0), head) * supervised
    per_token = Inkling2ChunkedLinearCrossEntropy(37, reduction="none", chunk_tokens=4)(hidden, labels, head)
    assert per_token.shape == labels.shape and per_token[1, 0] == 0
    torch.testing.assert_close(per_token, expected, rtol=1e-6, atol=1e-6)
    mean = Inkling2ChunkedLinearCrossEntropy(37, reduction="mean")(hidden, labels, head)
    torch.testing.assert_close(mean, expected.sum() / supervised.sum(), rtol=1e-6, atol=1e-6)
    weights = torch.rand(2, 5)
    weighted = Inkling2ChunkedLinearCrossEntropy(37)(hidden, labels, head, loss_weights=weights)
    torch.testing.assert_close(weighted, (expected * weights).sum(), rtol=1e-6, atol=1e-6)

    hidden.requires_grad_(True)
    ignored = Inkling2ChunkedLinearCrossEntropy(37)(hidden, torch.full_like(labels, -100), head, num_label_tokens=0)
    ignored.backward()
    assert ignored == 0 and torch.count_nonzero(hidden.grad) == 0


def test_chunked_loss_projects_the_logits_once_for_a_summed_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A summed loss takes both gradients from the forward; only token weights recompute the chunks."""
    import nemo_automodel.components.models.inkling2.loss as loss_module

    def recomputed(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("the backward recomputed a chunk")

    monkeypatch.setattr(loss_module, "_logit_gradients_core", recomputed)
    monkeypatch.setattr(loss_module, "_compiled_logit_gradients", recomputed)
    torch.manual_seed(227)
    head = torch.randn(40, 16, requires_grad=True)
    hidden = (4 * torch.randn(2, 9, 16)).to(torch.bfloat16).requires_grad_(True)
    labels = torch.randint(0, 37, (2, 9))
    loss_fn = Inkling2ChunkedLinearCrossEntropy(37, chunk_tokens=4)
    loss_fn(hidden, labels, head, num_label_tokens=18).backward()
    assert hidden.grad is not None and head.grad is not None
    # Token weights give every token another upstream gradient: that backward does recompute.
    with pytest.raises(AssertionError, match="recomputed a chunk"):
        loss_fn(hidden, labels, head, loss_weights=torch.rand(2, 9)).backward()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_loss_token_weights_reach_both_gradients(dtype: torch.dtype) -> None:
    """Token weights scale each token's contribution to the hidden and the head gradients."""
    torch.manual_seed(229)
    head = torch.randn(40, 16, requires_grad=True)
    hidden = (4 * torch.randn(2, 9, 16)).to(dtype).requires_grad_(True)
    labels = torch.randint(0, 37, (2, 9))
    labels[0, 3] = -100
    weights = torch.rand(2, 9)
    Inkling2ChunkedLinearCrossEntropy(37, chunk_tokens=4)(hidden, labels, head, loss_weights=weights).backward()

    expected_hidden = hidden.detach().clone().requires_grad_(True)
    expected_head = head.detach().clone().requires_grad_(True)
    supervised = labels != -100
    losses = _softcapped_token_losses(expected_hidden, labels.clamp_min(0), expected_head)
    (losses * weights)[supervised].sum().backward()
    torch.testing.assert_close(hidden.grad.float(), expected_hidden.grad.float(), rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(head.grad[:37], expected_head.grad[:37], rtol=1e-2, atol=1e-5)
    # Negative control: the unweighted gradient is another one.
    plain = hidden.detach().clone().requires_grad_(True)
    Inkling2ChunkedLinearCrossEntropy(37, chunk_tokens=4)(plain, labels, head.detach()).backward()
    assert not torch.allclose(plain.grad.float(), expected_hidden.grad.float(), rtol=1e-2, atol=1e-5)


def test_chunked_loss_second_backward_recomputes_the_same_gradients() -> None:
    """The forward's gradients are released by the first backward; a retained graph still differentiates."""
    torch.manual_seed(233)
    head = torch.randn(40, 16, requires_grad=True)
    hidden = (4 * torch.randn(2, 9, 16)).to(torch.bfloat16).requires_grad_(True)
    labels = torch.randint(0, 37, (2, 9))
    loss = Inkling2ChunkedLinearCrossEntropy(37, chunk_tokens=4)(hidden, labels, head, num_label_tokens=18)
    first = torch.autograd.grad(loss, (hidden, head), retain_graph=True)
    assert loss.grad_fn.next_functions is not None
    second = torch.autograd.grad(loss, (hidden, head))
    for a, b in zip(first, second):
        torch.testing.assert_close(a.float(), b.float(), rtol=1e-2, atol=1e-6)


def test_chunked_loss_rejects_invalid_configuration() -> None:
    """Construction and call errors name the offending argument."""
    for kwargs, match in (
        ({"unpadded_vocab_size": 0}, "unpadded_vocab_size must be positive"),
        ({"unpadded_vocab_size": 37, "logit_softcapping": 0}, "logit_softcapping must be positive"),
        ({"unpadded_vocab_size": 37, "chunk_tokens": 0}, "chunk_tokens must be positive"),
        ({"unpadded_vocab_size": 37, "reduction": "max"}, "Unsupported reduction"),
    ):
        with pytest.raises(ValueError, match=match):
            Inkling2ChunkedLinearCrossEntropy(**kwargs)
    head, hidden, labels = torch.randn(40, 16), torch.randn(1, 3, 16), torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="fewer than unpadded_vocab_size"):
        Inkling2ChunkedLinearCrossEntropy(41)(hidden, labels, head)
    with pytest.raises(ValueError, match="only supported when reduction is 'sum'"):
        Inkling2ChunkedLinearCrossEntropy(37, reduction="mean")(hidden, labels, head, num_label_tokens=3)
    with pytest.raises(ValueError, match="loss_weights.shape must match"):
        Inkling2ChunkedLinearCrossEntropy(37)(hidden, labels, head, loss_weights=torch.ones(3))


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


def test_bf16_head_owns_the_fp32_softcapped_masked_logits_and_gradients() -> None:
    model = Inkling2ForCausalLM(tiny_config(dtype="bfloat16"))
    head = model.get_output_embeddings()
    assert isinstance(head, Inkling2LMHead) and head.weight.dtype == torch.bfloat16
    generator = torch.Generator().manual_seed(13)
    hidden = torch.randn(2, 3, head.in_features, generator=generator, requires_grad=True)
    upstream = torch.randn(2, 3, head.out_features, generator=generator)

    cap = model.text_config.final_logit_softcapping
    expected = cap * torch.tanh(F.linear(hidden, head.weight.float()) / cap)
    padded = torch.arange(head.out_features) >= model.text_config.unpadded_vocab_size
    expected = expected.masked_fill(padded, float("-inf"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head(hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (hidden, head.weight)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, inputs, upstream), torch.autograd.grad(expected, inputs, upstream)
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_chunked_cross_entropy_reproduces_the_model_logits(dtype: str) -> None:
    torch.manual_seed(0)
    model = Inkling2ForCausalLM(tiny_config(dtype=dtype))
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[2, 3, 4, 5, 6, -100]])
    output = model(ids, output_hidden_states=True)
    expected = F.cross_entropy(output.logits.flatten(0, 1), labels.flatten(), reduction="sum")

    actual = calculate_loss(
        ChunkedCrossEntropy(2, compile=False),
        model=model,
        hidden_states=output.hidden_states,
        logits=output.logits,
        labels=labels,
    )

    torch.testing.assert_close(actual, expected)
    weight = model.get_output_embeddings().weight
    (actual_grad,) = torch.autograd.grad(actual, weight, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, weight)
    # Chunks accumulate the weight gradient in another order before the BF16 rounding.
    error = (actual_grad.float() - expected_grad.float()).norm() / expected_grad.float().norm()
    assert error < (1e-2 if dtype == "bfloat16" else 1e-5), error


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
