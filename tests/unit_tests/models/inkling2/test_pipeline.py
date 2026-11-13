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

"""Pipeline ownership, depth-state transport and gradients before distributed execution."""

import copy
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo_automodel.components.distributed.pipelining.functional import split_model_into_stages
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.unit_tests.models.inkling2.test_precision import initialize_nonuniform_queries
from tests.unit_tests.models.inkling2.test_reference import tiny_config

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def module_fqns(cuts: tuple[int, ...]) -> list[list[str]]:
    """Describe exclusive owners while preserving global layer FQNs."""
    stages = [[f"layers.{i}" for i in range(start, stop)] for start, stop in zip(cuts, cuts[1:])]
    stages[0].extend(["embed_tokens", "embed_norm"])
    stages[-1].extend(["output_query", "norm", "lm_head"])
    return stages


def partition(model: Inkling2ForCausalLM, cuts: tuple[int, ...], *, ac: bool = False) -> list[Inkling2ForCausalLM]:
    """Construct sequential stages with exact ownership; no communication is mocked here."""
    parts = []
    for start, stop in zip(cuts, cuts[1:]):
        part = copy.deepcopy(model)
        part.layers = nn.ModuleDict({str(i): part.layers[str(i)] for i in range(start, stop)})
        if start:
            part.embed_tokens = part.embed_norm = None
        if stop < model.config.num_hidden_layers:
            part.norm = part.lm_head = part.output_query = None
        if ac:
            for name, layer in part.layers.items():
                part.layers[name] = checkpoint_wrapper(layer)
        parts.append(part)
    return parts


def run(parts: list[Inkling2ForCausalLM], ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Execute stage owners in order with differentiable boundary tensors.

    Args:
        parts: Ordered pipeline stage modules.
        ids: Integer token IDs [batch, sequence].
        valid: Boolean real-token mask [batch, sequence].

    Returns:
        FP32 logits [batch, sequence, vocab], or final compute-dtype hidden states
        [batch, sequence, hidden] when enabled on the last stage. Inputs are unchanged.
    """
    value = parts[0](ids, attention_mask=valid)
    for part in parts[1:]:
        value = part(*value)
    return value


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("cuts", [(0, 6, 13), (0, 7, 13), (0, 3, 6, 10, 13)])
def test_all_gradients_across_residual_boundaries(dtype: str, ac: bool, cuts: tuple[int, ...]) -> None:
    torch.manual_seed(417)
    reference = Inkling2ForCausalLM(tiny_config(dtype=dtype))
    initialize_nonuniform_queries(reference)
    parts = partition(reference, cuts, ac=ac)
    ids = torch.randint(1, 37, (2, 9))
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, :2], valid[1, -2:] = False, False
    expected, actual = reference(ids, attention_mask=valid), run(parts, ids, valid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Two in-flight graphs, backward in reverse order, expose stale shared state.
    later_expected, later_actual = reference(ids.flip(1), attention_mask=valid), run(parts, ids.flip(1), valid)
    upstream, later_upstream = torch.randn_like(actual[..., :37]), torch.randn_like(actual[..., :37])
    later_actual[..., :37].backward(later_upstream)
    later_expected[..., :37].backward(later_upstream)
    actual[..., :37].backward(upstream)
    expected[..., :37].backward(upstream)
    owners = {}
    for part in parts:
        for name, parameter in part.named_parameters():
            name = canonical_parameter_fqn(name)
            assert name not in owners, name
            owners[name] = parameter
    wanted = dict(reference.named_parameters())
    assert owners.keys() == wanted.keys()
    for name, parameter in owners.items():
        actual_grad, expected_grad = parameter.grad, wanted[name].grad
        assert actual_grad is not None and expected_grad is not None, name
        assert torch.isfinite(actual_grad).all(), name
        if dtype == "float32":
            torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-5, atol=3e-6, msg=name)
        else:
            error = actual_grad.float() - expected_grad.float()
            relative = error.norm() / expected_grad.float().norm().clamp_min(1e-12)
            assert relative <= 0.08 or error.abs().max() <= 5e-4, (name, relative, error.abs().max())
    for name in (
        "embed_tokens.weight",
        "layers.0.attn.qkvr.weight",
        "layers.6.attn.qkvr.weight",
        "output_query.weight",
    ):
        assert owners[name].grad.float().norm() > 0, name


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_transport_metas_hidden_loss_and_checkpoint_union(dtype: str) -> None:
    model = Inkling2ForCausalLM(tiny_config(dtype=dtype))
    parts = partition(model, (0, 3, 6, 10, 13))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    value = (ids,)
    exported, native = {}, {}
    for index, part in enumerate(parts):
        inputs, outputs = part.get_pipeline_stage_metas(
            is_first=index == 0, microbatch_size=1, seq_len=5, dtype=getattr(torch, dtype)
        )
        assert [(t.shape, t.dtype) for t in value] == [(t.shape, t.dtype) for t in inputs]
        result = part(*value)
        value = result if isinstance(result, tuple) else (result,)
        assert [(t.shape, t.dtype) for t in value] == [(t.shape, t.dtype) for t in outputs]
        if index < len(parts) - 1:
            assert value[0].requires_grad and not value[1].requires_grad
        state = part.state_dict()
        assert not native.keys() & state.keys()
        native.update(state)
        converted = part.state_dict_adapter.to_hf(state)
        assert not exported.keys() & converted.keys()
        exported.update(converted)
    torch.testing.assert_close(native, model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(exported, model.state_dict_adapter.to_hf(model.state_dict()), rtol=0, atol=0)
    parts[-1]._pp_return_hidden_states = True
    torch.testing.assert_close(
        run(parts, ids, torch.ones_like(ids)), model(ids, output_hidden_states=True).hidden_states, rtol=0, atol=0
    )
    _, outputs = parts[-1].get_pipeline_stage_metas(
        is_first=False, microbatch_size=1, seq_len=5, dtype=getattr(torch, dtype)
    )
    assert outputs[0].shape == (1, 5, model.config.hidden_size)
    for part in parts:
        part.initialize_weights(torch.device("cpu"))
        assert all(torch.isfinite(p).all() for p in part.parameters())


def test_generated_split_uses_model_owned_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    # This test exercises actual trimming and hooks; the GPU suite exercises P2P.
    def stage_stub(
        submod: nn.Module, stage_index: int, num_stages: int, device: torch.device, group: None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(submod=submod, stage_index=stage_index)

    monkeypatch.setattr("nemo_automodel.components.distributed.pipelining.functional.PipelineStage", stage_stub)
    model = Inkling2ForCausalLM(tiny_config())
    parts = []
    for rank in range(2):
        mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: rank, get_group=lambda _: None)
        _, local = split_model_into_stages(model, mesh, "pp", "1f1b", torch.device("cpu"))
        parts.extend(local)
    assert parts[0]._pipeline_layer_ids() == list(range(7))
    assert parts[1]._pipeline_layer_ids() == list(range(7, 13))
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(run(parts, ids, torch.ones_like(ids)), model(ids), rtol=0, atol=0)


def test_invalid_ownership_and_transport_fail() -> None:
    model = Inkling2ForCausalLM(tiny_config())
    first, last = partition(model, (0, 6, 13))
    with pytest.raises(ValueError, match="complete residual"):
        last(torch.ones(1, 4, model.config.hidden_size))
    value = first(torch.ones(1, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="shape, dtype or device"):
        last(value[0][:, :, :1], value[1])
    with pytest.raises(ValueError, match="external pipeline loss"):
        first(torch.ones(1, 4, dtype=torch.long), logits_to_keep=1)
    last.embed_norm = copy.deepcopy(first.embed_norm)
    with pytest.raises(ValueError, match="exclusively own"):
        last.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=4, dtype=torch.float32)


def test_shared_trainer_padding_mask_contract() -> None:
    model = Inkling2ForCausalLM(tiny_config())
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    valid = torch.tensor([[False, True, True, True, False]])
    expected = model(ids, attention_mask=valid)
    torch.testing.assert_close(model(ids, padding_mask=~valid), expected, rtol=0, atol=0)
    first, last = partition(model, (0, 7, 13))
    state = first(ids, attention_mask=valid, padding_mask=~valid)
    # Repeated schedule kwargs cannot replace the per-microbatch carried layout.
    actual = last(*state, attention_mask=torch.ones_like(valid), padding_mask=torch.zeros_like(valid))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="padding_mask must be boolean"):
        first(ids, padding_mask=valid.long())
