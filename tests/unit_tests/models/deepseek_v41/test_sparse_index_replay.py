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
"""CSA2 sparse-index plans: built once per selection lineage and replayed across activation-checkpoint recompute."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import nemo_automodel.components.models.deepseek_v4.optimized_kernels as kernels
import nemo_automodel.components.models.deepseek_v41.attention as attention_mod
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionState,
    csa2_selection_replay,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.moe.parallelizer import _with_model_checkpoint_context, apply_ac
from tests.unit_tests.models.deepseek_v41.test_attention import _backend, _config
from tests.unit_tests.models.deepseek_v41.test_model import _build_model


def reference_sparse_kernel(
    q, kv, sinks, topk_idxs, sm_scale, *, backend, reference_rounding=False, all_rows_nonempty=False, plan=None
):
    """Torch stand-in for the sparse kernels that understands a prepared plan (flat cuDNN or TileLang layout)."""
    batch, sequence = q.shape[:2]
    if plan is not None:
        assert plan.backend == backend and plan.kv_sequence == kv.shape[1]
        if backend == "cudnn":
            flat = plan.indices.reshape(batch, sequence, -1).long()
            offsets = (torch.arange(batch) * kv.shape[1]).view(batch, 1, 1)
            topk_idxs = torch.where(flat >= 0, flat - offsets, torch.full_like(flat, -1))
            assert torch.equal(plan.topk_length.view(batch, sequence), (flat >= 0).sum(-1).to(torch.int32))
            assert (plan.valid_rows is None) == all_rows_nonempty
        else:
            topk_idxs = plan.indices
    return kernels.sparse_attention_torch(q.float(), kv.float(), sinks, topk_idxs.long(), sm_scale).to(q.dtype)


def _reference_indexer_scores(queries, keys, weights, allowed):
    """Eager stand-in for the TileLang indexer scoring (CUDA only) with the same masking contract."""
    scores = torch.einsum("bshd,btd->bsht", queries.float(), keys.float())
    scores = (scores.relu() * weights.float().unsqueeze(-1)).sum(dim=2)
    return scores.masked_fill(~allowed, -torch.inf).to(queries.dtype)


def _use_torch_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the sparse backends' CUDA kernels (attention core and indexer scoring) to torch references."""
    monkeypatch.setattr(attention_mod, "dsv4_sparse_attention", reference_sparse_kernel)
    monkeypatch.setattr(attention_mod, "indexer_scores", _reference_indexer_scores)


def _layers(backend: str) -> list[DeepseekV41Attention]:
    config = _config()
    torch.manual_seed(5)
    return [DeepseekV41Attention(config, index, _backend(backend)) for index in range(config.num_hidden_layers)]


def _run_chain(layers, hidden, *, strip_plan=False, attention_mask=None):
    state = DeepseekV41AttentionState()
    positions = torch.arange(hidden.shape[1])[None]
    outputs, states = [], []
    for layer in layers:
        if strip_plan:
            state = replace(state, sparse_plan=None)
        out = layer(hidden, position_ids=positions, state=state, attention_mask=attention_mask)
        outputs.append(out.hidden_states)
        states.append(out.state)
        state = out.state
    return outputs, states


def _count_plans(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = kernels.plan_dsv4_sparse_indices

    def counting(*args, **kwargs):
        calls.append(kwargs["backend"])
        return real(*args, **kwargs)

    monkeypatch.setattr(attention_mod, "plan_dsv4_sparse_indices", counting)
    return calls


@pytest.mark.parametrize("backend", ["cudnn", "tilelang"])
def test_plan_is_built_once_per_selection_lineage_and_reused_bitwise(backend, monkeypatch) -> None:
    """Six layers, four lineages ({0}, {1, 2}, {3}, {4, 5}): four plans; a reuse layer carries its producer's plan."""
    _use_torch_kernels(monkeypatch)
    calls = _count_plans(monkeypatch)
    layers = _layers(backend)
    hidden = torch.randn(2, 6, 16)
    outputs, states = _run_chain(layers, hidden)
    assert calls == [backend] * 4
    assert states[2].sparse_plan is states[1].sparse_plan and states[5].sparse_plan is states[4].sparse_plan
    assert states[3].sparse_plan is not states[2].sparse_plan  # the indexer at layer 3 selects anew
    expected_kv = [6 + 6 // layer.compress_ratio if layer.compress_ratio else 6 for layer in layers]
    assert [state.sparse_plan.kv_sequence for state in states] == expected_kv
    assert all(state.sparse_plan.backend == backend for state in states)
    calls.clear()
    rebuilt, _ = _run_chain(layers, hidden, strip_plan=True)
    assert calls == [backend] * 6
    for actual, expected in zip(outputs, rebuilt):
        assert torch.equal(actual, expected)


def test_padded_inputs_plan_the_nonempty_rows_once_per_lineage(monkeypatch) -> None:
    _use_torch_kernels(monkeypatch)
    layers = _layers("cudnn")
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    _, states = _run_chain(layers, torch.randn(2, 6, 16), attention_mask=mask)
    for state in states:
        plan = state.sparse_plan
        assert plan.valid_rows is not None and plan.valid_rows.dtype == torch.int64
        assert torch.equal(plan.valid_rows, torch.nonzero(mask.flatten()).flatten())
        assert bool((plan.topk_length.view(2, 6)[mask == 0] == 0).all())


def test_stale_or_foreign_plans_are_rebuilt_and_indexers_drop_them(monkeypatch) -> None:
    _use_torch_kernels(monkeypatch)
    calls = _count_plans(monkeypatch)
    layers = _layers("cudnn")
    hidden = torch.randn(2, 6, 16)
    positions = torch.arange(6)[None]
    _, states = _run_chain(layers, hidden)
    calls.clear()
    clean = layers[2](hidden, position_ids=positions, state=states[1])
    stale = replace(states[1], sparse_plan=replace(states[1].sparse_plan, kv_sequence=7))
    rebuilt = layers[2](hidden, position_ids=positions, state=stale)
    assert calls == ["cudnn"] and rebuilt.state.sparse_plan.kv_sequence == 9
    assert torch.equal(clean.hidden_states, rebuilt.hidden_states)
    calls.clear()
    reselected = layers[4](hidden, position_ids=positions, state=states[3])
    assert calls == ["cudnn"] and reselected.state.sparse_plan is not states[3].sparse_plan
    plan = states[0].sparse_plan
    q = torch.zeros(2, 6, 4, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="sparse-index plan"):
        kernels.dsv4_sparse_attention(
            q, torch.zeros(2, 7, 8), torch.zeros(4), plan.indices, 0.25, backend="cudnn", plan=plan
        )
    with pytest.raises(ValueError, match="sparse-index plan"):
        kernels.dsv4_sparse_attention(
            q, torch.zeros(2, 6, 8), torch.zeros(4), plan.indices, 0.25, backend="tilelang", plan=plan
        )


@pytest.mark.parametrize("backend", ["cudnn", "tilelang"])
def test_checkpoint_replay_serves_the_plan_to_the_recompute(backend, monkeypatch) -> None:
    _use_torch_kernels(monkeypatch)
    calls = _count_plans(monkeypatch)
    layers = _layers(backend)
    hidden = torch.randn(2, 6, 16, requires_grad=True)
    positions = torch.arange(6)[None]
    source = layers[1](hidden, position_ids=positions, state=DeepseekV41AttentionState())
    calls.clear()
    out = checkpoint(
        layers[3],
        hidden,
        position_ids=positions,
        state=source.state,
        use_reentrant=False,
        context_fn=csa2_selection_replay.checkpoint_context_fn(None),
    )
    out.hidden_states.square().mean().backward()
    assert calls == [backend], "the recompute takes the recorded plan"
    replayed_grad, hidden.grad = hidden.grad.clone(), None
    calls.clear()
    plain = checkpoint(layers[3], hidden, position_ids=positions, state=source.state, use_reentrant=False)
    plain.hidden_states.square().mean().backward()
    assert calls == [backend, backend], "without the channel the recompute plans again"
    # fp32 tolerance, not bitwise: the planned path allocates different intermediates, and CPU GEMMs round
    # differently with buffer alignment (the mechanism assertions above are the exact checks).
    torch.testing.assert_close(out.hidden_states, plain.hidden_states)
    torch.testing.assert_close(hidden.grad, replayed_grad)


def _count_indexer_calls(model, layer: str = "2") -> list[int]:
    calls = [0]
    model.model.layers[layer].attn.indexer.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    return calls


def _build_model_with_backend(attn: str):
    from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from tests.unit_tests.models.deepseek_v41.test_model import _tiny_config

    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch")
    model = DeepseekV41ForCausalLM(_tiny_config(), backend=backend)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model


@pytest.mark.parametrize("padded", [False, True], ids=["full", "right-padded"])
@pytest.mark.parametrize("backend", ["eager", "cudnn"])
def test_checkpoint_replay_runs_each_indexer_once_and_matches_the_unwrapped_model(monkeypatch, backend, padded):
    """apply_ac with and without the router policy: indexer once per step, loss and gradients match no-AC. The
    cudnn backend (torch stand-ins for the kernels) carries the plan through the replay, with and without padding."""
    if backend == "cudnn":
        _use_torch_kernels(monkeypatch)
        monkeypatch.setattr(kernels, "_should_use_tilelang", lambda *args, **kwargs: False)  # the mHC sinkhorn
    torch.manual_seed(21)
    pristine = _build_model_with_backend(backend)
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12], [12, 3, 4, 5, 6, 7, 8, 9]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]]) if padded else None

    # exact on the eager path (the same ops in the same order); the cudnn stand-in allocates different
    # intermediates with the plan, and CPU GEMMs round with buffer alignment. Its torch reference puts NaN into
    # the sink gradient of fully padded query rows (the real kernel zeroes those rows), identically on both sides.
    def close(actual, expected):
        if backend == "eager":
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, equal_nan=True)

    def run_step(model):
        model.zero_grad(set_to_none=True)
        loss = model(tokens, attention_mask=mask).logits.float().square().mean()
        loss.backward()
        grads = {
            name.replace("_checkpoint_wrapped_module.", ""): parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), grads

    reference = copy.deepcopy(pristine)
    reference_calls = _count_indexer_calls(reference)
    ref_loss, ref_grads = run_step(reference)
    assert reference_calls[0] == 1
    for ignore_router in (False, True):
        no_replay = copy.deepcopy(pristine)
        for block in no_replay.model.layers.values():
            block.attn.reuse_selection_on_recompute = False
        apply_ac(no_replay, ignore_router=ignore_router)
        no_replay_calls = _count_indexer_calls(no_replay)
        nr_loss, nr_grads = run_step(no_replay)
        assert no_replay_calls[0] == 2, "without replay the recompute reruns the indexer"
        replay = copy.deepcopy(pristine)
        apply_ac(replay, ignore_router=ignore_router)
        replay_calls = _count_indexer_calls(replay)
        rp_loss, rp_grads = run_step(replay)
        assert replay_calls[0] == 1, "with replay the recompute takes the recorded selection"
        close(rp_loss, ref_loss)
        close(nr_loss, ref_loss)
        assert rp_grads.keys() == ref_grads.keys() == nr_grads.keys()
        for name in ref_grads:
            close(rp_grads[name], ref_grads[name])
            close(nr_grads[name], ref_grads[name])


def test_config_knob_defaults_on_and_the_block_hook_honours_it() -> None:
    assert DeepseekV41TextConfig().reuse_sparse_selection_on_recompute is True
    off = DeepseekV41TextConfig(reuse_sparse_selection_on_recompute=False)
    assert off.to_dict()["reuse_sparse_selection_on_recompute"] is False
    block = _build_model().model.layers["2"]
    wrapped = _with_model_checkpoint_context(block, None)
    assert callable(wrapped)
    forward_ctx, recompute_ctx = wrapped()
    with forward_ctx:
        assert csa2_selection_replay.current()[1] == "record"
    with recompute_ctx:
        assert csa2_selection_replay.current()[1] == "replay"
    assert csa2_selection_replay.current() is None
    block.attn.reuse_selection_on_recompute = False
    sentinel = object()
    assert _with_model_checkpoint_context(block, sentinel) is sentinel
