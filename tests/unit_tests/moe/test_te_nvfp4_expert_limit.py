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

"""Transformer Engine experts under NVFP4: at most 64 experts per rank, and resident weights need those experts."""

import logging
import random
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import utils as common_utils
from nemo_automodel.components.models.common.utils import (
    HAVE_TE,
    BackendConfig,
    TENvfp4Config,
    _patch_te_modules,
    set_is_first_microbatch,
)
from nemo_automodel.components.moe import experts as experts_module
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsTE, _plan_dispatch_chunks
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.import_utils import is_te_min_version
from nemo_automodel.shared.recompute_replay import mark_checkpoint_recompute


class _FakeGroupedLinear(torch.nn.Module):
    """CPU stand-in for TE ``GroupedLinear``: one ``weight{i}`` per expert."""

    def __init__(
        self,
        *,
        first_expert,
        num_gemms,
        in_features,
        out_features,
        bias,
        params_dtype,
        device,
        use_grouped_tensor=False,
    ):
        super().__init__()
        self.num_gemms = num_gemms
        self.use_bias = bias
        for index in range(num_gemms):
            weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=params_dtype))
            self.register_parameter(f"weight{first_expert + index}", weight)


class _FakePadding(torch.nn.Module):
    def __init__(self, num_gemms, align_size=None):
        super().__init__()


class _FakeMesh:
    """Expert-parallel mesh of ``size`` ranks seen from rank 0."""

    def __init__(self, size: int) -> None:
        self._size = size

    def get_local_rank(self) -> int:
        return 0

    def size(self) -> int:
        return self._size

    def get_group(self) -> None:
        return None


@pytest.fixture
def fake_te(monkeypatch):
    """Stand in for Transformer Engine and the token dispatcher so the experts are built on CPU."""
    te = ModuleType("transformer_engine.pytorch")
    te.Fp8Padding, te.Fp8Unpadding = _FakePadding, _FakePadding
    monkeypatch.setattr("nemo_automodel.components.moe.experts._ExpertGroupedLinear", _FakeGroupedLinear)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te)
    monkeypatch.setattr("nemo_automodel.components.moe.experts.is_te_min_version", lambda version: True)
    monkeypatch.setattr("nemo_automodel.components.models.common.utils._patch_te_modules", lambda: None)
    monkeypatch.setattr(GroupedExpertsTE, "_normalize_moe_mesh", lambda self, mesh: mesh)
    monkeypatch.setattr(
        "nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher", lambda **kwargs: SimpleNamespace(**kwargs)
    )


def _moe_config(n_routed_experts: int) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=n_routed_experts,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=8,
        inter_dim=4,
        moe_inter_dim=4,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    ("quantization", "n_experts", "ep_size", "allowed"),
    [
        ({"te_nvfp4": {}}, 96, 1, False),
        ({"te_nvfp4": {}}, 130, 2, False),
        ({"te_nvfp4": {}}, 128, 2, True),
        ({"te_fp8": {"recipe": "mxfp8"}}, 96, 1, True),
        ({}, 96, 1, True),
    ],
)
def test_only_nvfp4_limits_the_experts_per_rank(fake_te, quantization, n_experts, ep_size, allowed):
    backend = BackendConfig(experts="te", dispatcher="deepep", **quantization)
    experts = GroupedExpertsTE(_moe_config(n_experts), backend=backend)

    if not allowed:
        with pytest.raises(ValueError, match="te_nvfp4 supports at most 64 experts per rank"):
            experts.init_token_dispatcher(_FakeMesh(ep_size))
        return

    experts.init_token_dispatcher(_FakeMesh(ep_size))
    assert experts.gate_up_linear.num_gemms == n_experts // ep_size
    assert experts.token_dispatcher.num_local_experts == n_experts // ep_size


@pytest.mark.parametrize(
    ("linear", "experts", "dispatcher", "world_size", "built"),
    [
        ("te", "torch_mm", "hybridep", 2, "GroupedExpertsDeepEP"),
        ("torch", "te", "hybridep", 1, "GroupedExperts"),
        ("te", "te", "torch", 2, "GroupedExperts"),
    ],
    ids=["torch experts", "one rank", "torch dispatcher"],
)
@pytest.mark.filterwarnings("ignore:.*dispatcher is enabled in config, but world size is 1")
def test_resident_weights_are_rejected_where_floating_point_experts_are_built(
    monkeypatch, linear, experts, dispatcher, world_size, built
):
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: world_size)
    choice = {"linear": linear, "experts": experts, "dispatcher": dispatcher}
    backend = BackendConfig(**choice, te_nvfp4={"resident_weights": True})

    with pytest.raises(ValueError, match=f"te_nvfp4.resident_weights needs .* this setup builds {built}\\."):
        MoE(_moe_config(8), backend)

    # The same setup without resident weights builds those experts.
    assert type(MoE(_moe_config(8), BackendConfig(**choice, te_nvfp4={})).experts).__name__ == built


# Transformer Engine experts under a quantization recipe: every dispatch runs on a path that is right for it.
#
# CPU tests of the size limits of the two GEMM paths under NVFP4, of the plan that cuts a dispatch above both into
# chunks of rows, and of the forward that follows them, against a stand-in for ``GroupedLinear``.


# A rank of 32 experts of 1024 -> 2688 -> 1024; 45,056 rows per expert are 22 routes for each of 8 x 8192 tokens.
RANK_EXPERTS = 32
RANK_WIDTH = 2688
# One quantization of all rows is right up to these element counts: 15/16 of the 2**31 and 2**32 elements at
# which the grouped quantization and the quantization of the per-expert path start to fail.
GROUPED_LIMIT = 2013265920
PER_EXPERT_LIMIT = 4026531840
# Rows that Transformer Engine pads every expert to under NVFP4.
NVFP4_ROW_ALIGN = 128


@dataclass
class _Recipe:
    """Stand-in for a Transformer Engine recipe: the predicates the experts read."""

    kind: str

    def nvfp4(self) -> bool:
        return self.kind == "nvfp4"

    def custom(self) -> bool:
        return self.kind == "custom"

    def mxfp8(self) -> bool:
        return self.kind == "mxfp8"

    def float8_block_scaling(self) -> bool:
        return False


@dataclass
class _Call:
    """One forward of the stand-in ``GroupedLinear``."""

    splits: list[int]
    splits_on_host: bool
    grouped_tensor: bool
    weight_cache: bool | None = None
    output_buffer: bool = False


class _RecordingGroupedLinear(torch.nn.Module):
    """CPU stand-in for TE ``GroupedLinear``: one FP32 GEMM per expert, and a record of every call."""

    def __init__(
        self,
        *,
        first_expert: int,
        num_gemms: int,
        in_features: int,
        out_features: int,
        bias: bool,
        params_dtype: torch.dtype,
        device: str,
        use_grouped_tensor: bool = False,
    ) -> None:
        super().__init__()
        self.num_gemms = num_gemms
        self.use_grouped_tensor = use_grouped_tensor
        self.calls: list[_Call] = []
        self.fail = False
        for index in range(num_gemms):
            weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=params_dtype) * 0.1)
            self.register_parameter(f"weight{first_expert + index}", weight)

    def forward(
        self,
        values: torch.Tensor,
        splits: torch.Tensor | list[int],
        is_first_microbatch: bool | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multiply the rows of every expert with its weight."""
        # Transformer Engine takes device split sizes on the grouped-tensor path and host ones otherwise.
        assert isinstance(splits, torch.Tensor) == self.use_grouped_tensor
        counts = splits.tolist() if isinstance(splits, torch.Tensor) else list(splits)
        assert sum(counts) == values.shape[0]
        self.calls.append(
            _Call(counts, isinstance(splits, list), self.use_grouped_tensor, is_first_microbatch, out is not None)
        )
        if self.fail:
            raise RuntimeError("GEMM failed")
        outputs, row = [], 0
        for expert, count in enumerate(counts):
            outputs.append(F.linear(values[row : row + count], getattr(self, f"weight{expert}")))
            row += count
        result = torch.cat(outputs)
        if out is not None:
            # Transformer Engine returns a view of the buffer; here the buffer gets the values and the caller
            # the tensor that carries the gradient.
            assert out.shape == result.shape and not out.requires_grad
            with torch.no_grad():
                out.copy_(result)
        return result


class _Padding(torch.nn.Module):
    """Stand-in for TE ``Fp8Padding``: nothing is padded; ``align_size`` is the row multiple it would pad to."""

    def __init__(self, num_gemms: int, align_size: int | None = None) -> None:
        super().__init__()
        self.align_size = 1 if align_size is None else align_size

    def forward(self, values: torch.Tensor, splits: list[int]) -> tuple[torch.Tensor, list[int]]:
        """Return the rows and their split sizes unchanged."""
        return values, splits


class _Unpadding(torch.nn.Module):
    """Stand-in for TE ``Fp8Unpadding``: nothing was padded."""

    def __init__(self, num_gemms: int, align_size: int | None = None) -> None:
        super().__init__()

    def forward(self, values: torch.Tensor, splits: list[int]) -> torch.Tensor:
        """Return the rows unchanged."""
        return values


@dataclass
class _GroupedInputDispatcher:
    """Keep already expert-grouped rows in place."""

    counts: list[int]

    def token_permutation2(
        self, hidden_states: torch.Tensor, num_local_tokens: int, token_probs: torch.Tensor, token_indices: torch.Tensor
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """Return the inputs without communication or permutation."""
        return hidden_states, self.counts, token_probs[:, 0]

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return expert-grouped outputs unchanged."""
        return hidden_states


EXPERT_DIM, INTER_DIM = 8, 16


@pytest.fixture
def fake_te_state(monkeypatch):
    """Stand in for Transformer Engine so the experts run on CPU; yields the quantization state to set a recipe."""
    state = Mock()
    state.is_fp8_enabled.return_value = False
    te = ModuleType("transformer_engine.pytorch")
    te.Fp8Padding, te.Fp8Unpadding = _Padding, _Unpadding
    quantization = ModuleType("transformer_engine.pytorch.quantization")
    quantization.FP8GlobalStateManager = state
    grouped_linear = ModuleType("transformer_engine.pytorch.module.grouped_linear")
    grouped_linear.is_module_grouped_tensor_path_supported = lambda recipe, dtype: True
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.quantization", quantization)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module.grouped_linear", grouped_linear)
    monkeypatch.setattr(experts_module, "_ExpertGroupedLinear", _RecordingGroupedLinear)
    monkeypatch.setattr(experts_module, "is_te_min_version", lambda version: True)
    monkeypatch.setattr("nemo_automodel.components.models.common.utils._patch_te_modules", lambda: None)
    # The counts behind the log lines are kept per process: every test starts from zero.
    monkeypatch.setattr(experts_module, "_DISPATCH_COUNTS", experts_module._QuantizedDispatchCounts())
    monkeypatch.setattr(common_utils, "IS_FIRST_MICROBATCH", None)
    return state


def _experts(counts: list[int], *, native: bool) -> GroupedExpertsTE:
    config = MoEConfig(
        n_routed_experts=len(counts),
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=EXPERT_DIM,
        inter_dim=INTER_DIM,
        moe_inter_dim=INTER_DIM,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="relu2",
        dtype=torch.float32,
    )
    backend = BackendConfig(experts="te", dispatcher="deepep", te_grouped_tensor=native)
    module = GroupedExpertsTE(config, backend=backend)
    module.ep_size = 1
    module.token_dispatcher = _GroupedInputDispatcher(counts)
    module.expert_activation = lambda values, probabilities: torch.relu(values).pow(2) * probabilities
    return module


def _set_recipe(state: Mock, kind: str | None) -> None:
    state.is_fp8_enabled.return_value = kind is not None
    state.get_fp8_recipe.return_value = _Recipe(kind) if kind is not None else None


def _run(
    module: GroupedExpertsTE, hidden: torch.Tensor, probs: torch.Tensor, upstream: torch.Tensor
) -> dict[str, torch.Tensor | None]:
    """Forward and backward of the experts on rows that are already grouped by expert."""
    module.zero_grad(set_to_none=True)
    for layer in (module.gate_up_linear, module.down_linear):
        layer.calls.clear()
    x = hidden.clone().requires_grad_()
    p = probs.clone().requires_grad_()
    rows = hidden.shape[0]
    output = module(x, torch.ones(rows, dtype=torch.bool), p, torch.zeros(rows, 1, dtype=torch.long))
    output.backward(upstream)
    values = {"output": output.detach(), "input_grad": x.grad, "probability_grad": p.grad}
    values.update({name: parameter.grad for name, parameter in module.named_parameters()})
    return values


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == experts_module.__name__]


@pytest.mark.cpu_only
def test_limits_stay_below_the_offsets_at_which_the_kernels_fail():
    assert experts_module._NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS == GROUPED_LIMIT == 2**31 * 15 // 16
    assert experts_module._NVFP4_SPLIT_QUANTIZE_MAX_ELEMENTS == PER_EXPERT_LIMIT == 2**32 * 15 // 16


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("rows_per_expert", "max_rows", "align", "expected"),
    [
        # Everything fits one chunk.
        ([4, 8, 0, 4], 16, 4, [[4, 8, 0, 4]]),
        # Two experts per chunk of 8 rows.
        ([4, 4, 4, 4], 8, 4, [[4, 4, 0, 0], [0, 0, 4, 4]]),
        # The third expert does not fit behind the first one any more and gets a chunk of its own.
        ([4, 0, 8, 4], 8, 4, [[4, 0, 0, 0], [0, 0, 8, 0], [0, 0, 0, 4]]),
        # The second expert is larger than a chunk: it fills the first chunk, two more, and nothing is left.
        ([4, 20, 4], 8, 4, [[4, 4, 0], [0, 8, 0], [0, 8, 0], [0, 0, 4]]),
        # A chunk holds a whole number of blocks: 10 rows are two blocks of 4.
        ([8, 8], 10, 4, [[8, 0], [0, 8]]),
        # One expert alone, cut into chunks of three blocks of 2 rows.
        ([14], 7, 2, [[6], [6], [2]]),
        # A dispatch without rows is one call without rows.
        ([0, 0], 8, 4, [[0, 0]]),
    ],
)
def test_chunk_plan_of_small_dispatches(rows_per_expert, max_rows, align, expected):
    assert _plan_dispatch_chunks(rows_per_expert, max_rows=max_rows, align=align) == expected


@pytest.mark.cpu_only
def test_chunk_plan_rejects_a_chunk_smaller_than_one_block():
    with pytest.raises(ValueError, match="at most 100 rows cannot hold a block of 128 rows"):
        _plan_dispatch_chunks([128, 256], max_rows=100, align=128)


@pytest.mark.cpu_only
def test_chunk_plan_covers_every_row_in_order_within_the_limit():
    generator = random.Random(7)
    for _ in range(300):
        align = generator.choice([1, 2, 16, 128])
        experts = generator.randint(1, 12)
        rows_per_expert = [generator.choice([0, 0, 1, 2, 3, 5, 9, 40]) * align for _ in range(experts)]
        max_rows = generator.randint(align, 24 * align)
        capacity = max_rows // align * align

        chunks = _plan_dispatch_chunks(rows_per_expert, max_rows=max_rows, align=align)

        # Every row of every expert is in exactly one chunk, in whole blocks, and no chunk is over the limit.
        assert [sum(chunk[expert] for chunk in chunks) for expert in range(experts)] == rows_per_expert
        assert all(len(chunk) == experts and sum(chunk) <= max_rows for chunk in chunks)
        assert all(rows % align == 0 for chunk in chunks for rows in chunk)
        # The chunks are consecutive row ranges of the dispatched tensor: listing (expert, rows) chunk by chunk
        # gives the experts in order, each one's pieces next to each other.
        pieces = [(expert, rows) for chunk in chunks for expert, rows in enumerate(chunk) if rows]
        assert [expert for expert, _ in pieces] == sorted(expert for expert, _ in pieces)
        # An expert that fits a chunk is not cut; a larger one is cut into full chunks except for its ends.
        for expert, rows in enumerate(rows_per_expert):
            parts = [chunk[expert] for chunk in chunks if chunk[expert]]
            if rows <= capacity:
                assert len(parts) <= 1
            else:
                assert all(part == capacity for part in parts[1:-1])
        # No chunk is empty unless the dispatch is, and no chunk could take the first piece of the next one.
        if sum(rows_per_expert):
            assert all(sum(chunk) > 0 for chunk in chunks)
        for chunk, following in zip(chunks, chunks[1:]):
            expert, first_piece = next((expert, rows) for expert, rows in enumerate(following) if rows)
            whole_expert_moved = chunk[expert] == 0 and rows_per_expert[expert] <= capacity
            assert sum(chunk) + (rows_per_expert[expert] if whole_expert_moved else first_piece) > capacity


def _expected_calls(kind: str | None, native: bool, m_splits: list[int]) -> list[list[int]] | None:
    """What a dispatch of the 32-expert rank needs, derived from its element count."""
    if kind is None:
        return None
    elements = sum(m_splits) * RANK_WIDTH
    if kind in ("nvfp4", "custom") and elements > PER_EXPERT_LIMIT:
        rows = max(m_splits)
        # Whole experts per chunk: the per-expert limit in rows, rounded down to blocks, holds this many.
        per_chunk = (PER_EXPERT_LIMIT // RANK_WIDTH // NVFP4_ROW_ALIGN * NVFP4_ROW_ALIGN) // rows
        assert per_chunk >= 1
        with_rows = [expert for expert, count in enumerate(m_splits) if count]
        return [
            [rows if expert in with_rows[start : start + per_chunk] else 0 for expert in range(RANK_EXPERTS)]
            for start in range(0, len(with_rows), per_chunk)
        ]
    if native and kind in ("nvfp4", "custom") and elements > GROUPED_LIMIT:
        return [m_splits]
    return None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("kind", "native", "rows_per_expert", "empty_experts", "expected_calls"),
    [
        # NVFP4 with te_grouped_tensor: grouped-tensor GEMMs up to their limit, one GEMM per expert above it,
        # chunks above the limit of that path.
        ("nvfp4", True, 22528, 0, None),
        ("nvfp4", True, 23296, 0, None),
        ("nvfp4", True, 23424, 0, 1),
        ("nvfp4", True, 45056, 0, 1),
        ("nvfp4", True, 46720, 0, 1),
        ("nvfp4", True, 46848, 0, 2),
        ("nvfp4", True, 100352, 0, 3),
        # NVFP4 with one GEMM per expert: nothing to change up to its limit, chunks above it.
        ("nvfp4", False, 45056, 0, None),
        ("nvfp4", False, 46720, 0, None),
        ("nvfp4", False, 46848, 0, 2),
        # A custom recipe can hold NVFP4 quantizers; it only runs with one GEMM per expert.
        ("custom", False, 46720, 0, None),
        ("custom", False, 46848, 0, 2),
        # MXFP8 and unquantized GEMMs are not limited.
        ("mxfp8", True, 100352, 0, None),
        ("mxfp8", False, 100352, 0, None),
        (None, True, 100352, 0, None),
        # Experts without rows change nothing: the size alone decides.
        ("nvfp4", True, 1024, 3, None),
        ("mxfp8", True, 1024, 1, None),
        ("nvfp4", False, 1024, 3, None),
        ("nvfp4", True, 45056, 3, 1),
        ("nvfp4", True, 100352, 16, 2),
        # Without a recipe the split sizes stay on the device and the dispatch is not inspected.
        (None, True, 1024, 3, None),
    ],
)
def test_calls_of_a_dispatch_by_recipe_path_and_size(
    fake_te_state, monkeypatch, kind, native, rows_per_expert, empty_experts, expected_calls
):
    module = _experts([0] * RANK_EXPERTS, native=native)
    # The widths of the rank above; the plan reads nothing else from the module.
    monkeypatch.setattr(module.config, "moe_latent_size", 1024)
    monkeypatch.setattr(module.config, "moe_inter_dim", RANK_WIDTH)
    recipe = _Recipe(kind) if kind is not None else None
    m_splits = [0] * empty_experts + [rows_per_expert] * (RANK_EXPERTS - empty_experts)
    expected = _expected_calls(kind, native, m_splits)
    assert (None if expected is None else len(expected)) == expected_calls

    assert module._per_expert_chunks(recipe, m_splits, NVFP4_ROW_ALIGN) == expected


@pytest.mark.cpu_only
def test_a_dispatch_too_large_for_grouped_quantization_runs_one_gemm_per_expert(fake_te_state, monkeypatch, caplog):
    torch.manual_seed(3)
    counts = [5, 4, 7, 3, 9, 2]
    module = _experts(counts, native=True)
    hidden = torch.randn(sum(counts), EXPERT_DIM)
    probs = torch.rand(sum(counts), 1)
    upstream = torch.randn(sum(counts), EXPERT_DIM)
    _set_recipe(fake_te_state, "nvfp4")
    layers = (module.gate_up_linear, module.down_linear)

    grouped = _run(module, hidden, probs, upstream)
    assert [layer.calls for layer in layers] == [[_Call(counts, False, True)]] * 2

    # 30 rows of the 16-wide intermediate are 480 elements: above the grouped limit, within the other.
    monkeypatch.setattr(experts_module, "_NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS", 30 * INTER_DIM - 1)
    monkeypatch.setattr(experts_module, "_NVFP4_SPLIT_QUANTIZE_MAX_ELEMENTS", 30 * INTER_DIM)
    with caplog.at_level(logging.WARNING, logger=experts_module.__name__):
        per_expert = _run(module, hidden, probs, upstream)
        _run(module, hidden, probs, upstream)

    # Both projections ran as modules without grouped-tensor GEMMs, with host split sizes, and are restored.
    assert [layer.calls for layer in layers] == [[_Call(counts, True, False)]] * 2
    assert all(layer.use_grouped_tensor for layer in layers)
    assert grouped.keys() == per_expert.keys()
    for name, expected in grouped.items():
        torch.testing.assert_close(per_expert[name], expected, rtol=1e-6, atol=1e-6, msg=name)
    # Two dispatches left the grouped-tensor path, one line says so.
    messages = _messages(caplog)
    assert len(messages) == 1, messages
    assert f"a dispatch of 30 rows holds {30 * INTER_DIM} elements, more than the {30 * INTER_DIM - 1}" in messages[0]
    assert "such dispatches run one GEMM per expert instead" in messages[0]

    # One row less fits the grouped quantization again.
    monkeypatch.setattr(experts_module, "_NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS", 30 * INTER_DIM)
    _run(module, hidden, probs, upstream)
    assert [layer.calls for layer in layers] == [[_Call(counts, False, True)]] * 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8"])
def test_a_dispatch_with_an_empty_expert_stays_on_the_path_the_module_is_built_for(fake_te_state, caplog, kind):
    torch.manual_seed(5)
    counts = [5, 0, 7, 3]
    module = _experts(counts, native=True)
    hidden = torch.randn(sum(counts), EXPERT_DIM)
    probs = torch.rand(sum(counts), 1)
    upstream = torch.randn(sum(counts), EXPERT_DIM)
    layers = (module.gate_up_linear, module.down_linear)

    unquantized = _run(module, hidden, probs, upstream)
    assert [layer.calls for layer in layers] == [[_Call(counts, False, True)]] * 2

    _set_recipe(fake_te_state, kind)
    with caplog.at_level(logging.WARNING, logger=experts_module.__name__):
        quantized = _run(module, hidden, probs, upstream)

    # Grouped-tensor GEMMs with device split sizes, as without a recipe, and nothing to report.
    assert [layer.calls for layer in layers] == [[_Call(counts, False, True)]] * 2
    for name, expected in unquantized.items():
        torch.testing.assert_close(quantized[name], expected, rtol=1e-6, atol=1e-6, msg=name)
    assert _messages(caplog) == []
    # The expert without rows gets zero gradients from the call.
    for layer in layers:
        assert torch.count_nonzero(layer.weight1.grad) == 0
        assert torch.count_nonzero(layer.weight0.grad) > 0


@pytest.mark.cpu_only
def test_the_grouped_tensor_path_is_restored_when_a_gemm_fails(fake_te_state, monkeypatch):
    counts = [4, 6]
    module = _experts(counts, native=True)
    _set_recipe(fake_te_state, "nvfp4")
    monkeypatch.setattr(experts_module, "_NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS", 9 * INTER_DIM)
    module.gate_up_linear.fail = True
    hidden = torch.randn(sum(counts), EXPERT_DIM)

    with pytest.raises(RuntimeError, match="GEMM failed"):
        module(hidden, torch.ones(10, dtype=torch.bool), torch.rand(10, 1), torch.zeros(10, 1, dtype=torch.long))

    assert module.gate_up_linear.calls == [_Call(counts, True, False)]
    assert module.gate_up_linear.use_grouped_tensor and module.down_linear.use_grouped_tensor


@pytest.mark.cpu_only
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("first_microbatch", [True, False, None])
def test_a_dispatch_too_large_for_one_nvfp4_quantization_runs_in_chunks(
    fake_te_state, monkeypatch, caplog, native, first_microbatch
):
    torch.manual_seed(11)
    # Blocks of 2 rows. 26 rows; a chunk holds 10: experts 0-1, expert 2 in three pieces of which the first fills
    # the first chunk, expert 4 behind its last piece, expert 5 alone.
    counts = [4, 2, 14, 0, 2, 4]
    chunks = [[4, 2, 4, 0, 0, 0], [0, 0, 10, 0, 0, 0], [0, 0, 0, 0, 2, 4]]
    module = _experts(counts, native=native)
    module.fp8_padding.align_size = 2
    hidden = torch.randn(sum(counts), EXPERT_DIM)
    probs = torch.rand(sum(counts), 1)
    upstream = torch.randn(sum(counts), EXPERT_DIM)
    _set_recipe(fake_te_state, "nvfp4")
    monkeypatch.setattr(common_utils, "IS_FIRST_MICROBATCH", first_microbatch)
    layers = (module.gate_up_linear, module.down_linear)

    # One call on the path the module is built for; here it only supplies the values.
    single = _run(module, hidden, probs, upstream)
    assert [len(layer.calls) for layer in layers] == [1, 1]

    # 10 rows of the 16-wide intermediate fit one quantization, 11 do not.
    monkeypatch.setattr(experts_module, "_NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS", 5 * INTER_DIM)
    monkeypatch.setattr(experts_module, "_NVFP4_SPLIT_QUANTIZE_MAX_ELEMENTS", 11 * INTER_DIM - 1)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=experts_module.__name__):
        chunked = _run(module, hidden, probs, upstream)

    # Every chunk went through both projections with one GEMM per expert and host split sizes. The first call
    # leaves the weight cache to the projection, the later ones reuse what the first quantized whenever weights
    # are cached at all; the down projections wrote into one buffer.
    later = {True: False, False: False, None: None}[first_microbatch]
    for layer, buffer in zip(layers, (False, True)):
        assert layer.calls == [
            _Call(chunk, True, False, None if index == 0 else later, buffer) for index, chunk in enumerate(chunks)
        ]
    assert all(layer.use_grouped_tensor is native for layer in layers)
    assert single.keys() == chunked.keys()
    for name, expected in single.items():
        torch.testing.assert_close(chunked[name], expected, rtol=1e-5, atol=1e-6, msg=name)
    messages = _messages(caplog)
    assert len(messages) == 1, messages
    assert f"a dispatch of 26 rows holds {26 * INTER_DIM} elements, more than the {11 * INTER_DIM - 1}" in messages[0]
    assert "in consecutive chunks of rows (3 here), with one GEMM per expert" in messages[0]


@pytest.mark.cpu_only
def test_the_counts_of_dispatches_off_the_configured_path_are_logged(fake_te_state, monkeypatch, caplog):
    torch.manual_seed(13)
    module = _experts([4, 4, 4], native=True)
    _set_recipe(fake_te_state, "nvfp4")
    # 8 rows fit the grouped quantization, 12 fit one quantization of the per-expert path.
    monkeypatch.setattr(experts_module, "_NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS", 8 * INTER_DIM)
    monkeypatch.setattr(experts_module, "_NVFP4_SPLIT_QUANTIZE_MAX_ELEMENTS", 12 * INTER_DIM)
    # 256 dispatches: 156 on grouped-tensor GEMMs (56 of them with an empty expert), 60 too large for them, 40 in
    # chunks (30 of 16 rows in 2 chunks and 10 of 28 rows in 3).
    dispatches = [[4, 2, 2]] * 100 + [[4, 4, 4]] * 60 + [[4, 0, 2]] * 56 + [[8, 4, 4]] * 30 + [[12, 12, 4]] * 10
    assert len(dispatches) == 256

    with caplog.at_level(logging.WARNING, logger=experts_module.__name__), torch.no_grad():
        for counts in dispatches:
            module.token_dispatcher = _GroupedInputDispatcher(counts)
            rows = sum(counts)
            module(
                torch.randn(rows, EXPERT_DIM),
                torch.ones(rows, dtype=torch.bool),
                torch.rand(rows, 1),
                torch.zeros(rows, 1, dtype=torch.long),
            )

    messages = _messages(caplog)
    # One line for the first dispatch of each kind, and the counts at the 256th dispatch.
    assert len(messages) == 3, messages
    assert "such dispatches run one GEMM per expert instead" in messages[0]
    assert "in consecutive chunks of rows (2 here)" in messages[1]
    assert messages[2] == (
        "TE experts: of the 256 quantized expert dispatches of this rank so far, 40 ran in chunks (90 chunks in "
        "all, the largest dispatch 28 rows), and 60 left the grouped-tensor GEMMs for one GEMM per expert because "
        "of their size."
    )


# Quantized weights of Transformer Engine GEMM modules across an activation-checkpoint recompute.
#
# FP8 with current scaling runs on every GPU Transformer Engine quantizes on; MXFP8 and NVFP4 need Blackwell.


def _block_scaling_capable() -> bool:
    return HAVE_TE and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


RecipeName = Literal["fp8", "mxfp8", "nvfp4"]
_blackwell_only = pytest.mark.skipif(not _block_scaling_capable(), reason="MXFP8 and NVFP4 GEMMs need Blackwell")
RECIPES = ["fp8", pytest.param("mxfp8", marks=_blackwell_only), pytest.param("nvfp4", marks=_blackwell_only)]
ModuleKind = Literal["grouped_linear", "linear"]
ContextFn = Callable[[], tuple[AbstractContextManager, AbstractContextManager]]

EXPERTS = 4
ROWS = 128  # per expert: the NVFP4 group alignment
IN_FEATURES = 128
OUT_FEATURES = 256


def _recipe(name: RecipeName):
    """A recipe without random rounding, so that equal operands give equal bits."""
    from transformer_engine.common.recipe import Float8CurrentScaling, MXFP8BlockScaling

    if name == "fp8":
        return Float8CurrentScaling()
    if name == "mxfp8":
        return MXFP8BlockScaling()
    return TENvfp4Config(stochastic_rounding=False).build_recipe()


def _module(kind: ModuleKind) -> torch.nn.Module:
    import transformer_engine.pytorch as te

    torch.manual_seed(0)
    if kind == "grouped_linear":
        return te.GroupedLinear(EXPERTS, IN_FEATURES, OUT_FEATURES, bias=False, params_dtype=torch.bfloat16).cuda()
    return te.Linear(IN_FEATURES, OUT_FEATURES, bias=False, params_dtype=torch.bfloat16).cuda()


def _count_weight_quantizations(kind: ModuleKind, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record, per weight handed to Transformer Engine's weight quantization, whether it was quantized."""
    import transformer_engine.pytorch.module.grouped_linear as grouped_linear_module
    import transformer_engine.pytorch.module.linear as linear_module

    te_module = grouped_linear_module if kind == "grouped_linear" else linear_module
    quantize_weight = te_module.quantize_weight
    quantized: list[bool] = []

    def counting(**kwargs):
        quantized.append(kwargs["workspace"] is None or kwargs["update_workspace"])
        return quantize_weight(**kwargs)

    monkeypatch.setattr(te_module, "quantize_weight", counting)
    return quantized


def _checkpointed_step(
    module: torch.nn.Module,
    kind: ModuleKind,
    recipe_name: RecipeName,
    context_fn: ContextFn | None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Forward and backward of ``module`` through a non-reentrant checkpoint, both inside the autocast."""
    from transformer_engine.pytorch.quantization import autocast as te_autocast

    generator = torch.Generator(device="cuda").manual_seed(1)
    rows = EXPERTS * ROWS
    inputs = torch.randn(rows, IN_FEATURES, device="cuda", generator=generator).bfloat16().requires_grad_(True)
    output_grad = torch.randn(rows, OUT_FEATURES, device="cuda", generator=generator).bfloat16()
    for parameter in module.parameters():
        parameter.grad = None

    def body(x: torch.Tensor) -> torch.Tensor:
        if kind == "grouped_linear":
            return module(x, [ROWS] * EXPERTS)
        return module(x)

    checkpoint_kwargs = {} if context_fn is None else {"context_fn": context_fn}
    with te_autocast(enabled=True, recipe=_recipe(recipe_name)):
        output = checkpoint(body, inputs, use_reentrant=False, **checkpoint_kwargs)
        output.backward(output_grad)
    return output.detach(), inputs.grad, [parameter.grad.clone() for parameter in module.parameters()]


def _assert_same(
    left: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],
    right: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],
) -> None:
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[1], right[1])
    assert len(left[2]) == len(right[2])
    for left_grad, right_grad in zip(left[2], right[2], strict=True):
        assert torch.equal(left_grad, right_grad)


@pytest.mark.skipif(
    not (HAVE_TE and torch.cuda.is_available() and is_te_min_version("2.19.0")),
    reason="requires Transformer Engine 2.19+ and a GPU",
)
class TestTEWeightReuseOnRecompute:
    @pytest.fixture(autouse=True)
    def _microbatch_flag(self):
        _patch_te_modules()
        yield
        set_is_first_microbatch(None)

    @pytest.mark.parametrize("recipe_name", RECIPES)
    @pytest.mark.parametrize("kind", ["grouped_linear", "linear"])
    def test_recompute_of_a_first_microbatch_reuses_the_weights_its_forward_quantized(
        self, kind: ModuleKind, recipe_name: RecipeName, monkeypatch: pytest.MonkeyPatch
    ):
        module = _module(kind)
        num_weights = len(list(module.parameters()))
        quantized = _count_weight_quantizations(kind, monkeypatch)

        # Without the mark the recompute quantizes every weight a second time.
        set_is_first_microbatch(True)
        unmarked = _checkpointed_step(module, kind, recipe_name, None)
        assert quantized == [True] * num_weights + [True] * num_weights

        quantized.clear()
        marked = _checkpointed_step(module, kind, recipe_name, mark_checkpoint_recompute(None))
        assert quantized == [True] * num_weights + [False] * num_weights
        _assert_same(marked, unmarked)

        # Later micro-batches reuse the copy in the forward and in the recompute, as before.
        quantized.clear()
        set_is_first_microbatch(False)
        later = _checkpointed_step(module, kind, recipe_name, mark_checkpoint_recompute(None))
        assert quantized == [False] * (2 * num_weights)
        _assert_same(later, unmarked)

    @pytest.mark.parametrize("recipe_name", RECIPES)
    @pytest.mark.parametrize("kind", ["grouped_linear", "linear"])
    def test_reused_weights_follow_an_optimizer_step(self, kind: ModuleKind, recipe_name: RecipeName):
        module = _module(kind)
        marked_context = mark_checkpoint_recompute(None)

        set_is_first_microbatch(True)
        before = _checkpointed_step(module, kind, recipe_name, marked_context)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.mul_(-1.5)

        # A forward that reuses the copy of the old weights is what a stale reuse would compute.
        set_is_first_microbatch(False)
        stale = _checkpointed_step(module, kind, recipe_name, marked_context)
        _assert_same(stale, before)

        # The first micro-batch after the update quantizes the new weights and its recompute reuses those.
        set_is_first_microbatch(True)
        after = _checkpointed_step(module, kind, recipe_name, marked_context)
        # Reference: no stored copy at all, every forward and recompute quantizes the weights it is given.
        set_is_first_microbatch(None)
        reference = _checkpointed_step(module, kind, recipe_name, None)
        _assert_same(after, reference)
        assert not torch.equal(after[0], before[0])

    def test_recompute_without_a_microbatch_flag_keeps_quantizing(self, monkeypatch: pytest.MonkeyPatch):
        module = _module("grouped_linear")
        quantized = _count_weight_quantizations("grouped_linear", monkeypatch)
        set_is_first_microbatch(None)
        _checkpointed_step(module, "grouped_linear", "fp8", mark_checkpoint_recompute(None))
        assert quantized == [True] * (2 * EXPERTS)
