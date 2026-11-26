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

"""Triton attention of Inkling2 against the eager reference of the same module."""

import copy
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2 import triton_attention
from nemo_automodel.components.models.inkling2.attention import AttentionSources, Inkling2Attention
from nemo_automodel.components.models.inkling2.config import Inkling2TextConfig
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.models.inkling2.triton_attention import HAVE_TRITON, relative_attention

needs_gpu_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="The Triton attention kernels need a GPU and Triton"
)

# Neither divides the kernel tiles (32 queries and keys in FP32; 64 or 128 in BF16), and
# both are shorter than SEQUENCE so that the window and the relative extent cut keys off.
WINDOW = 37
EXTENT = 48
SEQUENCE = 150
LOCAL_LAYER, FULL_LAYER = 0, 5
GRADIENTS = ("q", "k", "v", "r", "table", "sink")
# (relative, absolute as a fraction of the largest reference value) per input dtype. FP32 inputs
# differ by accumulation order only (1e-6 measured). With BF16 inputs the kernels round the
# probabilities to BF16 for the value product and return BF16 gradients: 1.4e-3 measured on
# the context and up to 1.2e-2 on a gradient, over these layouts and four seeds.
CONTEXT_TOLERANCE = {torch.float32: (1e-4, 1e-5), torch.bfloat16: (1e-2, 5e-3)}
GRADIENT_TOLERANCE = {torch.float32: (1e-4, 1e-5), torch.bfloat16: (5e-2, 3e-2)}


@pytest.fixture(autouse=True)
def exact_reference_products() -> Iterator[None]:
    """Keep TF32 out of the eager reference's matrix products."""
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


def _config() -> Inkling2TextConfig:
    return Inkling2TextConfig(
        hidden_size=32,
        num_hidden_layers=13,
        vocab_size=40,
        unpadded_vocab_size=37,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=3,
        rel_extent=EXTENT,
        sliding_window_size=WINDOW,
        n_routed_experts=12,
        num_experts_per_tok=8,
        intermediate_size=8,
        log_scaling_n_floor=4,
        dtype="float32",
        architectures=["Inkling2ForCausalLM"],
    )


def _attention(layer: int, backend: str, dtype: torch.dtype = torch.float32) -> Inkling2Attention:
    module = Inkling2Attention(_config(), layer, dtype=dtype, backend=backend)
    module.reset_parameters(0.3)
    return module


def _rows(rows: list[list[tuple[int, int]]], sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return validity and document positions [batch, sequence] of rows of (real, padded) documents."""
    valid = torch.zeros(len(rows), sequence, dtype=torch.bool)
    positions = torch.zeros(len(rows), sequence, dtype=torch.long)
    for row, documents in enumerate(rows):
        start = 0
        for real, padded in documents:
            valid[row, start : start + real] = True
            positions[row, start : start + real] = torch.arange(real)
            start += padded
        assert start <= sequence
    return valid, positions


@dataclass(frozen=True)
class Case:
    """Inputs of one attention call; q, r and sink follow the queries, k and v the keys."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    r: torch.Tensor
    table: torch.Tensor
    sink: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    key_valid: torch.Tensor
    key_positions: torch.Tensor
    query_offset: int


def _layout(kind: str, window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return query validity/positions, key validity/positions and the query offset of one layout.

    The context-parallel layouts describe one rank of a two-rank split of a 2 * SEQUENCE row: a
    full layer attends every rank's keys, a local layer the window before its shard and its shard.
    """
    if kind == "single":
        valid, positions = _rows([[(SEQUENCE, SEQUENCE)], [(97, SEQUENCE)]], SEQUENCE)
        return valid, positions, valid, positions, 0
    if kind == "packed":
        valid, positions = _rows([[(60, 64), (50, 50), (36, 36)], [(1, 1), (149, 149)]], SEQUENCE)
        return valid, positions, valid, positions, 0
    valid, positions = _rows([[(2 * SEQUENCE, 2 * SEQUENCE)], [(170, 170), (100, 100), (20, 30)]], 2 * SEQUENCE)
    first, second = slice(0, SEQUENCE), slice(SEQUENCE, 2 * SEQUENCE)
    if kind == "cp_full_first":
        return valid[:, first], positions[:, first], valid, positions, 0
    if kind == "cp_full_second":
        return valid[:, second], positions[:, second], valid, positions, SEQUENCE
    if kind == "cp_local_first":
        # No rank precedes the first one: its halo is zero-valued and invalid.
        halo = torch.zeros(valid.shape[0], window, dtype=torch.long)
        return (
            valid[:, first],
            positions[:, first],
            torch.cat((halo.bool(), valid[:, first]), dim=1),
            torch.cat((halo, positions[:, first]), dim=1),
            window,
        )
    assert kind == "cp_local_second"
    keys = slice(SEQUENCE - window, 2 * SEQUENCE)
    return valid[:, second], positions[:, second], valid[:, keys], positions[:, keys], window


def _case(module: Inkling2Attention, kind: str, dtype: torch.dtype, seed: int = 0) -> Case:
    valid, positions, key_valid, key_positions, query_offset = _layout(kind, module.window)
    batch, sequence = valid.shape
    generator = torch.Generator().manual_seed(seed)

    def leaf(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (scale * torch.randn(*shape, generator=generator)).to("cuda", dtype).requires_grad_(True)

    keys = leaf(2, batch, key_valid.shape[1], module.kv_heads, module.head_dim).detach()
    if kind == "cp_local_first":
        keys[:, :, : module.window] = 0
    return Case(
        q=leaf(batch, sequence, module.heads, module.head_dim),
        k=keys[0].clone().requires_grad_(True),
        v=keys[1].clone().requires_grad_(True),
        r=leaf(batch, sequence, module.heads, module.d_rel),
        table=leaf(module.d_rel, module.rel_extent, scale=0.5),
        sink=leaf(batch, sequence, module.heads),
        valid=valid.cuda(),
        positions=positions.cuda(),
        key_valid=key_valid.cuda(),
        key_positions=key_positions.cuda(),
        query_offset=query_offset,
    )


def _kernel(module: Inkling2Attention, case: Case) -> torch.Tensor:
    return relative_attention(
        case.q,
        case.k,
        case.v,
        relative=case.r,
        table=case.table,
        sink=case.sink,
        valid=case.valid,
        positions=case.positions,
        key_valid=case.key_valid,
        key_positions=case.key_positions,
        query_offset=case.query_offset,
        window=module.window if module.is_local else None,
    )


def _reference(module: Inkling2Attention, case: Case) -> torch.Tensor:
    return module._eager(
        case.q,
        case.k,
        case.v,
        profiles=(case.r @ case.table).transpose(1, 2),
        sink=case.sink,
        valid=case.valid,
        positions=case.positions,
        key_valid=case.key_valid,
        key_positions=case.key_positions,
        query_offset=case.query_offset,
    )


def _evaluate(
    attend: Callable[[Inkling2Attention, Case], torch.Tensor], module: Inkling2Attention, case: Case
) -> dict[str, torch.Tensor]:
    """Return the context of the valid queries and the gradients of one weighted sum of it."""
    keep = case.valid[:, :, None, None]
    out = attend(module, case).float() * keep
    weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(1)).to(out.device)
    gradients = torch.autograd.grad((out * weight).sum(), [vars(case)[name] for name in GRADIENTS])
    return {"out": out.detach()} | {name: gradient.float() for name, gradient in zip(GRADIENTS, gradients)}


def _assert_matches(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    for name, wanted in expected.items():
        rtol, atol = (CONTEXT_TOLERANCE if name == "out" else GRADIENT_TOLERANCE)[dtype]
        torch.testing.assert_close(
            actual[name], wanted, rtol=rtol, atol=atol * wanted.abs().max().clamp_min(1e-6).item(), msg=name
        )


LAYOUTS = [
    (LOCAL_LAYER, "single"),
    (LOCAL_LAYER, "packed"),
    (LOCAL_LAYER, "cp_local_first"),
    (LOCAL_LAYER, "cp_local_second"),
    (FULL_LAYER, "single"),
    (FULL_LAYER, "packed"),
    (FULL_LAYER, "cp_full_first"),
    (FULL_LAYER, "cp_full_second"),
]


@needs_gpu_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layer,kind", LAYOUTS)
def test_kernels_match_the_eager_reference(layer: int, kind: str, dtype: torch.dtype) -> None:
    module = _attention(layer, "eager")
    assert module.is_local == (layer == LOCAL_LAYER)
    case = _case(module, kind, dtype)
    reference = _evaluate(_reference, module, case)
    # Every layout must exercise the sink, the relative logits and at least one masked key.
    assert reference["sink"].abs().max() > 0 and reference["table"].abs().max() > 0
    _assert_matches(_evaluate(_kernel, module, case), reference, dtype)


@needs_gpu_triton
@pytest.mark.parametrize("layer,kind", LAYOUTS)
def test_backward_writes_every_element_of_its_buffers(layer: int, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The backward buffers start uninitialized: filled with NaN instead, none may reach a gradient."""
    module = _attention(layer, "eager")
    case = _case(module, kind, torch.float32)
    reference = _evaluate(_reference, module, case)
    uninitialized = torch.empty_like
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty_like", lambda tensor, **options: uninitialized(tensor, **options).fill_(torch.nan))
        actual = _evaluate(_kernel, module, case)
    _assert_matches(actual, reference, torch.float32)


def _shifted_layout(
    kind: str, window: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return a layout whose rows start with padding and a first document at a chosen position.

    A row is (padding, first position, document lengths): the documents follow the padding without
    gaps, and the first one starts at the given position, as a document does whose earlier tokens
    are not in the row.

    Returns:
        Query validity and positions [batch, sequence], key validity and positions
        [batch, key_sequence], the document number of every key [batch, key_sequence] (-1 for
        padding and for an empty halo slot) and the query offset. ``cp_first_shard`` is the first
        rank of a CP split: ``window`` empty halo slots precede its shard.
    """
    rows = {
        "from_zero": [(5, 0, [100]), (40, 0, [80, 20])],
        # Positions equal to the column index, and a first position that is neither zero nor the column.
        "left_padded": [(5, 5, [100]), (40, 3, [80, 20])],
        "cp_first_shard": [(2, 3, [120]), (0, 7, [60, 90])],
    }[kind]
    valid = torch.zeros(len(rows), SEQUENCE, dtype=torch.bool)
    positions = torch.zeros(len(rows), SEQUENCE, dtype=torch.long)
    document = torch.full((len(rows), SEQUENCE), -1)
    for row, (padding, first_position, lengths) in enumerate(rows):
        start = padding
        for number, length in enumerate(lengths):
            valid[row, start : start + length] = True
            positions[row, start : start + length] = torch.arange(length) + (first_position if number == 0 else 0)
            document[row, start : start + length] = number
            start += length
        assert start <= SEQUENCE
    if kind != "cp_first_shard":
        return valid, positions, valid, positions, document, 0
    halo = torch.zeros(len(rows), window, dtype=torch.long)
    return (
        valid,
        positions,
        torch.cat((halo.bool(), valid), dim=1),
        torch.cat((halo, positions), dim=1),
        torch.cat((halo - 1, document), dim=1),
        window,
    )


def _visible_keys(key_document: torch.Tensor, row: int, column: int, window: int | None) -> list[int]:
    """Return the key columns a query at a global column sees: its own document, causal, within the window.

    Args:
        key_document: Document number of every key [batch, key_sequence], -1 where there is no token.
        row: Batch row of the query.
        column: Global key column of the query.
        window: Largest number of keys a query sees, or None.
    """
    own = int(key_document[row, column])
    if own < 0:
        return []
    oldest = 0 if window is None else max(0, column - (window - 1))
    return [key for key in range(oldest, column + 1) if int(key_document[row, key]) == own]


@pytest.mark.parametrize("window", [None, WINDOW])
@pytest.mark.parametrize("kind", ["from_zero", "left_padded", "cp_first_shard"])
def test_key_ranges_hold_exactly_the_keys_of_each_query(kind: str, window: int | None) -> None:
    valid, positions, key_valid, key_positions, key_document, query_offset = _shifted_layout(kind, WINDOW)
    first, last = triton_attention.key_ranges(valid, positions, key_valid, key_positions, query_offset, window)
    for row in range(valid.shape[0]):
        last_query = [-1] * key_valid.shape[1]
        for query in range(valid.shape[1]):
            column = query + query_offset
            visible = _visible_keys(key_document, row, column, window)
            assert list(range(int(first[row, query]), column + 1)) == visible, (row, query)
            for key in visible:
                last_query[key] = query
        assert last[row].tolist() == last_query, row


def _from_the_definition(key_document: torch.Tensor) -> Callable[[Inkling2Attention, Case], torch.Tensor]:
    """Return the attention written out per query in FP64, with the documents given by number.

    Args:
        key_document: Document number of every key [batch, key_sequence], -1 where there is no token.

    Returns:
        A function of a module and a case that gives the context [batch, sequence, heads, head_dim]:
        over the keys of the query's own document up to the query (inside the window in a local
        layer), the softmax of the scaled QK product plus the relative logit of the distance,
        with one more zero-valued entry for the sink.
    """

    def attend(module: Inkling2Attention, case: Case) -> torch.Tensor:
        q, k, v, sink = (tensor.double().cpu() for tensor in (case.q, case.k, case.v, case.sink))
        # The relative logits are rounded to the input dtype before they enter the softmax.
        by_distance = (case.r @ case.table).double().cpu()
        extent = by_distance.shape[-1]
        groups = module.heads // module.kv_heads
        contexts = []
        for row in range(q.shape[0]):
            for query in range(q.shape[1]):
                column = query + case.query_offset
                visible = _visible_keys(key_document, row, column, module.window if module.is_local else None)
                if not visible:
                    contexts.append(q.new_zeros(module.heads, module.head_dim))
                    continue
                keys = torch.tensor(visible)
                scores = torch.einsum("hd,nhd->nh", q[row, query], k[row, keys].repeat_interleave(groups, dim=1))
                distance = column - keys
                relative = by_distance[row, query][:, distance.clamp(max=extent - 1)].T * (distance < extent)[:, None]
                logits = torch.cat((scores * module.head_dim**-0.5 + relative, sink[row, query][None]))
                weights = logits.softmax(0)[:-1]
                contexts.append(torch.einsum("nh,nhd->hd", weights, v[row, keys].repeat_interleave(groups, dim=1)))
        return torch.stack(contexts).view(*q.shape).to(case.q.device)

    return attend


def _shifted_case(module: Inkling2Attention, kind: str, dtype: torch.dtype) -> tuple[Case, torch.Tensor]:
    """Return a case of a shifted layout with random keys in every slot, and its key documents."""
    valid, positions, key_valid, key_positions, key_document, query_offset = _shifted_layout(kind, module.window)
    batch, sequence = valid.shape
    generator = torch.Generator().manual_seed(11)

    def leaf(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (scale * torch.randn(*shape, generator=generator)).to("cuda", dtype).requires_grad_(True)

    case = Case(
        q=leaf(batch, sequence, module.heads, module.head_dim),
        k=leaf(batch, key_valid.shape[1], module.kv_heads, module.head_dim),
        v=leaf(batch, key_valid.shape[1], module.kv_heads, module.head_dim),
        r=leaf(batch, sequence, module.heads, module.d_rel),
        table=leaf(module.d_rel, module.rel_extent, scale=0.5),
        sink=leaf(batch, sequence, module.heads),
        valid=valid.cuda(),
        positions=positions.cuda(),
        key_valid=key_valid.cuda(),
        key_positions=key_positions.cuda(),
        query_offset=query_offset,
    )
    return case, key_document


@needs_gpu_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["from_zero", "left_padded", "cp_first_shard"])
@pytest.mark.parametrize("layer", [LOCAL_LAYER, FULL_LAYER])
def test_kernels_attend_only_the_tokens_of_a_document_that_starts_at_a_nonzero_position(
    layer: int, kind: str, dtype: torch.dtype
) -> None:
    """Padding and empty halo slots before a row's first document hold no key, whatever its first position."""
    module = _attention(layer, "eager")
    case, key_document = _shifted_case(module, kind, dtype)
    expected = _evaluate(_from_the_definition(key_document), module, case)
    assert expected["sink"].abs().max() > 0 and expected["table"].abs().max() > 0
    _assert_matches(_evaluate(_kernel, module, case), expected, dtype)


def test_launch_shapes_fit_the_profile_layout() -> None:
    """The kernels that read whole tiles keep them inside one layout block and inside the rows."""
    launches = (triton_attention._LAUNCH["forward"], triton_attention._LAUNCH["query"], triton_attention._LAUNCH_FP32)
    for query_block, key_block, *_ in launches:
        assert triton_attention._ALIGN % query_block == 0
        assert triton_attention._MARGIN % key_block == 0


@pytest.mark.parametrize(
    "sequence,extent,query_offset",
    [(150, 23, 0), (128, 32, 64), (300, 23, 23), (257, 70, 148), (40, 200, 0)],
)
def test_aligned_profiles_hold_each_pair_once(sequence: int, extent: int, query_offset: int) -> None:
    """Every query/key pair within the extent sits at its column; all other elements are zero."""
    batch, heads, features = 2, 3, 4
    block, margin = triton_attention._ALIGN, triton_attention._MARGIN
    generator = torch.Generator().manual_seed(sequence + extent)
    relative = torch.randn(batch, sequence, heads, features, generator=generator, dtype=torch.float64)
    table = torch.randn(features, extent, generator=generator, dtype=torch.float64)
    profiles, shift = triton_attention._aligned_profiles(relative, table, query_offset)
    blocks = -(-sequence // block)
    assert profiles.shape[:2] == (block, batch * blocks * heads)
    assert profiles.shape[2] % margin == 0
    expected = torch.zeros_like(profiles)
    logits = relative @ table  # [batch, sequence, heads, extent], by distance
    for b in range(batch):
        for t in range(sequence):
            for h in range(heads):
                row = (b * blocks + t // block) * heads + h
                for distance in range(extent):
                    key = t + query_offset - distance
                    column = key - (t // block) * block + shift * margin
                    expected[t % block, row, column] = logits[b, t, h, distance]
    torch.testing.assert_close(profiles, expected, rtol=1e-12, atol=1e-12)


def _wrong_window(module: Inkling2Attention, case: Case) -> torch.Tensor:
    wider = copy.copy(module)
    wider.window = module.window + 1
    return _reference(wider, case)


def _no_relative_logits(module: Inkling2Attention, case: Case) -> torch.Tensor:
    return _reference(module, replace(case, table=case.table * 0))


def _wrong_query_offset(module: Inkling2Attention, case: Case) -> torch.Tensor:
    return _reference(module, replace(case, query_offset=case.query_offset + 1))


def _no_sink(module: Inkling2Attention, case: Case) -> torch.Tensor:
    return _reference(module, replace(case, sink=case.sink - 1e4))


def _one_document(module: Inkling2Attention, case: Case) -> torch.Tensor:
    columns = torch.arange(case.positions.shape[1], device=case.positions.device).expand_as(case.positions)
    return _reference(module, replace(case, positions=columns, key_positions=columns))


@needs_gpu_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "layer,kind,defect",
    [
        (LOCAL_LAYER, "single", _wrong_window),
        (LOCAL_LAYER, "single", _no_relative_logits),
        (FULL_LAYER, "single", _no_relative_logits),
        (LOCAL_LAYER, "cp_local_second", _wrong_query_offset),
        (FULL_LAYER, "cp_full_second", _wrong_query_offset),
        (FULL_LAYER, "single", _no_sink),
        (FULL_LAYER, "packed", _one_document),
    ],
)
def test_comparison_rejects_a_defective_reference(
    layer: int, kind: str, defect: Callable[[Inkling2Attention, Case], torch.Tensor], dtype: torch.dtype
) -> None:
    module = _attention(layer, "eager")
    case = _case(module, kind, dtype)
    actual = _evaluate(_kernel, module, case)
    _assert_matches(actual, _evaluate(_reference, module, case), dtype)
    with pytest.raises(AssertionError):
        _assert_matches(actual, _evaluate(defect, module, case), dtype)


def _module_pair(layer: int, device: str) -> tuple[Inkling2Attention, Inkling2Attention]:
    torch.manual_seed(layer)
    eager = _attention(layer, "eager").to(device)
    triton = _attention(layer, "triton").to(device)
    triton.load_state_dict(eager.state_dict())
    return eager, triton


def _module_inputs(device: str) -> tuple[AttentionSources, torch.Tensor, torch.Tensor]:
    valid, positions = _rows([[(60, 64), (50, 50), (36, 36)], [(1, 1), (149, 149)]], SEQUENCE)
    generator = torch.Generator().manual_seed(3)
    sources = [torch.randn(2, SEQUENCE, 32, generator=generator).to(device).requires_grad_(True) for _ in range(3)]
    return AttentionSources(*sources), valid.to(device), positions.to(device)


@needs_gpu_triton
@pytest.mark.parametrize("layer", [LOCAL_LAYER, FULL_LAYER])
def test_module_with_triton_backend_matches_eager_backend(layer: int) -> None:
    eager, triton = _module_pair(layer, "cuda")
    sources, valid, positions = _module_inputs("cuda")
    weight = torch.randn(2, SEQUENCE, 32, generator=torch.Generator().manual_seed(4)).cuda()
    results = []
    for module in (eager, triton):
        out = module(sources, valid, positions)
        leaves = [*sources, *module.parameters()]
        results.append((out.detach(), torch.autograd.grad((out * weight).sum(), leaves)))
    (expected, expected_gradients), (actual, actual_gradients) = results
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5 * expected.abs().max().item())
    names = ["q source", "k source", "v source", *(name for name, _ in eager.named_parameters())]
    for name, got, wanted in zip(names, actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(
            got, wanted, rtol=1e-3, atol=1e-4 * wanted.abs().max().clamp_min(1e-6).item(), msg=name
        )


@pytest.mark.parametrize("layer", [LOCAL_LAYER, FULL_LAYER])
def test_triton_backend_runs_the_eager_path_on_cpu(layer: int) -> None:
    eager, triton = _module_pair(layer, "cpu")
    assert triton.triton and not eager.triton and triton.flex is None
    sources, valid, positions = _module_inputs("cpu")
    assert torch.equal(triton(sources, valid, positions), eager(sources, valid, positions))


def _model(attn: str) -> Inkling2ForCausalLM:
    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    return Inkling2ForCausalLM(_config(), backend=backend)


def test_model_accepts_the_triton_backend() -> None:
    eager, triton = _model("eager"), _model("triton")
    assert all(layer.attn.triton for layer in triton.layers.values())
    assert not any(layer.attn.triton for layer in eager.layers.values())
    triton.load_state_dict(eager.state_dict())
    ids = torch.randint(0, 37, (2, 11), generator=torch.Generator().manual_seed(5))
    assert torch.equal(triton(ids), eager(ids))


@needs_gpu_triton
def test_model_with_triton_backend_matches_eager_backend_on_cuda() -> None:
    torch.manual_seed(6)
    eager, triton = _model("eager").cuda(), _model("triton").cuda()
    triton.load_state_dict(eager.state_dict())
    ids = torch.randint(0, 37, (2, 70), generator=torch.Generator().manual_seed(7)).cuda()
    expected, actual = eager(ids), triton(ids)
    torch.testing.assert_close(actual[..., :37], expected[..., :37], rtol=1e-3, atol=1e-4)
    actual[..., :37].logsumexp(-1).sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in triton.parameters())


@needs_gpu_triton
def test_recompute_reuses_the_recorded_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under the replay channel the second forward runs no forward kernel and gives the same gradients."""
    from nemo_automodel.components.models.inkling2 import triton_attention
    from nemo_automodel.shared.recompute_replay import RecomputeReplayRecorder

    torch.manual_seed(0)
    module = _attention(LOCAL_LAYER, "triton").cuda()
    valid, positions = _rows([[(90, 96), (50, 54)]], SEQUENCE)
    valid, positions = valid.cuda(), positions.cuda()
    sources = [torch.randn(1, SEQUENCE, 32, device="cuda") for _ in range(3)]
    weights = torch.randn(1, SEQUENCE, 32, device="cuda")

    def gradients() -> tuple[torch.Tensor, list[torch.Tensor]]:
        leaves = [source.clone().requires_grad_(True) for source in sources]
        out = module(AttentionSources(*leaves), valid, positions)
        return out.detach(), list(torch.autograd.grad((out * weights).sum(), leaves))

    reference_out, reference_grads = gradients()
    recorder: RecomputeReplayRecorder[tuple[torch.Tensor, torch.Tensor]] = RecomputeReplayRecorder()
    with triton_attention.attention_replay.scope(recorder, "record"):
        recorded_out, _ = gradients()
    assert len(recorder) == 1
    assert torch.equal(recorded_out, reference_out)

    class _NoKernel:
        def __getitem__(self, grid: tuple[int, int]) -> None:
            raise AssertionError("the forward kernel ran")

    monkeypatch.setattr(triton_attention, "_forward_kernel", _NoKernel())
    with pytest.raises(AssertionError, match="the forward kernel ran"):
        gradients()
    recorder.rewind()
    with triton_attention.attention_replay.scope(recorder, "replay"):
        replayed_out, replayed_grads = gradients()
    assert torch.equal(replayed_out, reference_out)
    for replayed, reference in zip(replayed_grads, reference_grads):
        assert torch.equal(replayed, reference)


def _recompute_everything() -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
    """Return activation-checkpoint contexts that save no operation."""
    return create_selective_checkpoint_contexts(lambda ctx, func, *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE)


@needs_gpu_triton
@pytest.mark.parametrize("layer", [LOCAL_LAYER, FULL_LAYER])
def test_recomputation_after_another_microbatch_gives_the_gradients_without_checkpointing(layer: int) -> None:
    """Two pipeline microbatches are in flight: every forward runs before the first backward.

    Two checkpointed layers share the token layout of a microbatch. When the first microbatch is
    recomputed, the key ranges that the kernels kept belong to the second one.
    """
    torch.manual_seed(layer)
    module = _attention(layer, "triton").cuda()
    microbatches = []
    for seed, documents in ((1, [(90, 96), (50, 54)]), (2, [(40, 40), (110, 110)])):
        valid, positions = _rows([documents], SEQUENCE)
        hidden = torch.randn(1, SEQUENCE, 32, generator=torch.Generator().manual_seed(seed)).cuda().requires_grad_(True)
        microbatches.append((hidden, valid.cuda(), positions.cuda()))
    weight = torch.randn(1, SEQUENCE, 32, generator=torch.Generator().manual_seed(3)).cuda()
    parameters = list(module.parameters())

    def gradients(
        attend: Callable[[AttentionSources, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> list[tuple[torch.Tensor, ...]]:
        losses = []
        for hidden, valid, positions in microbatches:
            first = attend(AttentionSources(hidden, hidden, hidden), valid, positions)
            second = attend(AttentionSources(first, first, first), valid, positions)
            losses.append((second * weight).sum())
        return [
            torch.autograd.grad(loss, [hidden, *parameters])
            for loss, (hidden, _, _) in zip(losses, microbatches, strict=True)
        ]

    def checkpointed(sources: AttentionSources, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return checkpoint(module, sources, valid, positions, use_reentrant=False, context_fn=_recompute_everything)

    expected = gradients(module)
    actual = gradients(checkpointed)
    for got_microbatch, wanted_microbatch in zip(actual, expected, strict=True):
        assert wanted_microbatch[0].abs().max() > 0
        for got, wanted in zip(got_microbatch, wanted_microbatch, strict=True):
            torch.testing.assert_close(got, wanted, rtol=1e-5, atol=1e-6 * wanted.abs().max().clamp_min(1e-6).item())
