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

"""CPU contracts for packed documents, pipeline residuals, Magi plans and CP/EP protocol controls."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.models.flint import magi as magi_cp
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.models.flint.packing import _prepare_inputs_for_cp
from tests.functional_tests.models.flint import run_packed_cp_logits as harness
from tests.unit_tests.models.flint.conftest import tiny_backend, tiny_config


def _model():
    torch.manual_seed(74375)
    model = tiny_config().build(backend=tiny_backend("sdpa"))
    model.initialize_weights()
    return model


def test_compact_documents_have_no_cross_document_attention_or_target() -> None:
    """Document-local RoPE and one shifted CE exclude document/padding edges."""
    model = _model()
    tokens = torch.tensor([[5, 2, 9, 4, 7, 3, 2]])
    documents = torch.tensor([[1, 1, 1, 2, 2, 2, 0]])
    out = model(tokens, attention_mask=documents, labels=tokens)
    changed = tokens.clone()
    changed[:, :3] = torch.tensor([[8, 6, 1]])
    after = model(changed, attention_mask=documents).logits
    torch.testing.assert_close(out.logits[:, 3:6], after[:, 3:6], atol=0, rtol=0)
    expected = F.cross_entropy(out.logits[:, [0, 1, 3, 4]].flatten(0, 1), tokens[:, [1, 2, 4, 5]].flatten())
    torch.testing.assert_close(out.loss, expected)
    # The compact stream agrees with separately evaluated complete documents.
    for start, end in ((0, 3), (3, 6)):
        separate = model(tokens[:, start:end]).logits
        torch.testing.assert_close(out.logits[:, start:end], separate, atol=2e-7, rtol=2e-5)


def test_chunked_linear_loss_uses_final_hidden_only_and_shifts_targets_once() -> None:
    """Actual ChunkedCrossEntropy matches full CE and every FP32 parameter adjoint."""
    complete, chunked = _model(), _model()
    tokens = torch.tensor([[5, 2, 9, 4, 7, 3]])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    targets[:, -1] = -100
    # Fail immediately if final-hidden extraction accidentally invokes the head.
    handle = chunked.lm_head.register_forward_pre_hook(
        lambda module, args: (_ for _ in ()).throw(AssertionError("head must not run for hidden-only loss"))
    )
    hidden = chunked(inputs, skip_logits=True).hidden_states
    assert isinstance(hidden, torch.Tensor) and hidden.shape == (1, 5, chunked.config.hidden_size)
    assert hidden.dtype == torch.float32
    actual = ChunkedCrossEntropy(chunk_len=2, compile=False, reduction="mean")(hidden, targets, chunked.lm_head.weight)
    expected = F.cross_entropy(complete(inputs).logits.flatten(0, 1), targets.flatten())
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)
    for (name, parameter), (other_name, other) in zip(
        chunked.named_parameters(), complete.named_parameters(), strict=True
    ):
        assert name == other_name and parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, atol=2e-7, rtol=2e-5, msg=name)
    handle.remove()
    recipe_output = chunked(inputs, output_hidden_states=True, logits_to_keep=1)
    assert recipe_output.logits.shape == (1, 1, chunked.config.vocab_size)
    torch.testing.assert_close(recipe_output.hidden_states, hidden, atol=0, rtol=0)


def test_two_native_stage_residuals_match_unsplit_model_and_linear_loss_boundary() -> None:
    """Global IDs enter stage zero once; later stages consume floating local hidden states."""
    complete = _model()
    first, last = copy.deepcopy(complete), copy.deepcopy(complete)
    first.model.layers = nn.ModuleDict({"0": first.model.layers[0]})
    first.model.norm, first.lm_head = None, None
    last.model.embed_tokens = None
    last.model.layers = nn.ModuleDict({"1": last.model.layers[1], "2": last.model.layers[2]})
    tokens = torch.tensor([[5, 2, 9, 4, 7]])
    residual = first(tokens).logits
    assert residual.shape == (1, 5, complete.config.hidden_size)
    torch.testing.assert_close(last(residual).logits, complete(tokens).logits, atol=0, rtol=0)
    last._pp_return_hidden_states = True
    hidden = last(residual)
    assert hidden.dtype == torch.float32 and hidden.shape == residual.shape
    expected = complete(tokens, skip_logits=True).hidden_states
    torch.testing.assert_close(hidden, expected, atol=0, rtol=0)
    inputs, outputs = last.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=5, dtype=torch.float32)
    assert inputs[0].shape == outputs[0].shape == (1, 5, complete.config.hidden_size)
    assert outputs[0].dtype == torch.float32


def test_packed_hf_labels_fail_before_an_implicit_local_shift() -> None:
    """External already-shifted labels own packed/CP document target boundaries."""
    with pytest.raises(ValueError, match="already-shifted"):
        _model()(torch.tensor([[1, 2]]), labels=torch.tensor([[2, 3]]), qkv_format="thd")


def test_thd_pipeline_rejects_multiple_physical_rows_per_microbatch() -> None:
    """PP metadata represents one packed row, rather than an implicit batch flatten."""
    first_stage = _model()
    first_stage.lm_head = None
    first_stage.backend = tiny_backend("te")
    with pytest.raises(ValueError, match="one physical packed row per microbatch"):
        _prepare_inputs_for_cp(
            first_stage,
            {"input_ids": torch.ones(4, 8, dtype=torch.long), "qkv_format": "thd"},
            num_chunks=2,
        )


@pytest.mark.parametrize("chunks", [1, 2])
@pytest.mark.parametrize("mask_as_argument", [False, True])
def test_real_thd_sharder_preserves_explicit_loss_masks(chunks: int, mask_as_argument: bool) -> None:
    """Shared THD conversion retains each external mask in its local token coordinates."""
    model = _model()
    model.backend = tiny_backend("te")
    if chunks > 1:
        model.lm_head = None
    ids = torch.arange(16).reshape(2, 8)
    mask = torch.arange(16, dtype=torch.float32).reshape_as(ids) / 16
    batch = {
        "input_ids": ids,
        "labels": ids + 1,
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]]).expand(2, -1),
        "seq_lens": torch.tensor([[5, 2], [5, 2]], dtype=torch.int32),
        "seq_lens_padded": torch.tensor([[6, 2], [6, 2]], dtype=torch.int32),
        "qkv_format": "thd",
    }
    if not mask_as_argument:
        batch["loss_mask"] = mask
    prepared = _prepare_inputs_for_cp(model, batch, num_chunks=chunks)
    _, local, _ = prepared["cp_sharder"].shard_batch(None, None, batch, loss_mask=mask if mask_as_argument else None)
    torch.testing.assert_close(local["loss_mask"], mask.reshape(chunks, -1), atol=0, rtol=0)


@pytest.mark.parametrize("cp_size", [1, 2])
def test_duplicate_dp_and_sharded_cp_have_identical_global_objective(cp_size: int) -> None:
    """Actual CE/autograd [5,7] logits prove WORLD compensation exactly once."""
    generator = torch.Generator().manual_seed(731)
    logits = torch.randn(5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, -100, 2, 5, 3])
    reference = F.cross_entropy(logits, labels, reduction="sum") / 4
    expected = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    local_losses, local_gradients = [], []
    for rank in range(2):
        selection = slice(None) if cp_size == 1 else slice(rank, None, 2)
        local = F.cross_entropy(logits[selection], labels[selection], reduction="sum")
        local /= harness.expected_global_targets(4, cp_size)
        local_losses.append(local)
        local_gradients.append(torch.autograd.grad(local * 2, logits, retain_graph=True)[0])
    # Different scalar summation orders can differ by one FP64 ULP; this
    # algebra control does not define or change the measured CUDA CE gate.
    torch.testing.assert_close(sum(local_losses), reference, atol=2 * torch.finfo(torch.float64).eps, rtol=0)
    torch.testing.assert_close(sum(local_gradients) / 2, expected, atol=0, rtol=0)
    topology = harness.comparison_topology(cp_size, 2)
    assert topology["ep_size"] == 2 and topology["pp_size"] == 1
    assert topology["dp_size"] * topology["cp_size"] == 2


def test_expert_sum_is_normalized_once_before_observation_and_direct_clip(monkeypatch) -> None:
    """Run real shared expert division; the norm stand-in observes CPU tensors."""
    from nemo_automodel.components.training import utils

    model = torch.nn.Module()
    model.dense = torch.nn.Parameter(torch.tensor([3.0]))
    model.experts = torch.nn.Linear(1, 1, bias=False)
    model.experts._nemo_ep_local_expert_params = True
    model.dense.grad = torch.tensor([3.0])
    model.experts.weight.grad = torch.tensor([[8.0]])
    calls = []

    def norm_only(max_norm, parts, **kwargs):
        calls.append(max_norm)
        return torch.stack([parameter.grad.square().sum() for parameter in parts[0].parameters()]).sum().sqrt()

    monkeypatch.setattr(utils, "clip_grad_norm", norm_only)
    shard = SimpleNamespace(size=lambda: 1)
    mesh = {"ep_shard": shard}

    class Mesh(dict):
        mesh_dim_names = ("ep_shard", "ep")

    context = SimpleNamespace(device_mesh=None, moe_mesh=Mesh(mesh))
    norm, options = harness.normalize_gradients(model, context)
    assert norm == 5 and calls == [float("inf")]
    assert model.dense.grad == 3 and model.experts.weight.grad == 4
    utils.clip_grad_norm(1.0, [model], **options)
    assert calls == [float("inf"), 1.0]
    assert model.experts.weight.grad == 4


@pytest.mark.parametrize("difference", ["missing", "extra"])
def test_gradient_fqn_mismatch_fails_before_any_global_tensor_read(tmp_path, monkeypatch, difference) -> None:
    model = torch.nn.Linear(2, 3)
    model(torch.ones(1, 2)).sum().backward()
    tensors = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    if difference == "missing":
        tensors.pop("bias")
    else:
        tensors["unrecognized.weight"] = torch.ones(1)
    archive = tmp_path / "gradients.safetensors"
    save_file(tensors, archive)

    def forbidden(value):
        raise AssertionError("must reject the FQN set before materializing gradients")

    monkeypatch.setattr(harness, "full_tensor", forbidden)
    with pytest.raises(AssertionError, match="parameter sets differ"):
        harness.capture_gradients(model, tmp_path / "unused", archive, 0)


def test_every_adam_destination_and_update_count_are_required() -> None:
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(model.parameters(), **harness.OPTIMIZER)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    receipt = harness.optimizer_update_receipt(model, optimizer, 1)
    assert receipt["parameter_count"] == 2
    with pytest.raises(AssertionError, match="count"):
        harness.optimizer_update_receipt(model, optimizer, 2)
    optimizer.state[model.weight]["exp_avg"][0, 0] = float("nan")
    with pytest.raises(AssertionError, match="nonfinite"):
        harness.optimizer_update_receipt(model, optimizer, 1)


class _FakeDeviceMesh:
    """Named CP-only mesh used to exercise public sharder resolution."""

    mesh_dim_names = ("cp",)

    def __init__(self, cp_mesh) -> None:
        self._cp_mesh = cp_mesh

    def __getitem__(self, name: str):
        if name != "cp":
            raise KeyError(name)
        return self._cp_mesh


@pytest.mark.parametrize("pipeline,rows", [(False, 1), (True, 1), (True, 3)])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_magi_shards_token_masks_and_retains_independent_microbatch_plans(
    monkeypatch, pipeline: bool, rows: int, explicit_mask: bool
) -> None:
    """Exercise real THD conversion and PyTorch microbatch splitting with distinct packs."""
    monkeypatch.delenv("MAGI_ATTENTION_DSINK_ALL_REDUCE_OP", raising=False)
    counts = [[2, 6], [3, 5], [4, 4]][:rows]
    ids = torch.arange(rows * 8).reshape(rows, 8)
    positions = torch.stack([torch.cat([torch.arange(n) for n in row]) for row in counts])
    batch = {
        "input_ids": ids,
        "labels": ids + 1,
        "position_ids": positions,
        "seq_lens": torch.tensor([[a, b - 1] for a, b in counts]),
        "loss_mask": (ids % 3 != 0).float(),
        "seq_lens_padded": torch.tensor(counts),
        "qkv_format": "thd",
    }
    partition_by_boundary = {
        2: torch.tensor([0, 1, 6, 7]),
        3: torch.tensor([0, 2, 5, 7]),
        4: torch.tensor([1, 3, 4, 6]),
    }

    def key_for(cu: torch.Tensor, *args, **kwargs):
        """Map CPU document boundaries [documents+1] to a fake CP partition."""
        return SimpleNamespace(indices=partition_by_boundary[int(cu[1])], total_seqlen_q=8, boundary=int(cu[1]))

    def dispatch(tensor: torch.Tensor, key, **kwargs) -> torch.Tensor:
        """Select CP-local [tokens] from the global [sequence] input."""
        return tensor[key.indices]

    api = SimpleNamespace(
        magi_attn_varlen_key=key_for,
        make_varlen_key_for_new_mask_after_dispatch=lambda cuq, cuk, key, **kw: key,
        get_position_ids=lambda key: key.indices,
        dispatch=dispatch,
    )
    configs = SimpleNamespace(DistAttnConfig=lambda **kw: kw, DispatchConfig=lambda **kw: kw)
    monkeypatch.setattr(magi_cp, "safe_import", lambda name: (True, api if name.endswith(".api") else configs))
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: group)
    model = SimpleNamespace(
        config=FlintConfig(),
        backend=SimpleNamespace(attn="magi"),
        supports_thd=True,
        _packed_cp_attn_backends=("te", "magi"),
        cp_mesh=mesh,
        _is_pipeline_parallel_stage=lambda: pipeline,
    )
    model.prepare_model_inputs_for_cp = lambda current_batch, num_chunks=1: magi_cp._prepare_magi_inputs(
        model, current_batch, num_chunks=num_chunks
    )
    sharder = ContextParallelSharder(model, _FakeDeviceMesh(mesh), batch, num_chunks=rows)
    source_mask = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) / (rows * 8)
    if explicit_mask:
        # The external argument overrides a conflicting batch field, preserving
        # non-binary weights and the document/CP token order.
        _, result, layout = sharder.shard_batch(mesh, None, batch, loss_mask=source_mask)
    else:
        _, result = sharder.shard(batch)
        layout = sharder.shard_layout
    assert result["padding_mask"].shape == (rows, 4)
    expected_padding = torch.stack([partition_by_boundary[row[0]] == 7 for row in counts])
    assert torch.equal(result["padding_mask"].reshape(rows, -1), expected_padding)
    expected_mask = source_mask if explicit_mask else batch["loss_mask"]
    assert torch.equal(
        result["loss_mask"].reshape(rows, -1),
        torch.stack([expected_mask[i, partition_by_boundary[row[0]]] for i, row in enumerate(counts)]),
    )
    if not pipeline:
        assert layout.input_row_shape == (1, 8)
        assert torch.equal(result["input_ids"], ids[:, partition_by_boundary[2]])
        return
    assert layout is None
    assert torch.equal(result["input_ids"], ids)
    input_ids = result.pop("input_ids")
    inputs, kwargs = split_args_kwargs_into_chunks((input_ids,), result, rows)
    for index in (*reversed(range(rows)), 0):
        chunk = kwargs[index]
        boundary = counts[index][0]
        local_indices = partition_by_boundary[boundary]
        plan = chunk["_flint_magi_plan"].select(chunk["_flint_magi_microbatch"])
        assert plan.keys[0].boundary == boundary
        assert torch.equal(chunk["position_ids"][0], positions[index, local_indices])
        assert torch.equal(chunk["labels"][0], ids[index, local_indices] + 1)
        assert torch.equal(chunk["loss_mask"][0], expected_mask[index, local_indices])
        assert torch.equal(inputs[index][0][0, local_indices], ids[index, local_indices])
        assert chunk["_flint_magi_microbatch"].device.type == "cpu"


def test_magi_pp_rejects_multiple_packs_per_microbatch(monkeypatch) -> None:
    """Keep the supported one-pack-per-microbatch contract explicit."""
    model = SimpleNamespace(config=FlintConfig(), _is_pipeline_parallel_stage=lambda: True)
    monkeypatch.setattr(magi_cp, "safe_import", lambda name: (False, None))
    batch = {"input_ids": torch.zeros(4, 8, dtype=torch.long), "qkv_format": "thd"}
    sharder = magi_cp._prepare_magi_inputs(model, batch, num_chunks=2)["cp_sharder"]
    with pytest.raises(ValueError, match="one packed row per microbatch"):
        sharder.shard_batch(None, None, batch)


def test_magi_without_pp_rejects_multiple_microbatches() -> None:
    """The flat non-PP token layout represents one pack only."""
    model = SimpleNamespace(config=FlintConfig(), _is_pipeline_parallel_stage=lambda: False)
    with pytest.raises(ValueError, match="without PP requires one packed microbatch"):
        magi_cp._prepare_magi_inputs(model, {"qkv_format": "thd"}, num_chunks=2)


@pytest.mark.parametrize("index", [-1, 2])
def test_magi_batch_plans_reject_out_of_range_index(index: int) -> None:
    """Reject invalid schedule identities instead of selecting another plan."""
    plan = magi_cp.FlintMagiPlan({0: SimpleNamespace(total_seqlen_q=8)})
    plans = magi_cp.FlintMagiBatchPlans((plan, plan))
    with pytest.raises(IndexError, match="outside"):
        plans.select(torch.tensor([index]))


def test_magi_attention_preserves_protected_sink_and_its_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    """A direct public-API sink argument preserves one-sink probability and gradients."""
    torch.manual_seed(4375)
    q = torch.randn(3, 2, 4, requires_grad=True)
    k = torch.randn(3, 1, 4, requires_grad=True)
    v = torch.randn(3, 1, 4, requires_grad=True)
    sink = torch.randn(2, requires_grad=True)
    calls = []

    def calc_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: object,
        *,
        sink: torch.Tensor,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, None]:
        """Stand in for the public optional kernel with independent scalar-head math.

        Args:
            query: Local queries [tokens, query_heads, head_dim].
            key: Local keys [tokens, kv_heads, head_dim].
            value: Local values [tokens, kv_heads, head_dim].
            plan: Opaque mask key.
            sink: Replicated FP32 learned logits [1, query_heads].
            softmax_scale: Query/key score multiplier.

        Returns:
            Differentiable local output [tokens, query_heads, head_dim] and no metadata.
        """
        calls.append((plan, sink.dtype, sink.shape))
        rows = []
        for token in range(query.shape[0]):
            heads = []
            for head in range(query.shape[1]):
                scores = (key[: token + 1, 0] @ query[token, head]) * softmax_scale
                probabilities = torch.cat([scores, sink[:, head]]).softmax(0)[:-1]
                heads.append(probabilities @ value[: token + 1, 0])
            rows.append(torch.stack(heads))
        return torch.stack(rows), None

    monkeypatch.setattr(magi_cp, "safe_import", lambda _: (True, SimpleNamespace(calc_attn=calc_attn)))
    key = SimpleNamespace(total_seqlen_q=3)
    output = magi_cp.FlintMagiPlan({0: key}).attention(q, k, v, sink=sink, window=0, scale=0.5)
    assert calls == [(key, torch.float32, torch.Size([1, 2]))]
    output.backward(torch.randn_like(output))
    assert sink.grad is not None and torch.count_nonzero(sink.grad) == 2
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in [q, k, v])
    with pytest.raises(ValueError, match="protected FP32"):
        magi_cp.FlintMagiPlan({0: key}).attention(q, k, v, sink=sink.bfloat16(), window=0, scale=0.5)
