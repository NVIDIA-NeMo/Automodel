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

"""Four-rank CPU/Gloo checks that linear CE streams a sharded LM head instead of gathering it.

One spawn runs every scenario and records its outcome; the tests assert on the
recorded outcomes. Each scenario compares the loss, both gradients, and the
parameters after one SGD step against a dense single-process FP32 reference,
with ``DTensor.full_tensor`` patched to raise and no ``[vocab, hidden]`` tensor
allowed among the autograd-saved tensors.
"""

from __future__ import annotations

import json
import os
import socket
import traceback
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

_WORLD_SIZE = 4
pytestmark = pytest.mark.cpu_only
_HIDDEN = 16
_SCENARIOS = (
    "fsdp_uneven_vocab_chunked_ce",
    "fsdp_uneven_vocab_fused_ce",
    "fsdp_empty_shard",
    "hsdp",
    "single_rank_mesh",
    "eval_without_group",
    "frozen_head",
    "mismatched_group_raises",
    "unsupported_placement_raises",
    "mtp_depths_share_one_resolution",
    "autocast_recompute",
    "padded_vocab_adapter",
    "inkling_chunked_head",
    "negative_infinity_chunk",
    "target_validation",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _data(vocab: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build identical global inputs on every rank.

    Returns:
        weight: Tensor of shape [vocab, hidden].
        hidden: Tensor of shape [ranks, 2, 5, hidden]; rank ``r`` owns ``hidden[r]``.
        labels: Tensor of shape [ranks, 2, 5]; rank 0 has ignored tokens and the
            last rank is fully ignored, so it still enters every collective with
            no supervised tokens.
    """
    generator = torch.Generator().manual_seed(vocab)
    weight = torch.randn(vocab, _HIDDEN, generator=generator)
    hidden = torch.randn(_WORLD_SIZE, 2, 5, _HIDDEN, generator=generator)
    labels = torch.randint(vocab, (_WORLD_SIZE, 2, 5), generator=generator)
    labels[0, 0, :3] = -100
    labels[-1] = -100
    return weight, hidden, labels


def _reference(weight: torch.Tensor, hidden: torch.Tensor, labels: torch.Tensor):
    """Dense FP32 loss and gradients over the global batch.

    Args:
        weight: Tensor of shape [vocab, hidden].
        hidden: Tensor of shape [ranks, 2, 5, hidden].
        labels: Tensor of shape [ranks, 2, 5].

    Returns:
        The token-normalized loss, the gradients of ``hidden`` and ``weight``,
        and the weight after one SGD step with learning rate 0.1.
    """
    ref_weight = weight.clone().requires_grad_()
    ref_hidden = hidden.clone().requires_grad_()
    normalizer = (labels != -100).sum()
    loss = F.cross_entropy(F.linear(ref_hidden, ref_weight).flatten(0, 2), labels.flatten(), reduction="sum")
    (loss / normalizer).backward()
    return loss.detach() / normalizer, ref_hidden.grad, ref_weight.grad, weight - 0.1 * ref_weight.grad


def _forbid_gather_and_full_saves(
    vocab: int, fn: Callable[[], torch.Tensor], *, allow_full_saves: bool = False
) -> torch.Tensor:
    """Run ``fn`` with ``DTensor.full_tensor`` disabled and, by default, full-size saves rejected.

    A one-rank mesh holds the whole ``[vocab, hidden]`` weight locally, so its
    parameter view may be saved; ``allow_full_saves`` permits that case.
    """
    from torch.distributed.tensor import DTensor

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        """Reject full-head saves.

        Args:
            tensor: Tensor of arbitrary shape saved by autograd.

        Returns:
            The same tensor, preserving its shape and storage.
        """
        if not allow_full_saves and tuple(tensor.shape) == (vocab, _HIDDEN):
            raise AssertionError("a [vocab, hidden] tensor was saved for backward")
        return tensor

    original = DTensor.full_tensor

    def forbidden(*args, **kwargs):
        raise AssertionError("DTensor.full_tensor was called")

    DTensor.full_tensor = forbidden
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            return fn()
    finally:
        DTensor.full_tensor = original


def _check_training_parity(rank, mesh, placements, vocab, loss_fn, reduce_group, reduce_ranks, data_rows=None):
    """Train one step on a sharded head and compare with the dense reference.

    Args:
        rank: Global rank.
        mesh: Device mesh of the LM head.
        placements: LM-head placements on ``mesh``.
        vocab: Vocabulary size.
        loss_fn: Linear CE loss under test.
        reduce_group: Group whose ranks own independent tokens.
        reduce_ranks: Global ranks of ``reduce_group``, in group order.
        data_rows: Sample row of ``_data`` owned by each of ``reduce_ranks``;
            defaults to the global ranks themselves.
    """
    from torch.distributed.tensor import distribute_tensor

    weight, hidden, labels = _data(vocab)
    rows = reduce_ranks if data_rows is None else data_rows
    hidden, labels = hidden[rows], labels[rows]
    local = reduce_ranks.index(rank)
    sharded = torch.nn.Parameter(distribute_tensor(weight, mesh, placements))
    local_hidden = hidden[local].clone().requires_grad_()
    normalizer = int((labels != -100).sum())

    def step() -> torch.Tensor:
        loss = loss_fn(
            local_hidden, labels[local], sharded, num_label_tokens=normalizer, grad_reduce_group=reduce_group
        )
        # Recipes scale the local loss by the reduction world size before backward.
        (loss * len(reduce_ranks)).backward()
        return loss

    loss = _forbid_gather_and_full_saves(vocab, step, allow_full_saves=mesh.size() == 1)
    ref_loss, ref_hidden_grad, ref_weight_grad, ref_updated = _reference(weight, hidden, labels)
    total = loss.detach().clone()
    dist.all_reduce(total, group=reduce_group)
    torch.testing.assert_close(total, ref_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(local_hidden.grad / len(reduce_ranks), ref_hidden_grad[local], rtol=1e-5, atol=1e-6)
    assert tuple(sharded.grad.placements) == tuple(placements)
    torch.testing.assert_close(sharded.grad.full_tensor(), ref_weight_grad, rtol=1e-5, atol=1e-6)
    torch.optim.SGD([sharded], lr=0.1).step()
    torch.testing.assert_close(sharded.full_tensor(), ref_updated, rtol=1e-5, atol=1e-6)


def _run_scenarios(rank: int) -> dict[str, str]:
    """Run every scenario on this rank and return its outcome."""
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    import nemo_automodel.components.loss.linear_ce as linear_ce
    import nemo_automodel.components.loss.linear_ce_base as linear_ce_base
    from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
    from nemo_automodel.components.loss.linear_ce_base import ShardedLMHeadWeight

    # Force several chunks per shard: three rows per rank per chunk in fp32.
    linear_ce_base._STREAM_CHUNK_BYTES = 3 * _WORLD_SIZE * _HIDDEN * 4
    # The fused loss's torch_compile kernel is exercised eagerly; Inductor is not under test.
    linear_ce_base._compiled_chunk_lse_and_target = linear_ce_base._chunk_lse_and_target
    linear_ce_base._compiled_chunk_logits = linear_ce_base._chunk_logits
    linear_ce.HAVE_CUT_CROSS_ENTROPY = True

    world = dist.group.WORLD
    ranks = list(range(_WORLD_SIZE))
    flat = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
    hsdp = init_device_mesh("cpu", (2, 2), mesh_dim_names=("replicate", "shard"))
    singletons = [dist.new_group([member]) for member in ranks]
    singleton_mesh = DeviceMesh.from_group(singletons[rank], "cpu")
    chunked = ChunkedCrossEntropy(3, compile=False)
    fused = linear_ce.FusedLinearCrossEntropy(impl="torch_compile")

    def eval_without_group() -> None:
        weight, hidden, labels = _data(37)
        sharded = torch.nn.Parameter(distribute_tensor(weight, flat, [Shard(0)]))
        normalizer = int((labels != -100).sum())
        with torch.no_grad():
            loss = _forbid_gather_and_full_saves(
                37, lambda: chunked(hidden[rank], labels[rank], sharded, num_label_tokens=normalizer)
            )
        dist.all_reduce(loss)
        torch.testing.assert_close(loss, _reference(weight, hidden, labels)[0], rtol=1e-5, atol=1e-6)

    def frozen_head() -> None:
        weight, hidden, labels = _data(37)
        sharded = torch.nn.Parameter(distribute_tensor(weight, flat, [Shard(0)]), requires_grad=False)
        local_hidden = hidden[rank].clone().requires_grad_()
        normalizer = int((labels != -100).sum())
        _forbid_gather_and_full_saves(
            37,
            lambda: chunked(
                local_hidden, labels[rank], sharded, num_label_tokens=normalizer, grad_reduce_group=world
            ).backward(),
        )
        assert sharded.grad is None
        torch.testing.assert_close(local_hidden.grad, _reference(weight, hidden, labels)[1][rank])

    def mismatched_group_raises() -> None:
        sharded = torch.nn.Parameter(distribute_tensor(_data(37)[0], flat, [Shard(0)]))
        with pytest.raises(ValueError, match="mesh size=4, reduction group size=1"):
            chunked.resolve_lm_weight(sharded, grad_reduce_group=singletons[rank])

    def unsupported_placement_raises() -> None:
        weight, hidden, labels = _data(37)
        sharded = torch.nn.Parameter(distribute_tensor(weight, flat, [Shard(1)]))
        resolved = chunked.resolve_lm_weight(sharded, grad_reduce_group=world)
        assert isinstance(resolved, ShardedLMHeadWeight)
        with pytest.raises(ValueError, match="one Shard\\(0\\) mesh dimension"):
            chunked(hidden[rank], labels[rank], resolved, grad_reduce_group=world)

    def mtp_depths_share_one_resolution() -> None:
        from nemo_automodel.components.loss.mtp import calculate_mtp_loss
        from nemo_automodel.components.loss.utils import calculate_loss, prepare_lm_weight

        weight, hidden, labels = _data(37)
        depth_hidden = [hidden, hidden.roll(1, dims=-1) * 0.5]
        normalizer = int((labels != -100).sum())

        def head(lm_weight: torch.Tensor) -> torch.nn.Module:
            """Wrap a projection weight in a model.

            Args:
                lm_weight: Tensor of global shape [vocab, hidden], possibly sharded.

            Returns:
                Module whose LM head owns the supplied weight.
            """
            model = torch.nn.Module()
            model.lm_head = torch.nn.Linear(_HIDDEN, 37, bias=False)
            model.lm_head.weight = torch.nn.Parameter(lm_weight)
            return model

        def main_plus_mtp(model, states, targets, group):
            """Compute the normalized main and two auxiliary CE losses.

            Args:
                model: Module owning a dense or FSDP/HSDP-sharded LM-head
                    weight of global shape [vocab, hidden].
                states: Two tensors of shape [batch, sequence, hidden]. The
                    first serves the main loss; the second is reused for both
                    MTP depths. Batch is global for the reference, otherwise local.
                targets: Integer tensor of shape [batch, sequence], with -100 ignored.
                group: Group owning independent token batches, or None for the reference.

            Returns:
                FP32 scalar containing the normalized main and auxiliary losses.
            """
            lm_weight = prepare_lm_weight(chunked, model, grad_reduce_group=group)
            loss = calculate_loss(
                chunked,
                model=model,
                hidden_states=states[0],
                labels=targets,
                lm_weight=lm_weight,
                num_label_tokens=normalizer,
                grad_reduce_group=group,
            )
            return loss + calculate_mtp_loss(
                chunked,
                mtp_per_depth_h=states[1:] * 2,
                labels=targets,
                model=model,
                scaling_factor=0.1,
                num_label_tokens=normalizer,
                grad_reduce_group=group,
            )

        # Dense reference: the same production loss path on the global batch with an unsharded head.
        dense = head(weight.clone())
        dense_states = [states.flatten(0, 1).clone().requires_grad_() for states in depth_hidden]
        reference = main_plus_mtp(dense, dense_states, labels.flatten(0, 1), None)
        reference.backward()

        sharded = head(distribute_tensor(weight, flat, [Shard(0)]))
        local_states = [states[rank].clone().requires_grad_() for states in depth_hidden]
        with patch.object(chunked, "prepare_lm_weight", wraps=chunked.prepare_lm_weight) as prepare:
            loss = _forbid_gather_and_full_saves(37, lambda: main_plus_mtp(sharded, local_states, labels[rank], world))
        (loss * _WORLD_SIZE).backward()
        # One weight preparation for the main loss and one shared across both MTP depths.
        assert prepare.call_count == 2
        total = loss.detach().clone()
        dist.all_reduce(total)
        torch.testing.assert_close(total, reference.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            sharded.lm_head.weight.grad.full_tensor(), dense.lm_head.weight.grad, rtol=1e-5, atol=1e-6
        )
        for local, dense_state in zip(local_states, dense_states):
            torch.testing.assert_close(
                local.grad / _WORLD_SIZE, dense_state.grad.unflatten(0, (_WORLD_SIZE, 2))[rank], rtol=1e-5, atol=1e-6
            )

    def autocast_recompute() -> None:
        weight, hidden, labels = _data(37)
        sharded = torch.nn.Parameter(distribute_tensor(weight, flat, [Shard(0)]))
        local_hidden = hidden[rank].clone().requires_grad_()
        dense_weight = weight.clone().requires_grad_()
        dense_hidden = hidden.clone().requires_grad_()
        # Use one projection tile in each path to isolate recomputation precision.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = ChunkedCrossEntropy(32, compile=False)(local_hidden, labels[rank], sharded, grad_reduce_group=world)
            reference = F.cross_entropy(
                F.linear(dense_hidden, dense_weight).flatten(0, 2), labels.flatten(), reduction="sum"
            )
        # Production backward runs after leaving autocast.
        (loss * _WORLD_SIZE).backward()
        reference.backward()
        dist.all_reduce(loss.detach())
        torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sharded.grad.full_tensor(), dense_weight.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(local_hidden.grad / _WORLD_SIZE, dense_hidden.grad[rank], rtol=2e-2, atol=2e-2)

    def padded_vocab_adapter() -> None:
        """Exclude padded rows without changing physical FSDP/HSDP shard boundaries."""
        from nemo_automodel.components.loss.utils import prepare_lm_weight
        from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy

        for dtype in (torch.float32, torch.bfloat16):
            for mesh, placements, vocab in (
                (flat, [Shard(0)], 37),
                (flat, [Shard(0)], 7),
                (flat, [Shard(0)], 1),
                (hsdp, [Replicate(), Shard(0)], 37),
            ):
                weight, hidden, labels = _data(43)
                weight = (weight * 0.1).to(dtype)
                hidden = hidden.bfloat16()
                labels = torch.where(labels == -100, labels, labels % vocab)
                normalizer = int((labels != -100).sum())
                dense_weight = weight.clone().requires_grad_()
                dense_hidden = hidden.clone().requires_grad_()
                logits = F.linear(dense_hidden, dense_weight[:vocab].bfloat16())
                logits = (logits / 30).tanh() * 30
                reference = F.cross_entropy(logits.float().flatten(0, 2), labels.flatten(), reduction="sum")
                reference = reference / normalizer
                reference.backward()

                model = torch.nn.Module()
                model.lm_head = torch.nn.Linear(_HIDDEN, 43, bias=False)
                model.lm_head.weight = torch.nn.Parameter(distribute_tensor(weight, mesh, placements))
                local_hidden = hidden[rank].clone().requires_grad_()
                loss_fn = Inkling2FusedLinearCrossEntropy(vocab)
                # Exercise streamed adapter semantics with the existing eager CPU
                # chunk kernel; CCE's CUDA kernel is covered by the GPU suite.
                loss_fn.impl = "torch_compile"

                def train() -> torch.Tensor:
                    operand = prepare_lm_weight(loss_fn, model, grad_reduce_group=world)
                    assert isinstance(operand, ShardedLMHeadWeight)
                    assert operand.weight is model.lm_head.weight
                    assert operand.dtype == torch.bfloat16 and operand.shape == (vocab, _HIDDEN)
                    loss = loss_fn(
                        local_hidden, labels[rank], operand, num_label_tokens=normalizer, grad_reduce_group=world
                    )
                    (loss * _WORLD_SIZE).backward()
                    return loss.detach()

                loss = _forbid_gather_and_full_saves(43, train)
                dist.all_reduce(loss)
                torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
                grad = model.lm_head.weight.grad.full_tensor()
                assert torch.count_nonzero(grad[vocab:]) == 0
                torch.testing.assert_close(grad, dense_weight.grad, rtol=2e-2, atol=1e-3)
                torch.testing.assert_close(
                    local_hidden.grad / _WORLD_SIZE, dense_hidden.grad[rank], rtol=2e-2, atol=1e-3
                )
                torch.optim.SGD(model.parameters(), lr=0.1).step()
                torch.testing.assert_close(
                    model.lm_head.weight.full_tensor(), weight - 0.1 * dense_weight.grad, rtol=2e-2, atol=1e-3
                )

    def inkling_chunked_head() -> None:
        """Compare streamed BF16 head gradients against dense softcapped FP32 logits with padding."""
        from nemo_automodel.components.models.inkling2.layers import Inkling2LMHead
        from tests.unit_tests.models.inkling2.test_reference import tiny_config

        weight, hidden, labels = _data(43)
        config = tiny_config(dtype="bfloat16")
        config.hidden_size, config.vocab_size, config.unpadded_vocab_size = _HIDDEN, 43, 37
        head = Inkling2LMHead(config, dtype=torch.bfloat16, logits_dtype=torch.float32)
        head.weight = torch.nn.Parameter(weight.bfloat16())
        labels = torch.where(labels == -100, labels, labels % config.unpadded_vocab_size)
        dense_hidden = hidden.bfloat16().requires_grad_()
        normalizer = int((labels != -100).sum())
        reference = F.cross_entropy(head(dense_hidden).flatten(0, 2), labels.flatten(), reduction="sum") / normalizer
        reference.backward()
        expected_grad = head.weight.grad
        head.weight = torch.nn.Parameter(distribute_tensor(head.weight.detach(), flat, [Shard(0)]))
        local_hidden = dense_hidden[rank].detach().requires_grad_()

        def train() -> torch.Tensor:
            loss = chunked(local_hidden, labels[rank], head.weight, normalizer, world, lm_head=head)
            (loss * _WORLD_SIZE).backward()
            return loss.detach()

        loss = _forbid_gather_and_full_saves(43, train)
        dist.all_reduce(loss)
        torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
        grad = head.weight.grad.full_tensor()
        torch.testing.assert_close(grad, expected_grad, rtol=2e-2, atol=1e-3)
        assert grad[config.unpadded_vocab_size :].count_nonzero() == 0
        torch.testing.assert_close(local_hidden.grad / _WORLD_SIZE, dense_hidden.grad[rank], rtol=2e-2, atol=1e-3)

    def negative_infinity_chunk() -> None:
        """A chunk with only -inf logits has finite dense-equivalent gradients."""
        old_budget = linear_ce_base._STREAM_CHUNK_BYTES
        linear_ce_base._STREAM_CHUNK_BYTES = _WORLD_SIZE * 2
        try:
            weight = torch.tensor([[-2.0], [0.0]] * _WORLD_SIZE, dtype=torch.float16)
            hidden = torch.full((1, 1), 65504.0, dtype=torch.float16, requires_grad=True)
            labels = torch.tensor([1])
            dense_weight = weight.clone().requires_grad_()
            dense_hidden = hidden.detach().clone().requires_grad_()
            expected = F.cross_entropy(F.linear(dense_hidden, dense_weight).float(), labels)
            expected.backward()
            sharded = torch.nn.Parameter(distribute_tensor(weight, flat, [Shard(0)]))
            actual = chunked(hidden, labels, sharded, grad_reduce_group=world)
            actual.backward()
            grad = sharded.grad.full_tensor()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(hidden.grad, dense_hidden.grad)
            torch.testing.assert_close(grad, dense_weight.grad)
        finally:
            linear_ce_base._STREAM_CHUNK_BYTES = old_budget

    def target_validation() -> None:
        """One bad rank fails all FSDP/HSDP peers before streaming; ignores stay valid."""
        for mesh, placements in ((flat, [Shard(0)]), (hsdp, [Replicate(), Shard(0)])):
            weight = torch.arange(18, dtype=torch.float32).reshape(9, 2) * 0.01
            sharded = torch.nn.Parameter(distribute_tensor(weight, mesh, placements))
            # Physical vocabulary 9 includes padding; only ids 0..6 are usable.
            operand = ShardedLMHeadWeight(sharded, dtype=torch.float32, vocab_size=7)
            for bad in (-1, 7, 1.5):
                labels = torch.tensor([bad if rank == 1 else -100])
                error, match = (RuntimeError, "dtype") if bad == 1.5 else (IndexError, "Target ids")
                with pytest.raises(error, match=match):
                    chunked(torch.ones(1, 2), labels, operand, grad_reduce_group=world)
            for ignore in (-7, 7):
                loss_fn = ChunkedCrossEntropy(1, compile=False, ignore_index=ignore)
                # Include a fully ignored rank and both vocabulary boundaries.
                labels = torch.tensor([ignore, ignore if rank == 3 else (0 if rank == 0 else 6)])
                hidden = torch.ones(2, 2, requires_grad=True)
                dense_hidden = hidden.detach().clone().requires_grad_()
                dense_weight = weight.clone().requires_grad_()
                expected = F.cross_entropy(
                    F.linear(dense_hidden, dense_weight[:7]), labels, ignore_index=ignore, reduction="sum"
                )
                expected.backward()
                dist.all_reduce(dense_weight.grad)
                dense_weight.grad.div_(_WORLD_SIZE)
                sharded.grad = None
                actual = loss_fn(hidden, labels, operand, grad_reduce_group=world)
                actual.backward()
                grad = sharded.grad.full_tensor()
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(hidden.grad, dense_hidden.grad)
                torch.testing.assert_close(grad, dense_weight.grad)

    checks = {
        # 37 rows over 4 ranks: shards of 10, 10, 10 and 7 rows, streamed in chunks of 3.
        "fsdp_uneven_vocab_chunked_ce": lambda: _check_training_parity(
            rank, flat, [Shard(0)], 37, chunked, world, ranks
        ),
        "fsdp_uneven_vocab_fused_ce": lambda: _check_training_parity(rank, flat, [Shard(0)], 37, fused, world, ranks),
        # 9 rows over 4 ranks: shards of 3, 3, 3 and 0 rows.
        "fsdp_empty_shard": lambda: _check_training_parity(rank, flat, [Shard(0)], 9, chunked, world, ranks),
        "hsdp": lambda: _check_training_parity(rank, hsdp, [Replicate(), Shard(0)], 37, chunked, world, ranks),
        # Each rank alone; every one trains on the supervised sample 0.
        "single_rank_mesh": lambda: _check_training_parity(
            rank, singleton_mesh, [Shard(0)], 37, chunked, singletons[rank], [rank], data_rows=[0]
        ),
        "eval_without_group": eval_without_group,
        "frozen_head": frozen_head,
        "mismatched_group_raises": mismatched_group_raises,
        "unsupported_placement_raises": unsupported_placement_raises,
        "mtp_depths_share_one_resolution": mtp_depths_share_one_resolution,
        "autocast_recompute": autocast_recompute,
        "padded_vocab_adapter": padded_vocab_adapter,
        "inkling_chunked_head": inkling_chunked_head,
        "negative_infinity_chunk": negative_infinity_chunk,
        "target_validation": target_validation,
    }
    assert tuple(checks) == _SCENARIOS
    outcomes = {}
    for name, check in checks.items():
        try:
            check()
            outcomes[name] = "ok"
        except Exception:
            outcomes[name] = traceback.format_exc()
        # A failed scenario must not desynchronize the collectives of the next one.
        dist.barrier()
    return outcomes


def _worker(rank: int, port: int, result_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        outcomes = _run_scenarios(rank)
    finally:
        dist.destroy_process_group()
    Path(result_dir, f"rank{rank}.json").write_text(json.dumps(outcomes))


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
    result_dir = tmp_path_factory.mktemp("linear_ce_streamed")
    mp.spawn(_worker, args=(_free_port(), str(result_dir)), nprocs=_WORLD_SIZE, join=True)
    per_rank = [json.loads(Path(result_dir, f"rank{rank}.json").read_text()) for rank in range(_WORLD_SIZE)]
    return {name: [outcome[name] for outcome in per_rank] for name in _SCENARIOS}


@pytest.mark.runtime_budget(
    30,
    hard_timeout=90,
    reason="spawns four Gloo ranks once; every scenario shares that spawn",
)
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_streamed_linear_ce_scenario(outcomes: dict[str, list[str]], scenario: str) -> None:
    failures = [f"rank {rank}:\n{outcome}" for rank, outcome in enumerate(outcomes[scenario]) if outcome != "ok"]
    assert not failures, "\n".join(failures)
