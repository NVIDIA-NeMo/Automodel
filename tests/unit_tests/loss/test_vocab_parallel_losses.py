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

"""Four-rank CPU/Gloo parity of logits losses on vocabulary-sharded (tensor-parallel) DTensor logits.

One spawn runs every scenario and records its outcome; the tests assert on the
recorded outcomes. Each scenario feeds a ``Shard(-1)`` DTensor over a four-rank
TP mesh and compares the loss and the logits gradient with the same loss on the
dense logits in a single process. The loss's forward and backward may not run
any gather collective (``CommDebugMode``), and no autograd-saved tensor may span
the whole vocabulary. Vocabularies of 13 and 9 ids give uneven shards
(4/4/4/1) and an empty shard (3/3/3/0).
"""

from __future__ import annotations

import json
import os
import socket
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_WORLD_SIZE = 4
_SCENARIOS = (
    "masked_ce_sum_with_mask",
    "masked_ce_mean_and_none_with_empty_shard",
    "masked_ce_loss_weights",
    "kd_loss",
    "kd_loss_rejects_mixed_layouts",
    "mdlm_and_encoder_ar",
    "hybrid_legacy_concat",
    "idlm",
    "scdd",
    "uno_tv_and_reverse_kl",
    "dflash_nll_and_argmax",
    "argmax_ties_and_layout_errors",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _vocab_range(vocab: int, rank: int) -> tuple[int, int]:
    """Return the ``torch.chunk`` vocabulary range ``[start, end)`` of ``rank``."""
    rows = -(-vocab // _WORLD_SIZE)
    start = min(rank * rows, vocab)
    return start, min(start + rows, vocab)


def _as_sharded(mesh, full: torch.Tensor, *, requires_grad: bool):
    """Wrap this rank's vocabulary slice of ``full`` as a ``Shard(-1)`` DTensor without communication.

    The local slice is a leaf, as the output of a tensor-parallel ``lm_head`` is
    produced locally; a leaf DTensor would instead redistribute its accumulated
    gradient.

    Args:
        mesh: One-dimensional tensor-parallel mesh.
        full: Tensor of shape [..., vocab], identical on every rank.
        requires_grad: Whether the local slice is a gradient leaf.

    Returns:
        The local leaf of shape [..., local_vocab] and the DTensor of global
        shape [..., vocab] built on it.
    """
    from torch.distributed.tensor import DTensor, Shard

    start, end = _vocab_range(full.shape[-1], dist.get_rank())
    local = full[..., start:end].detach().clone().requires_grad_(requires_grad)
    full = full.contiguous()
    sharded = DTensor.from_local(
        local, mesh, [Shard(full.ndim - 1)], run_check=False, shape=full.shape, stride=full.stride()
    )
    return local, sharded


def _guarded(vocab: int, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Run ``fn`` and fail if it gathers anything or saves a full-vocabulary tensor.

    Args:
        vocab: Global vocabulary size; no saved tensor may have it as its last axis.
        fn: Computes a loss (forward and, if needed, backward).

    Returns:
        The value returned by ``fn``.
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.debug import CommDebugMode

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        if local.ndim and local.shape[-1] == vocab:
            raise AssertionError(f"a full-vocabulary tensor of shape {tuple(local.shape)} was saved for backward")
        return tensor

    with CommDebugMode() as comm, torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = fn()
    gathers = {str(op): count for op, count in comm.get_comm_counts().items() if "gather" in str(op)}
    assert not gathers, f"the loss gathered: {gathers}"
    return result


def _check_parity(
    mesh,
    full_logits: torch.Tensor,
    loss: Callable[[torch.Tensor], torch.Tensor],
    *,
    upstream: torch.Tensor | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Compare ``loss`` on sharded logits with ``loss`` on the dense logits.

    Args:
        mesh: One-dimensional tensor-parallel mesh.
        full_logits: Tensor of shape [..., vocab], identical on every rank.
        loss: Maps logits (sharded DTensor or dense tensor) to a loss tensor.
        upstream: Optional gradient for a non-scalar loss, identical on every rank.
        rtol: Relative tolerance for the loss and gradient.
        atol: Absolute tolerance for the loss and gradient.
    """
    vocab = full_logits.shape[-1]
    local, sharded = _as_sharded(mesh, full_logits, requires_grad=True)

    def run() -> torch.Tensor:
        value = loss(sharded)
        value.backward(upstream)
        return value

    actual = _guarded(vocab, run)
    dense = full_logits.detach().clone().requires_grad_()
    expected = loss(dense)
    expected.backward(upstream)
    start, end = _vocab_range(vocab, dist.get_rank())
    error = None
    try:
        torch.testing.assert_close(actual.detach(), expected.detach(), rtol=rtol, atol=atol)
        torch.testing.assert_close(local.grad, dense.grad[..., start:end], rtol=rtol, atol=atol)
    except AssertionError as caught:
        error = caught
    # Fail on every rank together: a rank that raised alone would leave its peers in the next collective.
    failed = torch.tensor(int(error is not None))
    dist.all_reduce(failed)
    if failed:
        raise error or AssertionError("parity failed on another rank")


def _run_scenarios(rank: int) -> dict[str, str]:
    """Run every scenario on this rank and return its outcome."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    from nemo_automodel.components.loss import vocab_parallel
    from nemo_automodel.components.loss.dllm_loss import (
        DFlashDecayLoss,
        HybridDiffusionLLMLoss,
        IDLMLoss,
        MDLMCrossEntropyLoss,
        SCDDLoss,
        UnoDistillLoss,
        encoder_ar_loss,
    )
    from nemo_automodel.components.loss.kd_loss import KDLoss
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("tp",))
    generator = torch.Generator().manual_seed(1234)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    def randint(high: int, *shape: int) -> torch.Tensor:
        return torch.randint(high, shape, generator=generator)

    def masked_ce_sum_with_mask() -> None:
        logits = randn(2, 5, 13)
        labels = randint(13, 2, 5)
        labels[0, :2] = -100
        mask = torch.ones(2, 5, dtype=torch.long)
        mask[1, 4] = 0
        loss_fn = MaskedCrossEntropy()
        _check_parity(mesh, logits, lambda x: loss_fn(x, labels.clone(), mask=mask, num_label_tokens=7))

    def masked_ce_mean_and_none_with_empty_shard() -> None:
        logits = randn(2, 5, 9)
        labels = randint(9, 2, 5)
        labels[1, 1] = -100
        labels[0, 3] = 8  # the last id, owned by the third rank (ids 6..8); the fourth shard is empty
        _check_parity(mesh, logits, lambda x: MaskedCrossEntropy(reduction="mean")(x, labels.clone()))
        upstream = randn(10)
        _check_parity(
            mesh, logits, lambda x: MaskedCrossEntropy(reduction="none")(x, labels.clone()), upstream=upstream
        )

    def masked_ce_loss_weights() -> None:
        logits = randn(2, 5, 13)
        labels = randint(13, 2, 5)
        weights = torch.rand(2, 5, generator=generator)
        loss_fn = MaskedCrossEntropy()
        _check_parity(mesh, logits, lambda x: loss_fn(x, labels.clone(), loss_weights=weights, num_label_tokens=10))

    def kd_loss() -> None:
        student = randn(2, 5, 13)
        dense_teacher = randn(2, 5, 13)
        _, teacher = _as_sharded(mesh, dense_teacher, requires_grad=False)
        labels = randint(13, 2, 5)
        labels[1, :3] = -100
        loss_fn = KDLoss(temperature=2.0)

        def loss(x: torch.Tensor) -> torch.Tensor:
            return loss_fn(x, teacher if isinstance(x, DTensor) else dense_teacher, labels, 7)

        _check_parity(mesh, student, loss)

    def kd_loss_rejects_mixed_layouts() -> None:
        _, student = _as_sharded(mesh, randn(2, 5, 13), requires_grad=False)
        with pytest.raises(ValueError, match="one vocabulary layout"):
            KDLoss()(student, randn(2, 5, 13), randint(13, 2, 5))

    def mdlm_and_encoder_ar() -> None:
        logits = randn(2, 6, 13)
        targets = randint(13, 2, 6)
        noise_mask = torch.rand(2, 6, generator=generator) > 0.4
        p_mask = torch.rand(2, 6, generator=generator) * 0.9 + 0.1
        loss_mask = torch.ones(2, 6, dtype=torch.long)
        loss_mask[1, 0] = 0
        mdlm = MDLMCrossEntropyLoss()

        def loss(x: torch.Tensor) -> torch.Tensor:
            diffusion = mdlm(x, targets, noise_mask, p_mask, loss_mask, num_diffusion_tokens=9).total_loss
            return diffusion + encoder_ar_loss(x, targets, valid_mask=loss_mask.bool())

        _check_parity(mesh, logits, loss)

    def hybrid_legacy_concat() -> None:
        logits = randn(2, 10, 13)  # [x_t | causal] halves of 5 positions each
        targets = randint(13, 2, 5)
        noise_mask = torch.rand(2, 5, generator=generator) > 0.3
        p_mask = torch.rand(2, 5, generator=generator) * 0.9 + 0.1
        loss_mask = torch.ones(2, 5, dtype=torch.long)
        hybrid = HybridDiffusionLLMLoss(alpha=0.7)
        _check_parity(
            mesh,
            logits,
            lambda x: hybrid(x, targets, noise_mask, p_mask, loss_mask, loss_mask_ar=loss_mask).total_loss,
        )

    def idlm() -> None:
        logits = randn(2, 10, 13)
        targets = randint(13, 2, 5)
        answer = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1]], dtype=torch.bool)
        valid = torch.ones(2, 5, dtype=torch.long)
        loss_fn = IDLMLoss(auto_balance=True)
        _check_parity(mesh, logits, lambda x: loss_fn(x, targets, answer, valid, seq_len=5).total_loss)

    def scdd() -> None:
        # The [MASK] id 5 lives on the second rank (ids 4..7); chunks of 3 positions run checkpointed.
        logits = randn(2, 5, 13)
        clean = randint(13, 2, 5)
        noisy = clean.clone()
        noisy[0, 1] = 5
        noisy[1, 3] = 5
        noisy[1, 0] = (clean[1, 0] + 3) % 13
        p_mask = torch.tensor([[0.4] * 5, [0.8] * 5])
        loss_mask = torch.ones(2, 5, dtype=torch.long)
        scdd_loss = SCDDLoss(mask_token_id=5, num_timesteps=10, max_ratio=0.2, chunk_size=3)

        def loss(x: torch.Tensor) -> torch.Tensor:
            return scdd_loss(x, clean, None, p_mask, loss_mask, noisy_input_ids=noisy).total_loss

        _check_parity(mesh, logits, loss, rtol=1e-5, atol=1e-5)

    def uno_tv_and_reverse_kl() -> None:
        logits = randn(2, 10, 13)
        answer = torch.tensor([[1, 1, 0, 1, 1], [0, 1, 1, 1, 0]], dtype=torch.bool)
        valid = torch.ones(2, 5, dtype=torch.long)
        uno = UnoDistillLoss(tv_weight=1.0, kl_weight=0.5, chunk_size=3)
        _check_parity(mesh, logits, lambda x: uno(x, answer, valid, seq_len=5).total_loss)

    def dflash_nll_and_argmax() -> None:
        logits = randn(2, 3, 4, 13)
        logits[0, 0, 0, 2] = logits[0, 0, 0, 9] = 50.0  # tie across shards resolves to the first id
        targets = randint(13, 2, 3, 4)
        block_mask = torch.rand(2, 3, 4, generator=generator) > 0.2
        dflash = DFlashDecayLoss()
        _check_parity(
            mesh,
            logits,
            lambda x: dflash.forward_with_token_nll(x, targets, block_mask, block_size=5).output.total_loss,
        )
        _, sharded = _as_sharded(mesh, logits, requires_grad=False)
        details = _guarded(13, lambda: dflash.forward_with_token_nll(sharded, targets, block_mask, block_size=5))
        assert torch.equal(details.pred_ids, logits.argmax(dim=-1))
        assert details.pred_ids[0, 0, 0].item() == 2

    def argmax_ties_and_layout_errors() -> None:
        logits = randn(3, 9)
        logits[0, 1] = logits[0, 7] = 9.0  # first and third ranks tie
        logits[1, 5] = logits[1, 6] = 9.0  # tie within one shard
        shard = vocab_parallel.vocab_shard(_as_sharded(mesh, logits, requires_grad=False)[1])
        assert (shard.start, shard.local.shape[-1]) == ([0, 3, 6, 9][rank], [3, 3, 3, 0][rank])
        assert torch.equal(_guarded(9, lambda: vocab_parallel.argmax(shard)), logits.argmax(dim=-1))
        replicated = vocab_parallel.vocab_shard(distribute_tensor(logits, mesh, [Replicate()]))
        assert replicated.group is None and replicated.local.shape == logits.shape
        with pytest.raises(ValueError, match="sharded on the vocabulary axis"):
            vocab_parallel.vocab_shard(distribute_tensor(randn(8, 9), mesh, [Shard(0)]))

    checks = {
        "masked_ce_sum_with_mask": masked_ce_sum_with_mask,
        "masked_ce_mean_and_none_with_empty_shard": masked_ce_mean_and_none_with_empty_shard,
        "masked_ce_loss_weights": masked_ce_loss_weights,
        "kd_loss": kd_loss,
        "kd_loss_rejects_mixed_layouts": kd_loss_rejects_mixed_layouts,
        "mdlm_and_encoder_ar": mdlm_and_encoder_ar,
        "hybrid_legacy_concat": hybrid_legacy_concat,
        "idlm": idlm,
        "scdd": scdd,
        "uno_tv_and_reverse_kl": uno_tv_and_reverse_kl,
        "dflash_nll_and_argmax": dflash_nll_and_argmax,
        "argmax_ties_and_layout_errors": argmax_ties_and_layout_errors,
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
    result_dir = tmp_path_factory.mktemp("vocab_parallel_losses")
    mp.spawn(_worker, args=(_free_port(), str(result_dir)), nprocs=_WORLD_SIZE, join=True)
    per_rank = [json.loads(Path(result_dir, f"rank{rank}.json").read_text()) for rank in range(_WORLD_SIZE)]
    return {name: [outcome[name] for outcome in per_rank] for name in _SCENARIOS}


@pytest.mark.runtime_budget(
    30,
    hard_timeout=90,
    reason="spawns four Gloo ranks once; every scenario shares that spawn",
)
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_vocab_parallel_loss_scenario(outcomes: dict[str, list[str]], scenario: str) -> None:
    failures = [f"rank {rank}:\n{outcome}" for rank, outcome in enumerate(outcomes[scenario]) if outcome != "ok"]
    assert not failures, "\n".join(failures)
