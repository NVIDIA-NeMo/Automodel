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

"""Packed context parallelism (Gloo ranks) against independent single-process oracles, forward and backward.

The pack stresses the boundaries of a four-rank split (24 tokens per rank): documents ending just before, on and just
after a boundary, and one spanning three ranks that is longer than ``index_topk`` and the sliding window.
"""

from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.models.hy_v4_flash.attention import HYV4FlashAttention
from tests.unit_tests.models.hy_v4_flash.conftest import (
    TOLERANCE,
    assert_documents_match_alone,
    build_model,
    pack,
    random_documents,
)

DOC_LENGTHS = (23, 1, 1, 52, 3, 13)  # starts 0, 23, 24, 25, 77, 80; rank boundaries at 24, 48, 72
PADDING = 3
WORLD = 4


def _objective(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Sum of squared logits [1, tokens, vocab] over the real tokens of ``attention_mask`` [1, tokens]."""
    return (logits.square().sum(-1) * attention_mask).sum()


def _single_rank() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model = build_model()
    batch = pack(random_documents(DOC_LENGTHS), PADDING)
    logits = model(**batch).logits
    _objective(logits, batch["attention_mask"]).backward()
    return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _drop_remote_keys(rank: int, local_tokens: int) -> None:
    """Test-only broken variant: the cross-rank key gather returns zeros for every token of another rank."""
    original = HYV4FlashAttention._gather_keys

    def gather(
        self: HYV4FlashAttention,
        kv_local: torch.Tensor,
        token_keys: torch.Tensor | None,
        cp_group: dist.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        kv, keys = original(self, kv_local, token_keys, cp_group)
        own = torch.zeros(kv.shape[0], 1, dtype=torch.bool)
        own[rank * local_tokens : (rank + 1) * local_tokens] = True
        return kv * own, None if keys is None else keys * own

    HYV4FlashAttention._gather_keys = gather


def _worker(rank: int, rendezvous: str, result_path: str) -> None:
    torch.set_num_threads(1)  # four workers share the runner's cores
    # Built before the process group exists so the Engram tables stay complete and local (no owner sharding).
    model = build_model()
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=240)
    )
    try:
        # Wired the way the shared parallelizer's ``apply_cp`` does it: the mesh goes on the model, and only blocks
        # whose attention type is ``sliding_attention`` are offered ``setup_cp_attention``. Sparse blocks are never
        # called, so a layer that kept its own copy of the group would not gather keys there.
        mesh = init_device_mesh("cpu", (WORLD,), mesh_dim_names=("cp",))
        model.cp_mesh = mesh
        for layer in model.model.layers:
            if layer.attention_type == "sliding_attention":
                layer.self_attn.setup_cp_attention(mesh)
        batch = pack(random_documents(DOC_LENGTHS), PADDING)
        local_tokens = batch["input_ids"].shape[1] // WORLD
        shard = {k: v[:, rank * local_tokens : (rank + 1) * local_tokens] for k, v in batch.items()}
        out = model(**shard)
        grads = {}
        # the per-rank objectives sum to the single-rank objective
        _objective(out.logits, shard["attention_mask"]).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                grad = parameter.grad.clone()
                dist.all_reduce(grad)
                grads[name] = grad
        # Negative control in the same processes: the same forward without the cross-rank key gather.
        _drop_remote_keys(rank, local_tokens)
        with torch.no_grad():
            broken = model(**shard).logits
        torch.save({"logits": out.logits.detach(), "grads": grads, "broken": broken}, result_path)
    finally:
        dist.destroy_process_group()


@dataclass(frozen=True)
class _RankResults:
    """One four-rank run: logits [1, tokens, vocab], summed gradients, and the logits without the key gather."""

    logits: torch.Tensor
    grads: dict[str, torch.Tensor]
    broken_logits: torch.Tensor


@pytest.fixture(scope="module")
def four_ranks(tmp_path_factory: pytest.TempPathFactory) -> _RankResults:
    """Run the pack once over four Gloo ranks (real cross-rank forward and backward) for both tests."""
    tmp_path = tmp_path_factory.mktemp("cp4")
    context = mp.get_context("spawn")
    paths = [str(tmp_path / f"rank{rank}.pt") for rank in range(WORLD)]
    processes = [
        context.Process(target=_worker, args=(rank, str(tmp_path / "init"), paths[rank])) for rank in range(WORLD)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=400)
        assert process.exitcode == 0
    results = [torch.load(path) for path in paths]
    return _RankResults(
        torch.cat([r["logits"] for r in results], dim=1),
        results[0]["grads"],
        torch.cat([r["broken"] for r in results], dim=1),
    )


def _assert_documents_match_alone(logits: torch.Tensor) -> None:
    """Compare the reassembled logits [1, tokens, vocab] document by document with each document run alone."""
    documents = random_documents(DOC_LENGTHS)
    per_document = list(logits[0, : sum(DOC_LENGTHS)].split(list(DOC_LENGTHS)))
    assert_documents_match_alone(build_model(), documents, per_document)


@pytest.mark.cpu_only
def test_cp4_matches_unpacked_documents_and_single_rank_gradients(four_ranks: _RankResults) -> None:
    _assert_documents_match_alone(four_ranks.logits)
    reference_logits, reference_grads = _single_rank()
    real = sum(DOC_LENGTHS)
    torch.testing.assert_close(four_ranks.logits[:, :real], reference_logits[:, :real], **TOLERANCE)
    assert four_ranks.grads.keys() == reference_grads.keys()
    for name, grad in reference_grads.items():
        torch.testing.assert_close(
            four_ranks.grads[name], grad, rtol=2e-3, atol=2e-4, msg=lambda m, name=name: f"grad {name}: {m}"
        )


@pytest.mark.cpu_only
def test_cp4_without_the_cross_rank_key_gather_is_detected(four_ranks: _RankResults) -> None:
    with pytest.raises(AssertionError):
        _assert_documents_match_alone(four_ranks.broken_logits)
