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

The pack is laid out to stress the rank boundaries of a four-rank split (24 tokens per rank): a document ending one
token before a boundary, a one-token document on the last token of a rank, a one-token document on the first token of
the next rank, a document spanning three ranks (longer than ``index_topk`` and the sliding window, so selection and the
window both drop keys while the keys live on other ranks), short documents and trailing padding.
"""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.models.hy_v4_flash.attention import HYV4FlashAttention
from tests.unit_tests.models.hy_v4_flash.tiny import (
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
    """Sum of squared logits of the real tokens.

    Args:
        logits: Tensor of shape [1, tokens, vocab].
        attention_mask: Bool tensor of shape [1, tokens], False at padding.
    """
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


def _worker(rank: int, rendezvous: str, result_path: str, broken: bool) -> None:
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
        if broken:
            _drop_remote_keys(rank, local_tokens)
        shard = {k: v[:, rank * local_tokens : (rank + 1) * local_tokens] for k, v in batch.items()}
        out = model(**shard)
        grads = {}
        if not broken:  # the negative control only compares logits
            # the per-rank objectives sum to the single-rank objective
            _objective(out.logits, shard["attention_mask"]).backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad = parameter.grad.clone()
                    dist.all_reduce(grad)
                    grads[name] = grad
        torch.save({"logits": out.logits.detach(), "grads": grads}, result_path)
    finally:
        dist.destroy_process_group()


def _run_ranks(tmp_path: Path, broken: bool) -> list[dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
    context = mp.get_context("spawn")
    paths = [str(tmp_path / f"rank{rank}.pt") for rank in range(WORLD)]
    processes = [
        context.Process(target=_worker, args=(rank, str(tmp_path / "init"), paths[rank], broken))
        for rank in range(WORLD)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=400)
        assert process.exitcode == 0
    results = [torch.load(path) for path in paths]
    return torch.cat([r["logits"] for r in results], dim=1), results[0]["grads"]


def _assert_documents_match_alone(logits: torch.Tensor) -> None:
    """Compare the reassembled logits [1, tokens, vocab] document by document with each document run alone."""
    documents = random_documents(DOC_LENGTHS)
    per_document = list(logits[0, : sum(DOC_LENGTHS)].split(list(DOC_LENGTHS)))
    assert_documents_match_alone(build_model(), documents, per_document)


@pytest.mark.cpu_only
@pytest.mark.runtime_budget(120, reason="spawns four Gloo workers that each build the tiny model")
def test_cp4_matches_unpacked_documents_and_single_rank_gradients(tmp_path: Path) -> None:
    logits, grads = _run_ranks(tmp_path, broken=False)
    _assert_documents_match_alone(logits)
    reference_logits, reference_grads = _single_rank()
    real = sum(DOC_LENGTHS)
    torch.testing.assert_close(logits[:, :real], reference_logits[:, :real], **TOLERANCE)
    assert grads.keys() == reference_grads.keys()
    for name, grad in reference_grads.items():
        torch.testing.assert_close(
            grads[name], grad, rtol=2e-3, atol=2e-4, msg=lambda m, name=name: f"grad {name}: {m}"
        )


@pytest.mark.cpu_only
@pytest.mark.runtime_budget(120, reason="spawns four Gloo workers that each build the tiny model")
def test_cp4_without_the_cross_rank_key_gather_is_detected(tmp_path: Path) -> None:
    logits, _ = _run_ranks(tmp_path, broken=True)
    with pytest.raises(AssertionError):
        _assert_documents_match_alone(logits)
