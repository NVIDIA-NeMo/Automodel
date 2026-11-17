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

"""Context-parallel shards of a packed ERNIE 5.5 row against one process: Gloo ranks on CPU, no GPU needed.

Every rank builds the same small FP32 model on the eager attention path, with dense feed-forward layers only, so
that no expert parallelism is needed, and packs the same three documents into one row of 768 tokens. The second
document starts on the first rank of two and ends on the second.

* ``test_keys_reach_the_attention_in_row_order`` gives every key its row offset as content and records what the
  attention cores receive after the exchange between ranks: an HCA query reads the window keys and the complete
  compressed keys of its own document, across the shard boundary, and a DSA layer scores and attends over the keys
  of the whole row in row order.
* ``test_shards_match_one_process`` compares the logits and the gradients, summed over the ranks, of two and of
  four shards with one process that holds the whole row.
"""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5 import hca as hca_module
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM

VOCAB, RATIO, WINDOW = 64, 128, 8
# Row offsets of the documents once each starts at a multiple of 128: 0, 256 and 640.
LENGTHS = (130, 300, 60)
STARTS = (0, 256, 640)
ROW = 768
RATIOS = (128, 128, -2, 128, -2)
# Largest differences measured between the shards and one process: 6.0e-8 in the logits, and 1.7e-5 of the largest
# entry of a gradient.
LOGIT_ATOL, GRADIENT_RTOL = 1e-6, 1e-3


def _model() -> Ernie5_5_MoeForCausalLM:
    """Build a seeded FP32 model with dense feed-forward layers and an indexer whose scores have no ties."""
    config = Ernie5_5Config(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=len(RATIOS),
        max_position_embeddings=1024,
        compress_ratios=list(RATIOS),
        first_k_dense_replace=len(RATIOS),
        num_nextn_predict_layers=0,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=6,
        o_lora_rank=4,
        o_groups=2,
        sliding_window=WINDOW,
        hybrid_mla_num_attention_heads=2,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=4,
        hybrid_mla_qk_nope_head_dim=4,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=3,
        index_n_heads=32,
        index_head_dim=16,
        index_topk=3,
        rope_interleave=True,
        hc_mult=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        moe_latent_size=8,
        pad_token_id=0,
        torch_dtype="float32",
    )
    model = Ernie5_5_MoeForCausalLM(
        config, BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")
    )
    torch.manual_seed(5591)
    model.initialize_weights(dtype=torch.float32)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("vha_postmix_V", "sinks_param.weight", "softmax_offset")):
                parameter.normal_(std=0.5)
    for layer in model.model.layers:
        if layer.is_dsa:
            # The one-process reference holds the whole row.
            layer.attention.max_eager_sequence = ROW
    return model


def _batch() -> dict[str, torch.Tensor | str]:
    """Return the packed row of the three documents as the THD collater emits it, with random targets."""
    generator = torch.Generator().manual_seed(3)
    tokens = sum(LENGTHS)
    return {
        "input_ids": torch.randint(1, VOCAB, (1, tokens), generator=generator),
        "labels": torch.randint(1, VOCAB, (1, tokens), generator=generator),
        "seq_lens": torch.tensor([LENGTHS]),
        "qkv_format": "thd",
    }


def _layout(first: int, sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the positions and the documents, counted from one, of ``sequence`` row offsets from ``first``."""
    positions = torch.zeros(1, ROW, dtype=torch.int64)
    documents = torch.zeros(1, ROW, dtype=torch.int64)
    for document, (start, length) in enumerate(zip(STARTS, LENGTHS), start=1):
        positions[0, start : start + length] = torch.arange(length)
        documents[0, start : start + length] = document
    return positions[:, first : first + sequence], documents[:, first : first + sequence]


def _check_hca_keys(rank: int, mesh, model: Ernie5_5_MoeForCausalLM) -> list[str]:
    """Record the keys and key indices of a packed HCA call on this rank and compare them with the row."""
    attention = model.model.layers[0].attention
    attention.setup_cp_attention(mesh)
    sequence = ROW // mesh.size()
    first = rank * sequence
    position_ids, packed_seq_ids = _layout(first, sequence)
    offsets = torch.arange(first, first + sequence, dtype=torch.float32)
    # Key i holds its row offset in its first channel. The compressor pools (offset, 1) of every token uniformly,
    # so the ratio of the first two channels of a compressed key is the mean row offset of its 128 tokens.
    keys = torch.zeros(1, sequence, attention.head_dim)
    keys[0, :, 0] = offsets
    hidden = torch.zeros(1, sequence, attention.config.hidden_size)
    hidden[0, :, 0], hidden[0, :, 1] = offsets, 1.0
    with torch.no_grad():
        attention.compressor.wgate.weight.zero_()
        attention.compressor.wkv.weight.zero_()
        attention.compressor.wkv.weight[0, 0] = attention.compressor.wkv.weight[1, 1] = 1.0
    seen = {}

    def record(query, kv, sinks, indices, scale, **kwargs):
        seen.update(kv=kv, indices=indices)
        return torch.zeros_like(query)

    original, hca_module.dsv4_sparse_attention = hca_module.dsv4_sparse_attention, record
    try:
        rotary = model.model._hca_rotary(torch.device("cpu"))
        with torch.no_grad():
            attention._attend_packed(
                hidden,
                torch.zeros(1, sequence, attention.config.q_lora_rank),
                torch.zeros(1, attention.num_heads, sequence, attention.head_dim),
                keys,
                rotary(torch.empty(1, sequence, 1), position_ids),
                rotary,
                position_ids,
                packed_seq_ids,
            )
    finally:
        hca_module.dsv4_sparse_attention = original
    kv, indices = seen["kv"][0], seen["indices"][0]
    names = torch.where(kv[:, 1] != 0, kv[:, 0] / kv[:, 1], kv[:, 0])
    failures = []
    for local in range(sequence):
        position, document = position_ids[0, local].item(), packed_seq_ids[0, local].item()
        offset = first + local
        if document == 0:
            expected = [float(offset)]
        else:
            start = offset - position
            expected = [float(key) for key in range(max(offset - WINDOW + 1, start), offset + 1)]
            expected += [start + RATIO * window + (RATIO - 1) / 2 for window in range((position + 1) // RATIO)]
        read = sorted(names[index].item() for index in indices[local] if index >= 0)
        if read != pytest.approx(sorted(expected), abs=1e-3):
            failures.append(f"HCA query at row offset {offset} reads {read}, expected {sorted(expected)}")
    return failures


class _Recorder(nn.Module):
    """Stand-in for the FP32 parameter holder of a DSA layer that keeps the arguments of its call."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.call: tuple[tuple, dict] | None = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Record the call and return zeros of shape [batch, sequence, width]."""
        self.call = (args, kwargs)
        return torch.zeros(*args[0].shape[:2], self.width)


def _check_dsa_keys(rank: int, mesh, model: Ernie5_5_MoeForCausalLM) -> list[str]:
    """Record what the eager DSA core receives on this rank and compare it with the row."""
    attention = model.model.layers[2].attention
    attention.setup_cp_attention(mesh)
    sequence = ROW // mesh.size()
    first = rank * sequence
    position_ids, packed_seq_ids = _layout(first, sequence)
    offsets = torch.arange(first, first + sequence, dtype=torch.float32)
    index_heads, index_dim = attention.indexer.num_heads, attention.indexer.head_dim

    def project(hidden, positions):
        # The latent key and the indexer key of token i hold i, and the index score of every query for key i is i.
        latent_kv = torch.zeros(1, sequence, attention.kv_rank + attention.rope_dim)
        latent_kv[0, :, 0] = offsets
        index_q = torch.zeros(1, sequence, index_heads, index_dim)
        index_q[..., 0] = 1.0
        index_k = torch.zeros(1, sequence, index_dim)
        index_k[0, :, 0] = offsets
        weights = torch.full((1, sequence, index_heads), 1.0 / index_heads)
        query = torch.zeros(1, sequence, attention.num_heads, attention.kv_rank + attention.rope_dim)
        return torch.zeros(1, sequence, attention.q_a_proj.out_features), query, latent_kv, index_q, index_k, weights

    attention._project = project
    attention.fp32_params = _Recorder(attention.num_heads * attention.value_dim)
    with torch.no_grad():
        attention(torch.zeros(1, sequence, 16), position_ids=position_ids, document_ids=packed_seq_ids - 1)
    (_, latent_kv, scores, document_ids, *_), kwargs = attention.fp32_params.call
    row = torch.arange(ROW, dtype=torch.float32)
    _, row_documents = _layout(0, ROW)
    failures = []
    if not torch.equal(latent_kv[0, :, 0], row):
        failures.append("DSA latent keys are not in row order")
    if not torch.allclose(scores[0], row.expand(sequence, ROW), atol=1e-4):
        failures.append("DSA indexer keys are not in row order")
    if not torch.equal(kwargs["key_document_ids"], row_documents - 1) or not torch.equal(
        document_ids, packed_seq_ids - 1
    ):
        failures.append("DSA document ids do not follow the row")
    if kwargs["first_query"] != first:
        failures.append(f"DSA first query is {kwargs['first_query']}, expected {first}")
    return failures


def _check_shards(rank: int, mesh, model: Ernie5_5_MoeForCausalLM) -> list[str]:
    """Compare the logits and the summed gradients of the shards with one process that holds the whole row."""
    batch = _batch()
    reference = model(batch["input_ids"], seq_lens=batch["seq_lens"]).logits
    F.cross_entropy(reference[0], batch["labels"][0], reduction="sum").backward()
    expected = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    model.zero_grad()

    model.cp_mesh = mesh
    for layer in model.model.layers:
        layer.self_attn.setup_cp_attention(mesh)
    sharder = ContextParallelSharder(model, mesh, batch)
    _, local = sharder.shard(batch)
    logits = model(
        local["input_ids"],
        attention_mask=local["attention_mask"],
        position_ids=local["position_ids"],
        packed_seq_ids=local["packed_seq_ids"],
    ).logits
    F.cross_entropy(logits[0], local["labels"][0], reduction="sum").backward()
    # Every collective runs before the first comparison, so that a failure on one rank cannot block another.
    row = sharder.gather_token_tensor(logits.detach(), seq_dim=1)[0]
    parameters = dict(model.named_parameters())
    gradients = {}
    for name in expected:
        gradient = parameters[name].grad
        gradients[name] = torch.zeros_like(expected[name]) if gradient is None else gradient.clone()
        dist.all_reduce(gradients[name])
    failures = [] if len(expected) > 100 else [f"only {len(expected)} tensors have a gradient"]
    difference = (row[sharder.shard_layout.input_token_stream_positions[0]] - reference[0].detach()).abs().max()
    if difference > LOGIT_ATOL:
        failures.append(f"logits differ by {difference:.3e}")
    for name, gradient in gradients.items():
        error, scale = (gradient - expected[name]).abs().max(), expected[name].abs().max()
        if error > GRADIENT_RTOL * scale:
            failures.append(f"gradient of {name} differs by {error:.3e} at scale {scale:.3e}")
    return failures


def _worker(rank: int, world_size: int, init_method: str, check: str) -> None:
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        torch.set_num_threads(1)
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
        model = _model()
        if check == "keys":
            failures = _check_hca_keys(rank, mesh, model) + _check_dsa_keys(rank, mesh, model)
        else:
            failures = _check_shards(rank, mesh, model)
        assert not failures, failures[:5]
    finally:
        dist.destroy_process_group()


def _run(world_size: int, check: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        # All ranks run on this host.
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_worker, args=(world_size, (tmp_path / "rendezvous").as_uri(), check), nprocs=world_size, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.timeout(300)
def test_keys_reach_the_attention_in_row_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _run(2, "keys", monkeypatch, tmp_path)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.timeout(300)
@pytest.mark.parametrize("world_size", [2, 4])
def test_shards_match_one_process(world_size: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _run(world_size, "shards", monkeypatch, tmp_path)
