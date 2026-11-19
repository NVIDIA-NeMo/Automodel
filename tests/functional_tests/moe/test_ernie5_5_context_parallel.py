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
that no expert parallelism is needed, and packs the same three documents into one row of 768 tokens. With four
ranks the second document starts on the second rank and ends on the third.

The test compares the logits and the gradients, summed over the ranks, of four shards with one process that holds
the whole row. The shards run with activation checkpointing, as in the packed recipes, so every layer runs again
during backward, with its exchange between the ranks.
"""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.moe.parallelizer import checkpoint_moe_block

VOCAB, WINDOW = 64, 8
# Once each document starts at a multiple of 128, the documents start at row offsets 0, 256 and 640.
LENGTHS = (130, 300, 60)
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
    for index, layer in enumerate(model.model.layers):
        layer.self_attn.setup_cp_attention(mesh)
        model.model.layers[index] = checkpoint_moe_block(layer, router_policy=None)
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
    parameters = {name.replace("._checkpoint_wrapped_module", ""): p for name, p in model.named_parameters()}
    gradients = {}
    for name in expected:
        gradient = parameters[name].grad
        gradients[name] = torch.zeros_like(expected[name]) if gradient is None else gradient.clone()
        dist.all_reduce(gradients[name])
    failures = [] if len(expected) > 100 else [f"only {len(expected)} tensors have a gradient"]
    difference = (row[sharder.shard_layout.input_token_stream_positions[0]] - reference[0].detach()).abs().max()
    # A NaN compares false with every tolerance, so a non-finite difference is a failure of its own.
    if not torch.isfinite(difference) or difference > LOGIT_ATOL:
        failures.append(f"logits differ by {difference:.3e}")
    for name, gradient in gradients.items():
        error, scale = (gradient - expected[name]).abs().max(), expected[name].abs().max()
        if not torch.isfinite(error) or error > GRADIENT_RTOL * scale:
            failures.append(f"gradient of {name} differs by {error:.3e} at scale {scale:.3e}")
    return failures


def _worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        torch.set_num_threads(1)
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
        model = _model()
        failures = _check_shards(rank, mesh, model)
        assert not failures, failures[:5]
    finally:
        dist.destroy_process_group()


def _run(world_size: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        # All ranks run on this host.
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_worker, args=(world_size, (tmp_path / "rendezvous").as_uri()), nprocs=world_size, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.timeout(300)
def test_four_shards_match_one_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _run(4, monkeypatch, tmp_path)
