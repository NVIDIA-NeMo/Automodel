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

"""Correctness tests for Inkling's head-chunked UPipe context parallelism.

Four layers of checking, cheapest first:

1. Head geometry -- pure arithmetic, no torch.distributed. Every ``(stage, rank)`` pair
   must hand a rank a query head belonging to the KV group of its co-resident KV head.
2. All-to-all -- two gloo ranks on CPU. ``cp2hp`` must deliver the right global slice and
   ``hp2cp`` must invert it, with gradients flowing back to the right rank.
3. FlexAttention translation -- one GPU. The ``score_mod`` / ``mask_mod`` pair must
   reproduce ``InklingRelativeLogits`` plus ``eager_attention_forward`` densely.
4. Module parity -- two GPUs, CP=2. The gate: UPipe ``InklingAttention`` forward *and*
   backward must match the single-device module on the same weights and inputs.
"""

from __future__ import annotations

import copy
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.distributed.context_parallel.upipe import UPipeHeadGeometry

CUDA_DEVICES = torch.cuda.device_count()

SEQ_LEN = 256
BATCH = 2
HIDDEN = 128
SLIDING_WINDOW = 64


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _init_pg(rank: int, world_size: int, port: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


# ---------------------------------------------------------------------------
# 1. Head geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "cp_size"),
    [
        (4, 2, 2),  # the parity-test config
        (64, 8, 2),  # Inkling global layer
        (64, 8, 8),
        (64, 16, 4),  # Inkling sliding layer
        (64, 8, 16),  # KV replication kicks in
        (64, 8, 32),
        (8, 8, 2),  # MHA, no GQA
        (8, 1, 4),  # MQA, heavy replication
    ],
)
def test_head_geometry_pairs_queries_with_their_kv_group(num_heads, num_kv_heads, cp_size):
    geometry = UPipeHeadGeometry.build(cp_size=cp_size, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=128)
    geometry.validate()

    real_gqa_ratio = num_heads // num_kv_heads
    covered = []
    for stage in range(geometry.pipe_degree):
        for rank in range(cp_size):
            query = geometry.query_head(stage, rank)
            # The pre-replication KV head this rank will actually convolve and attend to
            # must be the one true GQA would have paired with this query head.
            assert geometry.source_kv_head(stage, rank) == query // real_gqa_ratio
            covered.append(query)

    assert sorted(covered) == list(range(num_heads))
    assert geometry.pipe_degree * cp_size == num_heads


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "cp_size"),
    [
        (6, 2, 4),  # num_heads not divisible by cp_size
        (64, 12, 8),  # num_kv_heads not divisible by cp_size
        (64, 6, 16),  # cp_size not divisible by num_kv_heads (replication impossible)
    ],
)
def test_head_geometry_rejects_unschedulable_configs(num_heads, num_kv_heads, cp_size):
    with pytest.raises(ValueError):
        UPipeHeadGeometry.build(cp_size=cp_size, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=128)


def test_head_order_round_trips_through_its_inverse():
    geometry = UPipeHeadGeometry.build(cp_size=4, num_heads=64, num_kv_heads=8, head_dim=128)
    order = geometry.head_order()
    inverse = geometry.inverse_head_order()

    assert sorted(order.tolist()) == list(range(64))
    # Concatenating stage outputs yields heads in `order`; index_select with `inverse`
    # must restore natural order.
    emitted = torch.arange(64)[order]
    torch.testing.assert_close(emitted[inverse], torch.arange(64))


# ---------------------------------------------------------------------------
# 2. All-to-all
# ---------------------------------------------------------------------------


def _all_to_all_worker(rank: int, world_size: int, port: int) -> None:
    try:
        _init_pg(rank, world_size, port, "gloo")
        torch.set_num_threads(1)
        from nemo_automodel.components.distributed.context_parallel.upipe import cp2hp, hp2cp

        group = dist.group.WORLD
        batch, seq_full, heads, head_dim = 2, 8, 4, 3
        seq_local = seq_full // world_size
        heads_local = heads // world_size

        # Distinct value per (batch, position, head, channel) so a misroute cannot alias.
        base = torch.arange(batch * seq_full * heads * head_dim, dtype=torch.float32)
        global_tensor = base.reshape(batch, seq_full, heads, head_dim)

        local = global_tensor[:, rank * seq_local : (rank + 1) * seq_local].clone().requires_grad_(True)

        gathered = cp2hp(local, group)
        expected = global_tensor[:, :, rank * heads_local : (rank + 1) * heads_local]
        assert gathered.shape == expected.shape
        torch.testing.assert_close(gathered, expected)

        # cp2hp's gradient must land back on the rank that owned each token. Checking this
        # direction on its own matters: the round-trip below can only report *that*
        # something is wrong, not which half.
        #
        # Weight every global (position, head) slot distinctly, then hand each rank the
        # slice of those weights covering the heads it received. local[b, s, h, d] lands
        # on rank h // heads_local at global position rank * seq_local + s, where it picks
        # up weight[b, rank * seq_local + s, h, d].
        weights = base.reshape(batch, seq_full, heads, head_dim)
        (gathered * weights[:, :, rank * heads_local : (rank + 1) * heads_local]).sum().backward()
        torch.testing.assert_close(local.grad, weights[:, rank * seq_local : (rank + 1) * seq_local])

        # hp2cp must be the exact inverse, and the pair gradient-transparent. An
        # all-to-all writing into a mis-strided buffer still round-trips forward, so the
        # gradient half of this is the part that bites.
        round_trip_input = local.detach().clone().requires_grad_(True)
        restored = hp2cp(cp2hp(round_trip_input, group), group)
        torch.testing.assert_close(restored, round_trip_input)

        seed = torch.randn(restored.shape, generator=torch.Generator().manual_seed(rank))
        (restored * seed).sum().backward()
        torch.testing.assert_close(round_trip_input.grad, seed)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_all_to_all_moves_sequence_to_heads_and_back():
    mp.spawn(_all_to_all_worker, args=(2, _free_port()), nprocs=2, join=True)


# ---------------------------------------------------------------------------
# Shared config + module construction for the attention tests
# ---------------------------------------------------------------------------


def _tiny_text_config():
    """Two layers: index 0 global (with log scaling), index 1 sliding."""
    from transformers.models.inkling.configuration_inkling import InklingTextConfig

    return InklingTextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=32,
        sliding_window_size=SLIDING_WINDOW,
        d_rel=8,
        rel_extent=128,
        # Exercised only on the global layer; makes the tau path part of the gate.
        log_scaling_n_floor=32,
        log_scaling_alpha=0.1,
        layer_types=["hybrid", "hybrid_sliding"],
        max_position_embeddings=512,
        conv_kernel_size=4,
        attention_dropout=0.0,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        _attn_implementation="eager",
    )


def _build_attention(config, layer_idx: int, device, seed: int = 0):
    """Build an ``InklingAttention`` with deterministic weights and AutoModel's convs."""
    import torch.nn as nn
    from transformers.models.inkling.modeling_inkling import InklingAttention

    from nemo_automodel.components.models.inkling.layers import InklingShortConvolution

    torch.manual_seed(seed)
    attn = InklingAttention(config, layer_idx)
    # `proj` is allocated with torch.empty, so it must be filled explicitly.
    nn.init.normal_(attn.rel_logits_proj.proj, std=0.05)
    for name in ("k_sconv", "v_sconv"):
        setattr(attn, name, InklingShortConvolution(getattr(attn, name)))
    return attn.to(device=device, dtype=torch.float32).train()


def _dense_additive_mask(seq_len: int, sliding_window: int | None, device) -> torch.Tensor:
    """The causal (plus sliding) mask as the additive form ``eager_attention_forward`` wants."""
    positions = torch.arange(seq_len, device=device)
    distance = positions[:, None] - positions[None, :]
    allowed = distance >= 0
    if sliding_window is not None:
        allowed = allowed & (distance < sliding_window)
    return torch.where(allowed, 0.0, float("-inf")).view(1, 1, seq_len, seq_len)


# ---------------------------------------------------------------------------
# 3. FlexAttention translation of the relative-position bias
# ---------------------------------------------------------------------------


@pytest.mark.skipif(CUDA_DEVICES < 1, reason="FlexAttention needs CUDA")
@pytest.mark.parametrize("sliding_window", [None, SLIDING_WINDOW])
def test_flex_mods_reproduce_dense_relative_bias(sliding_window):
    """The score_mod/mask_mod pair must equal HF's dense bias + eager attention."""
    from types import SimpleNamespace

    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    from transformers.models.inkling.modeling_inkling import (
        InklingRelativeLogits,
        eager_attention_forward,
    )

    from nemo_automodel.components.models.inkling.cp_attention import (
        make_inkling_mask_mod,
        make_rel_bias_score_mod,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    heads, head_dim, d_rel = 2, 32, 8
    rel_extent = SLIDING_WINDOW if sliding_window is not None else 128
    scaling = 1.0 / head_dim

    query = torch.randn(BATCH, heads, SEQ_LEN, head_dim, device=device)
    key = torch.randn(BATCH, heads, SEQ_LEN, head_dim, device=device)
    value = torch.randn(BATCH, heads, SEQ_LEN, head_dim, device=device)
    relative = torch.randn(BATCH, SEQ_LEN, heads, d_rel, device=device)

    rel_module = InklingRelativeLogits(d_rel, rel_extent).to(device)
    torch.nn.init.normal_(rel_module.proj, std=0.05)
    positions = torch.arange(SEQ_LEN, device=device)

    reference, _ = eager_attention_forward(
        SimpleNamespace(num_key_value_groups=1, training=False),
        query,
        key,
        value,
        _dense_additive_mask(SEQ_LEN, sliding_window, device),
        scaling,
        position_bias=rel_module(relative, positions, positions),
    )
    reference = reference.transpose(1, 2)  # eager returns [B, S, H, D]

    rel_logits = (relative @ rel_module.proj).transpose(1, 2)
    block_mask = create_block_mask(
        make_inkling_mask_mod(sliding_window, None), B=None, H=None, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, device=device
    )
    actual = flex_attention(
        query,
        key,
        value,
        score_mod=make_rel_bias_score_mod(rel_logits, rel_extent),
        block_mask=block_mask,
        scale=scaling,
    )

    diff = (actual - reference).abs()
    assert torch.isfinite(actual).all()
    assert diff.mean().item() < 1e-5, f"mean diff {diff.mean().item():.3e}"
    assert diff.max().item() < 1e-3, f"max diff {diff.max().item():.3e}"


# ---------------------------------------------------------------------------
# 4. Module-level CP parity (the gate)
# ---------------------------------------------------------------------------


def _attention_parity_worker(rank: int, world_size: int, port: int, layer_idx: int) -> None:
    try:
        _init_pg(rank, world_size, port, "nccl")
        from torch.distributed.device_mesh import init_device_mesh

        from nemo_automodel.components.models.inkling.cp_attention import attach_inkling_upipe_attention

        device = torch.device(f"cuda:{rank}")
        config = _tiny_text_config()

        # Same seed on every rank, so both the CP module and the reference hold
        # bit-identical weights without any broadcast.
        reference = _build_attention(config, layer_idx, device)
        cp_module = copy.deepcopy(reference)
        cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))["cp"]
        attach_inkling_upipe_attention(cp_module)
        cp_module.setup_cp_attention(cp_mesh)

        torch.manual_seed(1234)
        hidden_full = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=device)
        grad_seed = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=device)

        seq_local = SEQ_LEN // world_size
        local_slice = slice(rank * seq_local, (rank + 1) * seq_local)

        # --- reference: full sequence, single device ---
        x_full = hidden_full.clone().requires_grad_(True)
        ref_out, _ = reference(
            x_full,
            attention_mask=_dense_additive_mask(SEQ_LEN, reference.sliding_window, device),
            conv_mask=None,
        )
        (ref_out * grad_seed).sum().backward()

        # --- CP: local shard, UPipe attention ---
        x_local = hidden_full[:, local_slice].clone().requires_grad_(True)
        cp_out, _ = cp_module(x_local, attention_mask=None, conv_mask=None)
        (cp_out * grad_seed[:, local_slice]).sum().backward()

        assert cp_out.shape == (BATCH, seq_local, HIDDEN)
        assert torch.isfinite(cp_out).all()

        expected_out = ref_out[:, local_slice]
        out_diff = (cp_out - expected_out).abs()
        assert out_diff.mean().item() < 1e-4, f"forward mean diff {out_diff.mean().item():.3e}"
        assert out_diff.max().item() < 5e-3, f"forward max diff {out_diff.max().item():.3e}"

        expected_grad = x_full.grad[:, local_slice]
        grad_diff = (x_local.grad - expected_grad).abs()
        assert grad_diff.mean().item() < 1e-4, f"input-grad mean diff {grad_diff.mean().item():.3e}"
        assert grad_diff.max().item() < 5e-3, f"input-grad max diff {grad_diff.max().item():.3e}"

        # Every rank holds a replica of the weights and covers a disjoint set of query
        # tokens, so the true parameter gradient is the sum of the per-rank ones.
        ref_grads = dict(reference.named_parameters())
        for name, param in cp_module.named_parameters():
            assert param.grad is not None, f"no gradient reached {name}"
            summed = param.grad.clone()
            dist.all_reduce(summed, op=dist.ReduceOp.SUM)
            expected = ref_grads[name].grad
            scale = max(expected.abs().max().item(), 1.0)
            diff = (summed - expected).abs().max().item() / scale
            assert diff < 5e-3, f"gradient mismatch for {name}: relative max diff {diff:.3e}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(CUDA_DEVICES < 2, reason="UPipe CP parity needs 2 GPUs")
@pytest.mark.parametrize(("layer_idx", "label"), [(0, "global"), (1, "sliding")])
def test_upipe_attention_matches_single_device(layer_idx, label):
    del label
    mp.spawn(_attention_parity_worker, args=(2, _free_port(), layer_idx), nprocs=2, join=True)
