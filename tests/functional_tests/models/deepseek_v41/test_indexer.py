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

"""GPU scoring and packed CP2 replacement parity; requires at most two GPUs."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention, DeepseekV41AttentionState
from nemo_automodel.components.models.deepseek_v41.indexer import indexer_scores
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache
from tests.unit_tests.models.deepseek_v41.test_attention import _backend, _config


@pytest.mark.parametrize("batch,sequence,width,heads,dim", [(1, 17, 131, 2, 32), (2, 33, 257, 32, 128)])
def test_fused_scores_match_reference_with_masks_and_tile_tails(
    batch: int, sequence: int, width: int, heads: int, dim: int
) -> None:
    torch.manual_seed(412)
    q = quantize_cache(
        torch.randn(batch, sequence, heads, dim, device="cuda", dtype=torch.bfloat16), format="mxfp4", block_size=32
    )
    k = quantize_cache(
        torch.randn(batch, width, dim, device="cuda", dtype=torch.bfloat16), format="mxfp4", block_size=32
    )
    weights = torch.randn(batch, sequence, heads, device="cuda", dtype=torch.bfloat16) / (dim * heads) ** 0.5
    allowed = torch.rand(batch, sequence, width, device="cuda") > 0.2
    allowed[:, 0] = False
    # Literal released arithmetic: GEMM and weighted products round to BF16.
    products = torch.einsum("bshd,btd->bsht", q, k)
    expected = (products.relu() * weights.unsqueeze(-1)).sum(2).masked_fill(~allowed, -torch.inf)
    actual = indexer_scores(q, k, weights, allowed)
    assert torch.equal(torch.isneginf(actual), ~allowed)
    # FP32 head-reduction order can change the final result by one BF16 ULP.
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1 / 128)
    count = min(16, width)
    torch.testing.assert_close(
        actual.topk(count, dim=-1, sorted=False).indices.sort(-1).values,
        expected.topk(count, dim=-1, sorted=False).indices.sort(-1).values,
        atol=0,
        rtol=0,
    )


def test_fused_scores_validate_dtype_and_handle_empty_axes() -> None:
    q = torch.empty(1, 0, 2, 32, device="cuda", dtype=torch.bfloat16)
    keys = torch.empty(1, 3, 32, device="cuda", dtype=torch.bfloat16)
    weights = torch.empty(1, 0, 2, device="cuda", dtype=torch.bfloat16)
    allowed = torch.empty(1, 0, 3, device="cuda", dtype=torch.bool)
    assert indexer_scores(q, keys, weights, allowed).shape == (1, 0, 3)
    with pytest.raises(ValueError, match="requires BF16"):
        indexer_scores(q.float(), keys, weights, allowed)


def _cp_worker(rank: int, rendezvous: str, ac: bool) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(minutes=10),
        device_id=device,
    )
    try:
        torch.manual_seed(719)
        config = _config("bfloat16")
        config.hidden_size = 128
        config.head_dim = 64
        config.qk_rope_head_dim = 32
        config.q_lora_rank = 32
        config.o_lora_rank = 32
        config.index_n_heads = 32
        config.index_head_dim = 128
        reference = torch.nn.ModuleList([DeepseekV41Attention(config, i, _backend("tilelang")) for i in range(6)]).to(
            device
        )
        fused = torch.nn.ModuleList([DeepseekV41Attention(config, i, _backend("tilelang")) for i in range(6)]).to(
            device
        )
        fused.load_state_dict(reference.state_dict())
        for layer in reference:
            if layer.indexer is not None:
                layer.indexer.attn_backend = "eager"
        source = torch.randn(1, 256, 128, device=device, dtype=torch.bfloat16)
        ids = torch.zeros(1, 256, device=device, dtype=torch.long)
        positions = torch.zeros_like(ids)
        offset = 0
        for doc, length, span in [(1, 1, 2), (2, 63, 64), (3, 189, 190)]:
            ids[:, offset : offset + length] = doc
            positions[:, offset : offset + length] = torch.arange(length, device=device)
            offset += span
        shard = slice(rank * 128, (rank + 1) * 128)
        ref_input = source[:, shard].clone().requires_grad_()
        new_input = source[:, shard].clone().requires_grad_()
        ref_hidden, new_hidden = ref_input, new_input
        ref_state, new_state = DeepseekV41AttentionState(), DeepseekV41AttentionState()
        for left, right in zip(reference, fused):
            kwargs = dict(
                position_ids=positions[:, shard],
                packed_seq_ids=ids[:, shard],
                attention_mask=ids[:, shard] != 0,
                cp_group=dist.group.WORLD,
            )
            if ac:
                before = checkpoint(left, ref_hidden, state=ref_state, **kwargs, use_reentrant=False)
                after = checkpoint(right, new_hidden, state=new_state, **kwargs, use_reentrant=False)
            else:
                before = left(ref_hidden, state=ref_state, **kwargs)
                after = right(new_hidden, state=new_state, **kwargs)
            torch.testing.assert_close(after.hidden_states, before.hidden_states, atol=0, rtol=0)
            for name in ("index_keys", "compressed_kv", "topk_indices", "candidates"):
                a, b = getattr(after.state, name), getattr(before.state, name)
                if a is None:
                    assert b is None
                else:
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            selected = after.state.topk_indices
            if selected is not None:
                valid = selected >= 0
                key_docs = after.state.compressed_seq_ids[:, None].expand(-1, 128, -1).gather(-1, selected.clamp_min(0))
                assert torch.all(~valid | (key_docs == ids[:, shard, None]))
                physical = torch.arange(rank * 128, (rank + 1) * 128, device=device)[None, :, None]
                assert torch.all(~valid | ((selected + 1) * after.state.compression_ratio <= physical + 1))
                assert torch.all(~valid | (ids[:, shard, None] > 0))
            ref_hidden, new_hidden = before.hidden_states, after.hidden_states
            ref_state, new_state = before.state, after.state
        upstream = torch.randn_like(new_hidden)
        ref_hidden.backward(upstream)
        new_hidden.backward(upstream)
        torch.testing.assert_close(new_input.grad, ref_input.grad, atol=0, rtol=0)
        for (name, a), (other, b) in zip(reference.named_parameters(), fused.named_parameters()):
            assert name == other
            if a.grad is None:
                assert b.grad is None
            elif name.endswith("sinks_param.weight"):
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5, msg=name)
            else:
                torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ac", [False, True])
def test_packed_cp2_replacement_preserves_selection_outputs_and_gradients(tmp_path: Path, ac: bool) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Real CP2 collective and checkpoint ordering require two CUDA devices.")
    mp.spawn(_cp_worker, args=(str(tmp_path / "rendezvous"), ac), nprocs=2, join=True)
