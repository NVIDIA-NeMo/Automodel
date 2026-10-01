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

"""Real CP collectives: CSA2 sharing, padding, gradients and optimizer updates."""

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionState,
)
from nemo_automodel.components.models.deepseek_v41.cp import (
    gather_sequence,
    shard_cp_batch,
)
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41NgramHash
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from tests.unit_tests.models.deepseek_v41.test_attention import _backend, _config
from tests.unit_tests.models.deepseek_v41.test_engram import (
    _tiny_config as _hash_config,
)
from tests.unit_tests.models.deepseek_v41.test_engram import _tokenizer


def _run_stack(layers, hidden, positions, mask, group, ac):
    """Run CSA2 source/reindex/reuse layers with local or global token axes.

    Args:
        layers: Six attention modules with matching parameters.
        hidden: Tensor of shape [batch, sequence, hidden].
        positions: Global token positions [1, sequence].
        mask: Boolean validity mask [batch, sequence].
        group: CP group, or None for unsharded reference.
        ac: Whether to recompute each layer during backward.

    Returns:
        Final hidden [batch, sequence, hidden] and six immutable state snapshots.
    """
    state = DeepseekV41AttentionState()
    states = []
    for layer in layers:
        args = dict(position_ids=positions, state=state, attention_mask=mask, cp_group=group)
        out = checkpoint(layer, hidden, use_reentrant=False, **args) if ac else layer(hidden, **args)
        hidden = hidden + out.hidden_states
        state = out.state
        states.append(state)
    return hidden, states


def _attention_parity(group, device, backend="eager", ac=False):
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    torch.manual_seed(130)
    layers = torch.nn.ModuleList([DeepseekV41Attention(_config(), i, _backend(backend)) for i in range(6)]).to(device)
    for layer in layers:
        layer.reset_parameters()
    reference = copy.deepcopy(layers)
    sequence = 16
    torch.manual_seed(140)
    full = torch.randn(2, sequence, 16, device=device)
    upstream = torch.randn_like(full)
    mask = torch.arange(sequence, device=device)[None] < torch.tensor([13, 5], device=device)[:, None]
    positions = torch.arange(sequence, device=device)[None]
    shard = slice(rank * sequence // size, (rank + 1) * sequence // size)
    local = full[:, shard].detach().clone().requires_grad_()
    full = full.detach().clone().requires_grad_()
    expected, expected_states = _run_stack(reference, full, positions, mask, None, ac)
    actual, states = _run_stack(layers, local, positions[:, shard], mask[:, shard], group, ac)
    torch.testing.assert_close(actual, expected[:, shard], atol=2e-6, rtol=2e-5)
    for state, ref in zip(states, expected_states):
        if state.compressed_kv is not None:
            torch.testing.assert_close(state.compressed_kv, ref.compressed_kv, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(state.compressed_valid, ref.compressed_valid, atol=0, rtol=0)
            torch.testing.assert_close(state.index_keys, ref.index_keys, atol=0, rtol=0)
            torch.testing.assert_close(state.topk_indices, ref.topk_indices[:, shard], atol=0, rtol=0)
        if state.candidates is not None:
            torch.testing.assert_close(state.candidates, ref.candidates[:, shard], atol=0, rtol=0)
    expected.backward(upstream)
    actual.backward(upstream[:, shard])
    torch.testing.assert_close(local.grad, full.grad[:, shard], atol=3e-6, rtol=3e-5)
    for (name, param), (_, ref) in zip(layers.named_parameters(), reference.named_parameters()):
        if ref.grad is None:
            assert param.grad is None, name
            continue
        assert param.grad is not None, name
        dist.all_reduce(param.grad, group=group)
        torch.testing.assert_close(param.grad, ref.grad, atol=2e-5, rtol=2e-4, msg=lambda detail: f"{name}: {detail}")
    norm = torch.nn.utils.clip_grad_norm_(layers.parameters(), 0.5)
    ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
    torch.testing.assert_close(norm, ref_norm, atol=2e-5, rtol=2e-5)
    torch.optim.SGD(layers.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    for (name, param), (_, ref) in zip(layers.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(param, ref, atol=2e-6, rtol=2e-5, msg=lambda detail: f"{name}: {detail}")


def _cpu_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=120)
    )
    try:
        for backend in ("eager", "sdpa"):
            for ac in (False, True):
                _attention_parity(dist.group.WORLD, torch.device("cpu"), backend, ac)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("cp",))
        ids = torch.tensor([[7, 1, 4, 3, 8, 7, 2, 5, 1, 4, 7, 3, 2]])
        labels = ids.clone()
        ctx, local_batch, layout = shard_cp_batch(mesh, None, {"input_ids": ids, "labels": labels}, pad_multiple=2)
        assert layout.original_seq_len == 13 and layout.padded_seq_len == 16
        assert local_batch["input_ids"].shape == (1, 8)
        assert local_batch["position_ids"][0, 0] == rank * 8
        assert local_batch["attention_mask"].sum() == (8 if rank == 0 else 5)
        hasher = DeepseekV41NgramHash(_hash_config(), _tokenizer())
        restored = gather_sequence(local_batch["input_ids"], dist.group.WORLD)
        restored_mask = gather_sequence(local_batch["attention_mask"], dist.group.WORLD)
        expected_hashes = hasher(ids)
        hashes = hasher(restored, token_mask=restored_mask)
        torch.testing.assert_close(hashes[:, :13], expected_hashes, atol=0, rtol=0)
        with pytest.raises(ValueError, match="complete compression groups"):
            layer = DeepseekV41Attention(_config(), 1, _backend())
            layer(
                torch.randn(1, 3, 16),
                position_ids=torch.arange(3)[None],
                state=DeepseekV41AttentionState(),
                cp_group=dist.group.WORLD,
            )
        with pytest.raises(ValueError, match="physical spans"):
            shard_cp_batch(mesh, None, {"input_ids": ids, "labels": labels, "seq_lens": torch.tensor([14])})
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="two spawned workers import the model stack and check real CP gradients and optimizer updates",
)
def test_cp2_attention_gradients_clipping_update_and_padding(tmp_path: Path):
    mp.spawn(_cpu_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_cp_capability_is_declared():
    from nemo_automodel._transformers.capabilities import ModelSupports
    from tests.unit_tests.models.deepseek_v41.test_model import _tiny_config

    assert DeepseekV41ForCausalLM.ModelCapabilities().supports_cp
    model = DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    assert ModelSupports(model).supports_cp
