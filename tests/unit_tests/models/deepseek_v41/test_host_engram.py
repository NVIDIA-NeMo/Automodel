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

"""Native frozen lookup parity, trainable gradients and external checkpoint contract."""

import json

import pytest
import torch
from safetensors.torch import save_file

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41Engram
from nemo_automodel.components.models.deepseek_v41.host_engram import HostEngramTableConfig
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import dequantize_checkpoint_weight


def _checkpoint(root):
    torch.manual_seed(17)
    values = torch.randn(17, 64).to(torch.float8_e4m3fn)
    scales = torch.randint(123, 130, (17, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    tensors = {"layers.1.engram.embed.weight": values, "layers.1.engram.embed.scale": scales}
    save_file(tensors, root / "table.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "table.safetensors" for name in tensors}})
    )
    return values, scales


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lookup_matches_existing_dequantizer_and_owns_no_device_tensors(tmp_path, dtype):
    weight, scale = _checkpoint(tmp_path)
    with torch.device("meta"):
        host = HostEngramTableConfig(str(tmp_path), 1, 17, 64).build(dtype=dtype)
    host.to_empty(device="cpu").to(dtype=torch.float64)
    ids = torch.tensor([[16, 0, 16], [2, 2, 0]], dtype=torch.int32)
    dense = dequantize_checkpoint_weight(weight, scale, dtype=dtype, rowwise=True)
    torch.testing.assert_close(host(ids), dense[ids.long()], rtol=0, atol=0)
    assert not list(host.parameters()) and not list(host.buffers()) and not host.state_dict()
    assert host(torch.empty(2, 0, dtype=torch.int64)).shape == (2, 0, 64)
    with pytest.raises(ValueError, match="out of range"):
        host(torch.tensor([17]))
    with pytest.raises(ValueError, match="int32 or int64"):
        host(torch.tensor([1.0]))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_engram_forward_gradients_and_resume(tmp_path, dtype):
    weight, scale = _checkpoint(tmp_path)
    config = DeepseekV41TextConfig(
        hidden_size=8,
        hc_mult=2,
        engram_layer_ids=[1],
        engram_num_embeddings=[17],
        engram_head_dim=64,
        engram_n_heads=2,
        engram_max_ngram_size=2,
        dtype=dtype,
        initializer_range=0.1,
    )
    dense = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
    with torch.no_grad():
        dense.embed.weight.copy_(dequantize_checkpoint_weight(weight, scale, dtype=getattr(torch, dtype), rowwise=True))
    dense.embed.weight.requires_grad_(False)
    config.engram_host_checkpoint = str(tmp_path)
    host = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
    trainable = {name: tensor for name, tensor in dense.state_dict().items() if name != "embed.weight"}
    host.load_state_dict(trainable, strict=True)
    ids = torch.tensor([[[0, 16], [2, 2], [0, 2]], [[16, 0], [0, 0], [16, 16]]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    x = torch.randn(2, 3, 2, 8, dtype=getattr(torch, dtype), requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = dense(x, ids, token_mask=mask)
    actual = host(y, ids, token_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for name, param in host.named_parameters():
        assert param.requires_grad and param.grad is not None
        torch.testing.assert_close(param.grad, dict(dense.named_parameters())[name].grad, rtol=0, atol=0)
    before = host.embed(ids).clone()
    optimizer = torch.optim.AdamW(host.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(host.wkv.weight, dense.wkv.weight)
    torch.testing.assert_close(host.embed(ids), before, rtol=0, atol=0)
    saved = tmp_path / "train.pt"
    torch.save(host.state_dict(), saved)
    restored = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    torch.testing.assert_close(restored(y, ids), host(y, ids), rtol=0, atol=0)


def test_wrong_shape_and_truncation_fail_at_construction(tmp_path):
    _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Invalid native"):
        HostEngramTableConfig(str(tmp_path), 1, 18, 64).build(dtype=torch.float32)
    shard = tmp_path / "table.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Invalid native"):
        HostEngramTableConfig(str(tmp_path), 1, 17, 64).build(dtype=torch.float32)


def _distributed_host_worker(rank, checkpoint, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        torch.manual_seed(17)
        config = DeepseekV41TextConfig(
            hidden_size=8,
            hc_mult=2,
            engram_layer_ids=[1],
            engram_num_embeddings=[17],
            engram_head_dim=64,
            engram_n_heads=2,
            engram_max_ngram_size=2,
            dtype="float32",
        )
        dense = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
        config.engram_host_checkpoint = checkpoint
        host = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"), process_group=dist.group.WORLD)
        host.load_state_dict({k: v for k, v in dense.state_dict().items() if k != "embed.weight"})
        with torch.no_grad():
            dense.embed.weight.copy_(host.embed(torch.arange(17)))
        dense.embed.weight.requires_grad_(False)
        ddp = DistributedDataParallel(host)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD([p for p in dense.parameters() if p.requires_grad], lr=0.01)
        for step in range(2):
            # A rank with no requests must not enter owner collectives.
            length = 3 if rank == 0 else step
            ids = torch.zeros((1, length, 2), dtype=torch.int64)
            hidden = torch.randn(1, length, 2, 8)
            actual, expected = ddp(hidden, ids), dense(hidden, ids)
            upstream = torch.randn_like(actual)
            actual.backward(upstream)
            expected.backward(upstream)
            for name, param in host.named_parameters():
                reference = dict(dense.named_parameters())[name]
                dist.all_reduce(reference.grad)
                reference.grad.div_(2)
                torch.testing.assert_close(param.grad, reference.grad, rtol=1e-6, atol=1e-7)
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            for name, param in host.named_parameters():
                torch.testing.assert_close(param, dict(dense.named_parameters())[name], rtol=1e-6, atol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(60)
def test_two_rank_frozen_lookup_and_projection_updates(tmp_path):
    import torch.multiprocessing as mp

    _checkpoint(tmp_path)
    mp.spawn(_distributed_host_worker, args=(str(tmp_path), str(tmp_path / "gloo")), nprocs=2, join=True)
