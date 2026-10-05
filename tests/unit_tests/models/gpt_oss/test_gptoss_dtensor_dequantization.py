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

"""Exercise MXFP4 chunk writes through real distributed tensor storage."""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from transformers import GptOssConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss.state_dict_adapter import GPTOSSStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


def _dequantize_worker(rank: int, rendezvous: str) -> None:
    """Check expert and feature sharding with writes that cross chunk boundaries."""
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=8,
            moe_inter_dim=8,
            norm_topk_prob=False,
        )
        adapter = GPTOSSStateDictAdapter(GptOssConfig(), config, BackendConfig(attn="flex"))
        blocks = torch.arange(4 * 8 * 2 * 16).reshape(4, 8, 2, 16).to(torch.uint8)
        scales = (torch.arange(4 * 8 * 2).reshape(4, 8, 2) % 5 + 125).to(torch.uint8)
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        unpacked = torch.stack((lut[(blocks & 15).long()], lut[(blocks >> 4).long()]), dim=-1)
        expected = torch.ldexp(unpacked, (scales.int() - 127)[..., None, None])
        expected = expected.reshape(4, 8, 64).transpose(1, 2).contiguous()
        original_empty = torch.distributed.tensor.empty

        def poisoned_empty(*args, **kwargs):
            """Fill the real DTensor output allocation so unwritten rows fail deterministically."""
            output = original_empty(*args, **kwargs)
            output.to_local().fill_(float("nan"))
            return output

        for mesh_shape in ((2, 1), (1, 2)):
            mesh = DeviceMesh("cpu", torch.arange(2).reshape(mesh_shape), mesh_dim_names=("ep", "ep_shard"))
            # HF load metadata replicates feature shards before MXFP4 decoding.
            distributed_blocks = distribute_tensor(blocks, mesh, [Shard(0), Replicate()])
            distributed_scales = distribute_tensor(scales, mesh, [Shard(0), Replicate()])
            for dtype in (torch.float32, torch.bfloat16):
                for chunk_rows in (7, 32, 128):
                    # Keep this real Gloo/DTensor test on CPU, including on GPU CI runners.
                    with (
                        patch("torch.cuda.is_available", return_value=False),
                        patch("torch.distributed.tensor.empty", side_effect=poisoned_empty),
                    ):
                        actual = adapter._convert_moe_packed_tensors(
                            distributed_blocks, distributed_scales, dtype=dtype, rows_per_chunk=chunk_rows
                        )
                    assert actual.placements == (Shard(0), Shard(2))
                    torch.testing.assert_close(actual.full_tensor(), expected.to(dtype), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    20, hard_timeout=70, reason="Starts two fresh PyTorch workers and checks real Gloo/DTensor collectives."
)
def test_chunked_mxfp4_dequantization_writes_local_dtensor_storage(tmp_path) -> None:
    """A partial DTensor slice must not redirect MXFP4 output writes into a temporary."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with Gloo is unavailable")
    mp.spawn(_dequantize_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
