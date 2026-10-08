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

"""Real CUDA FP8 loading with EP2, including already-sharded input tensors."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.model import Step5ForCausalLM


def main() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
    config = Step5Config(
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=192,
        vocab_size=32,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=128,
        share_expert_dim=128,
        moe_layer_list=[0],
        dtype="bfloat16",
        num_nextn_predict_layers=0,
        sparse_config={"enabled": False},
    )
    model = Step5ForCausalLM(
        config, backend=BackendConfig(attn="sdpa", linear="torch", experts="torch_mm", dispatcher="torch")
    )
    adapter = model.state_dict_adapter
    values = torch.arange(4 * 128 * 128, device="cuda", dtype=torch.float32).sin().reshape(4, 128, 128)
    source = values.to(torch.float8_e4m3fn)
    scales = torch.tensor([0.125, 0.25, 0.5, 1.0], device="cuda").reshape(4, 1, 1)
    bias = torch.tensor([34.03125, 34.09375, 34.15625, 34.21875], device="cuda")
    start = dist.get_rank() * 2
    expected = (source.float() * scales).to(torch.bfloat16)[start : start + 2].transpose(1, 2)
    for already_sharded in (False, True):
        weight, scale = source, scales
        if already_sharded:
            weight = DTensor.from_local(source[start : start + 2], mesh, [Shard(0)])
            scale = DTensor.from_local(scales[start : start + 2], mesh, [Shard(0)])
        checkpoint = {"model.layers.0.moe.router_bias": bias}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.0.moe.{projection}.weight"
            checkpoint[name] = weight
            checkpoint[name + "_scale_inv"] = scale
        loaded = adapter.from_hf(checkpoint, device_mesh=mesh)
        merged = loaded["model.layers.0.moe.experts.gate_and_up_projs"]
        down = loaded["model.layers.0.moe.experts.down_projs"]
        assert isinstance(merged, DTensor) and isinstance(down, DTensor)
        assert merged.shape == (4, 128, 256)
        assert merged.to_local().shape == (2, 128, 256)
        torch.testing.assert_close(merged.to_local(), torch.cat((expected, expected), -1), atol=0, rtol=0)
        torch.testing.assert_close(down.to_local(), expected, atol=0, rtol=0)
        loaded_bias = loaded["model.layers.0.moe.gate.e_score_correction_bias"]
        assert loaded_bias.dtype == torch.float32
        torch.testing.assert_close(loaded_bias, bias, atol=0, rtol=0)
    if dist.get_rank() == 0:
        print("FP8 EP2 regular and DTensor loading: passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
