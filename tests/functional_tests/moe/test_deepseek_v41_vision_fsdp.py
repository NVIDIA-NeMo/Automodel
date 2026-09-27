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

"""Two-GPU DSV4.1 vision FSDP/EP forward and gradient regression."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.moe.parallelizer import parallelize_model


def _worker(rank: int, port: int, activation_checkpointing: bool) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        torch.manual_seed(39)
        config = DeepseekV41Config(
            text_config=DeepseekV41TextConfig(
                vocab_size=32,
                hidden_size=16,
                moe_intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                head_dim=8,
                qk_rope_head_dim=4,
                q_lora_rank=8,
                o_lora_rank=8,
                o_groups=1,
                hc_mult=2,
                n_routed_experts=4,
                num_experts_per_tok=2,
                compress_ratios=[0],
                kv_source_layer_ids=[],
                index_source_layer_ids=[],
                candidate_source_layer_id=-1,
                engram_layer_ids=[],
                num_nextn_predict_layers=0,
                dspark_block_size=0,
                dspark_noise_token_id=0,
                dtype="bfloat16",
            ),
            vision_config=DeepseekV41VisionConfig(
                num_hidden_layers=2,
                hidden_size=16,
                num_attention_heads=2,
                intermediate_size=32,
                patch_size=2,
                downsample_ratio=2,
            ),
            image_token_id=0,
            dtype="bfloat16",
        )
        model = DeepseekV41ForCausalLM(
            config,
            backend=BackendConfig(
                attn="eager",
                linear="torch",
                rms_norm="torch_fp32",
                experts="torch_mm",
                dispatcher="torch",
                enable_hf_state_dict_adapter=False,
            ),
        ).cuda(rank)
        model.initialize_weights(torch.device("cuda", rank), dtype=torch.bfloat16)
        # Odd 3x5 patch grid -> two rows, three merged tokens and a newline per row.
        types = torch.tensor([[-1, 0, 1, 1, 1, 2, 1, 1, 1, 2, 3, -1, -1]], device=rank)
        ids = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3]], device=rank)
        pixels = torch.randn(15, 3, 2, 2, device=rank, dtype=torch.bfloat16, requires_grad=True)
        batch = dict(
            input_ids=ids,
            pixel_values=pixels,
            image_grid_hws=torch.tensor([[3, 5]], device=rank),
            vision_token_types=types,
        )
        expected = model(**batch).logits
        upstream = torch.randn_like(expected)
        (expected * upstream).sum().backward()
        expected = expected.detach()
        expected_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if name.startswith(("model.vision.", "model.aligner.", "model.image_"))
        }
        expected_pixel_grad = pixels.grad.detach().clone()
        assert expected_grads and all(torch.isfinite(g).all() for g in expected_grads.values())
        for name in ("image_start", "image_end", "image_newline"):
            assert expected_grads[f"model.{name}"].abs().sum() > 0
        model.zero_grad(set_to_none=True)
        pixels.grad = None
        mesh = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(dp_size=2, ep_size=2),
            world_size=2,
        )
        parallelize_model(
            model,
            mesh.device_mesh,
            mesh.moe_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
            ),
            lm_head_precision=torch.float32,
            activation_checkpointing=activation_checkpointing,
            **mesh.parallelize_axis_kwargs(),
        )
        for name, p in model.named_parameters():
            if name.startswith("model.vision.") and any(n in name for n in (".norm.", ".norm1.", ".norm2.")):
                assert p.dtype == torch.float32, name
        for _ in range(2):
            actual = model(**batch).logits
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)
            (actual * upstream).sum().backward()
            for name, expected_grad in expected_grads.items():
                grad = model.get_parameter(name).grad
                assert grad is not None, name
                if isinstance(grad, DTensor):
                    grad = grad.full_tensor()
                torch.testing.assert_close(grad, expected_grad, rtol=0.03, atol=0.003, msg=name)
            torch.testing.assert_close(pixels.grad, expected_pixel_grad, rtol=0.03, atol=0.003)
            model.zero_grad(set_to_none=True)
            pixels.grad = None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.timeout(180)
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_vision_fsdp_ep_matches_unsharded_forward_and_gradients(activation_checkpointing: bool) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, activation_checkpointing), nprocs=2, join=True)
