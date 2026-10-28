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

"""Two-rank numerical regression for GLM-5.3's FP32-holder FSDP policy under EP2 x DP2."""

from __future__ import annotations

import copy
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.models.glm5_next.layers import (
    Glm5NextHyperConnectionFp32Params,
    Glm5NextKDAFp32Params,
)
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
from nemo_automodel.components.models.glm5_next.parallelization import PARALLELIZER
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config

_WORLD_SIZE = 2
_ROWS = 4
_SEQ_LEN = 12
_MAX_NORM = 0.5
_HOLDERS = (Glm5NextHyperConnectionFp32Params, Glm5NextKDAFp32Params)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _strip(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "")


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _worker(rank: int, port: int, activation_checkpointing: bool) -> None:
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(_WORLD_SIZE))
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=_WORLD_SIZE)
        device = torch.device("cuda", rank)

        torch.manual_seed(1234)
        config = tiny_glm5_next_config()
        # FLA's CUDA KDA kernels need a head width of at least 8.
        config.text_config.linear_head_dim = 32
        config.text_config.hc_mult = 4
        model = Glm5NextForConditionalGeneration(config, backend=tiny_backend(adapter=False))
        model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
        model.to(device)
        reference = copy.deepcopy(model)
        mesh_context = MeshContext.build(
            FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)),
            ParallelismSizes(dp_size=_WORLD_SIZE, ep_size=_WORLD_SIZE),
            moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=True),
            activation_checkpointing=activation_checkpointing,
            world_size=_WORLD_SIZE,
        )
        model = PARALLELIZER.parallelize(model, mesh_context)
        holder_names = {_strip(name) for name, module in model.named_modules() if isinstance(module, _HOLDERS)}
        assert holder_names, "the tiny model must contain mHC and KDA FP32 holders"

        # Every rank draws the same global batch [rows, sequence]; each owns two rows, one per microbatch.
        torch.manual_seed(307)
        ids = torch.randint(1, 60, (_ROWS, _SEQ_LEN), device=device)
        docs = torch.stack([torch.cat((torch.ones(cut), torch.full((_SEQ_LEN - cut,), 2))) for cut in (6, 4, 5, 7)]).to(
            device=device, dtype=torch.int32
        )
        labels = ids.roll(-1, dims=1)
        labels[:, -1] = -100
        labels[1, 3:6] = -100
        count = int((labels != -100).sum())

        reference_loss = torch.zeros((), device=device)
        for row in range(_ROWS):
            logits = reference(ids[row : row + 1], attention_mask=docs[row : row + 1]).logits
            loss = F.cross_entropy(logits.float().flatten(0, 1), labels[row].flatten(), reduction="sum") / count
            reference_loss += loss.detach()
            loss.backward()
        expected_norm = (
            torch.stack([p.grad.double().square().sum() for p in reference.parameters() if p.grad is not None])
            .sum()
            .sqrt()
            .float()
        )
        coefficient = (_MAX_NORM / (expected_norm + 1e-6)).clamp(max=1.0)
        for parameter in reference.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)

        local_loss = torch.zeros((), device=device)
        for microbatch, row in enumerate((2 * rank, 2 * rank + 1)):
            logits = model(ids[row : row + 1], attention_mask=docs[row : row + 1]).logits
            if microbatch == 0:
                # Holders stay gathered between forward and backward; ordinary modules follow the caller's
                # reshard_after_forward=True and return to their sharded DTensor parameters.
                for name, module in model.named_modules():
                    if isinstance(module, _HOLDERS):
                        assert all(not isinstance(p, DTensor) for p in module.parameters(recurse=False)), name
                mlp = next(m for n, m in model.named_modules() if _strip(n).endswith("layers.0.mlp"))
                assert all(isinstance(p, DTensor) for p in mlp.parameters()), "decoder block was not resharded"
            loss = F.cross_entropy(logits.float().flatten(0, 1), labels[row].flatten(), reduction="sum") / count
            local_loss += loss.detach()
            (loss * _WORLD_SIZE).backward()
        dist.all_reduce(local_loss)
        torch.testing.assert_close(local_loss, reference_loss, rtol=1e-5, atol=1e-6)

        actual_norm = scale_grads_and_clip_grad_norm(
            _MAX_NORM,
            [model],
            device_mesh=mesh_context.device_mesh,
            moe_mesh=mesh_context.moe_mesh,
            ep_axis_name="ep",
            num_label_tokens=count,
            dp_group_size=_WORLD_SIZE,
        )
        torch.testing.assert_close(actual_norm.float(), expected_norm, rtol=1e-5, atol=1e-6)

        expected = dict(reference.named_parameters())
        compared_holders = set()
        for name, parameter in model.named_parameters():
            reference_parameter = expected[_strip(name)]
            assert (parameter.grad is None) == (reference_parameter.grad is None), name
            if parameter.grad is None:
                continue
            assert parameter.grad.dtype == reference_parameter.grad.dtype, name
            torch.testing.assert_close(
                _full(parameter.grad), reference_parameter.grad, rtol=1e-4, atol=1e-6, msg=lambda m: f"{name}: {m}"
            )
            compared_holders.update(holder for holder in holder_names if _strip(name).startswith(holder + "."))
        assert compared_holders == holder_names, holder_names - compared_holders

        optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.125)
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                _full(parameter.detach()), expected[_strip(name)].detach(), rtol=1e-5, atol=1e-6, msg=name
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason="requires two CUDA devices",
)
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_glm5_next_fp32_holders_stay_gathered_and_reduce_once(activation_checkpointing: bool) -> None:
    """EP2 x DP2 gradients, global norm and SGD update match a single-process FP32 reference."""
    mp.spawn(_worker, args=(_free_port(), activation_checkpointing), nprocs=_WORLD_SIZE, join=True)
