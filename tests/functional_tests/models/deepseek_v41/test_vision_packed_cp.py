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

"""Real two-GPU CP image boundaries, frame sharding, and FSDP/EP gradients."""

import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed import FSDP2Config, MeshContext, MoEParallelizerConfig, ParallelismSizes
from nemo_automodel.components.distributed.cp_vision_frame_shard import (
    CpVisionFrameShardingConfig,
    reset_cp_vision_group,
    set_cp_vision_group,
)
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.models.deepseek_v41.cp import shard_cp_batch
from nemo_automodel.components.moe.parallelizer import apply_ac
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from tests.unit_tests.models.deepseek_v41.test_vision_packing import _batch, _model, _sample


def _worker(rank: int, port: int, fsdp: bool, ac: bool, ratio: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        cp_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
        torch.manual_seed(193)
        reference = _model(ratio=ratio).cuda(rank)
        model = copy.deepcopy(reference)
        if fsdp:
            mesh = MeshContext.build(
                FSDP2Config(
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.float32,
                        reduce_dtype=torch.float32,
                        output_dtype=None,
                        cast_forward_inputs=False,
                    )
                ),
                ParallelismSizes(dp_size=1, cp_size=2, ep_size=2),
                moe_parallel_config=MoEParallelizerConfig(lm_head_precision=torch.float32),
                activation_checkpointing=ac,
                world_size=2,
            )
            parallelize_model(model, mesh)
        elif ac:
            apply_ac(model)
        # Move the image through every kind of CP cut, then exercise uneven
        # image counts, multiple documents, no images, and an empty shifted doc.
        prefixes = (9,) if fsdp else (5, 6, 7, 8, 9, 10, 15, 16)
        torch.manual_seed(73)
        cases = [([_sample(((3, 5),), prefix=p, ratio=ratio)], 32, f"cut-{p}") for p in prefixes]
        cases += [
            ([_sample(((1, 1),), ratio=ratio)], 32, "one-image"),
            ([_sample(((1, 1), (5, 7), (2, 4)), ratio=ratio)], 64, "uneven-images"),
            (
                [_sample(((2, 3),), ratio=ratio), _sample(()), _sample(((1, 7), (3, 5)), ratio=ratio)],
                96,
                "mixed-documents",
            ),
            ([_sample(())], 32, "text-only"),
            ([_sample((), prefix=0, suffix=1)], 32, "empty-after-shift"),
        ]
        for enabled in (False, True):
            for samples, length, case in cases:
                raw = _batch(samples, length=length)
                raw = {k: v.cuda(rank) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
                _, global_batch, _ = shard_cp_batch(None, None, copy.deepcopy(raw), packed_alignment=2)
                labels = global_batch.pop("labels")
                global_batch.pop("n_images_per_sample", None)
                reference.zero_grad(set_to_none=True)
                expected_out = reference(**global_batch, output_residual_streams=False)
                expected = expected_out.logits
                expected_loss = F.cross_entropy(
                    expected.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
                ) / (labels != -100).sum().clamp_min(1)
                expected_loss.backward()
                expected_grads = {
                    n: p.grad.detach().clone() for n, p in reference.named_parameters() if p.grad is not None
                }

                model.zero_grad(set_to_none=True)
                _, local, _ = shard_cp_batch(cp_mesh, None, copy.deepcopy(raw), packed_alignment=2, pad_multiple=2)
                local_labels = local.pop("labels")
                local.pop("n_images_per_sample", None)
                token = set_cp_vision_group(
                    cp_mesh.get_group(),
                    config=CpVisionFrameShardingConfig(
                        enabled=enabled, min_tokens=0, min_frames=0, min_local_patch_rows=96
                    ),
                )
                try:
                    actual_out = model(**local, output_residual_streams=False)
                    actual = actual_out.logits
                    torch.testing.assert_close(
                        actual, expected[:, rank * (length // 2) : (rank + 1) * (length // 2)], atol=2e-5, rtol=2e-4
                    )
                    loss = F.cross_entropy(
                        actual.flatten(0, 1), local_labels.flatten(), ignore_index=-100, reduction="sum"
                    )
                    loss = loss / (labels != -100).sum().clamp_min(1)
                    (loss * (2 if fsdp else 1)).backward()
                finally:
                    reset_cp_vision_group(token)
                if fsdp:
                    # Match native VLM finalization: FSDP averages dense gradients,
                    # while EP-owned expert gradients need the shared EP scaling.
                    scale_grads_and_clip_grad_norm(
                        None,
                        [model],
                        device_mesh=mesh.device_mesh,
                        moe_mesh=mesh.moe_mesh,
                        ep_axis_name="ep",
                        dp_group_size=2,
                    )
                total_loss = loss.detach().clone()
                dist.all_reduce(total_loss)
                torch.testing.assert_close(total_loss, expected_loss, atol=2e-6, rtol=2e-5)
                for name, param in model.named_parameters():
                    canonical_name = name.replace("_checkpoint_wrapped_module.", "")
                    grad = param.grad
                    if fsdp:
                        if isinstance(grad, DTensor):
                            grad = grad.full_tensor()
                        if grad is None:
                            grad = torch.zeros_like(reference.get_parameter(canonical_name))
                    else:
                        grad = torch.zeros_like(param) if grad is None else grad.clone()
                        dist.all_reduce(grad)
                    expected_grad = expected_grads.get(canonical_name, torch.zeros_like(grad))
                    torch.testing.assert_close(
                        grad,
                        expected_grad,
                        atol=5e-5,
                        rtol=8e-4,
                        msg=f"{name}, frame={enabled}, case={case}, ratio={ratio}",
                    )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices for actual CP collectives")
@pytest.mark.timeout(360)
@pytest.mark.parametrize("fsdp,ac", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("ratio", [2, 3])
def test_cp_image_split_points_and_frame_sharding_gradients(fsdp: bool, ac: bool, ratio: int) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, fsdp, ac, ratio), nprocs=2, join=True)
