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

"""Two-GPU FSDP2, compiled attention, and optimizer-resume regression tests."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")


def _worker(rank: int, work_dir: str, lora: bool) -> None:
    """Compare sharded and unsharded updates from [1, 16, 2, 4, 4] latents."""
    from diffusers import WanAnimate2Transformer3DModel

    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.distributed.config import FSDP2Config
    from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
    from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
    from nemo_automodel.components.models.wan_animate2.adapter import WanAnimate2Adapter
    from nemo_automodel.components.training.utils import clip_grad_norm

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"file://{work_dir}/init", rank=rank, world_size=2)
    try:
        torch.manual_seed(43)
        model = WanAnimate2Transformer3DModel(
            dim=64,
            ffn_dim=128,
            freq_dim=16,
            text_dim=16,
            text_len=8,
            num_heads=2,
            num_layers=2,
            use_img_emb=True,
        ).to(device=device, dtype=torch.bfloat16 if lora else torch.float32)
        if lora:
            apply_lora_to_linear_modules(
                model,
                PeftConfig(
                    dim=4,
                    alpha=4,
                    dropout=0.0,
                    lora_dtype=torch.bfloat16,
                    use_memory_efficient_lora=False,
                    target_modules=["*.self_attn.to_q", "*.self_attn.to_k", "*.self_attn.to_v"],
                ),
                skip_freeze=True,
            )
        reference = deepcopy(model)
        mesh = init_device_mesh("cuda", (1, 2, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"))
        manager = FSDP2Manager(
            FSDP2Config(
                activation_checkpointing=True,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            ),
            device_mesh=mesh,
        )
        model = manager.parallelize(model)
        if lora:
            # Match from_pretrained: FSDP records all parameters as trainable,
            # then the base is frozen before the first forward and optimizer.
            for module in (model, reference):
                for name, parameter in module.named_parameters():
                    parameter.requires_grad_("lora_" in name)
        assert any(isinstance(parameter, DTensor) for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, foreach=False)
        torch.manual_seed(101 + rank)
        latents = torch.randn(1, 16, 2, 4, 4, device=device)
        batch = {
            "video_latents": latents,
            "reference_latents": torch.randn(1, 16, 1, 4, 4, device=device),
            "driving_latents": torch.randn_like(latents),
            "cond_zero_latents": torch.randn_like(latents),
            "clip_fea": torch.randn(1, 3, 1280, device=device),
            "clip_fea_ref": torch.randn(1, 3, 1280, device=device),
            "text_embeddings": torch.randn(1, 5, 16, device=device),
            "prompt_ref_embeddings": torch.randn(1, 4, 16, device=device),
        }
        latents = WanAnimate2Adapter().prepare_latents(latents, batch)
        context = FlowMatchingContext(
            batch=batch,
            latents=latents,
            noisy_latents=torch.randn_like(latents),
            task_type="i2v",
            data_type="video",
            timesteps=torch.tensor([500.0], device=device),
            sigma=torch.tensor([0.5], device=device),
            device=device,
            dtype=torch.bfloat16,
        )
        adapter = WanAnimate2Adapter()
        inputs = adapter.prepare_inputs(context)
        target = torch.randn_like(latents)

        def backward(
            module: torch.nn.Module, opt: torch.optim.Optimizer, *, average_gradients: bool = False
        ) -> torch.Tensor:
            """Backpropagate the fixed velocity regression.

            Args:
                module: Sharded or unsharded transformer.
                opt: Optimizer whose previous gradients are cleared.
                average_gradients: Average the unsharded gradients across ranks.

            Returns:
                Detached velocities [1, 16, 3, 4, 4], including the reference slot.
            """
            opt.zero_grad(set_to_none=True)
            prediction = adapter.forward(module, inputs)
            loss = (prediction - target).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            for parameter in module.parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
                assert torch.isfinite(gradient).all()
                if average_gradients:
                    # Match FSDP's fp32 reduction even when LoRA gradients are
                    # stored in bf16, avoiding reduction-rounding drift.
                    reduced = parameter.grad.float()
                    dist.all_reduce(reduced)
                    parameter.grad.copy_(reduced.div_(2))
            return prediction.detach()

        def compare_gradients() -> None:
            """Compare global gradients, including unused and frozen parameters."""
            actual_parameters = {
                name.replace("._checkpoint_wrapped_module", ""): parameter
                for name, parameter in model.named_parameters()
            }
            expected_parameters = dict(reference.named_parameters())
            assert actual_parameters.keys() == expected_parameters.keys()
            for name, parameter in actual_parameters.items():
                expected_gradient = expected_parameters[name].grad
                assert (parameter.grad is None) == (expected_gradient is None), name
                if expected_gradient is None:
                    continue
                actual_gradient = (
                    parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
                )
                # Scale the absolute tolerance by this parameter's own gradient
                # magnitude; a fixed tolerance can hide missing LoRA gradients.
                tolerance = 0.03 * expected_gradient.float().abs().max().item() + 1e-8
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=tolerance, msg=name)

        for index in range(2):
            actual = backward(model, optimizer)
            expected = backward(reference, ref_optimizer, average_gradients=True)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-3)
            compare_gradients()
            reference_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.float().norm()
                        for parameter in reference.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
            assert torch.isfinite(reference_norm) and reference_norm > 0
            # First update computes the norm without clipping; second clips
            # every nonzero gradient using an independently computed threshold.
            max_norm = float("inf") if index == 0 else reference_norm.item() / 2
            actual_norm = clip_grad_norm(max_norm, [model], device_mesh=mesh, foreach=False)
            torch.testing.assert_close(actual_norm.float(), reference_norm, rtol=2e-2, atol=1e-7)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm, foreach=False)
            compare_gradients()
            optimizer.step()
            ref_optimizer.step()
            if index == 0:
                model_state, optimizer_state = get_state_dict(model, optimizer)
                dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=f"{work_dir}/checkpoint")
        after_second_step = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model_state, optimizer_state = get_state_dict(model, optimizer)
        dcp.load({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=f"{work_dir}/checkpoint")
        set_state_dict(model, optimizer, model_state_dict=model_state, optim_state_dict=optimizer_state)
        replayed = backward(model, optimizer)
        clip_grad_norm(max_norm, [model], device_mesh=mesh, foreach=False)
        optimizer.step()
        torch.testing.assert_close(replayed, actual, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            local = value.to_local() if isinstance(value, DTensor) else value
            saved = after_second_step[name]
            saved = saved.to_local() if isinstance(saved, DTensor) else saved
            torch.testing.assert_close(local, saved, rtol=0, atol=0, msg=name)
            full = value.full_tensor() if isinstance(value, DTensor) else value
            torch.testing.assert_close(full, reference.state_dict()[name], rtol=2e-2, atol=3e-3, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.timeout(1200)
def test_fsdp_training_and_resume(tmp_path: Path, lora: bool) -> None:
    """Whole-block checkpointing works with compiled flex attention and FSDP2."""
    pytest.importorskip("diffusers", minversion="0.40.0")
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path), lora), nprocs=2, join=True)
