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

"""Two-rank FSDP2 gradient, accumulation, checkpoint, and resume parity."""

from __future__ import annotations

import copy
import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    """Reserve a free localhost TCP port for process-group rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous_socket:
        rendezvous_socket.bind(("127.0.0.1", 0))
        return int(rendezvous_socket.getsockname()[1])


def _two_rank_worker(
    rank: int,
    world_size: int,
    port: int,
    result_dir: str,
    activation_checkpointing: bool,
) -> None:
    """Compare accumulated FSDP2 updates and DCP continuation with a reference."""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        import torch.distributed.checkpoint as dcp
        from diffusers import QwenImageTransformer2DModel
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
        from nemo_automodel.components.checkpoint.config import CheckpointingConfig
        from nemo_automodel.components.distributed.utils import get_sync_ctx
        from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
        from nemo_automodel.components.models.qwen_image_edit.adapter import QwenImageEditAdapter
        from nemo_automodel.components.distributed.parallelizer import QwenImageEditParallelizationStrategy
        from nemo_automodel.components.training.utils import (
            prepare_after_first_microbatch,
            prepare_for_final_backward,
            prepare_for_grad_accumulation,
        )

        def assert_model_state_close(actual_model, expected_model, failure: str) -> None:
            """Compare every full model tensor after materializing FSDP2 shards."""
            actual_state = actual_model.state_dict()
            expected_state = expected_model.state_dict()
            if actual_state.keys() != expected_state.keys():
                raise AssertionError(f"{failure}: state-dict keys differ")
            for name in actual_state:
                actual_value = actual_state[name]
                expected_value = expected_state[name]
                if hasattr(actual_value, "full_tensor"):
                    actual_value = actual_value.full_tensor()
                if hasattr(expected_value, "full_tensor"):
                    expected_value = expected_value.full_tensor()
                torch.testing.assert_close(actual_value, expected_value, atol=5e-5, rtol=5e-5, msg=f"{failure}: {name}")

        def assert_gradients_close(actual_model, expected_gradients: dict[str, torch.Tensor | None]) -> None:
            """Compare every full gradient and its global L2 norm.

            Args:
                actual_model: FSDP2 model whose parameters may hold sharded
                    gradient tensors of arbitrary global shape.
                expected_gradients: Mapping from parameter name to either a
                    full tensor of the parameter's global shape or ``None``.
            """
            actual_parameters = {
                name.replace("._checkpoint_wrapped_module", ""): parameter
                for name, parameter in actual_model.named_parameters()
            }
            if actual_parameters.keys() != expected_gradients.keys():
                raise AssertionError("FSDP2 changed or omitted named parameters")

            actual_norm_squared = torch.zeros((), dtype=torch.float32, device=device)
            expected_norm_squared = torch.zeros((), dtype=torch.float32, device=device)
            for name, expected_gradient in expected_gradients.items():
                actual_gradient = actual_parameters[name].grad
                if expected_gradient is None:
                    if actual_gradient is not None:
                        raise AssertionError(f"FSDP2 unexpectedly produced a gradient for {name}")
                    continue
                if actual_gradient is None:
                    raise AssertionError(f"FSDP2 omitted the gradient for {name}")
                if hasattr(actual_gradient, "full_tensor"):
                    actual_gradient = actual_gradient.full_tensor()
                if not torch.isfinite(actual_gradient).all():
                    raise AssertionError(f"FSDP2 produced a non-finite gradient for {name}")
                torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-5, rtol=3e-5)
                actual_norm_squared += actual_gradient.float().square().sum()
                expected_norm_squared += expected_gradient.float().square().sum()
            torch.testing.assert_close(
                actual_norm_squared.sqrt(),
                expected_norm_squared.sqrt(),
                atol=3e-5,
                rtol=3e-5,
            )

        torch.manual_seed(123)
        reference = QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=1,
            joint_attention_dim=12,
            axes_dims_rope=(2, 2, 4),
            zero_cond_t=True,
        ).to(device)
        initial_model = copy.deepcopy(reference)
        sharded = copy.deepcopy(reference)
        canonical_keys = set(reference.state_dict())

        generator = torch.Generator(device=device).manual_seed(456)
        adapter = QwenImageEditAdapter()
        examples = []
        for index in range(3):
            context = FlowMatchingContext(
                noisy_latents=torch.randn((1, 4, 4, 4), generator=generator, device=device),
                latents=torch.randn((1, 4, 4, 4), generator=generator, device=device),
                timesteps=torch.tensor([250.0 + 125.0 * index], device=device),
                sigma=torch.tensor([0.25 + 0.125 * index], device=device),
                task_type="image-edit",
                data_type="image",
                device=device,
                dtype=torch.float32,
                batch={
                    "context_latents": [torch.randn((1, 4, 4, 4), generator=generator, device=device)],
                    "text_embeddings": torch.randn((1, 5, 12), generator=generator, device=device),
                    "text_attention_mask": torch.tensor([[1, 1, 1, 1, 0]], device=device),
                },
            )
            examples.append(
                (
                    adapter.prepare_inputs(context),
                    torch.randn((1, 4, 4, 4), generator=generator, device=device),
                )
            )

        mesh = init_device_mesh(
            "cuda",
            (1, world_size, 1),
            mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"),
        )
        policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        sharded = QwenImageEditParallelizationStrategy().parallelize(
            model=sharded,
            device_mesh=mesh,
            mp_policy=policy,
            activation_checkpointing=activation_checkpointing,
            enable_fsdp2_prefetch=False,
        )
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(sharded.parameters(), lr=1e-3)
        if set(sharded.state_dict()) != canonical_keys:
            raise AssertionError("FSDP2 changed or omitted upstream Diffusers state-dict keys")

        # The first optimizer update uses two microbatches. The real recipe
        # helpers defer FSDP synchronization until the second backward.
        reference_optimizer.zero_grad(set_to_none=True)
        reference_predictions = []
        for inputs, target_velocity in examples[:2]:
            reference_prediction = adapter.forward(reference, inputs)
            reference_predictions.append(reference_prediction.detach().clone())
            (adapter.compute_loss(reference_prediction, target_velocity).mean() / 2).backward()
        reference_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
        }

        optimizer.zero_grad(set_to_none=True)
        prepare_for_grad_accumulation([sharded], pp_enabled=False)
        for microbatch_index, (inputs, target_velocity) in enumerate(examples[:2]):
            is_final_microbatch = microbatch_index == 1
            if is_final_microbatch:
                prepare_for_final_backward([sharded], pp_enabled=False)
            with get_sync_ctx(sharded, is_final_microbatch, defer_fsdp_grad_sync=True):
                sharded_prediction = adapter.forward(sharded, inputs)
                torch.testing.assert_close(
                    sharded_prediction,
                    reference_predictions[microbatch_index],
                    atol=2e-5,
                    rtol=2e-5,
                )
                (adapter.compute_loss(sharded_prediction, target_velocity).mean() / 2).backward()
            if microbatch_index == 0:
                prepare_after_first_microbatch()

        assert_gradients_close(sharded, reference_gradients)

        required_branch_gradients = {
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
            "transformer_blocks.0.txt_mlp.net.0.proj.weight",
        }
        if any(reference_gradients[name] is None for name in required_branch_gradients):
            raise AssertionError("Reference loss did not reach every required dual-stream branch")

        reference_optimizer.step()
        optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        assert_model_state_close(sharded, reference, "FSDP2 post-step parameters differ from fp32 reference")

        model_state = get_model_state_dict(sharded)
        checkpoint_dir = Path(result_dir, "dcp_checkpoint")
        dcp.save({"model": model_state}, checkpoint_id=checkpoint_dir / "model_state")
        checkpointer = Checkpointer(
            CheckpointingConfig(
                checkpoint_dir=checkpoint_dir,
                model_save_format="torch_save",
                model_cache_dir=Path(result_dir, "cache"),
                model_repo_id="test/qwen-image-edit",
                save_consolidated=False,
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            process_group=dist.group.WORLD,
        )
        checkpointer.save_optimizer(optimizer, sharded, str(checkpoint_dir))

        resumed = copy.deepcopy(initial_model)
        resumed = QwenImageEditParallelizationStrategy().parallelize(
            model=resumed,
            device_mesh=mesh,
            mp_policy=policy,
            activation_checkpointing=activation_checkpointing,
            enable_fsdp2_prefetch=False,
        )
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        resumed_model_state = get_model_state_dict(resumed)
        dcp.load({"model": resumed_model_state}, checkpoint_id=checkpoint_dir / "model_state")
        set_model_state_dict(resumed, resumed_model_state)
        checkpointer.load_optimizer(resumed_optimizer, resumed, str(checkpoint_dir))
        if set(resumed.state_dict()) != canonical_keys:
            raise AssertionError("DCP reload changed or omitted upstream Diffusers state-dict keys")
        assert_model_state_close(resumed, sharded, "DCP reload did not restore the saved parameters")

        # Continue with a single-microbatch group. This is both accumulation=1
        # and the partial-final-window case for the preceding accumulation=2
        # contract, while exercising restored Adam moments.
        continuation_inputs, continuation_target = examples[2]
        reference_prediction = adapter.forward(reference, continuation_inputs)
        adapter.compute_loss(reference_prediction, continuation_target).mean().backward()
        continuation_reference_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
        }

        for model, model_optimizer in ((sharded, optimizer), (resumed, resumed_optimizer)):
            prepare_for_grad_accumulation([model], pp_enabled=False)
            prepare_for_final_backward([model], pp_enabled=False)
            with get_sync_ctx(model, True, defer_fsdp_grad_sync=True):
                prediction = adapter.forward(model, continuation_inputs)
                torch.testing.assert_close(prediction, reference_prediction, atol=2e-5, rtol=2e-5)
                adapter.compute_loss(prediction, continuation_target).mean().backward()
            assert_gradients_close(model, continuation_reference_gradients)
            model_optimizer.step()
            model_optimizer.zero_grad(set_to_none=True)

        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        assert_model_state_close(sharded, reference, "Continued FSDP2 parameters differ from fp32 reference")
        assert_model_state_close(resumed, reference, "DCP-resumed parameters differ after continuation")
        checkpointer.close()

        Path(result_dir, f"rank_{rank}.ok").write_text("ok\n", encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


_DIFFUSERS_AVAILABLE = importlib.util.find_spec("diffusers") is not None

pytestmark = [
    pytest.mark.skipif(not _DIFFUSERS_AVAILABLE, reason="diffusers is not installed"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="requires two CUDA devices",
    ),
]


@pytest.mark.parametrize("activation_checkpointing", [False, True], ids=["ac_off", "ac_full"])
def test_two_rank_fsdp2_matches_single_process_gradients(
    tmp_path: Path,
    activation_checkpointing: bool,
) -> None:
    """Compare gradients, updates, DCP reload, and continuation on two NCCL ranks."""
    world_size = 2
    mp.spawn(
        _two_rank_worker,
        args=(world_size, _free_port(), str(tmp_path), activation_checkpointing),
        nprocs=world_size,
        join=True,
    )
    assert sorted(path.name for path in tmp_path.glob("rank_*.ok")) == ["rank_0.ok", "rank_1.ok"]
