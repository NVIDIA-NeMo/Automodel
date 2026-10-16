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

"""Train a tiny Wan through the diffusion pipeline's LoRA/FSDP lifecycle."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _train_wan(rank: int, world_size: int, init_file: str, force_backport: bool) -> None:
    from diffusers import WanTransformer3DModel
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    from torch.distributed.tensor import DTensor

    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config
    from nemo_automodel.shared import import_utils

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(42)
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            text_dim=8,
            freq_dim=8,
            ffn_dim=32,
            num_layers=2,
            rope_max_seq_len=16,
        ).to(device=device, dtype=torch.bfloat16)
        reference = deepcopy(transformer)
        peft_cfg = PeftConfig(target_modules=["*"], dim=4)
        torch.manual_seed(43)
        apply_lora_to_linear_modules(reference, peft_cfg)
        reference.train()
        mesh_context = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
                enable_fsdp2_prefetch=False,
            ),
            world_size=world_size,
        ).mesh_context
        pipe = SimpleNamespace(transformer=transformer, components={"transformer": transformer})
        loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: pipe)
        original_version_check = import_utils.is_torch_min_version

        def version_check(version):
            # Exercise the actual backport on this runtime; this does not claim
            # to emulate the rest of PyTorch 2.6/2.7.
            if force_backport and version == "2.8.0":
                return False
            return original_version_check(version)

        torch.manual_seed(43)
        # Replace only checkpoint loading; injection, sharding, freezing and
        # training all use the production diffusion pipeline and Wan sidecar.
        module_path = "nemo_automodel._diffusers.auto_diffusion_pipeline"
        with (
            patch(f"{module_path}.DiffusionPipeline", loader),
            patch(f"{module_path}.resolve_diffusion_model_dir", lambda path: path),
            patch.object(import_utils, "is_torch_min_version", version_check),
        ):
            pipe = NeMoAutoDiffusionPipeline.from_pretrained(
                "unused-local-checkpoint",
                device=device,
                torch_dtype=torch.bfloat16,
                mesh_context=mesh_context,
                peft_cfg=peft_cfg,
                model_type="wan",
                load_for_training=True,
            )
        model = pipe.transformer
        if force_backport:
            assert FSDPParamGroup._init_mp_dtypes._automodel_mixed_frozen_storage
        adapters = {name: param for name, param in model.named_parameters() if "lora_" in name}
        assert adapters
        assert all(param.dtype == torch.float32 and param.requires_grad for param in adapters.values())
        assert all(
            param.dtype == torch.bfloat16 and not param.requires_grad
            for name, param in model.named_parameters()
            if "lora_" not in name
        )
        assert all(isinstance(param, DTensor) == (world_size > 1) for param in adapters.values())
        reference_adapters = {name: param for name, param in reference.named_parameters() if "lora_" in name}
        optimizer = torch.optim.AdamW(adapters.values(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference_adapters.values(), lr=1e-3)
        initial_adapters = {name: param.detach().clone() for name, param in reference_adapters.items()}
        initial_base = {
            name: param.detach().clone() for name, param in reference.named_parameters() if "lora_" not in name
        }
        torch.manual_seed(44)
        inputs = torch.randn(world_size, 4, 1, 4, 4, device=device, dtype=torch.bfloat16)
        text = torch.randn(world_size, 3, 8, device=device, dtype=torch.bfloat16)
        timestep = torch.full((world_size,), 100, device=device, dtype=torch.long)
        target = torch.randn_like(inputs)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(inputs[rank : rank + 1], timestep[rank : rank + 1], text[rank : rank + 1]).sample
                reference_output = reference(inputs, timestep, text).sample
                loss = (output.float() - target[rank : rank + 1].float()).square().mean()
                reference_loss = (reference_output.float() - target.float()).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            reference_loss.backward()
            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss)
            torch.testing.assert_close(mean_loss / world_size, reference_loss, rtol=0.02, atol=0.002)
            for name, param in adapters.items():
                assert param.grad is not None and param.grad.dtype == torch.float32
                grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
                assert torch.isfinite(grad).all()
                torch.testing.assert_close(grad, reference_adapters[name].grad, rtol=0.05, atol=0.002)
            optimizer.step()
            reference_optimizer.step()
            for name, param in adapters.items():
                value = param.full_tensor() if isinstance(param, DTensor) else param
                torch.testing.assert_close(value, reference_adapters[name], rtol=0.03, atol=0.0002)
                for key in ("exp_avg", "exp_avg_sq"):
                    assert optimizer.state[param][key].dtype == torch.float32
        assert any(not torch.equal(param, initial_adapters[name]) for name, param in reference_adapters.items())
        for name, param in model.named_parameters():
            if name in initial_base:
                value = param.full_tensor() if isinstance(param, DTensor) else param
                torch.testing.assert_close(value, initial_base[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size, force_backport", [(1, False), (2, False), (2, True)])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs for FSDP2")
def test_diffusion_lora_fsdp_storage(tmp_path, world_size, force_backport):
    mp.spawn(
        _train_wan,
        args=(world_size, str(tmp_path / "process-group"), force_backport),
        nprocs=world_size,
        join=True,
    )
