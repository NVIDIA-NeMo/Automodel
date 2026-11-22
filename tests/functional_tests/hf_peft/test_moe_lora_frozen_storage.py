# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen MoE base storage stays BF16 through real FSDP2 training and state loading.

Two ranks cover both expert sharding (EP2) and expert data sharding (EP1/DP2).
The worker also accepts four ranks for EP2/DP2 validation outside CI. Compare
against FP32 base storage with identical BF16-representable weights: the storage
optimization must preserve logits, adapter gradients, and optimizer updates.
"""

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import Qwen3MoeConfig

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard of an arbitrary-shaped parameter or gradient.

    Args:
        tensor: Global DTensor with EP/DP placements, or a plain local tensor.

    Returns:
        Local tensor with the same dtype, aliasing the input's storage.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _build_model(setup: DistributedSetup, dispatcher: str, *, fp32_base: bool = False):
    torch.manual_seed(1234)
    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        use_cache=False,
    )
    # Disable only storage preparation for the reference; every other production
    # path (including real LoRA patching, EP, FSDP, and checkpointing) is shared.
    context = (
        patch("nemo_automodel._transformers.infrastructure.prepare_frozen_parameters_for_sharding")
        if fp32_base
        else contextlib.nullcontext()
    )
    with context:
        return NeMoAutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float32,
            distributed_setup=setup,
            peft_config=PeftConfig(target_modules=["*"], dim=16, alpha=32, dropout=0.0),
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch_fp32",
                rope_fusion=False,
                experts="torch_mm" if dispatcher == "deepep" else "torch",
                dispatcher=dispatcher,
                fake_balanced_gate=True,
                enable_hf_state_dict_adapter=False,
            ),
        )


def _assert_storage(model, frozen_dtype: torch.dtype) -> None:
    expert_bases = expert_adapters = 0
    for name, parameter in model.named_parameters():
        assert isinstance(parameter, DTensor), name
        expected = torch.float32 if parameter.requires_grad else frozen_dtype
        assert parameter.dtype == expected, (name, parameter.dtype, expected)
        assert _local(parameter).dtype == expected, name
        if ".experts." in name:
            expert_adapters += parameter.requires_grad
            expert_bases += not parameter.requires_grad
    assert expert_bases and expert_adapters


def _check_training(ep_size: int, dispatcher: str) -> None:
    # Reuse the mesh: DeepEP owns one communication buffer per process group.
    setup = DistributedSetup.build(
        strategy="fsdp2", parallelism_sizes=ParallelismSizes(ep_size=ep_size), activation_checkpointing=True
    )
    model = _build_model(setup, dispatcher)
    reference = _build_model(setup, dispatcher, fp32_base=True)
    _assert_storage(model, torch.bfloat16)
    _assert_storage(reference, torch.float32)
    # Nonzero B exercises both adapter factors on the very first backward.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora" in name and "_B" in name:
                _local(parameter).normal_(std=0.02)
        reference.load_state_dict(model.state_dict())
    _assert_storage(reference, torch.float32)
    params = dict(model.named_parameters())
    ref_params = dict(reference.named_parameters())
    assert params.keys() == ref_params.keys()
    frozen_before = {name: _local(p).detach().clone() for name, p in params.items() if not p.requires_grad}
    optimizers = [
        torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3) for m in (model, reference)
    ]
    # Rank-distinct input exercises real reductions, rather than identical-rank gradients.
    torch.manual_seed(4242 + dist.get_rank())
    inputs = torch.randint(0, 512, (1, 32), device="cuda")
    for _ in range(2):
        outputs = []
        for current, optimizer in zip((model, reference), optimizers):
            optimizer.zero_grad(set_to_none=True)
            logits = current(inputs).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, 512), inputs[:, 1:].reshape(-1))
            loss.backward()
            outputs.append(logits.detach())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for name, parameter in params.items():
            if not parameter.requires_grad:
                assert parameter.grad is None, name
                torch.testing.assert_close(_local(parameter), frozen_before[name], rtol=0, atol=0)
                continue
            assert parameter.grad is not None, name
            gradient = _local(parameter.grad)
            assert torch.isfinite(gradient).all(), name
            assert gradient.abs().max() > 0, name
            torch.testing.assert_close(gradient, _local(ref_params[name].grad), rtol=0, atol=0)
        for optimizer in optimizers:
            optimizer.step()
        for name, parameter in params.items():
            if parameter.requires_grad:
                torch.testing.assert_close(_local(parameter), _local(ref_params[name]), rtol=0, atol=0)
        _assert_storage(model, torch.bfloat16)
    # Loading FP32 checkpoint tensors must copy into the BF16 base allocation.
    model.load_state_dict(reference.state_dict())
    _assert_storage(model, torch.bfloat16)
    for name, parameter in params.items():
        torch.testing.assert_close(_local(parameter).float(), _local(ref_params[name]), rtol=0, atol=0)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"FROZEN_STORAGE_PASS ep={ep_size} world={dist.get_world_size()} dispatcher={dispatcher}", flush=True)


def _run_worker(dispatcher: str) -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    try:
        # DeepEP requires an expert-parallel mesh; EP1 uses the torch dispatcher.
        for ep_size in (2,) if dispatcher == "deepep" else (1, 2):
            _check_training(ep_size, dispatcher)
    finally:
        if dispatcher == "deepep":
            from nemo_automodel.components.moe.megatron.fused_a2a import free_buffer

            free_buffer()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices for EP and FSDP")
def test_moe_lora_frozen_storage_two_rank() -> None:
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            "--worker",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("FROZEN_STORAGE_PASS") == 2, result.stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dispatcher", choices=("torch", "deepep"), default="torch")
    args = parser.parse_args()
    if args.worker:
        _run_worker(args.dispatcher)
