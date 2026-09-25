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

"""Two-rank VL retrieval validation and optimizer parity with rank-asymmetric images."""

import copy
import os
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel._transformers.retrieval import BiEncoderModel, CrossEncoderModel
from nemo_automodel.components.distributed.parallelizer import apply_fsdp2_sharding_recursively
from nemo_automodel.components.models.ministral_bidirectional.model import (
    Mistral3BidirectionalConfig,
    Mistral3BidirectionalModel,
    Mistral3VLBidirectionalForSequenceClassification,
)


def _tiny_config() -> Mistral3BidirectionalConfig:
    config = Mistral3BidirectionalConfig(
        text_config={
            "model_type": "ministral3",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "sliding_window": None,
            "attention_dropout": 0.0,
            "use_cache": False,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "num_channels": 3,
            "attention_dropout": 0.0,
        },
        image_token_index=10,
        spatial_merge_size=1,
        num_labels=1,
        pooling="avg",
        temperature=0.02,
    )
    config._attn_implementation = "eager"
    return config


def _run_case(reranker: bool) -> None:
    """Compare [batch, hidden] embeddings or [batch, 1] scores and two Adam updates."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(42)
    if reranker:
        model = CrossEncoderModel(Mistral3VLBidirectionalForSequenceClassification(_tiny_config()))
    else:
        model = BiEncoderModel(Mistral3BidirectionalModel(_tiny_config()), pooling="avg", l2_normalize=False)
    model = model.to(device)
    reference = copy.deepcopy(model)
    initial_state = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
    policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    apply_fsdp2_sharding_recursively(model, mesh, policy)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
    # Match recipe setup: optimizer construction precedes any forward, while
    # FSDP exposes sharded parameters rather than temporarily gathered views.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    ids = torch.tensor([[10, 10, 10, 10, 1, 2]] if dist.get_rank() else [[1, 2, 3, 4]], device=device)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if dist.get_rank():
        inputs.update(
            pixel_values=torch.arange(192, dtype=torch.float32, device=device).reshape(1, 3, 8, 8) / 255,
            image_sizes=torch.tensor([[8, 8]], device=device),
        )

    def outputs(module: torch.nn.Module, batch: dict) -> torch.Tensor:
        """Return embeddings [1, 16] or reranker scores [1, 1] for this rank's batch."""
        result = module(batch)
        return result.logits if reranker else result

    model.eval()
    reference.eval()
    with torch.no_grad():
        expected = outputs(reference, inputs)
        # The default and even an explicit false cannot skip sharded vision
        # collectives when another rank executes real image inputs.
        for dummy_policy in (None, False, True):
            batch = dict(inputs)
            if dummy_policy is not None:
                batch["run_dummy_vision"] = dummy_policy
            torch.testing.assert_close(outputs(model, batch), expected, rtol=1e-5, atol=1e-5)

    model.train()
    reference.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual = outputs(model, inputs)
        expected = outputs(reference, inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        # Match FSDP's mean gradient across the two different modality batches.
        for parameter in reference.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(2)
        for name, parameter in model.named_parameters():
            expected_gradient = dict(reference.named_parameters())[name].grad
            assert parameter.grad is not None, name
            gradient = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
            torch.testing.assert_close(
                gradient, expected_gradient, rtol=1e-4, atol=1e-8, msg=lambda detail: f"{name}: {detail}"
            )
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in model.named_parameters():
            actual_state = optimizer.state[parameter]
            reference_state = reference_optimizer.state[dict(reference.named_parameters())[name]]
            assert actual_state.keys() == reference_state.keys(), name
            for key, value in actual_state.items():
                value = value.full_tensor() if isinstance(value, DTensor) else value
                torch.testing.assert_close(
                    value,
                    reference_state[key],
                    rtol=1e-4,
                    atol=1e-8,
                    msg=lambda detail: f"{name}/{key}: {detail}",
                )
        for name, value in model.state_dict().items():
            value = value.full_tensor() if isinstance(value, DTensor) else value
            torch.testing.assert_close(
                value, reference.state_dict()[name], rtol=1e-4, atol=1e-5, msg=lambda detail: f"{name}: {detail}"
            )

    changed = {name for name, value in reference.state_dict().items() if not torch.equal(value, initial_state[name])}
    for tower in ("vision_tower", "multi_modal_projector", "language_model"):
        assert any(tower in name for name in changed), f"No real optimizer update in {tower}"
    if reranker:
        assert any("score.weight" in name for name in changed)
    model.eval()
    reference.eval()
    with torch.no_grad():
        torch.testing.assert_close(outputs(model, inputs), outputs(reference, inputs), rtol=1e-4, atol=1e-5)
    dist.barrier()


def _run_worker() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=45))
    try:
        for reranker in (False, True):
            _run_case(reranker)
        if dist.get_rank() == 0:
            print("MISTRAL3_VL_FSDP_PASS", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_mistral3_vl_fsdp_mixed_modality_validation_and_updates() -> None:
    """Both retrieval wrappers preserve distributed validation, gradients, and nonzero updates."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    # DETAIL's process-group wrapper lacks DTensor's coalesced gather in some
    # supported PyTorch builds. The process-group timeout still detects hangs.
    env["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "MISTRAL3_VL_FSDP_PASS" in completed.stdout


if __name__ == "__main__" and "--worker" in sys.argv:
    _run_worker()
