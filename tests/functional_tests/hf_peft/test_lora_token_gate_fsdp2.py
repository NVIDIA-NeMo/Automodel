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

"""2-GPU FSDP2 test for per-token LoRA gating (``lora_token_gate``).

A tiny random Qwen3 causal LM gets LoRA on every ``*_proj`` and is parallelized
through the production ``fsdp2_strategy_parallelize`` entry point (with and
without activation checkpointing, fp32 and bf16 mixed precision). Two
microbatches with different per-token gates run with gradient sync deferred to
the last one. The reduce-scattered LoRA gradients must match a single-process
reference over the global batch, and a loss confined to gated-off tokens must
leave every adapter gradient exactly zero.
"""

import copy
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules, lora_token_gate
from nemo_automodel.components.distributed.parallelizer import fsdp2_strategy_parallelize

pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs (FSDP2 over NCCL)"),
    pytest.mark.timeout(180),
]

_WORLD_SIZE = 2
_LOCAL_BATCH = 2
_SEQ = 16
_VOCAB = 128


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _build_model(device: torch.device) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.float32)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["*_proj"], dim=4, alpha=8, use_triton=False))
    # Non-zero lora_B so the adapter changes the forward and gradients reach lora_A.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.05)
    return model


def _global_microbatches(device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Two global microbatches of ``(input_ids, gate, upstream)``, identical on every rank.

    Returns:
        List of tuples with input_ids: Tensor of shape [global_batch, sequence]; gate: bool Tensor of
        shape [global_batch, sequence]; upstream: Tensor of shape [global_batch, sequence, vocab] used
        as the gradient of the logits.
    """
    generator = torch.Generator().manual_seed(1)
    global_batch = _WORLD_SIZE * _LOCAL_BATCH
    microbatches = []
    for _ in range(2):
        input_ids = torch.randint(0, _VOCAB, (global_batch, _SEQ), generator=generator)
        gate = torch.rand(global_batch, _SEQ, generator=generator) < 0.5
        upstream = torch.randn(global_batch, _SEQ, _VOCAB, generator=generator)
        microbatches.append((input_ids.to(device), gate.to(device), upstream.to(device)))
    return microbatches


def _lora_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            grads[name.replace("_checkpoint_wrapped_module.", "")] = grad.float()
    return grads


def _run_rank(rank: int, port: int, activation_checkpointing: bool, bf16: bool) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=_WORLD_SIZE)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    try:
        microbatches = _global_microbatches(device)
        model = _build_model(device)

        # Single-process fp32 reference: sum of both microbatches' gated backward over the global batch.
        reference = copy.deepcopy(model)
        for input_ids, gate, upstream in microbatches:
            with lora_token_gate(reference, gate):
                reference(input_ids=input_ids).logits.backward(upstream)
        reference_grads = _lora_grads(reference)

        dtype = torch.bfloat16 if bf16 else torch.float32
        device_mesh = init_device_mesh(
            "cuda", (1, _WORLD_SIZE, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp")
        )
        model = fsdp2_strategy_parallelize(
            model,
            device_mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32, output_dtype=dtype),
            activation_checkpointing=activation_checkpointing,
        )

        local = slice(rank * _LOCAL_BATCH, (rank + 1) * _LOCAL_BATCH)
        for step, (input_ids, gate, upstream) in enumerate(microbatches):
            model.set_requires_gradient_sync(step == len(microbatches) - 1)
            with lora_token_gate(model, gate[local]):
                logits = model(input_ids=input_ids[local]).logits
                assert torch.isfinite(logits).all()
                # FSDP2 averages gradients over ranks; scale so the result is the global sum.
                logits.float().backward(upstream[local] * _WORLD_SIZE)
        leaked = [
            name for name, module in model.named_modules() if getattr(module, "_lora_token_gate", None) is not None
        ]
        assert not leaked, f"gate leaked past the context on {leaked}"

        tolerance = dict(rtol=5e-2, atol=5e-3) if bf16 else dict(rtol=1e-4, atol=1e-6)
        grads = _lora_grads(model)
        assert grads.keys() == reference_grads.keys()
        for name, grad in grads.items():
            assert torch.isfinite(grad).all(), name
            torch.testing.assert_close(grad, reference_grads[name], **tolerance, msg=name)

        # A loss confined to gated-off tokens must give exactly zero adapter gradients. Attention mixes
        # tokens, so the loss uses a row whose every token is gated off.
        model.zero_grad(set_to_none=False)
        model.set_requires_gradient_sync(True)
        input_ids, _, upstream = microbatches[0]
        gate = torch.zeros(_LOCAL_BATCH, _SEQ, dtype=torch.bool, device=device)
        gate[0] = True  # row 0 adapted, row 1 fully base
        with lora_token_gate(model, gate):
            logits = model(input_ids=input_ids[local]).logits
            (logits.float()[1] * upstream[local][1]).sum().backward()
        for name, grad in _lora_grads(model).items():
            assert torch.count_nonzero(grad) == 0, name
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("activation_checkpointing", "bf16"),
    [(False, False), (True, False), (False, True)],
    ids=["fp32", "fp32-activation-checkpointing", "bf16-mixed-precision"],
)
def test_lora_token_gate_fsdp2_matches_single_process_reference(activation_checkpointing: bool, bf16: bool) -> None:
    mp.spawn(_run_rank, args=(_free_port(), activation_checkpointing, bf16), nprocs=_WORLD_SIZE, join=True)
