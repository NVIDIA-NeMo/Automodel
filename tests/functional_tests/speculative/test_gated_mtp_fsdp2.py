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

"""2-GPU FSDP2 test for gated-LoRA MTP training through the fine-tuning recipe's training step.

A tiny random Qwen3 causal LM gets LoRA on every ``*_proj`` and is parallelized through the production
``fsdp2_strategy_parallelize`` entry point (with and without activation checkpointing, fp32 and bf16 mixed
precision). Each rank runs the recipe's real non-PP ``_forward_backward_step`` with gated MTP and soft-MTP
distillation on two padded microbatches, with gradient sync deferred to the last one. In fp32 the reduce-scattered
LoRA gradients must match the same step run in a single process over the global batch; in every variant they must
be finite and the token gate must be removed afterwards.
"""

import os
import socket
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.distributed.parallelizer import fsdp2_strategy_parallelize
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.speculative.gated_mtp import GatedMTPConfig
from nemo_automodel.recipes.llm import train_ft
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs (FSDP2 over NCCL)"),
    pytest.mark.timeout(240),
]

_WORLD_SIZE = 2
_LOCAL_BATCH = 2
_SEQ = 8
_VOCAB = 128
_MASK_IDS = [125, 126, 127]


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
        attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.float32)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["*_proj"], dim=4, alpha=8, use_triton=False))
    # Non-zero lora_B so the adapter changes the forward and gradients reach lora_A.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.05)
    return model


def _global_microbatches(device: torch.device) -> list[dict[str, torch.Tensor]]:
    """Two global microbatches, identical on every rank, with right padding on some rows.

    Returns:
        List of batches with input_ids, labels, and attention_mask, each a Tensor of shape
        [global_batch, sequence]; ``labels[t]`` is token ``t + 1`` and -100 on padding and the last real token.
    """
    generator = torch.Generator().manual_seed(1)
    global_batch = _WORLD_SIZE * _LOCAL_BATCH
    microbatches = []
    for _ in range(2):
        input_ids = torch.randint(0, _VOCAB - len(_MASK_IDS), (global_batch, _SEQ), generator=generator)
        lengths = torch.randint(_SEQ // 2, _SEQ + 1, (global_batch,), generator=generator)
        attention_mask = (torch.arange(_SEQ).unsqueeze(0) < lengths.unsqueeze(1)).long()
        labels = torch.cat([input_ids[:, 1:], torch.full((global_batch, 1), -100)], dim=1)
        labels[torch.arange(_SEQ).unsqueeze(0) >= (lengths - 1).unsqueeze(1)] = -100
        microbatches.append(
            {
                "input_ids": input_ids.to(device),
                "labels": labels.to(device),
                "attention_mask": attention_mask.to(device),
            }
        )
    return microbatches


def _make_recipe(model: torch.nn.Module, device: torch.device, *, dp_size: int, mp_policy) -> object:
    """Build a minimal harness around the real non-PP ``_forward_backward_step`` with soft gated MTP."""
    recipe = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    object.__setattr__(
        recipe, "cfg", SimpleNamespace(mtp=SimpleNamespace(distillation_mode="soft", scaling_factor=None))
    )
    object.__setattr__(recipe, "dist_env", SimpleNamespace(device=device))
    object.__setattr__(recipe, "device_mesh", object())
    object.__setattr__(recipe, "pp_enabled", False)
    object.__setattr__(recipe, "tokenizer", SimpleNamespace(pad_token_id=0))
    object.__setattr__(recipe, "te_fp8", None)
    object.__setattr__(recipe, "model_parts", [model])
    object.__setattr__(recipe, "distributed_config", SimpleNamespace(defer_fsdp_grad_sync=True, mp_policy=mp_policy))
    object.__setattr__(recipe, "loss_fn", MaskedCrossEntropy())
    object.__setattr__(recipe, "domain_mixture", None)
    object.__setattr__(recipe, "_get_cp_group_size", lambda: 1)
    object.__setattr__(recipe, "_get_dp_group_size", lambda include_cp=False: dp_size)
    object.__setattr__(recipe, "gated_mtp", GatedMTPConfig(mask_token_ids=_MASK_IDS, scaling_factor=0.5).build())
    recipe._mtp_accept_metrics = []
    return recipe


def _run_step(recipe: object, microbatches: list[dict[str, torch.Tensor]], num_label_tokens: int) -> None:
    for idx, batch in enumerate(microbatches):
        recipe._forward_backward_step(
            idx,
            dict(batch),
            loss_buffer=[],
            num_label_tokens=num_label_tokens,
            num_batches=len(microbatches),
            is_train=True,
        )


def _lora_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            grads[name.replace("_checkpoint_wrapped_module.", "")] = grad.float()
    return grads


class _IdentitySharder:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def shard(self, batch):
        return nullcontext, batch


def _run_rank(rank: int, port: int, activation_checkpointing: bool, bf16: bool, device_type: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=_WORLD_SIZE)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    train_ft.ContextParallelSharder = _IdentitySharder  # cp_size == 1
    try:
        microbatches = _global_microbatches(device)
        num_label_tokens = sum(int((batch["labels"] != -100).sum()) for batch in microbatches)

        # Single-process fp32 reference: the same step over the global batch.
        reference = _build_model(device)
        _run_step(_make_recipe(reference, device, dp_size=1, mp_policy=None), microbatches, num_label_tokens)
        reference_grads = _lora_grads(reference)

        model = _build_model(device)
        dtype = torch.bfloat16 if bf16 else torch.float32
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32, output_dtype=dtype)
        device_mesh = init_device_mesh(
            device_type, (1, _WORLD_SIZE, 1), mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp")
        )
        model = fsdp2_strategy_parallelize(
            model, device_mesh, mp_policy=mp_policy, activation_checkpointing=activation_checkpointing
        )
        local = slice(rank * _LOCAL_BATCH, (rank + 1) * _LOCAL_BATCH)
        local_microbatches = [{key: value[local] for key, value in batch.items()} for batch in microbatches]
        # The recipe scales the local loss by the DP size, since FSDP2 averages gradients over ranks.
        _run_step(
            _make_recipe(model, device, dp_size=_WORLD_SIZE, mp_policy=mp_policy),
            local_microbatches,
            num_label_tokens,
        )
        leaked = [
            name for name, module in model.named_modules() if getattr(module, "_lora_token_gate", None) is not None
        ]
        assert not leaked, f"gate leaked past the step on {leaked}"

        grads = _lora_grads(model)
        assert grads.keys() == reference_grads.keys()
        for name, grad in grads.items():
            assert torch.isfinite(grad).all(), name
            assert torch.count_nonzero(grad) > 0, name
            # bf16 compute is not elementwise comparable to the fp32 reference.
            # fp32 differs only by reduce-scatter and microbatch summation order.
            if not bf16:
                torch.testing.assert_close(
                    grad, reference_grads[name], rtol=1e-3, atol=1e-5, msg=lambda msg: f"{name}: {msg}"
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("activation_checkpointing", "bf16"),
    [(False, False), (True, False), (False, True)],
    ids=["fp32", "fp32-activation-checkpointing", "bf16-mixed-precision"],
)
def test_gated_mtp_fsdp2_matches_single_process_reference(activation_checkpointing: bool, bf16: bool) -> None:
    mp.spawn(_run_rank, args=(_free_port(), activation_checkpointing, bf16, "cuda"), nprocs=_WORLD_SIZE, join=True)
