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

"""UPipe context-parallel parity for the native Llama model.

Compares a full-sequence single-rank forward/backward against UPipe at each CP
size the world supports. Gradient parity is the load-bearing assertion: UPipe's
backward is hand-written and recomputes the projections from ``x``, so matching
logits alone would not exercise it.

Launch with ``torchrun --nproc_per_node=N``.
"""

import argparse
import os

import torch
import torch.distributed as dist
from transformers import LlamaConfig

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.models.llama.model import LlamaAttention, LlamaForCausalLM

SEQ_LEN = 512
BATCH = 1
DTYPE = torch.bfloat16

# bf16 attention accumulates differently once the sequence is split across
# ranks, so parity is judged on relative Frobenius error rather than allclose.
LOGITS_TOLERANCE = 3e-2
GRAD_TOLERANCE = 5e-2


def build_config() -> LlamaConfig:
    """Small Llama with GQA, chosen so cp_size up to 8 divides num_key_value_heads."""
    return LlamaConfig(
        vocab_size=512,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=SEQ_LEN,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
        # The rotary cache is built in the config dtype; leaving it float32 would
        # silently upcast the whole attention block and break the o_proj matmul.
        torch_dtype=DTYPE,
        # Eager is the unambiguous reference and keeps the comparison independent
        # of which SDPA backend the installed cuDNN picks for GQA.
        _attn_implementation="eager",
    )


def build_model(device) -> LlamaForCausalLM:
    """Build the model identically on every rank."""
    torch.manual_seed(1234)
    config = build_config()
    backend = BackendConfig(attn="upipe", linear="torch", rms_norm="torch", rope="torch", rope_fusion=False)
    model = LlamaForCausalLM(config, backend=backend)
    return model.to(device=device, dtype=DTYPE)


def set_cp_group(model, cp_group, cp_size) -> None:
    """Bind or clear the CP group on every attention module."""
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module.bind_upipe_cp_group(cp_group)
            if cp_group is not None:
                assert module._upipe_head_perm_size == cp_size


def forward_backward(model, input_ids, position_ids, grad_seed):
    """Run one forward/backward and collect logits plus attention weight grads."""
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None).logits
    logits.backward(grad_seed)

    grads = {}
    for name, param in model.named_parameters():
        if any(proj in name for proj in ("q_proj", "k_proj", "v_proj", "o_proj")):
            grads[name] = param.grad.detach().clone()
    return logits.detach(), grads


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / (expected.float().norm() + 1e-12)).item()


def run_cp_size(model, cp_size, input_ids, position_ids, grad_seed, reference_logits, reference_grads, rank, world):
    """Check UPipe at one CP size against the single-rank reference.

    Ranks outside the first CP group idle at the barrier so that world sizes
    that are not a multiple of ``cp_size`` still terminate cleanly.
    """
    ranks = list(range(cp_size))
    cp_group = dist.new_group(ranks)
    participating = rank in ranks

    failures = []
    if participating:
        local_len = SEQ_LEN // cp_size
        window = slice(rank * local_len, (rank + 1) * local_len)

        set_cp_group(model, cp_group, cp_size)
        logits, grads = forward_backward(
            model,
            input_ids[:, window],
            position_ids[:, window],
            grad_seed[:, window],
        )
        set_cp_group(model, None, cp_size)

        logits_error = relative_error(logits, reference_logits[:, window])
        if logits_error > LOGITS_TOLERANCE:
            failures.append(f"logits rel_err={logits_error:.3e} > {LOGITS_TOLERANCE}")

        # Each rank's weight gradient is a partial sum over its sequence shard.
        for name in sorted(grads):
            dist.all_reduce(grads[name], group=cp_group)
            grad_error = relative_error(grads[name], reference_grads[name])
            if grad_error > GRAD_TOLERANCE:
                failures.append(f"{name} rel_err={grad_error:.3e} > {GRAD_TOLERANCE}")

        if rank == 0:
            worst = max(relative_error(grads[n], reference_grads[n]) for n in grads)
            print(f"  cp_size={cp_size}: logits={logits_error:.3e} worst_grad={worst:.3e}", flush=True)

    dist.destroy_process_group(cp_group)
    dist.barrier()

    if failures:
        raise AssertionError(f"[rank {rank}] cp_size={cp_size}: " + "; ".join(failures))
    del world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-sizes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda")

    model = build_model(device)

    torch.manual_seed(7)
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ_LEN), device=device)
    position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0).expand(BATCH, -1)
    grad_seed = torch.randn(BATCH, SEQ_LEN, model.config.vocab_size, device=device, dtype=DTYPE) * 0.01

    # Reference: UPipe dormant (no CP group bound), so the ordinary SDPA path
    # runs over the full sequence -- exactly the model UPipe must reproduce.
    set_cp_group(model, None, 1)
    reference_logits, reference_grads = forward_backward(model, input_ids, position_ids, grad_seed)
    if rank == 0:
        print(f"reference logits {tuple(reference_logits.shape)}, {len(reference_grads)} attention grads", flush=True)

    cp_sizes = args.cp_sizes or [size for size in (2, 4, 8) if size <= world]
    for cp_size in cp_sizes:
        run_cp_size(
            model,
            cp_size,
            input_ids,
            position_ids,
            grad_seed,
            reference_logits,
            reference_grads,
            rank,
            world,
        )

    if rank == 0:
        print(f"UPipe CP parity passed for cp_sizes={cp_sizes}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
