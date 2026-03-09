# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""FlashOptim + DCP checkpoint roundtrip test.

Trains a model for N steps with FlashAdamW, saves optimizer + model state via
DCP, loads into a fresh optimizer, and compares continued-training losses to
verify checkpoint fidelity.

FlashOptim >= 0.1.3 includes native DTensor support for DCP compatibility.

Usage (L2 CI via shell script wrapper):
    See tests/functional_tests/hf_dcp/L2_FlashOptim_DCP_Roundtrip.sh

Manual usage:
    torchrun --nproc-per-node=2 tests/functional_tests/checkpoint/test_flashoptim_dcp_roundtrip.py \
        --model $TEST_DATA_DIR/hf_mixtral_2l/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashoptim import FlashAdamW

from nemo_automodel.components.models.common.utils import cast_model_to_dtype

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_CKPT_DIR = "/tmp/flashoptim_dcp_roundtrip"
SEQ_LEN = 512
BATCH_SIZE = 8
STEPS = 20
MASTER_WEIGHT_BITS = 24
LR = 1e-4
# Threshold for max loss delta between original and resumed training.
# Accounts for quantization noise from int8->bf16->int8 round-trip of
# FlashAdamW's compressed state dicts.
LOSS_DELTA_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic_batches(vocab_size, num_batches, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    """Generate synthetic random token batches for training."""
    batches = []
    g = torch.Generator().manual_seed(42)
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
        labels = input_ids.clone()
        batches.append({"input_ids": input_ids, "labels": labels})
    return batches


# ---------------------------------------------------------------------------
# Training / checkpoint helpers
# ---------------------------------------------------------------------------
def train(model, optimizer, batches, n, start=0):
    """Run n training steps, returning per-step losses."""
    losses = []
    for i in range(n):
        batch = {k: v.to("cuda") for k, v in batches[start + i].items()}
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(out.loss.item())
    return losses


def save_checkpoint(model, optimizer, path):
    """Save model + optimizer state via DCP."""
    os.makedirs(path, exist_ok=True)
    model_sd = get_model_state_dict(model)
    optim_sd = get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=path)


def load_checkpoint(model, optimizer, path):
    """Load model + optimizer state from DCP checkpoint."""
    model_sd = get_model_state_dict(model)
    optim_sd = get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )
    sd = {"model": model_sd, "optim": optim_sd}
    dcp.load(sd, checkpoint_id=path)
    set_model_state_dict(model, sd["model"])
    set_optimizer_state_dict(
        model,
        optimizer,
        optim_state_dict=sd["optim"],
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )


def build(model_name, mesh):
    """Build model + FlashAdamW optimizer with FSDP2."""
    torch.manual_seed(42)
    m = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    # Use Automodel's cast_model_to_dtype which respects _keep_in_fp32_modules
    cast_model_to_dtype(m, torch.bfloat16)
    fully_shard(m, mesh=mesh)
    o = FlashAdamW(
        [p for p in m.parameters() if p.requires_grad],
        lr=LR,
        master_weight_bits=MASTER_WEIGHT_BITS,
    )
    return m, o


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FlashOptim + DCP checkpoint roundtrip test"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=DEFAULT_CKPT_DIR,
        help=f"Checkpoint directory (default: {DEFAULT_CKPT_DIR})",
    )
    parser.add_argument(
        "--steps", type=int, default=STEPS, help=f"Training steps per phase (default: {STEPS})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LOSS_DELTA_THRESHOLD,
        help=f"Max allowed loss delta for PASS (default: {LOSS_DELTA_THRESHOLD})",
    )
    args = parser.parse_args()

    model_name = args.model

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Get vocab size from tokenizer for synthetic data generation
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    total_batches = 2 * args.steps
    batches = generate_synthetic_batches(vocab_size, total_batches)

    if rank == 0:
        print(f"Model:   {model_name}")
        print(f"Batches: {len(batches)} (synthetic, seq_len={SEQ_LEN})")
        print(f"Steps:   {args.steps} per phase")
        print()

    # Clean checkpoint dir
    if rank == 0:
        shutil.rmtree(args.ckpt_dir, ignore_errors=True)
    dist.barrier()

    # --- Phase 1: train + save ---
    model, optim = build(model_name, mesh)
    losses1 = train(model, optim, batches, args.steps, start=0)
    if rank == 0:
        print("Phase 1 - train")
        for i, loss in enumerate(losses1, 1):
            print(f"  step {i:3d}: loss={loss:.4f}")

    save_checkpoint(model, optim, args.ckpt_dir)
    dist.barrier()

    # --- Phase 2: fresh build, load checkpoint, continue ---
    model2, optim2 = build(model_name, mesh)
    load_checkpoint(model2, optim2, args.ckpt_dir)

    # Continue training on both the original and resumed models
    losses_orig = train(model, optim, batches, args.steps, start=args.steps)
    losses_resumed = train(model2, optim2, batches, args.steps, start=args.steps)

    mx = max(abs(a - b) for a, b in zip(losses_orig, losses_resumed))
    passed = mx < args.threshold

    if rank == 0:
        print()
        print("Phase 2 - continued training (original vs resumed)")
        print(f"  {'step':<6} {'original':<12} {'resumed':<12} {'delta':<12}")
        for i, (a, b) in enumerate(
            zip(losses_orig, losses_resumed), args.steps + 1
        ):
            d = abs(a - b)
            flag = " <<<" if d > args.threshold else ""
            print(f"  {i:<6} {a:<12.4f} {b:<12.4f} {d:<12.6f}{flag}")
        print()
        print(f"  max delta: {mx:.6f}  {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(
                f"  FAIL: max loss delta {mx:.6f} exceeds threshold {args.threshold}"
            )

    dist.barrier()
    if rank == 0:
        shutil.rmtree(args.ckpt_dir, ignore_errors=True)
    dist.destroy_process_group()

    # Exit with non-zero code on failure (useful for CI)
    if rank == 0 and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
