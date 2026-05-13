#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Standalone 2-GPU FSDP2 functional test for Gemma4 + MTP.

What this validates that single-GPU unit tests cannot:

1. ``Gemma4WithMTPCausalLMOutput`` survives FSDP2's mixed-precision output
   reconstruction. FSDP2 rebuilds ``ModelOutput`` instances from declared
   dataclass fields only — auxiliary fields tacked on with setattr() get
   silently dropped (the regression behind NemotronV3 PR #2161 commit
   0b2889ab). On a single GPU there is no output cast, so this invariant is
   only exercised under FSDP2.
2. The MTP head's parameters reduce-scatter correctly through FSDP2: the
   per-depth CE loss must be finite on every rank and gradients must be
   non-zero on at least one MTP parameter after backward.
3. The recipe-equivalent path (model forward → main CE + ``calculate_mtp_loss``
   → backward → optimizer step) runs end to end without raising.

Construction is intentionally minimal: a dense (non-MoE) tiny Gemma4 backbone
with two transformer layers, two MTP depths, and a small vocab. We wrap
every leaf decoder layer + every MTP sublayer with ``fully_shard`` and then
the root model, matching the FSDP2 wrap pattern the production parallelizer
applies.

Usage:
    torchrun --nproc_per_node=2 \\
        tests/functional_tests/vlm_mtp/run_gemma4_mtp_fsdp2.py
"""

from __future__ import annotations

import dataclasses
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def _init_distributed() -> tuple[int, int, torch.device]:
    """Initialize NCCL process group and bind device.

    Returns:
        ``(rank, world_size, device)``.
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this functional test")
    if not dist.is_initialized():
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "This script must be launched via torchrun with --nproc_per_node>=2; RANK/WORLD_SIZE not in env."
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return rank, world, device


def _build_tiny_dense_gemma4_with_mtp():
    """Build a tiny dense Gemma4 with MTP enabled, on ``meta`` device first.

    Returns ``(model, text_config)``.
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

    text_cfg = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        enable_moe_block=False,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=16,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="bfloat16",
    )
    cfg = Gemma4Config(text_config=text_cfg)
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = Gemma4ForConditionalGeneration.from_config(
        cfg,
        backend=backend,
        mtp_num_layers=2,
        mtp_layer_pattern="*",
        mtp_loss_scaling_factor=0.3,
    )
    return model, text_cfg


def _shard_model(model: nn.Module, mesh) -> None:
    """Apply ``fully_shard`` to MTP sublayers, backbone decoder layers, and root.

    We wrap leaves first then the root, mirroring the order ``apply_fsdp2``
    uses for VLM models in the production parallelizer.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    if model.mtp is not None:
        for sl in model.mtp.layers:
            fully_shard(sl, mesh=mesh, mp_policy=mp_policy)

    text_backbone = model.model.language_model
    layers = getattr(text_backbone, "layers", None)
    if layers is not None:
        for layer in layers.values() if isinstance(layers, nn.ModuleDict) else layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)

    fully_shard(model, mesh=mesh, mp_policy=mp_policy)


def main() -> int:
    rank, world, device = _init_distributed()
    if world < 2:
        if rank == 0:
            print("SKIP: this test requires at least 2 GPUs", file=sys.stderr)
        return 0

    torch.manual_seed(1234 + rank)
    torch.cuda.manual_seed_all(1234)

    model, text_cfg = _build_tiny_dense_gemma4_with_mtp()
    model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    model = model.to(device)
    model.train()

    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    _shard_model(model, mesh)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    B, S = 2, 8
    input_ids = torch.randint(0, text_cfg.vocab_size, (B, S), device=device)
    labels = input_ids.clone()

    out = model(input_ids=input_ids)

    from nemo_automodel.components.models.gemma4_moe.model import Gemma4WithMTPCausalLMOutput

    if not isinstance(out, Gemma4WithMTPCausalLMOutput):
        if rank == 0:
            print(f"FAIL: forward returned {type(out).__name__}, expected Gemma4WithMTPCausalLMOutput", file=sys.stderr)
        return 1

    declared = {f.name for f in dataclasses.fields(Gemma4WithMTPCausalLMOutput)}
    if not {"mtp_per_depth_h", "mtp_loss_scaling_factor"}.issubset(declared):
        if rank == 0:
            print(
                "FAIL: Gemma4WithMTPCausalLMOutput missing required dataclass fields; "
                "FSDP2 will drop them on the way out.",
                file=sys.stderr,
            )
        return 1

    if out.mtp_per_depth_h is None:
        if rank == 0:
            print(
                "FAIL: mtp_per_depth_h was dropped by FSDP2's output cast — "
                "verify the field is declared on Gemma4WithMTPCausalLMOutput.",
                file=sys.stderr,
            )
        return 1
    if len(out.mtp_per_depth_h) != 2:
        if rank == 0:
            print(f"FAIL: expected 2 MTP depths, got {len(out.mtp_per_depth_h)}", file=sys.stderr)
        return 1
    for k, h in enumerate(out.mtp_per_depth_h):
        if tuple(h.shape) != (B, S, text_cfg.hidden_size):
            if rank == 0:
                print(
                    f"FAIL: MTP depth {k} hidden state shape {tuple(h.shape)} != ({B}, {S}, {text_cfg.hidden_size})",
                    file=sys.stderr,
                )
            return 1

    # ------------------------------------------------------------------
    # Loss + backward + optimizer step
    # ------------------------------------------------------------------
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
    from nemo_automodel.recipes.vlm.finetune import calculate_mtp_loss

    loss_fn = MaskedCrossEntropy()
    num_tokens = int((labels != -100).sum().item())

    logits = out.logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    main_loss = loss_fn(logits=logits, labels=shifted_labels, num_label_tokens=num_tokens)

    mtp_loss = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_h=out.mtp_per_depth_h,
        labels=labels,
        model=model,
        scaling_factor=out.mtp_loss_scaling_factor,
        num_label_tokens=num_tokens,
    )
    total = main_loss + mtp_loss
    if not torch.isfinite(total):
        if rank == 0:
            print(f"FAIL: non-finite combined loss: main={main_loss.item()} mtp={mtp_loss.item()}", file=sys.stderr)
        return 1

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    total.backward()

    # Confirm at least one MTP parameter received a non-trivial gradient
    # after FSDP2's reduce-scatter on this rank.
    saw_mtp_grad = False
    for name, p in model.named_parameters():
        if not (name.startswith("mtp.") or ".mtp." in name):
            continue
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            continue
        # FSDP2 may return DTensor grads; pull a local shard if so.
        local = g.to_local() if hasattr(g, "to_local") else g
        if torch.isfinite(local).all() and local.abs().sum().item() > 0:
            saw_mtp_grad = True
            break

    saw_local = torch.tensor([1 if saw_mtp_grad else 0], device=device, dtype=torch.long)
    dist.all_reduce(saw_local, op=dist.ReduceOp.MAX)
    if saw_local.item() == 0:
        if rank == 0:
            print("FAIL: no rank observed a non-zero MTP-parameter gradient", file=sys.stderr)
        return 1

    optimizer.step()

    if rank == 0:
        print(
            "PASS: Gemma4 + MTP forward/backward under FSDP2 (2 GPUs) — "
            f"main_loss={main_loss.item():.4f}, mtp_loss={mtp_loss.item():.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
