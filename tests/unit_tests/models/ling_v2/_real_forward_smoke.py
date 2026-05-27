#!/usr/bin/env python
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

"""Single-GPU real-checkpoint forward smoke test for BailingMoeV2.

Strategy: load weights on CPU first (the framework's expert-tensor stacking
needs ~2x peak working memory which OOMs an 80 GB GPU for the 16B-A1.4B Mini
when load+stack happen on-device), assemble the full NeMo model in CPU RAM,
then move to GPU for inference.  Verifies that the real checkpoint loads
without missing/unexpected keys and that the forward pass produces finite,
non-degenerate logits.

Run inside the dev container::

    cd /work && HF_HOME=/work/hf_cache python \
        tests/unit_tests/models/ling_v2/_real_forward_smoke.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="inclusionAI/Ling-mini-2.0")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--out-device", default="cuda:0")
    args = parser.parse_args(argv)

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
    from nemo_automodel.components.models.ling_v2.model import BailingMoeV2ForCausalLM
    from nemo_automodel.components.models.ling_v2.state_dict_adapter import BailingMoeV2StateDictAdapter
    from nemo_automodel.components.moe.config import MoEConfig

    t0 = time.time()
    cfg = BailingMoeV2Config.from_pretrained(args.hf_model)
    print(
        f"config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"experts={cfg.num_experts} first_k_dense={cfg.first_k_dense_replace} "
        f"partial_rotary={cfg.partial_rotary_factor}"
    )

    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
        rope_fusion=False,
    )

    # Construct the NeMo model directly on CPU.  We must NOT go through
    # NeMoAutoModelForCausalLM.from_pretrained because that pulls in the full
    # infrastructure layer (FSDP2 manager, mesh, etc.) and lands tensors on GPU
    # early, which triggers the expert-stack OOM on a single 80 GB device.
    print("\nbuilding empty NeMo model on CPU ...")
    moe_cfg = MoEConfig(
        dim=cfg.hidden_size,
        inter_dim=cfg.intermediate_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_routed_experts=cfg.num_experts,
        n_shared_experts=cfg.num_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        n_expert_groups=cfg.n_group,
        n_limited_groups=cfg.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.0,
        force_e_score_correction_bias=bool(cfg.moe_router_enable_expert_bias),
        score_func=cfg.score_function,
        route_scale=cfg.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=cfg.norm_topk_prob,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        shared_expert_inter_dim=cfg.moe_intermediate_size,
        shared_expert_activation="swiglu",
        softmax_before_topk=False,
        dtype=torch.bfloat16,
    )
    model = BailingMoeV2ForCausalLM(cfg, moe_config=moe_cfg, backend=backend)
    model = model.to(dtype=torch.bfloat16)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e9:.2f} B")

    print("\nloading + grouping HF safetensors ...")
    ckpt_dir = snapshot_download(args.hf_model, allow_patterns=["*.safetensors", "*.json"])
    shards = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    hf_sd: dict[str, torch.Tensor] = {}
    for s in shards:
        hf_sd.update(load_file(s, device="cpu"))
    print(f"  {len(hf_sd)} HF tensors")

    adapter = BailingMoeV2StateDictAdapter(cfg, moe_cfg, backend, dtype=torch.bfloat16)
    native_sd = adapter.from_hf(hf_sd, device_mesh=None)
    print(f"  {len(native_sd)} native tensors after grouping")
    del hf_sd

    missing, unexpected = model.load_state_dict(native_sd, strict=False)
    print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    missing example: {missing[:3]}")
    if unexpected:
        print(f"    unexpected example: {unexpected[:3]}")

    print(f"\nmoving model to {args.out_device} ...")
    model = model.to(args.out_device).eval()
    print(f"GPU mem after move: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("\nforward pass ...")
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=args.out_device)
    with torch.no_grad():
        logits = model(input_ids)
    logits = logits.float()

    finite = torch.isfinite(logits).all().item()
    log_sm = torch.log_softmax(logits, dim=-1)
    avg_neg_log_p = -log_sm.mean().item()
    top1 = logits.argmax(dim=-1)
    top1_unique = top1.unique().numel()

    elapsed = time.time() - t0
    print(f"\nlogits: shape={tuple(logits.shape)} dtype={logits.dtype}")
    print(f"  finite={finite}  avg(-log p)={avg_neg_log_p:.3f}  top1 unique={top1_unique}")
    print(f"  argmax sample (first 10): {top1[0, :10].tolist()}")
    print(f"\ndone in {elapsed:.1f}s")

    ok = finite and missing == [] and unexpected == [] and top1_unique > 1
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
