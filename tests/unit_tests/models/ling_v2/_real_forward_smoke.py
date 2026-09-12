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
without missing/unexpected keys, that the model-owned router precision policy
(Param / Proj / Score / Out) survives a real load and cast, and that the forward
pass produces finite, non-degenerate logits.

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
    from nemo_automodel.components.models.common.utils import cast_model_to_dtype
    from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
    from nemo_automodel.components.models.ling_v2.model import BailingMoeV2ForCausalLM
    from nemo_automodel.components.models.ling_v2.state_dict_adapter import BailingMoeV2StateDictAdapter

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

    # Build directly on CPU: NeMoAutoModelForCausalLM.from_pretrained pulls in the
    # infrastructure layer (FSDP2 manager, mesh) and lands tensors on GPU early,
    # which OOMs the expert stack on a single 80 GB device.
    #
    # No moe_config on purpose: BailingMoeV2Model resolves
    # ``moe_config or MoEConfig(**moe_defaults)``, so passing one bypasses
    # moe_defaults and silently restores router_weights_fp32=False -- the bug this
    # run exists to disprove. The adapter reuses the model's resolved config below.
    print("\nbuilding empty NeMo model on CPU ...")
    model = BailingMoeV2ForCausalLM(cfg, backend=backend)
    # Not model.to(dtype=...): raw .to() casts every float buffer and would demote
    # e_score_correction_bias, defeating _keep_in_fp32_modules_strict.
    cast_model_to_dtype(model, torch.bfloat16)
    moe_cfg = model.model.moe_config

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

    # Router precision policy on the real checkpoint's own gate, after a real
    # load. tests/unit_tests/models/test_ling_v2_gate_precision_policy.py pins the
    # same four stages on tiny configs; this is the checkpoint-side counterpart.
    first_moe_layer = str(cfg.first_k_dense_replace)
    gate = model.model.layers[first_moe_layer].mlp.gate
    router_stages = {
        "Param(weight)": gate.weight.dtype is torch.bfloat16,
        "Param(bias)": gate.e_score_correction_bias.dtype is torch.float32,
        "Proj": gate.gate_precision is torch.float32,
        "Score": gate.score_dtype is torch.float32,
        "Out": bool(gate.router_weights_fp32),
    }
    router_ok = all(router_stages.values())
    print(f"\nrouter policy @ layer {first_moe_layer} (first MoE layer):")
    print(
        f"  weight={gate.weight.dtype} bias={gate.e_score_correction_bias.dtype} "
        f"proj={gate.gate_precision} score={gate.score_dtype} out_fp32={gate.router_weights_fp32}"
    )
    if not router_ok:
        print(f"  FAILED stages: {[k for k, v in router_stages.items() if not v]}")

    print(f"\nmoving model to {args.out_device} ...")
    model = model.to(args.out_device).eval()
    print(f"GPU mem after move: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("\nforward pass ...")
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=args.out_device)
    with torch.no_grad():
        logits = model(input_ids).logits
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

    ok = finite and missing == [] and unexpected == [] and top1_unique > 1 and router_ok
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
