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

"""GPU smoke test for the vLLM EAGLE-3 target backend (not part of CI).

Run on the training server to validate that vLLM's ``extract_hidden_states``
path lines up with the supervision contract in ``vllm_runner.py`` (aux-layer
capture, final-hidden logits reconstruction). Two stages:

1. vLLM only: build the target, run ``generate_batch`` on a tiny batch, and
   assert shapes / dtype / finiteness of aux hidden states and logits.
2. ``--compare-hf``: also run the co-located HuggingFace backend on the SAME
   inputs and report how closely they agree (next-token argmax match rate over
   the loss positions + mean cosine similarity of the aux hidden states). The
   contract claims numerical equivalence, so these should be high; on real text
   the argmax match is ~0.98 and the aux cosine ~1.0 (small argmax flips at
   near-tie positions come from bf16 kernel differences).

vLLM v1 spawns its engine-core subprocess and re-imports this module, so the
work lives under ``if __name__ == "__main__"`` (and ``main``); importing it from
a subprocess must not build the engine.

Example (single GPU):
    VLLM_USE_MODELSCOPE=0 HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=0 \
        python scripts/smoke_vllm_target.py --target /path/to/Qwen3-4B --compare-hf
"""

from __future__ import annotations

import argparse

import torch


def _parse_args():
    p = argparse.ArgumentParser(description="vLLM EAGLE-3 target backend smoke test.")
    p.add_argument("--target", default="Qwen/Qwen3-4B", help="Target model path.")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=48)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM KV pool fraction (lower leaves room for the HF compare).",
    )
    p.add_argument("--compare-hf", action="store_true", help="Also run the HF co-located backend and compare.")
    return p.parse_args()


def main():
    """Build the vLLM target, validate the supervision batch, optionally compare against HF."""
    args = _parse_args()
    assert torch.cuda.is_available(), "smoke test needs a GPU"
    device = torch.device("cuda")

    from nemo_automodel.components.speculative.eagle.vllm_target import VLLMEagle3TargetModel

    print(f"[smoke] building vLLM target from {args.target} (tp_size={args.tp_size}) ...")
    vllm_target = VLLMEagle3TargetModel.from_pretrained(
        args.target,
        dtype=torch.bfloat16,
        tp_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    cfg = vllm_target.model.config
    hidden, vocab = cfg.hidden_size, cfg.vocab_size
    print(f"[smoke] aux_layer_ids={vllm_target.aux_layer_ids} hidden_size={hidden} vocab_size={vocab}")

    # Real, equal-length token ids (no padding): vLLM treats each row as one full
    # causal prefill, so all rows must share a length.
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (args.batch, args.seq), device=device)
    attention_mask = torch.ones(args.batch, args.seq, device=device, dtype=torch.long)
    loss_mask = torch.ones(args.batch, args.seq, device=device, dtype=torch.long)

    batch = vllm_target.generate_batch(input_ids, attention_mask, loss_mask)
    print(f"[smoke] aux {tuple(batch.aux_hidden_states.shape)} {batch.aux_hidden_states.dtype}")
    print(f"[smoke] logits {tuple(batch.logits.shape)} {batch.logits.dtype}")

    assert batch.aux_hidden_states.shape == (args.batch, args.seq, 3 * hidden), "aux shape mismatch"
    assert batch.logits.shape == (args.batch, args.seq, vocab), "logits shape mismatch"
    assert batch.target_probs is None and batch.position_mask is None, "expected full-logits encoding"
    assert torch.isfinite(batch.aux_hidden_states).all(), "aux has NaN/Inf"
    assert torch.isfinite(batch.logits).all(), "logits has NaN/Inf"
    print("[smoke] STAGE 1 OK: vLLM target produces well-formed supervision")

    if not args.compare_hf:
        print("[smoke] done (pass --compare-hf for the HF equivalence check)")
        return

    aux_layer_ids = vllm_target.aux_layer_ids
    vllm_target.close()

    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
    from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel

    print("[smoke] building HF co-located target for comparison ...")
    hf_model = NeMoAutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code
    ).to(device)
    hf_model.requires_grad_(False)
    hf = HFEagle3TargetModel(hf_model, aux_layer_ids=aux_layer_ids)
    hf_batch = hf.generate_batch(input_ids, attention_mask, loss_mask)

    # Next-token argmax agreement over the supervised (shifted) positions.
    valid = hf_batch.loss_mask.bool()
    vllm_top = batch.logits.argmax(-1)[valid]
    hf_top = hf_batch.logits.argmax(-1)[valid]
    match = (vllm_top == hf_top).float().mean().item()
    # Mean cosine similarity of the aux hidden states.
    a = batch.aux_hidden_states.float().flatten(0, 1)
    b = hf_batch.aux_hidden_states.float().flatten(0, 1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()

    print(f"[smoke] logits argmax match rate (loss positions): {match:.4f}")
    print(f"[smoke] aux hidden-state mean cosine similarity:    {cos:.4f}")
    if match >= 0.95 and cos >= 0.99:
        print("[smoke] STAGE 2 OK: vLLM and HF backends agree")
    else:
        print("[smoke] STAGE 2 WARNING: agreement lower than expected; inspect aux-layer capture / shift")


if __name__ == "__main__":
    main()
