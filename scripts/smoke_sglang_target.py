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

"""GPU smoke test for the SGLang EAGLE-3 target backend (not part of CI).

Run on the training server to validate that sglang 0.5.9's internal API really
lines up with the ported forward in ``sglang_runner.py`` (ModelRunner build,
``set_eagle3_layers_to_capture``, ``CaptureHiddenMode.FULL``, the private
``LogitsProcessor`` helpers). Two stages:

1. SGLang only: build the target, run ``generate_batch`` on a tiny batch, and
   assert shapes / dtype / finiteness of aux hidden states and logits.
2. ``--compare-hf``: also run the co-located HuggingFace backend on the SAME
   inputs and report how closely they agree (next-token argmax match rate over
   the loss positions + mean cosine similarity of the aux hidden states). The
   contract claims numerical equivalence, so these should be high; small gaps
   from kernel/dtype differences are expected, large gaps are a red flag.

``--init-dist`` initializes ``torch.distributed`` (single-process NCCL group)
BEFORE building SGLang, mirroring ``target_model_backend: sglang`` inside the
training recipe, where torchrun owns the process group and SGLang must attach
to it instead of creating its own. Run the smoke once without and once with
this flag to validate both the ``serve_target`` and the co-located paths.

Example (single GPU):
    HF_HOME=/llm-align/liuchonghan/hf_cache CUDA_VISIBLE_DEVICES=0 \
        python scripts/smoke_sglang_target.py --target /llm-align/open_models/Qwen3/Qwen3-4B --compare-hf --init-dist
"""

from __future__ import annotations

import argparse
import os

import torch


def _parse_args():
    p = argparse.ArgumentParser(description="SGLang EAGLE-3 target backend smoke test.")
    p.add_argument("--target", default="/llm-align/open_models/Qwen3/Qwen3-4B", help="Target model path.")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.4,
        help="SGLang KV pool fraction (lower leaves room for the HF compare).",
    )
    p.add_argument("--compare-hf", action="store_true", help="Also run the HF co-located backend and compare.")
    p.add_argument(
        "--init-dist", action="store_true", help="Pre-initialize torch.distributed like the training recipe does."
    )
    return p.parse_args()


def main():
    """Build the SGLang target, validate the supervision batch, optionally compare against HF."""
    args = _parse_args()
    assert torch.cuda.is_available(), "smoke test needs a GPU"
    device = torch.device("cuda")

    if args.init_dist:
        import torch.distributed as dist

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group(
            backend="nccl", world_size=1, rank=0, device_id=torch.device("cuda", torch.cuda.current_device())
        )
        print("[smoke] torch.distributed pre-initialized (world_size=1), as in the training recipe")

    from nemo_automodel.components.speculative.eagle.sglang_target import SGLangEagle3TargetModel

    print(f"[smoke] building SGLang target from {args.target} (tp_size={args.tp_size}) ...")
    sgl = SGLangEagle3TargetModel.from_pretrained(
        args.target,
        dtype=torch.bfloat16,
        tp_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
        mem_fraction_static=args.mem_fraction_static,
    )
    cfg = sgl.model.config
    hidden, vocab = cfg.hidden_size, cfg.vocab_size
    print(f"[smoke] aux_layer_ids={sgl.aux_layer_ids} hidden_size={hidden} vocab_size={vocab}")

    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (args.batch, args.seq), device=device)
    attention_mask = torch.ones(args.batch, args.seq, device=device, dtype=torch.long)
    loss_mask = torch.ones(args.batch, args.seq, device=device, dtype=torch.long)

    batch = sgl.generate_batch(input_ids, attention_mask, loss_mask)
    print(f"[smoke] aux {tuple(batch.aux_hidden_states.shape)} {batch.aux_hidden_states.dtype}")
    print(f"[smoke] logits {tuple(batch.logits.shape)} {batch.logits.dtype}")

    assert batch.aux_hidden_states.shape == (args.batch, args.seq, 3 * hidden), "aux shape mismatch"
    assert batch.logits.shape == (args.batch, args.seq, vocab), "logits shape mismatch"
    assert batch.target_probs is None and batch.position_mask is None, "expected full-logits encoding"
    assert torch.isfinite(batch.aux_hidden_states).all(), "aux has NaN/Inf"
    assert torch.isfinite(batch.logits).all(), "logits has NaN/Inf"
    print("[smoke] STAGE 1 OK: SGLang target produces well-formed supervision")

    if not args.compare_hf:
        print("[smoke] done (pass --compare-hf for the HF equivalence check)")
        return

    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
    from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel

    print("[smoke] building HF co-located target for comparison ...")
    hf_model = NeMoAutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code
    ).to(device)
    hf_model.requires_grad_(False)
    hf = HFEagle3TargetModel(hf_model, aux_layer_ids=sgl.aux_layer_ids)
    hf_batch = hf.generate_batch(input_ids, attention_mask, loss_mask)

    # Next-token argmax agreement over the supervised (shifted) positions.
    valid = hf_batch.loss_mask.bool()
    sgl_top = batch.logits.argmax(-1)[valid]
    hf_top = hf_batch.logits.argmax(-1)[valid]
    match = (sgl_top == hf_top).float().mean().item()
    # Mean cosine similarity of the aux hidden states.
    a = batch.aux_hidden_states.float().flatten(0, 1)
    b = hf_batch.aux_hidden_states.float().flatten(0, 1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()

    print(f"[smoke] logits argmax match rate (loss positions): {match:.4f}")
    print(f"[smoke] aux hidden-state mean cosine similarity:    {cos:.4f}")
    if match >= 0.95 and cos >= 0.99:
        print("[smoke] STAGE 2 OK: SGLang and HF backends agree")
    else:
        print("[smoke] STAGE 2 WARNING: agreement lower than expected; inspect aux-layer capture / shift")


if __name__ == "__main__":
    main()
