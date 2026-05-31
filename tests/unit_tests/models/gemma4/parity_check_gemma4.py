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

"""Logit parity check: NeMoAutoModel (Gemma4) vs plain HuggingFace.

Loads google/gemma-4-E4B-it (or a local path) via both NeMoAutoModel and
transformers, runs the same text-only token sequence through both, and checks
that max |logit diff| is within --atol.

Usage (single GPU, text-only):
    python examples/vlm_finetune/gemma4/parity_check_gemma4.py \
        --hf-dir path/to/gemma-4-E4B-it/model \
        --atol 0.01

For bf16 (looser tolerance):
    python ... --bf16 --atol 1.0
"""

import argparse
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from nemo_automodel._transformers import NeMoAutoModelForImageTextToText


SEQ_LEN = 16


def _load_hf(model_path: str, dtype: torch.dtype, device: torch.device) -> AutoModelForImageTextToText:
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    return model.to(device).eval()


def _load_nemo(model_path: str, dtype: torch.dtype, device: torch.device) -> NeMoAutoModelForImageTextToText:
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
        use_liger_kernel=False,
        use_sdpa_patching=False,
    )
    return model.to(device).eval()


def _check_weights(hf_model: AutoModelForImageTextToText, nemo_model: NeMoAutoModelForImageTextToText) -> None:
    """Verify all shared parameters are bit-for-bit identical."""
    hf_sd = hf_model.state_dict()
    nemo_sd = nemo_model.state_dict()

    missing = set(hf_sd.keys()) - set(nemo_sd.keys())
    extra = set(nemo_sd.keys()) - set(hf_sd.keys())

    mismatched = []
    for key in hf_sd:
        if key not in nemo_sd:
            continue
        diff = (hf_sd[key].float() - nemo_sd[key].float()).abs().max().item()
        if diff > 0.0:
            mismatched.append((key, diff))

    print(f"  Weight check — missing={len(missing)}  extra={len(extra)}  mismatched={len(mismatched)}")
    if missing:
        print(f"    Missing keys (first 5): {list(missing)[:5]}")
    if extra:
        print(f"    Extra keys  (first 5): {list(extra)[:5]}")
    if mismatched:
        worst = sorted(mismatched, key=lambda x: -x[1])[:5]
        print(f"    Mismatched (worst 5): {worst}")
        print("  WARNING: weight mismatch detected — logit diff may be large")
    else:
        print("  Weight check PASSED (all shared weights identical)")


def _run_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma4 NeMo vs HF logit parity check")
    parser.add_argument("--hf-dir", default="google/gemma-4-E4B-it", help="HF model dir or hub ID")
    parser.add_argument("--atol", type=float, default=0.01, help="Max allowed |logit diff|")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (default: float32)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device(args.device)

    print("=" * 60)
    print("  Gemma4 parity check: NeMoAutoModel vs HuggingFace")
    print(f"  model  : {args.hf_dir}")
    print(f"  dtype  : {dtype}  device: {device}")
    print(f"  atol   : {args.atol}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Build a fixed token sequence (no tokenizer needed for parity check) #
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir)
    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, SEQ_LEN), device=device)

    # ------------------------------------------------------------------ #
    # Load models                                                          #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading HuggingFace model …")
    hf_model = _load_hf(args.hf_dir, dtype, device)

    print("[2/4] Loading NeMoAutoModel …")
    nemo_model = _load_nemo(args.hf_dir, dtype, device)

    # ------------------------------------------------------------------ #
    # Weight parity                                                        #
    # ------------------------------------------------------------------ #
    print("[3/4] Checking weights …")
    _check_weights(hf_model, nemo_model)

    # ------------------------------------------------------------------ #
    # Diagnostics: which class is actually loaded?                         #
    # ------------------------------------------------------------------ #
    print(f"  HF   model class : {type(hf_model).__name__}")
    print(f"  NeMo model class : {type(nemo_model).__name__}")
    # Check if forward method has been monkey-patched
    hf_fwd_code = getattr(hf_model.forward, "__func__", hf_model.forward).__code__
    nemo_fwd_code = getattr(nemo_model.forward, "__func__", nemo_model.forward).__code__
    print(f"  Same forward code: {hf_fwd_code is nemo_fwd_code}")
    # Check config differences that could affect numerics
    hf_cfg = getattr(hf_model.config, "text_config", hf_model.config)
    nemo_cfg = getattr(nemo_model.config, "text_config", nemo_model.config)
    for attr in ("use_cache", "attn_implementation", "sliding_window", "final_logit_softcapping"):
        hf_v = getattr(hf_cfg, attr, "N/A")
        nemo_v = getattr(nemo_cfg, attr, "N/A")
        marker = " ←DIFF" if hf_v != nemo_v else ""
        print(f"  config.{attr:35s}: hf={hf_v}  nemo={nemo_v}{marker}")
    # Verify HF model is deterministic (forward twice → should be identical)
    hf_logits2 = _run_forward(hf_model, input_ids)
    hf_self_diff = (hf_logits2.float() - _run_forward(hf_model, input_ids).float()).abs().max().item()
    print(f"  HF deterministic (should be 0.0): {hf_self_diff}")
    # Trace where the patched forward came from
    nemo_fwd = getattr(nemo_model.forward, "__func__", nemo_model.forward)
    print(f"  NeMo forward qualname : {nemo_fwd.__qualname__}")
    print(f"  NeMo forward module   : {getattr(nemo_fwd, '__module__', 'unknown')}")
    # Check top-level model and language_model forward
    for name, submod in [("language_model", getattr(nemo_model, "language_model", None)),
                          ("model",          getattr(nemo_model, "model", None))]:
        if submod is None:
            continue
        sf = getattr(submod.forward, "__func__", submod.forward)
        print(f"  NeMo {name}.forward qualname: {sf.__qualname__}")

    # ------------------------------------------------------------------ #
    # Forward-pass logit comparison                                        #
    # ------------------------------------------------------------------ #
    print("[4/4] Forward pass …")
    # Baseline: HF called directly
    hf_logits = _run_forward(hf_model, input_ids)

    # Test A: NeMo called normally
    nemo_logits = _run_forward(nemo_model, input_ids)

    # Diagnose: which language model backend does NeMo use?
    hf_lm  = getattr(getattr(hf_model,   "model", None), "language_model", None)
    nemo_lm = getattr(getattr(nemo_model, "model", None), "language_model", None)
    if hf_lm is not None and nemo_lm is not None:
        hf_lm_fwd   = getattr(hf_lm.forward,   "__func__", hf_lm.forward)
        nemo_lm_fwd = getattr(nemo_lm.forward,  "__func__", nemo_lm.forward)
        print(f"  HF   language_model class : {type(hf_lm).__name__}")
        print(f"  NeMo language_model class : {type(nemo_lm).__name__}  "
              f"(module={getattr(nemo_lm_fwd, '__module__', '?')})")
        print(f"  Same language_model code  : {hf_lm_fwd.__code__ is nemo_lm_fwd.__code__}")

        # Level-2 parity: compare text backbone outputs directly. Gemma4 needs
        # input_ids to build per-layer inputs, so do not pass inputs_embeds only.
        with torch.no_grad():
            hf_lm_out = hf_lm(input_ids=input_ids).last_hidden_state
            nemo_lm_out = nemo_lm(input_ids=input_ids).last_hidden_state
        lm_diff = (hf_lm_out.float() - nemo_lm_out.float()).abs()
        print(f"\n  Language-model hidden-state parity:")
        print(f"    max  |diff| : {lm_diff.max().item():.6f}")
        print(f"    mean |diff| : {lm_diff.mean().item():.6f}")
        if lm_diff.max().item() < 0.01:
            print("    --> language_model backbone MATCHES HF  ✓")
        else:
            print("    --> language_model backbone DIFFERS from HF  ✗")

        with torch.no_grad():
            hf_manual_logits = hf_model.lm_head(hf_lm_out)
            nemo_manual_logits = nemo_model.lm_head(nemo_lm_out)
            text_cfg = getattr(hf_model.config, "text_config", hf_model.config)
            if (softcap := getattr(text_cfg, "final_logit_softcapping", None)) is not None:
                hf_manual_logits = torch.tanh(hf_manual_logits / softcap) * softcap
                nemo_manual_logits = torch.tanh(nemo_manual_logits / softcap) * softcap
        manual_diff = (hf_manual_logits.float() - nemo_manual_logits.float()).abs()
        print(f"\n  Manual lm_head parity from language_model outputs:")
        print(f"    max  |diff| : {manual_diff.max().item():.6f}")
        print(f"    mean |diff| : {manual_diff.mean().item():.6f}")

    diff = (hf_logits.float() - nemo_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Show where the worst diffs are
    flat_diff = diff.view(-1)
    top5_vals, top5_idx = flat_diff.topk(5)
    vocab_size = hf_logits.shape[-1]
    print("\n  Top-5 worst positions (token_pos, vocab_idx):")
    for val, idx in zip(top5_vals.tolist(), top5_idx.tolist()):
        pos = idx // vocab_size
        vocab = idx % vocab_size
        hf_v = hf_logits[0, pos, vocab].item()
        nemo_v = nemo_logits[0, pos, vocab].item()
        print(f"    pos={pos:3d} vocab={vocab:6d}  hf={hf_v:8.3f}  nemo={nemo_v:8.3f}  diff={val:.4f}")

    print()
    print(f"  max  |diff| : {max_diff:.6f}")
    print(f"  mean |diff| : {mean_diff:.6f}")
    print(f"  atol        : {args.atol}")

    if max_diff <= args.atol:
        print(f"\n  --> PASSED  (max diff {max_diff:.6f} <= atol {args.atol})")
    else:
        print(f"\n  --> FAILED  (max diff {max_diff:.6f} > atol {args.atol})")
        sys.exit(1)


if __name__ == "__main__":
    main()
