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
"""Runbook rung 9: load an exported checkpoint with plain ``transformers`` and sanity-check it.

Checks, on GPU(s):

* it loads as ``Qwen3_5MoeForConditionalGeneration`` with the base's parameter count,
* the frozen vision tower is bit-identical to the base,
* a sample of trained tensors actually differs from the base,
* greedy generation produces text.

Usage (inside the training container, with the base snapshot in ``$HF_HOME``)::

    python examples/vlm_finetune/qwen3_5_moe/affine/check_export.py exports/<name>
"""

import argparse
import glob
import json
import os

import torch
import transformers
from safetensors import safe_open

PROBE = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.0.mlp.experts.gate_up_proj",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "lm_head.weight",
)


def _load_named(snapshot: str, names: list[str]) -> dict[str, torch.Tensor]:
    """Read the named tensors from a safetensors snapshot without loading the whole model."""
    with open(os.path.join(snapshot, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    out = {}
    for name in names:
        with safe_open(os.path.join(snapshot, weight_map[name]), "pt") as f:
            out[name] = f.get_tensor(name)
    return out


def main() -> None:
    """Load the export, compare against the base, and generate once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir")
    parser.add_argument("--base", default=None, help="Base snapshot dir; defaults to the HF cache entry.")
    parser.add_argument("--prompt", default="In one sentence, what does `git rebase --onto` do?")
    args = parser.parse_args()
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    base = args.base or sorted(glob.glob(os.path.join(hf_home, "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/")))[0]

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.export_dir, dtype=torch.bfloat16, device_map="auto"
    )
    print(f"LOAD ok: {type(model).__name__} (transformers {transformers.__version__})", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B", flush=True)
    sd = model.state_dict()
    print(f"tensors in state_dict: {len(sd)}", flush=True)

    visual = [k for k in sd if k.startswith("model.visual.")]
    base_visual = _load_named(base, visual)
    same = sum(torch.equal(sd[k].cpu(), base_visual[k]) for k in visual)
    print(f"visual (frozen): {same}/{len(visual)} tensors bit-identical to base", flush=True)

    probe = [p for p in PROBE if p in sd]
    base_probe = _load_named(base, probe)
    for k in probe:
        diff = (sd[k].float().cpu() - base_probe[k].float()).abs()
        print(
            f"trained {k}: changed={not torch.equal(sd[k].cpu(), base_probe[k])} max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e}",
            flush=True,
        )

    tok = transformers.AutoTokenizer.from_pretrained(args.export_dir)
    # transformers 5 returns a BatchEncoding here; index it rather than treating it as a tensor.
    enc = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    ids = enc["input_ids"].to(model.device)
    out = model.generate(
        input_ids=ids, attention_mask=enc["attention_mask"].to(model.device), max_new_tokens=60, do_sample=False
    )
    print("gen:", repr(tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)), flush=True)
    ok = same == len(visual) and all(not torch.equal(sd[k].cpu(), base_probe[k]) for k in probe if "A_log" not in k)
    print("RUNG9_OK" if ok else "RUNG9_FAIL", flush=True)


if __name__ == "__main__":
    main()
