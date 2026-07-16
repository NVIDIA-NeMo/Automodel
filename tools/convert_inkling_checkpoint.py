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

"""Streaming conversion of a raw Thinking-Machines Inkling checkpoint to HF-module layout.

The published ``thinkingmachines/Inkling`` checkpoint stores weights in the raw
SGLang layout (``model.llm.*``, ``attn.wq_du``, ``mlp.experts.w13_weight`` with
interleaved gate/up, ``unembed``, ...). NeMo AutoModel's DCP loader loads routed
experts in-place through non-contiguous strided views, which requires the
contiguous HF-module layout (``model.language_model.*.mlp.experts.gate_up_proj``);
the raw interleaved gate/up cannot be a writable view. This tool applies the
exact transformers ``inkling_mm_model`` conversion (key renames + Interleave/Chunk
of the fused gate/up) one safetensors shard at a time, so a 1.9 TB checkpoint is
converted without ever materializing the full model in memory. Each input shard
produces one output shard; a ``model.safetensors.index.json`` is written at the end.

Usage:
    python tools/convert_inkling_checkpoint.py --src <raw_ckpt_dir> --dst <out_dir> \
        [--num-layers N]   # keep only the first N decoder layers (for small test runs)
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Ordered (raw-substring -> module-substring) renames, mirroring transformers
# conversion_mapping.py "inkling_mm_model". Applied as sequential re.sub. The three
# fused gate/up tensors (experts w13_weight, shared_w13_weight, dense w13_dn) are
# NOT renamed here; they are handled by _convert_fused below.
_RENAMES: list[tuple[str, str]] = [
    (r"model\.llm\.layers", "model.language_model.layers"),
    (r"model\.llm\.embed_norm\.weight", "model.language_model.embed_norm.weight"),
    (r"model\.llm\.embed\.weight", "model.language_model.embed_tokens.weight"),
    (r"model\.llm\.norm\.weight", "model.language_model.norm.weight"),
    (r"model\.llm\.unembed\.weight", "lm_head.weight"),
    (r"model\.audio\.", "model.audio_tower."),
    (r"model\.visual", "model.vision_tower"),
    (r"vision_tower\.layers\.linear_(\d+)", r"vision_tower.encoder_layers.\1.projection"),
    (r"vision_tower\.layers\.norm_(\d+)", r"vision_tower.encoder_layers.\1.layer_norm"),
    (r"audio_tower\.encoder\.weight", "audio_tower.embed_audio_tokens.embed_audio_tokens.weight"),
    (r"audio_tower\.final_norm\.weight", "audio_tower.norm.weight"),
    (r"mlp\.experts\.w2_weight", "mlp.experts.down_proj"),
    (r"shared_w2_weight", "down_proj"),
    (r"mlp\.w2_md\.weight", "mlp.down_proj.weight"),
    (r"mlp\.gate\.bias", "mlp.gate.e_score_correction_bias"),
    (r"attn\.wq_du", "self_attn.q_proj"),
    (r"attn\.wk_dv", "self_attn.k_proj"),
    (r"attn\.wv_dv", "self_attn.v_proj"),
    (r"attn\.wr_du", "self_attn.r_proj"),
    (r"attn\.wo_ud", "self_attn.o_proj"),
    (r"\.attn\.q_norm", ".self_attn.q_norm"),
    (r"\.attn\.k_norm", ".self_attn.k_norm"),
    (r"\.attn\.k_sconv", ".self_attn.k_sconv.conv1d"),
    (r"\.attn\.v_sconv", ".self_attn.v_sconv.conv1d"),
    (r"\.attn\.rel_logits_proj", ".self_attn.rel_logits_proj"),
    (r"attn_sconv\.weight$", "attn_sconv.conv1d.weight"),
    (r"mlp_sconv\.weight$", "mlp_sconv.conv1d.weight"),
    (r"mlp_norm", "post_attention_layernorm"),
    (r"attn_norm", "input_layernorm"),
]

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _rename(key: str) -> str:
    for pat, repl in _RENAMES:
        key = re.sub(pat, repl, key)
    return key


def _interleave(t: torch.Tensor, dim: int) -> torch.Tensor:
    """De-interleave fused gate/up along ``dim`` (transformers Interleave, forward).

    Raw stores [g0, u0, g1, u1, ...]; this returns the contiguous [g0..gN, u0..uN].
    """
    shape = list(t.shape)
    shape[dim : dim + 1] = [shape[dim] // 2, 2]
    return t.reshape(shape).transpose(dim, dim + 1).reshape(t.shape).contiguous()


def _convert_fused(module_key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Handle the three fused gate/up tensors; return [] if ``module_key`` is not one."""
    if module_key.endswith(".mlp.experts.w13_weight"):
        mk = module_key.replace(".mlp.experts.w13_weight", ".mlp.experts.gate_up_proj")
        return [(mk, _interleave(tensor, dim=1))]
    if module_key.endswith(".mlp.shared_experts.shared_w13_weight"):
        base = module_key.replace(".mlp.shared_experts.shared_w13_weight", ".mlp.shared_experts.")
        gate, up = _interleave(tensor, dim=1).chunk(2, dim=1)
        return [(base + "gate_proj", gate.contiguous()), (base + "up_proj", up.contiguous())]
    if module_key.endswith(".mlp.w13_dn.weight"):
        base = module_key.replace(".mlp.w13_dn.weight", ".mlp.")
        gate, up = _interleave(tensor, dim=0).chunk(2, dim=0)
        return [(base + "gate_proj.weight", gate.contiguous()), (base + "up_proj.weight", up.contiguous())]
    return []


def convert_raw_tensor(raw_key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Convert a single raw (key, tensor) to one or two HF-module (key, tensor) pairs."""
    module_key = _rename(raw_key)
    fused = _convert_fused(module_key, tensor)
    return fused if fused else [(module_key, tensor)]


def _keep_key(raw_key: str, num_layers: int | None) -> bool:
    if raw_key.startswith("model.mtp."):
        return False  # MTP head is not instantiated by the model class
    if num_layers is None:
        return True
    m = _LAYER_RE.search(raw_key)
    if m is None:
        return True  # non-layer key (embed / norm / unembed / vision / audio)
    return int(m.group(1)) < num_layers


def main() -> None:
    """Stream-convert a raw Inkling checkpoint to HF-module layout, one shard at a time."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw Inkling checkpoint dir")
    ap.add_argument("--dst", required=True, help="output HF-module checkpoint dir")
    ap.add_argument("--num-layers", type=int, default=None, help="keep only the first N decoder layers")
    ap.add_argument("--pad-token", default="<|endoftext|>", help="pad/eos token to inject (Inkling has none)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.src, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Group required raw keys by source shard.
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if _keep_key(key, args.num_layers):
            shard_to_keys.setdefault(shard, []).append(key)

    module_weight_map: dict[str, str] = {}
    total_bytes = 0
    shards = sorted(shard_to_keys)
    for i, shard in enumerate(shards):
        out_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(os.path.join(args.src, shard), framework="pt", device="cpu") as f:
            for raw_key in shard_to_keys[shard]:
                tensor = f.get_tensor(raw_key)
                for mk, mt in convert_raw_tensor(raw_key, tensor):
                    out_tensors[mk] = mt
                    module_weight_map[mk] = out_name
                    total_bytes += mt.numel() * mt.element_size()
        save_file(out_tensors, os.path.join(args.dst, out_name), metadata={"format": "pt"})
        print(f"[{i + 1}/{len(shards)}] {shard} -> {out_name} ({len(out_tensors)} tensors)", flush=True)
        del out_tensors

    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": module_weight_map}, f, indent=2)

    # Config + processor files.
    for fn in os.listdir(args.src):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = os.path.join(args.src, fn)
        dst = os.path.join(args.dst, fn)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)

    # Truncate the config's layer count for a subset conversion.
    cfg_path = os.path.join(args.dst, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if args.num_layers is not None:
        cfg.setdefault("text_config", {})["num_hidden_layers"] = args.num_layers
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Inject a pad/eos token (Inkling's tokenizer ships without one).
    tok_cfg_path = os.path.join(args.dst, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path) and args.pad_token:
        with open(tok_cfg_path) as f:
            tok_cfg = json.load(f)
        tok_cfg["pad_token"] = args.pad_token
        tok_cfg["eos_token"] = args.pad_token
        with open(tok_cfg_path, "w") as f:
            json.dump(tok_cfg, f, indent=2)

    print(
        f"Done. {len(module_weight_map)} module tensors, {total_bytes / 1e9:.1f} GB, {len(shards)} shards -> {args.dst}"
    )


if __name__ == "__main__":
    main()
