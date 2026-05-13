#!/usr/bin/env python3
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

"""Stream-merge a NEMO PEFT LoRA adapter into a base HuggingFace model.

Companion to ``tools/merge_lora.py``. Where ``merge_lora.py`` materializes the
full base model in memory through HuggingFace ``peft.PeftModel`` and saves the
merged result, this tool walks the base model's safetensors shards one at a
time, applies any matching ``B @ A * (alpha / r)`` delta from the NEMO adapter
file, and writes a new shard with the same name. Memory footprint stays at
roughly one shard (~4-5 GiB) at a time, and the tool has no runtime dependency
on ``peft``.

The output directory is a drop-in HF model directory (config.json + sharded
safetensors + tokenizer/processor bundle) so it can be loaded by
``NeMoAutoModelForMultimodalLM.from_pretrained(<out_dir>)`` or any equivalent
HuggingFace ``AutoModel*`` class.

Adapter key naming
------------------

NEMO saves PEFT adapters with HuggingFace-PEFT-style key naming so that
downstream HF tools can read them. A typical key looks like::

    thinker.base_model.model.<INNER>.lora_{A,B}.weight

The ``base_model.model.`` segment is the HF-PEFT inner-model wrapper, and the
matching plain HF base-model weight lives at::

    thinker.<INNER>.weight

(For non-Qwen3-Omni models the ``thinker.`` prefix is absent; the
``base_model.model.`` strip still applies.)

Usage
-----

::

    python tools/merge_lora_streaming.py \\
        --base-dir <hf_snapshot_dir_of_base_model> \\
        --adapter-dir <nemo_peft_ckpt>/model \\
        --out-dir <output_dir> \\
        --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Set

import torch
from safetensors.torch import safe_open, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_HF_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors.index.json",
)


def adapter_key_to_base_key(adapter_key: str) -> str:
    """Translate a NEMO/HF-PEFT adapter key to its base-model key.

    Strips the HF-PEFT ``base_model.model.`` wrapper segment and the
    ``lora_{A,B}.weight`` suffix.

    Args:
        adapter_key: A key like
            ``thinker.base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``.

    Returns:
        The matching base-model key, e.g.
        ``thinker.model.layers.0.self_attn.q_proj.weight``.

    Raises:
        ValueError: If the key does not end in ``.lora_A.weight`` or
            ``.lora_B.weight``.
    """
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if adapter_key.endswith(suffix):
            stripped = adapter_key[: -len(suffix)] + ".weight"
            return stripped.replace("base_model.model.", "")
    raise ValueError(
        f"adapter key {adapter_key!r} does not end in '.lora_A.weight' or '.lora_B.weight'"
    )


def build_deltas(adapter_path: Path, *, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Materialize each LoRA ``B @ A * (alpha / r)`` delta keyed by base-model name.

    Args:
        adapter_path: NEMO PEFT checkpoint ``model/`` directory containing
            ``adapter_config.json`` and ``adapter_model.safetensors``.
        dtype: Target dtype for the materialized deltas. The matmul itself is
            performed in ``float32`` for numerical stability before casting.

    Returns:
        Dict mapping base-model key (e.g. ``thinker.model.layers.0.self_attn.q_proj.weight``)
        to a CPU tensor of shape ``[out_features, in_features]`` in ``dtype``.

    Raises:
        ValueError: When a ``lora_B`` key has no matching ``lora_A`` partner.
    """
    with open(adapter_path / "adapter_config.json") as f:
        cfg = json.load(f)
    alpha = float(cfg["lora_alpha"])
    r = float(cfg["r"])
    scale = alpha / r

    a_buffer: Dict[str, torch.Tensor] = {}
    deltas: Dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_path / "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith(".lora_A.weight"):
                a_buffer[k] = f.get_tensor(k)
        for k in keys:
            if not k.endswith(".lora_B.weight"):
                continue
            a_key = k[: -len(".lora_B.weight")] + ".lora_A.weight"
            if a_key not in a_buffer:
                raise ValueError(f"adapter missing lora_A pair for {k}")
            a = a_buffer[a_key].to(torch.float32)
            b = f.get_tensor(k).to(torch.float32)
            delta = (b @ a) * scale
            base_key = adapter_key_to_base_key(k)
            deltas[base_key] = delta.to(dtype)
    return deltas


def stream_merge_shards(
    base_dir: Path, adapter_path: Path, out_dir: Path, *, dtype: torch.dtype
) -> Set[str]:
    """Walk every base safetensors shard, apply matching adapter deltas, write outputs.

    Args:
        base_dir: HF snapshot directory containing ``model.safetensors.index.json``
            and the referenced shard files.
        adapter_path: NEMO PEFT adapter directory (see :func:`build_deltas`).
        out_dir: Output directory; created if missing.
        dtype: Output tensor dtype. The merge addition is performed in ``float32``
            and cast down at write time.

    Returns:
        The set of base-model keys that received an adapter delta. Useful for
        sanity-checks in callers.

    Raises:
        ValueError: When the adapter references base keys that do not exist in
            the base ``weight_map``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = build_deltas(adapter_path, dtype=dtype)
    logger.info("adapter deltas computed: %d tensors (alpha/r scale applied)", len(deltas))

    with open(base_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))

    merged_keys: Set[str] = set()
    for shard_name in shards:
        logger.info("processing %s", shard_name)
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(str(base_dir / shard_name), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k in deltas:
                    t = (t.to(torch.float32) + deltas[k].to(torch.float32)).to(dtype)
                    merged_keys.add(k)
                tensors[k] = t
        save_file(tensors, str(out_dir / shard_name), metadata={"format": "pt"})

    missing = set(deltas.keys()) - merged_keys
    if missing:
        raise ValueError(
            f"adapter referenced {len(missing)} base keys that do not exist in the index; "
            f"first 3 missing: {sorted(missing)[:3]}"
        )
    logger.info("merged %d base tensors with adapter deltas", len(merged_keys))
    return merged_keys


def copy_hf_metadata(base_dir: Path, adapter_path: Path, out_dir: Path) -> None:
    """Copy HF metadata files (config.json, tokenizer, processor, ...) from the base snapshot.

    The recipe sometimes writes a richer ``chat_template.jinja`` alongside the
    saved adapter (it matches the processor that produced training data); if
    present, it is preferred over any older ``chat_template.json`` from the
    base snapshot. Otherwise both files come from the base snapshot.
    """
    for fname in _HF_METADATA_FILES:
        src = base_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    extra = adapter_path / "chat_template.jinja"
    if extra.exists():
        shutil.copy2(extra, out_dir / "chat_template.jinja")


def merge_streaming(
    base_dir: Path | str, adapter_dir: Path | str, out_dir: Path | str, *, dtype: torch.dtype
) -> None:
    """End-to-end streaming merge: deltas + shard rewrite + metadata copy."""
    base = Path(base_dir).resolve()
    adapter = Path(adapter_dir).resolve()
    out = Path(out_dir).resolve()

    if not (base / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"base safetensors index missing under {base}")
    if not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"adapter file missing under {adapter}")

    stream_merge_shards(base, adapter, out, dtype=dtype)
    copy_hf_metadata(base, adapter, out)
    logger.info("wrote merged export to %s", out)


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True, help="HF snapshot dir of the base model")
    parser.add_argument("--adapter-dir", required=True, help="NEMO PEFT checkpoint model/ directory")
    parser.add_argument("--out-dir", required=True, help="Output directory for the merged export")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=list(_DTYPE_MAP.keys()),
        help="Output weight dtype (default: bfloat16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_streaming(args.base_dir, args.adapter_dir, args.out_dir, dtype=_DTYPE_MAP[args.dtype])


if __name__ == "__main__":
    main()
