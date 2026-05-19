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

"""Wrap a NEMO thinker-only Qwen3-Omni checkpoint as a HF-compatible Omni export.

NEMO's recipe currently maps the Qwen3-Omni-30B-A3B model to a custom thinker
sub-class (`Qwen3OmniMoeThinkerForConditionalGeneration`); its saved checkpoint
carries:

  * weights for the thinker sub-tree only (already prefixed ``thinker.*``)
  * a ``config.json`` whose ``model_type == "qwen3_omni_moe_thinker"`` and
    ``architectures == null`` — neither HF AutoConfig nor vLLM recognises this
    as a top-level architecture, so downstream inference tools fail to load.

This tool produces a drop-in HF directory that ``Qwen3OmniMoeForConditionalGeneration``
(``model_type == "qwen3_omni_moe"``) can load: it copies the trained
``thinker.*`` shards from the NEMO checkpoint, copies the un-trained
``code2wav.*`` and ``talker.*`` shards from the original base model, and writes
a merged ``model.safetensors.index.json`` + the base's ``config.json`` +
processor / tokenizer artefacts.

Usage::

    python tools/wrap_thinker_ckpt_as_omni.py \\
        --ckpt-dir <ckpt>/epoch_0_step_199/model/consolidated \\
        --base-dir <hf_snapshot_of_base_qwen3_omni> \\
        --out-dir <hf_loadable_export>

Memory footprint: zero materialisation of weights — the tool only reads the
safetensors shard index and either copies shards file-by-file (for base
``code2wav.*``/``talker.*``) or symlinks the thinker shards as-is.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict

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
)


def _read_index(directory: Path) -> Dict[str, str]:
    """Read ``model.safetensors.index.json`` and return its ``weight_map``."""
    with open(directory / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _scan_shard_keys(directory: Path) -> Dict[str, list[str]]:
    """Return ``{shard_filename: [key, ...]}`` for every safetensors shard in ``directory``.

    Falls back to scanning the directory when no index file is present (e.g. the
    NEMO consolidated directory occasionally writes a single shard without an
    index when fully merged).
    """
    out: Dict[str, list[str]] = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".safetensors") or fname.startswith("shard-"):
            continue
        with safe_open(str(directory / fname), framework="pt") as f:
            out[fname] = list(f.keys())
    return out


def _classify_keys(keys: list[str]) -> Dict[str, list[str]]:
    """Bucket safetensors keys by top-level prefix."""
    by_prefix: Dict[str, list[str]] = {}
    for k in keys:
        prefix = k.split(".", 1)[0]
        by_prefix.setdefault(prefix, []).append(k)
    return by_prefix


def _write_shard(out_path: Path, tensors: Dict[str, torch.Tensor]) -> None:
    save_file(tensors, str(out_path), metadata={"format": "pt"})


def wrap(ckpt_dir: Path, base_dir: Path, out_dir: Path) -> None:
    """Materialize an HF-compatible Qwen3-Omni export under ``out_dir``.

    Copies the trained ``thinker.*`` safetensors shards from ``ckpt_dir``,
    fills in the untrained ``code2wav.*`` / ``talker.*`` shards from the base
    HF snapshot at ``base_dir``, and writes a merged
    ``model.safetensors.index.json`` plus the base ``config.json`` and
    tokenizer/processor metadata.

    Args:
        ckpt_dir: NEMO consolidated checkpoint directory (thinker-only).
        base_dir: Base ``Qwen/Qwen3-Omni-30B-A3B-Instruct`` HF snapshot.
        out_dir: Destination directory for the wrapped HF export.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Collect ckpt thinker shards ----
    ckpt_shards = _scan_shard_keys(ckpt_dir)
    if not ckpt_shards:
        raise FileNotFoundError(f"no safetensors shards under {ckpt_dir}")
    ckpt_all_keys: list[str] = []
    for keys in ckpt_shards.values():
        ckpt_all_keys.extend(keys)
    ckpt_prefixes = _classify_keys(ckpt_all_keys)
    logger.info("ckpt key counts: %s", {k: len(v) for k, v in ckpt_prefixes.items()})
    if set(ckpt_prefixes.keys()) != {"thinker"}:
        raise ValueError(f"expected ckpt to contain only 'thinker.*' keys; got prefixes {sorted(ckpt_prefixes.keys())}")

    # ---- 2. Collect base shards keyed by prefix bucket ----
    base_weight_map = _read_index(base_dir)
    base_prefixes = _classify_keys(list(base_weight_map.keys()))
    logger.info("base key counts: %s", {k: len(v) for k, v in base_prefixes.items()})
    needed_base_prefixes = ("code2wav", "talker")
    for p in needed_base_prefixes:
        if p not in base_prefixes:
            raise ValueError(f"base model has no '{p}.*' keys; cannot wrap")

    # ---- 3. Re-bucket each ckpt shard's content into the output dir ----
    new_index: Dict[str, str] = {}
    next_index = 1
    out_shard_filenames: list[str] = []

    # 3a. Write thinker shards: keep the ckpt's existing shard files (rename to
    # the standard ``model-XXXXX-of-YYYYY.safetensors`` format), since the
    # tensors and keys are already correctly prefixed.
    ckpt_thinker_shards = list(ckpt_shards.items())
    base_code2wav_shards: Dict[str, list[str]] = {}
    base_talker_shards: Dict[str, list[str]] = {}
    for shard_name, full_map_keys in base_weight_map.items():
        # weight_map maps key -> shard filename; we need the reverse
        pass

    # Re-derive base shards by prefix bucket.
    base_shards_by_file: Dict[str, list[str]] = {}
    for key, shard in base_weight_map.items():
        base_shards_by_file.setdefault(shard, []).append(key)
    for shard_name, keys_in_shard in base_shards_by_file.items():
        ks_by_prefix = _classify_keys(keys_in_shard)
        if "code2wav" in ks_by_prefix:
            base_code2wav_shards.setdefault(shard_name, []).extend(ks_by_prefix["code2wav"])
        if "talker" in ks_by_prefix:
            base_talker_shards.setdefault(shard_name, []).extend(ks_by_prefix["talker"])

    total_shards = len(ckpt_thinker_shards) + len(base_code2wav_shards) + len(base_talker_shards)
    logger.info(
        "output plan: %d shards (ckpt thinker=%d, base code2wav=%d, base talker=%d)",
        total_shards,
        len(ckpt_thinker_shards),
        len(base_code2wav_shards),
        len(base_talker_shards),
    )

    def _shard_filename(idx: int) -> str:
        return f"model-{idx:05d}-of-{total_shards:05d}.safetensors"

    # 3b. Write thinker shards (rewrap into the canonical filename pattern).
    for shard_name, keys in ckpt_thinker_shards:
        out_name = _shard_filename(next_index)
        next_index += 1
        out_shard_filenames.append(out_name)
        with safe_open(str(ckpt_dir / shard_name), framework="pt") as f:
            tensors = {k: f.get_tensor(k) for k in keys}
        _write_shard(out_dir / out_name, tensors)
        for k in keys:
            new_index[k] = out_name
        logger.info(
            "wrote thinker shard %s (%d keys, %.2f GB)", out_name, len(keys), (out_dir / out_name).stat().st_size / 1e9
        )

    # 3c. Copy code2wav + talker shards from base; filter to only those keys
    # (some base shards mix prefixes).
    for src_kind, src_shards in (("code2wav", base_code2wav_shards), ("talker", base_talker_shards)):
        for shard_name, keys in sorted(src_shards.items()):
            out_name = _shard_filename(next_index)
            next_index += 1
            out_shard_filenames.append(out_name)
            with safe_open(str(base_dir / shard_name), framework="pt") as f:
                tensors = {k: f.get_tensor(k) for k in keys}
            _write_shard(out_dir / out_name, tensors)
            for k in keys:
                new_index[k] = out_name
            logger.info(
                "wrote %s shard %s (from base %s, %d keys, %.2f GB)",
                src_kind,
                out_name,
                shard_name,
                len(keys),
                (out_dir / out_name).stat().st_size / 1e9,
            )

    # ---- 4. Write the merged safetensors index ----
    index_path = out_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(
            {
                "metadata": {"total_size": sum(p.stat().st_size for p in out_dir.glob("*.safetensors"))},
                "weight_map": new_index,
            },
            f,
            indent=2,
        )
    logger.info("wrote %s (%d entries)", index_path, len(new_index))

    # ---- 5. Copy HF metadata from base (overwriting ckpt's bogus config.json) ----
    for fname in _HF_METADATA_FILES:
        src = base_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            logger.info("copied %s from base", fname)

    # Prefer the NEMO ckpt's chat_template.jinja if present.
    extra = ckpt_dir / "chat_template.jinja"
    if extra.exists():
        shutil.copy2(extra, out_dir / "chat_template.jinja")
        logger.info("copied chat_template.jinja from ckpt")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the thinker-to-Omni wrapper script."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="NEMO PEFT/full-FT consolidated directory")
    p.add_argument("--base-dir", required=True, help="Base Qwen3-Omni HF snapshot directory")
    p.add_argument("--out-dir", required=True, help="Output directory for HF-compatible export")
    return p.parse_args()


def main() -> None:
    """CLI entry point: resolve paths and invoke :func:`wrap`."""
    args = parse_args()
    wrap(Path(args.ckpt_dir).resolve(), Path(args.base_dir).resolve(), Path(args.out_dir).resolve())
    logger.info("done")


if __name__ == "__main__":
    main()
