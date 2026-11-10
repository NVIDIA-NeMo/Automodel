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

# Code provenance: the merge follows the reference fine-tuning merge script (W += (alpha / r) * B @ A in fp32,
# complete standalone checkpoint, shared-context flag removed). See serving_prompt.py for the provenance note.

"""Export a trained LoRA adapter as a complete, servable Deciffusion checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import safetensors.torch as _st

from nemo_automodel.components.models.deciffusion.scorer import SHARED_CONTEXT_FLAG


def checkpoint_fingerprint(checkpoint_dir: str | Path, *, window: int = 1 << 20) -> str:
    """Cheap content fingerprint of a local checkpoint: ``config.json`` plus size, head and tail bytes of every shard.

    Distinguishes checkpoints whose tensor names and shapes agree but whose weights differ, without reading every
    weight. Not a cryptographic proof of identical weights.

    Raises:
        FileNotFoundError: If the directory has no ``*.safetensors`` shard.
    """
    root = Path(checkpoint_dir)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{root} has no *.safetensors weights")
    digest = hashlib.sha256((root / "config.json").read_bytes() if (root / "config.json").exists() else b"")
    for shard in shards:
        size = shard.stat().st_size
        digest.update(f"{shard.name}|{size}".encode())
        with open(shard, "rb") as handle:
            digest.update(handle.read(window))
            handle.seek(max(0, size - window))
            digest.update(handle.read(window))
    return digest.hexdigest()


def strip_shared_context(checkpoint_dir: str | Path) -> bool:
    """Remove ``deciffusion_shared_context`` from ``config.json``.

    Fine-tuning here reads one question per sequence, never shared-context packing; with the flag left in place the
    server's automatic packing would read the fine-tuned model in a mode it was not trained for.

    Returns:
        ``True`` when the flag was present and removed.
    """
    path = Path(checkpoint_dir) / "config.json"
    if not path.exists():
        return False
    config = json.loads(path.read_text())
    if SHARED_CONTEXT_FLAG not in config:
        return False
    del config[SHARED_CONTEXT_FLAG]
    path.write_text(json.dumps(config, indent=2) + "\n")
    return True


def merge_lora_checkpoint(base_dir: str | Path, adapter_dir: str | Path, output_dir: str | Path) -> dict:
    """Merge a PEFT LoRA adapter into the base weights and write a complete checkpoint.

    Args:
        base_dir: Checkpoint the adapter was trained on.
        adapter_dir: Directory with ``adapter_config.json`` and ``adapter_model.safetensors``.
        output_dir: New directory (must not exist).

    Returns:
        Receipt ``{"base", "adapter", "lora_modules", "r", "alpha", "scale"}`` (also written as ``merge-receipt.json``).

    Raises:
        ValueError: For DoRA/rank-pattern adapters, adapter weights missing from the base, or shape mismatch.
        FileExistsError: If ``output_dir`` exists.
    """
    base_dir, adapter_dir, output_dir = Path(base_dir), Path(adapter_dir), Path(output_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("DoRA / rank_pattern / alpha_pattern adapters are not supported")
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(rank) if cfg.get("use_rslora") else alpha / rank
    adapter = _st.load_file(str(adapter_dir / "adapter_model.safetensors"))
    weights: dict = {}
    for shard in sorted(base_dir.glob("*.safetensors")):
        weights.update(_st.load_file(str(shard)))
    merged = 0
    for key, lora_a in adapter.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key[: -len(".lora_A.weight")]
        lora_b = adapter[module + ".lora_B.weight"]
        name = module.removeprefix("base_model.model.") + ".weight"
        if name not in weights:
            raise ValueError(f"{name} is not in the base checkpoint (adapter trained on a different base?)")
        delta = (lora_b.float() @ lora_a.float()) * scale
        if delta.shape != weights[name].shape:
            raise ValueError(f"{name}: adapter delta {tuple(delta.shape)} != base {tuple(weights[name].shape)}")
        weights[name] = (weights[name].float() + delta).to(weights[name].dtype)
        merged += 1
    if merged == 0:
        raise ValueError("the adapter contains no LoRA weights")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    shutil.copytree(
        base_dir,
        output_dir,
        ignore=shutil.ignore_patterns("*.safetensors", "*.safetensors.index.json", ".cache", "merge-receipt.json"),
    )
    _st.save_file(
        {k: v.contiguous() for k, v in weights.items()},
        str(output_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )
    strip_shared_context(output_dir)
    receipt = {
        "base": str(base_dir),
        "adapter": str(adapter_dir),
        "lora_modules": merged,
        "r": rank,
        "alpha": alpha,
        "scale": scale,
    }
    (output_dir / "merge-receipt.json").write_text(json.dumps(receipt))
    return receipt


def export_full_checkpoint(base_dir: str | Path, weights_dir: str | Path, output_dir: str | Path) -> dict:
    """Write a complete checkpoint from consolidated full-fine-tuning weights, cast to the base dtypes.

    Args:
        base_dir: Checkpoint the run started from (configuration, tokenizer and model code are copied from it).
        weights_dir: Directory of consolidated ``*.safetensors`` weights of the trained model.
        output_dir: New directory (must not exist).

    Returns:
        Receipt ``{"base", "weights", "tensors"}`` (also written as ``export-receipt.json``).

    Raises:
        ValueError: If the trained tensor names or shapes differ from the base.
        FileExistsError: If ``output_dir`` exists.
    """
    base_dir, weights_dir, output_dir = Path(base_dir), Path(weights_dir), Path(output_dir)
    base: dict = {}
    for shard in sorted(base_dir.glob("*.safetensors")):
        base.update(_st.load_file(str(shard)))
    trained: dict = {}
    for shard in sorted(weights_dir.glob("*.safetensors")):
        trained.update(_st.load_file(str(shard)))
    if set(trained) != set(base):
        missing, extra = sorted(set(base) - set(trained))[:5], sorted(set(trained) - set(base))[:5]
        raise ValueError(f"trained weights do not match the base tensors (missing {missing}, unexpected {extra})")
    for name, tensor in trained.items():
        if tensor.shape != base[name].shape:
            raise ValueError(f"{name}: trained shape {tuple(tensor.shape)} != base {tuple(base[name].shape)}")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    shutil.copytree(
        base_dir,
        output_dir,
        ignore=shutil.ignore_patterns("*.safetensors", "*.safetensors.index.json", ".cache", "merge-receipt.json"),
    )
    _st.save_file(
        {k: v.to(base[k].dtype).contiguous() for k, v in trained.items()},
        str(output_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )
    strip_shared_context(output_dir)
    receipt = {"base": str(base_dir), "weights": str(weights_dir), "tensors": len(trained)}
    (output_dir / "export-receipt.json").write_text(json.dumps(receipt))
    return receipt
