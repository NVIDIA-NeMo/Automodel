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

"""Fold the adapters of a LoRA run into the weights and write a Praxis checkpoint.

The checkpointer saves a LoRA run as adapters (``adapter_model.safetensors`` next to the PEFT configs);
the reference runtime loads full checkpoints. This module merges the adapters into the linears they adapt
and writes the result in the checkpoint layout (``config.json``, ``model-XXXXX-of-NNNNN.safetensors``
shards, ``model.safetensors.index.json``, ``SHA256SUMS``)::

    python -m nemo_automodel.components.models.praxis.lora_export --checkpoint runwayml/praxis-1-droid \\
        --adapters runs/libero_spatial_lora/epoch_0_step_599/model --out runs/libero_spatial_lora/merged_600
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components.checkpoint.addons import load_automodel_peft_config_dict
from nemo_automodel.components.models.praxis.hub import resolve_repo
from nemo_automodel.components.models.praxis.model import PraxisPolicy, praxis_backend
from nemo_automodel.components.models.praxis.state_dict_adapter import PraxisStateDictAdapter
from nemo_automodel.shared.utils import dtype_from_str

ADAPTER_FILE = "adapter_model.safetensors"
MAX_SHARD_BYTES = 5 * 1024**3
_LORA_A = re.compile(r"^(?P<base>.+)\.lora_A\.weight$")


def _natural_key(name: str) -> list[int | str]:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


_EXPERT_LORA = {"lora_gate_and_up": "gate_and_up_projs", "lora_down": "down_projs"}
_EXPERT_LORA_A = re.compile(r"^(?P<base>.+)\.(?P<kind>lora_gate_and_up|lora_down)_A$")


def adapter_scales(model: nn.Module) -> dict[str, float]:
    """The scale ``alpha / rank`` of every adapter module of ``model``, by module name.

    Expert adapters may run at a smaller rank than the linear adapters (``moe_rank_scaling``), so the
    scale is read from each module rather than derived from the PEFT config.
    """
    return {
        name: module.scale
        for name, module in model.named_modules()
        if isinstance(module, (LinearLoRA, GroupedExpertsLoRA, GroupedExpertsDeepEPLoRA))
    }


def merge_lora(state: dict[str, Tensor], scales: Mapping[str, float]) -> dict[str, Tensor]:
    """Fold the adapters into the weights they adapt and drop the adapter tensors.

    Linear adapters are ``<m>.lora_A.weight`` of shape [rank, in] and ``<m>.lora_B.weight`` of shape
    [out, rank] next to ``<m>.weight``; grouped-expert adapters are ``<m>.lora_gate_and_up_{A,B}`` and
    ``<m>.lora_down_{A,B}`` of shapes [experts, in, rank] and [experts, rank, out] next to
    ``<m>.gate_and_up_projs`` and ``<m>.down_projs``.

    Args:
        state: Module state dict with the adapters.
        scales: The scale ``alpha / rank`` of each adapted module, by module name (:func:`adapter_scales`).

    Returns:
        A new state dict with ``weight + scale * B @ A`` (``scale * A @ B`` per expert) in the weight's dtype.
    """
    out: dict[str, Tensor] = {}
    linears: set[str] = set()
    experts: set[tuple[str, str]] = set()
    for key, value in state.items():
        m = _LORA_A.match(key)
        if m is not None:
            linears.add(m.group("base"))
            continue
        e = _EXPERT_LORA_A.match(key)
        if e is not None:
            experts.add((e.group("base"), e.group("kind")))
            continue
        if key.endswith((".lora_B.weight", "_B")) and (
            ".lora_B." in key or ".lora_gate_and_up_B" in key or ".lora_down_B" in key
        ):
            continue
        if ".lora_magnitude" in key:
            continue
        out[key] = value
    for base in sorted(linears):
        a = state[f"{base}.lora_A.weight"].float()
        b = state[f"{base}.lora_B.weight"].float()
        w = out[f"{base}.weight"]
        out[f"{base}.weight"] = (w.float() + scales[base] * (b @ a)).to(w.dtype)
    for base, kind in sorted(experts):
        a = state[f"{base}.{kind}_A"].float()
        b = state[f"{base}.{kind}_B"].float()
        name = f"{base}.{_EXPERT_LORA[kind]}"
        w = out[name]
        out[name] = (w.float() + scales[base] * torch.bmm(a, b)).to(w.dtype)
    return out


def load_adapters(policy: PraxisPolicy, model_dir: str | Path) -> dict[str, Tensor]:
    """The adapter tensors of a checkpointer's PEFT save, under the parameter names of ``policy``.

    Args:
        policy: The model with its ``LinearLoRA`` layers applied.
        model_dir: The ``model`` directory of the checkpoint step.

    Returns:
        Every ``lora_A`` / ``lora_B`` parameter of ``policy`` with its saved value.
    """
    names = set(policy.state_dict())
    out: dict[str, Tensor] = {}
    for key, value in load_file(str(Path(model_dir) / ADAPTER_FILE)).items():
        name = key.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
        for prefix in ("base_model.model.", "model."):
            if name not in names and name.startswith(prefix):
                name = name[len(prefix) :]
        if name not in names:
            raise KeyError(f"adapter tensor {key} matches no parameter of the model")
        out[name] = value
    missing = sorted(name for name in names if ".lora_" in name and name not in out)
    if missing:
        raise KeyError(f"{len(missing)} adapter parameters are not in {ADAPTER_FILE}, e.g. {missing[:3]}")
    return out


def save_checkpoint(
    path: str | Path,
    native_state: dict[str, Tensor],
    adapter: PraxisStateDictAdapter,
    source_config: str | Path,
    max_shard_bytes: int = MAX_SHARD_BYTES,
) -> None:
    """Write ``native_state`` as a Praxis checkpoint directory.

    Tensors are stored in the checkpoint dtype (``torch_dtype`` of the config), except the
    names matching ``fp32_tensors`` which stay float32.

    Args:
        path: Output directory (created; existing shards are replaced).
        native_state: Full module state dict (CPU tensors) in module key names.
        adapter: Converts module keys to checkpoint keys.
        source_config: ``config.json`` of the checkpoint the weights derive from; copied verbatim.
        max_shard_bytes: Shard size limit.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(source_config).read_text())
    store_dtype = dtype_from_str(str(config.get("torch_dtype", "bfloat16")), torch.bfloat16)
    fp32_patterns = [str(p) for p in config.get("fp32_tensors", ["*.ffn.expert_bias"])]
    state = adapter.to_hf({k: v.detach().cpu().contiguous() for k, v in native_state.items()})
    for key in list(state):
        target = torch.float32 if any(fnmatch.fnmatch(key, p) for p in fp32_patterns) else store_dtype
        state[key] = state[key].to(target).contiguous()
    names = sorted(state, key=_natural_key)
    shards: list[list[str]] = [[]]
    size = 0
    for name in names:
        nbytes = state[name].numel() * state[name].element_size()
        if shards[-1] and size + nbytes > max_shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(name)
        size += nbytes
    weight_map: dict[str, str] = {}
    total = 0
    for i, keys in enumerate(shards):
        shard = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: state[k] for k in keys}, str(root / shard), metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = shard
            total += state[k].numel() * state[k].element_size()
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1)
    )
    shutil.copyfile(source_config, root / "config.json")
    lines = []
    for name in ["config.json", "model.safetensors.index.json", *sorted(set(weight_map.values()))]:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def export_merged(checkpoint: str | Path, adapters: str | Path, out: str | Path) -> None:
    """Merge the adapters of a LoRA run into the base checkpoint and write the result to ``out``.

    Args:
        checkpoint: The base checkpoint directory or its Hugging Face repo id.
        adapters: The ``model`` directory of the LoRA run's checkpoint step.
        out: Output directory of the merged checkpoint.
    """
    root = resolve_repo(checkpoint)
    peft = PeftConfig(**load_automodel_peft_config_dict(str(adapters)))
    policy = PraxisPolicy.from_pretrained(root, backend=praxis_backend(), device="cpu")
    apply_lora_to_linear_modules(policy, peft)
    state = policy.state_dict()
    state.update(load_adapters(policy, adapters))
    save_checkpoint(out, merge_lora(state, adapter_scales(policy)), policy.state_dict_adapter, root / "config.json")


def main() -> int:
    """Command line: merge the adapters of a LoRA run into the base checkpoint and write the result."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="base checkpoint directory or Hugging Face repo id")
    ap.add_argument("--adapters", required=True, help="the LoRA run's <step>/model directory")
    ap.add_argument("--out", required=True, help="output directory of the merged checkpoint")
    args = ap.parse_args()
    export_merged(args.checkpoint, args.adapters, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
