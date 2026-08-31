# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a few-layer HY V4 checkpoint for reference-logit parity or training smoke tests.

The public checkpoint is too large for a single-GPU reference process. This
utility copies the real embedding, selected decoder layers, final norm, and LM
head into new safetensors shards and trims the layer-count config fields. By
default no tensor is initialized or numerically transformed. ``--num-experts``
optionally keeps a prefix of the checkpoint's routed experts so a scaled EP
topology can exercise the production dispatcher without making a few-layer
proxy larger per rank than the full model.

Example:

    python tests/functional_tests/models/hy_v4/build_reference_proxy.py \
        --source ../Hy4-preview \
        --output ../Hy4-preview-1l-reference \
        --num-layers 1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

VLLM_REFERENCE_COMMIT = "b2f685834a6456197e7033966fdef52a23f1abcd"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _is_selected_weight(name: str, num_layers: int, *, include_mtp: bool) -> bool:
    if name.startswith(("model.embed_tokens.", "model.hc_head.", "model.norm.", "lm_head.")):
        return True
    if include_mtp and name.startswith("model.mtp_layers."):
        return True
    for layer_idx in range(num_layers):
        if name.startswith(f"model.layers.{layer_idx}."):
            return True
    return False


def _trim_expert_tensor(name: str, tensor, num_experts: int | None):
    if num_experts is None:
        return tensor
    is_expert_axis = ".mlp.experts." in name or name.endswith((".mlp.gate.weight", ".mlp.gate.e_score_correction_bias"))
    if not is_expert_axis:
        return tensor
    if tensor.shape[0] < num_experts:
        raise ValueError(f"Tensor {name} has only {tensor.shape[0]} experts; requested {num_experts}.")
    return tensor[:num_experts].contiguous()


def build_proxy(
    source: Path,
    output: Path,
    num_layers: int,
    *,
    include_mtp: bool = False,
    num_experts: int | None = None,
) -> None:
    if num_layers < 1:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = _load_json(source / "config.json")
    original_layers = int(config["num_hidden_layers"])
    if num_layers > original_layers:
        raise ValueError(f"Requested {num_layers} layers from a {original_layers}-layer checkpoint.")
    config["num_hidden_layers"] = num_layers
    for field in ("mlp_layer_types", "layer_types", "indexer_types"):
        values = config.get(field)
        if values is not None:
            config[field] = values[:num_layers]
    if not include_mtp:
        config["num_nextn_predict_layers"] = 0
    if num_experts is not None:
        original_experts = int(config["n_routed_experts"])
        experts_per_token = int(config["num_experts_per_tok"])
        if not experts_per_token <= num_experts <= original_experts:
            raise ValueError(f"num_experts must be in [{experts_per_token}, {original_experts}], got {num_experts}.")
        config["n_routed_experts"] = num_experts
    _write_json(output / "config.json", config)

    index = _load_json(source / "model.safetensors.index.json")
    selected_map = {
        name: shard
        for name, shard in index["weight_map"].items()
        if _is_selected_weight(name, num_layers, include_mtp=include_mtp)
    }
    if not selected_map:
        raise RuntimeError("No HY V4 weights matched the requested proxy prefixes.")

    by_source_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in selected_map.items():
        by_source_shard[shard].append(name)

    output_weight_map: dict[str, str] = {}
    total_size = 0
    shard_count = len(by_source_shard)
    for shard_idx, (source_name, names) in enumerate(sorted(by_source_shard.items()), start=1):
        output_name = f"model-{shard_idx:05d}-of-{shard_count:05d}.safetensors"
        with safe_open(source / source_name, framework="pt", device="cpu") as handle:
            tensors = {name: _trim_expert_tensor(name, handle.get_tensor(name), num_experts) for name in sorted(names)}
            for tensor in tensors.values():
                total_size += tensor.numel() * tensor.element_size()
            save_file(tensors, output / output_name, metadata={"format": "pt"})
        output_weight_map.update(dict.fromkeys(names, output_name))
        print(f"wrote {output_name}: {len(names)} tensors")

    _write_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": output_weight_map},
    )
    _write_json(
        output / "reference_provenance.json",
        {
            "source": str(source.resolve()),
            "num_hidden_layers": num_layers,
            "num_routed_experts": num_experts or int(config["n_routed_experts"]),
            "include_mtp": include_mtp,
            "tensor_count": len(output_weight_map),
            "total_size": total_size,
            "vllm_reference_commit": VLLM_REFERENCE_COMMIT,
        },
    )
    print(f"proxy ready at {output}: {len(output_weight_map)} tensors, {total_size / 1024**3:.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--include-mtp",
        action="store_true",
        help="Keep the checkpoint-native MTP layers for an end-to-end training proxy.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        help="Keep the first N routed experts for a scaled expert-parallel smoke test.",
    )
    args = parser.parse_args()
    build_proxy(
        args.source,
        args.output,
        args.num_layers,
        include_mtp=args.include_mtp,
        num_experts=args.num_experts,
    )


if __name__ == "__main__":
    main()
