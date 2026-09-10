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

"""Audit the complete text checkpoint using meta tensors and safetensors headers.

No weight payload is read and no GPU/distributed allocation is made. EP owner
shapes are arithmetic projections of the validated global tensors; this schema
audit does not replace a distributed load or forward test.
"""

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from run_reference_parity import _native_source_manifest
from transformers import AutoTokenizer

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM


def audit(checkpoint: Path, output_dir: Path, ep_size: int, experts: str = "torch_linear") -> None:
    """Compare every native destination against the complete pinned shard index."""
    source_manifest = _native_source_manifest()
    if ep_size < 1:
        raise ValueError("ep_size must be positive")
    config = DeepseekV41Config(**json.loads((checkpoint / "config.json").read_text()))
    config.vision_config.num_hidden_layers = 0
    text = config.text_config
    if text.n_routed_experts % ep_size:
        raise ValueError("The checkpoint's routed expert count must divide evenly across EP ranks")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False, use_fast=True)
    backend = BackendConfig(
        attn="tilelang",
        linear="torch",
        rms_norm="torch_fp32",
        experts=experts,
        dispatcher="hybridep",
        gate_precision="float32",
        enable_hf_state_dict_adapter=True,
    )
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config, backend=backend, tokenizer=tokenizer)
    state = model.state_dict()
    if any(not tensor.is_meta for tensor in state.values()):
        raise AssertionError("Every persistent model destination must be a meta tensor")
    adapter = model.state_dict_adapter
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    header_hashes = {}
    header_bytes = 0
    errors = []
    for filename in sorted(set(weight_map.values())):
        with (checkpoint / filename).open("rb") as stream:
            prefix = stream.read(8)
            length = struct.unpack("<Q", prefix)[0]
            if length > 100 * 1024 * 1024:
                raise ValueError(f"Unreasonably large safetensors header: {filename}")
            raw = stream.read(length)
        header_bytes += 8 + length
        header_hashes[filename] = hashlib.sha256(prefix + raw).hexdigest()
        for key, value in json.loads(raw).items():
            if key == "__metadata__":
                continue
            if key in headers:
                errors.append({"kind": "duplicate_header_key", "key": key})
            headers[key] = value
            if weight_map.get(key) != filename:
                errors.append({"kind": "header_index_mismatch", "key": key, "shard": filename})
    for key in weight_map.keys() - headers.keys():
        errors.append({"kind": "index_key_missing_from_headers", "key": key})

    output_dir.mkdir(parents=True, exist_ok=True)
    expected = set()
    layer_counts = Counter()
    expected_dtypes = Counter()
    quantized_dtypes = {torch.int8: "I8", torch.float8_e4m3fn: "F8_E4M3", torch.float8_e8m0fnu: "F8_E8M0"}
    expert_counts = Counter()
    with (output_dir / "destinations.jsonl").open("w") as destinations:
        for fqn, value in state.items():
            for key, target in adapter.convert_single_tensor_to_hf(
                fqn, value, quantization=True, for_checkpoint_load=True
            ):
                if not target.is_meta:
                    raise AssertionError(f"Adapter allocated a non-meta destination for {key}")
                if key in expected:
                    errors.append({"kind": "duplicate_destination", "key": key, "native_fqn": fqn})
                expected.add(key)
                source = headers.get(key)
                if source is None:
                    errors.append({"kind": "missing_source", "key": key, "native_fqn": fqn})
                elif tuple(source["shape"]) != tuple(target.shape):
                    errors.append(
                        {
                            "kind": "shape_mismatch",
                            "key": key,
                            "actual": source["shape"],
                            "expected": list(target.shape),
                            "native_fqn": fqn,
                        }
                    )
                if source and target.dtype in quantized_dtypes and source["dtype"] != quantized_dtypes[target.dtype]:
                    errors.append(
                        {
                            "kind": "quantized_dtype_mismatch",
                            "key": key,
                            "actual": source["dtype"],
                            "expected": quantized_dtypes[target.dtype],
                        }
                    )
                if source:
                    expected_dtypes[source["dtype"]] += 1
                layer = re.match(r"layers\.(\d+)\.", key)
                if layer:
                    layer_counts[int(layer[1])] += 1
                expert = re.fullmatch(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.(weight|scale)", key)
                if expert:
                    expert_counts[(int(expert[1]), int(expert[2]))] += 1
                destinations.write(
                    json.dumps(
                        {
                            "native_fqn": fqn,
                            "source_key": key,
                            "shape": list(target.shape),
                            "target_dtype": str(target.dtype),
                            "source_dtype": source["dtype"] if source else None,
                        }
                    )
                    + "\n"
                )
    for layer in range(text.num_hidden_layers):
        for expert in range(text.n_routed_experts):
            if expert_counts[(layer, expert)] != 6:
                errors.append(
                    {
                        "kind": "incomplete_expert",
                        "layer": layer,
                        "expert": expert,
                        "weight_and_scale_count": expert_counts[(layer, expert)],
                    }
                )
    excluded = {"dspark": [], "vision": [], "unexpected": []}
    for key in sorted(headers.keys() - expected):
        if key.startswith("mtp."):
            excluded["dspark"].append(key)
        elif key.startswith(("vision.", "aligner.")) or key in ("image_start", "image_end", "image_newline"):
            excluded["vision"].append(key)
        else:
            excluded["unexpected"].append(key)
    if excluded["unexpected"]:
        errors.append({"kind": "unexpected_backbone_sources", "keys": excluded["unexpected"]})
    engram_owners = []
    for layer, rows in zip(text.engram_layer_ids, text.engram_num_embeddings):
        owner_rows = math.ceil(rows / ep_size)
        engram_owners.append(
            {
                "layer": layer,
                "logical_rows": rows,
                "head_dim": text.engram_head_dim,
                "owner_rows": owner_rows,
                "padded_global_rows": owner_rows * ep_size,
                "physical_padding_rows": owner_rows * ep_size - rows,
                "owner_bf16_bytes": owner_rows * text.engram_head_dim * 2,
            }
        )
    summary = {
        "passed": not errors,
        "checkpoint": str(checkpoint.resolve()),
        "revision_directory": checkpoint.name,
        "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        "native_config": config.to_dict(),
        "backend": {
            key: str(value).removeprefix("torch.") if isinstance(value, torch.dtype) else value
            for key, value in asdict(backend).items()
        },
        "native_source_hashes_before_audit": source_manifest,
        "audit_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "header_sha256": header_hashes,
        "weight_payload_bytes_read": 0,
        "safetensors_header_bytes_read": header_bytes,
        "shards": len(header_hashes),
        "native_tensor_count": len(state),
        "checkpoint_tensor_count": len(headers),
        "matched_destination_count": len(expected),
        "checkpoint_dtypes_in_scope": expected_dtypes,
        "num_hidden_layers": text.num_hidden_layers,
        "routed_experts_per_layer": text.n_routed_experts,
        "expert_weight_and_scale_count": sum(expert_counts.values()),
        "per_layer_destination_count": dict(sorted(layer_counts.items())),
        "ep_size_projection_only": ep_size,
        "experts_per_ep_rank": text.n_routed_experts // ep_size,
        "engram_owners": engram_owners,
        "excluded_counts": {key: len(value) for key, value in excluded.items()},
        "errors": errors,
    }
    if _native_source_manifest() != source_manifest:
        raise AssertionError("Native source changed during the schema audit; discard this result")
    summary["source_manifest_unchanged"] = True
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "excluded_keys.json").write_text(json.dumps(excluded, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "header_sha256"}), flush=True)
    if errors:
        raise AssertionError(
            f"Checkpoint schema audit found {len(errors)} errors; inspect {output_dir / 'summary.json'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=128)
    parser.add_argument("--experts", choices=("torch_linear", "torch_mm", "torch"), default="torch_linear")
    args = parser.parse_args()
    audit(args.checkpoint, args.output_dir, args.ep_size, args.experts)
