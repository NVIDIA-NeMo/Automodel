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

"""GPU parity of released Engram weights using a compact exact requested-row table.

Only storage is reduced: every requested global row retains its checkpoint FP8
weight/scale and receives a unique compact ID. Both Engram forward methods remain
unchanged. This component diagnostic does not validate full-table or distributed
execution; the official inference FP8 table is frozen for gradient comparisons.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41Engram, DeepseekV41NgramHash
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import dequantize_checkpoint_weight

LOGGER = logging.getLogger(__name__)
REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"
REFERENCE_HASHES = {
    "model.py": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
}


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    """Measure two identically laid-out component tensors.

    Args:
        actual: Native tensor of arbitrary shape.
        expected: Reference tensor with the same shape and dtype.

    Returns:
        Scalar difference metrics and bitwise equality.
    """
    delta = actual.float() - expected.float()
    return {
        "exact": torch.equal(actual, expected),
        "max_absolute_difference": delta.abs().max().item(),
        "mean_absolute_difference": delta.abs().mean().item(),
        "cosine_similarity": F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item(),
    }


def main() -> None:
    """Run the selected-row, released-weight Engram diagnostic on one GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--layer-id", type=int, choices=(1, 14), default=1)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.sequence_length < 32:
        raise ValueError("This diagnostic needs at least 32 positions for text and image-span cases")
    for filename, expected_hash in REFERENCE_HASHES.items():
        actual_hash = hashlib.sha256((args.reference_dir / filename).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Official reference {filename} differs from revision {REVISION}")
    sys.path.insert(0, str(args.reference_dir))
    spec = importlib.util.spec_from_file_location("official_engram_model", args.reference_dir / "model.py")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    config = DeepseekV41Config.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    reference_args = reference.ModelArgs(**json.loads((args.reference_dir / "config.json").read_text()))
    reference_args = replace(reference_args, dtype="bf16", max_batch_size=2, max_seq_len=args.sequence_length)
    original_layout = reference.EngramLayout.from_args(reference_args)
    layer_index = original_layout.layer_ids.index(args.layer_id)
    text = (
        "Engram stores conditional memory indexed by normalized token n-grams. "
        "The same checkpoint weights and image masks must produce matching residuals. "
        "模型验证比较相同输入、哈希值、隐藏状态以及梯度。 "
    )
    base_tokens = tokenizer.encode(text, add_special_tokens=False)
    repeated = (base_tokens * ((args.sequence_length + len(base_tokens) - 1) // len(base_tokens)))[
        : args.sequence_length
    ]
    input_ids = torch.tensor([repeated, repeated[::-1]], dtype=torch.long, device=device)
    image_ids = input_ids.clone()
    image_mask = torch.ones_like(input_ids, dtype=torch.bool)
    image_mask[0, 8:15] = False
    image_mask[1, 1:4] = False
    image_ids.masked_fill_(~image_mask, config.image_token_id)
    with torch.device(device):
        official_hash = reference.NgramHashState(reference_args, original_layout, tokenizer)
        native_hash = DeepseekV41NgramHash(config.text_config, tokenizer)
    masks = [torch.ones_like(input_ids, dtype=torch.bool), image_mask, torch.zeros_like(input_ids, dtype=torch.bool)]
    token_cases = [input_ids, image_ids, image_ids]
    case_names = ["text", "image_spans", "all_masked"]
    hashes = []
    for tokens, mask in zip(token_cases, masks):
        official_ids = official_hash(tokens, 0, mask)
        native_ids = native_hash(tokens, token_mask=mask)
        torch.testing.assert_close(native_ids, official_ids, rtol=0, atol=0)
        hashes.append(official_ids[:, :, layer_index].clone())
    unique_rows = torch.unique(torch.cat([ids.flatten() for ids in hashes]), sorted=True)
    compact_hashes = [torch.searchsorted(unique_rows, ids) for ids in hashes]
    row_ids = unique_rows.cpu().tolist()
    LOGGER.info(
        "Reading %d distinct layer-%d rows from %d logical rows",
        len(row_ids),
        args.layer_id,
        config.text_config.engram_num_embeddings[layer_index],
    )
    weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{args.layer_id}.engram."
    shard_names = {
        weight_map[prefix + name]
        for name in ("embed.weight", "embed.scale", "wkv.weight", "wkv.scale", "q_weight", "k_weight")
    }
    if len(shard_names) != 1:
        raise ValueError(f"The pinned layer-{args.layer_id} Engram weights should occupy one source shard")
    with safe_open(str(args.checkpoint / shard_names.pop()), framework="pt", device="cpu") as checkpoint:
        weight_view = checkpoint.get_slice(prefix + "embed.weight")
        scale_view = checkpoint.get_slice(prefix + "embed.scale")
        # Byte concatenation preserves E4M3/E8M0 bits without requesting unsupported
        # CPU arithmetic on these storage dtypes. SafeTensors reads only each row.
        compact_weight = (
            torch.cat([weight_view[row : row + 1].view(torch.uint8) for row in row_ids])
            .view(torch.float8_e4m3fn)
            .to(device)
        )
        compact_scale = (
            torch.cat([scale_view[row : row + 1].view(torch.uint8) for row in row_ids])
            .view(torch.float8_e8m0fnu)
            .to(device)
        )
        projection_weight = checkpoint.get_tensor(prefix + "wkv.weight").to(device)
        projection_scale = checkpoint.get_tensor(prefix + "wkv.scale").to(device)
        q_weight = checkpoint.get_tensor(prefix + "q_weight").to(device)
        k_weight = checkpoint.get_tensor(prefix + "k_weight").to(device)

    compact_rows = list(original_layout.num_embeddings)
    compact_rows[layer_index] = len(row_ids)
    compact_layout = replace(original_layout, num_embeddings=tuple(compact_rows))
    compact_config = copy.deepcopy(config.text_config)
    compact_config.engram_num_embeddings = compact_rows
    reference.world_size, reference.rank = 1, 0
    reference.default_dtype = torch.bfloat16
    with torch.device(device), reference.set_dtype(torch.bfloat16):
        official = reference.Engram(reference_args, args.layer_id, compact_layout)
        native = DeepseekV41Engram(compact_config, args.layer_id, BackendConfig(linear="torch"))
    official.embed.requires_grad_(False)
    row_block = projection_weight.shape[0] // projection_scale.shape[0]
    column_block = projection_weight.shape[1] // projection_scale.shape[1]
    # Independent reference decoding: source scale axes identify rectangular
    # blocks, while the native path uses the production adapter implementation.
    official_projection = (
        (
            projection_weight.unflatten(0, (-1, row_block)).unflatten(-1, (-1, column_block)).float()
            * projection_scale.float()[:, None, :, None]
        )
        .flatten(0, 1)
        .flatten(-2)
        .to(torch.bfloat16)
    )
    with torch.no_grad():
        official.embed.weight.copy_(compact_weight)
        official.embed.scale.copy_(compact_scale)
        official.wkv.weight.copy_(official_projection)
        native.embed.weight.copy_(dequantize_checkpoint_weight(compact_weight, compact_scale, rowwise=True))
        native.wkv.weight.copy_(dequantize_checkpoint_weight(projection_weight, projection_scale))
        official.q_weight.copy_(q_weight)
        official.k_weight.copy_(k_weight)
        native.q_weight.copy_(q_weight)
        native.k_weight.copy_(k_weight)
    torch.testing.assert_close(native.wkv.weight, official.wkv.weight, rtol=0, atol=0)
    hidden_values = (
        torch.randn(
            2,
            args.sequence_length,
            config.text_config.hc_mult,
            config.text_config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    upstream = torch.randn_like(hidden_values) / hidden_values.numel() ** 0.5
    results = {}
    all_exact = True
    for name, ids, mask in zip(case_names, compact_hashes, masks):
        official.zero_grad(set_to_none=True)
        native.zero_grad(set_to_none=True)
        hidden = hidden_values.detach().clone().requires_grad_()
        official_hidden = hidden_values.detach().clone().requires_grad_()
        expected_lookup = official.embed(ids)
        actual_lookup = native.embed(ids)
        torch.testing.assert_close(actual_lookup, expected_lookup, rtol=0, atol=0)
        expected = official(official_hidden, ids, mask)
        actual = native(hidden, ids, token_mask=mask)
        expected.backward(upstream)
        actual.backward(upstream)
        gradient_metrics = {"hidden_states": _metrics(hidden.grad, official_hidden.grad)}
        for parameter_name in ("wkv.weight", "q_weight", "k_weight"):
            gradient_metrics[parameter_name] = _metrics(
                native.get_parameter(parameter_name).grad, official.get_parameter(parameter_name).grad
            )
        result = {
            "hashes_all_layers_exact": True,
            "lookup_exact": True,
            "residual": _metrics(actual, expected),
            "gradients": gradient_metrics,
            "native_table_gradient_finite": torch.isfinite(native.embed.weight.grad).all().item(),
        }
        all_exact = (
            all_exact and result["residual"]["exact"] and all(item["exact"] for item in gradient_metrics.values())
        )
        results[name] = result
        LOGGER.info("%s: %s", name, json.dumps(result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output.with_suffix(".inputs.safetensors")
    save_file(
        {
            "input_ids": input_ids.cpu(),
            "image_input_ids": image_ids.cpu(),
            "image_token_mask": image_mask.cpu(),
            "requested_global_rows": unique_rows.cpu(),
        },
        str(inputs_path),
    )
    report = {
        "reference_revision": REVISION,
        "reference_hashes": REFERENCE_HASHES,
        "reference_kind": "unchanged official Engram.forward and NgramHashState.forward",
        "checkpoint": str(args.checkpoint),
        "layer": args.layer_id,
        "sequence_length": args.sequence_length,
        "compute_dtype": "bfloat16",
        "seed": args.seed,
        "table_storage": "exact requested FP8/E8M0 checkpoint rows with bijective compact row-ID remapping",
        "logical_table_rows": config.text_config.engram_num_embeddings[layer_index],
        "requested_rows": len(row_ids),
        "gradient_scope": "hidden states, BF16 wkv projection, q_weight and k_weight; official FP8 lookup storage frozen",
        "scope": "Selected-row component diagnostic only; does not validate full-table owner sharding or end-to-end model outputs",
        "inputs": str(inputs_path),
        "results": results,
        "all_compared_tensors_exact": all_exact,
        "gpu": torch.cuda.get_device_name(),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all_exact:
        raise AssertionError(f"Engram component mismatch; details written to {args.output}")
    LOGGER.info("PASS: exact Engram residual/hash/gradient parity; report %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
