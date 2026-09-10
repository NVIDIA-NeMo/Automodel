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

"""Locate MoE precision differences using authentic official prefix activations.

The primary oracle is the unchanged released MoE.forward. A separate replay of
unchanged Expert.forward calls exposes its FP32 routed sum. Existing native
single-GPU grouped and loop methods expose the corresponding FP32 sums, before
final shared-expert addition. No production or oracle operation is replaced.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from run_decoder_reference_parity import _holder, _native_skeleton, _Options, _reference_runtime
from run_reference_parity import REFERENCE_HASHES, REVISION, _import_reference, _load_official, _tensor_metrics
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config


def _metrics(reference: torch.Tensor, native: torch.Tensor) -> dict:
    """Measure equal-layout precision variants without applying acceptance thresholds."""
    return {
        **asdict(_tensor_metrics(reference, native)),
        "exact": torch.equal(reference, native),
        "changed_elements": (reference != native).sum().item(),
    }


def _moe_holder(moe: nn.Module, layer: int, *, native: bool) -> nn.Module:
    """Preserve the original checkpoint path for an isolated MoE submodule."""
    block = nn.Module()
    block.ffn = moe
    return _holder(block, layer, native=native)


@torch.inference_mode()
def _compare(
    official: nn.Module,
    native: nn.Module,
    hidden: torch.Tensor,
    expected: torch.Tensor,
    contributions_artifact: Path | None = None,
) -> dict:
    """Compare complete MoE, routed kernels and final addition on identical input."""
    values = hidden.flatten(0, 1)
    mask = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    official_complete = official(hidden, None)
    torch.testing.assert_close(official_complete, expected, rtol=0, atol=0)
    weights, indices = official.gate(values, None)
    native.gate.set_routing_context(None, None)
    actual_weights, actual_indices, _ = native.gate(values, mask, None)
    torch.testing.assert_close(actual_weights, weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_indices, indices, rtol=0, atol=0)
    shared = official.shared_experts(values)
    native_shared = native.shared_experts(values)
    torch.testing.assert_close(native_shared, shared, rtol=0, atol=0)

    # Literal official routed accumulation, retaining unchanged Expert.forward.
    # This replay is validated against the unchanged complete MoE above/below.
    official_routed = torch.zeros_like(values, dtype=torch.float32)
    contributions = torch.zeros((*indices.shape, values.shape[-1]), dtype=values.dtype, device=values.device)
    active_experts = torch.unique(indices).tolist()
    for expert_id in active_experts:
        row, slot = torch.where(indices == expert_id)
        expert_output = official.experts[expert_id](values[row], weights[row, slot, None])
        official_routed[row] += expert_output
        contributions[row, slot] = expert_output
    replayed_complete = (official_routed + shared.float()).to(values.dtype).view_as(expected)
    torch.testing.assert_close(replayed_complete, official_complete, rtol=0, atol=0)
    if contributions_artifact is not None:
        contributions_artifact.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "hidden": values.cpu(),
                "indices": indices.cpu(),
                "weights": weights.cpu(),
                "routed_outputs": contributions.cpu(),
                "shared_output": shared.cpu(),
                "complete_output": official_complete.flatten(0, 1).cpu(),
            },
            str(contributions_artifact),
            metadata={
                "reference_revision": REVISION,
                "scope": "Unchanged Expert.forward contributions; complete sum validated against unchanged MoE.forward",
            },
        )

    experts = native.experts
    grouped = experts._forward_grouped_mm(
        values,
        mask,
        actual_weights,
        actual_indices,
        experts.gate_and_up_projs,
        experts.down_projs,
        None,
        None,
        experts.n_routed_experts,
        0,
    )
    loop = experts._forward_loop(
        values,
        actual_weights,
        actual_indices,
        mask,
        experts.gate_and_up_projs,
        experts.down_projs,
        None,
        None,
        experts.n_routed_experts,
        0,
        experts.n_routed_experts,
    )
    variants = {}
    for name, routed in (("official_routed", official_routed), ("native_grouped", grouped), ("native_loop", loop)):
        assert routed.dtype == torch.float32
        variants[name] = {
            "routed_fp32": _metrics(official_routed, routed),
            "shared_added_before_bf16_cast": _metrics(
                official_complete, (routed + native_shared.float()).to(values.dtype).view_as(expected)
            ),
            "shared_added_after_bf16_cast": _metrics(
                official_complete, (routed.to(values.dtype) + native_shared).view_as(expected)
            ),
        }
    return {
        "authentic_oracle_artifact_replayed_exactly": True,
        "routed_replay_reconstructs_primary_oracle_exactly": True,
        "router_weights_and_indices_exact": True,
        "shared_experts_exact": True,
        "active_experts": len(active_experts),
        "actual_native_complete_moe": _metrics(official_complete, native(hidden)),
        "precision_variants": variants,
    }


def main() -> int:
    """Load one released MoE pair and diagnose its arithmetic boundaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--contributions-artifact", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one free GPU for this isolated MoE diagnostic")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    with safe_open(str(args.reference_artifact), framework="pt", device="cpu") as artifact:
        hidden = artifact.get_tensor(f"layer.{args.layer}.ffn_norm").to(device)
        expected = artifact.get_tensor(f"layer.{args.layer}.ffn").to(device)
    options = _Options(args.checkpoint, args.reference_dir, args.output, sequence_length=hidden.shape[1])
    reference = _import_reference(args.reference_dir)
    reference_args = _reference_runtime(reference, options)
    backend = BackendConfig(
        attn="tilelang", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch"
    )
    config = DeepseekV41Config.from_pretrained(args.checkpoint, local_files_only=True)
    blocks, adapter = _native_skeleton(config, backend, layer_ids=(args.layer,))
    native = blocks[str(args.layer)].ffn.to_empty(device=device).eval().requires_grad_(False)
    with torch.device(device), reference.set_dtype(torch.bfloat16):
        official = reference.MoE(args.layer, reference_args).eval().requires_grad_(False)
    official_audit = _load_official(_moe_holder(official, args.layer, native=False), args.checkpoint)
    native_audit = adapter.load_from_checkpoint(_moe_holder(native, args.layer, native=True), args.checkpoint)
    with torch.device(device), reference.set_dtype(torch.bfloat16):
        results = _compare(official, native, hidden, expected, args.contributions_artifact)
    report = {
        "reference_revision": REVISION,
        "reference_source_hashes": REFERENCE_HASHES,
        "checkpoint": str(args.checkpoint),
        "reference_artifact": str(args.reference_artifact),
        "layer": args.layer,
        "input_shape": list(hidden.shape),
        "weight_compute_dtype": "bfloat16",
        "native_dispatcher": "torch; isolated single-GPU component diagnostic",
        "state_audit": {"official": asdict(official_audit), "native": asdict(native_audit)},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
