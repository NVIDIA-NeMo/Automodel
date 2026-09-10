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

"""Compare every decoder attention role on authentic identical source inputs.

Unchanged official blocks generate the real stream entering each layer. Native
attention independently retains the KV/index state created at20 and consumes the
same per-layer query input. No native hidden-state substitution is used to claim
end-to-end parity; this is an explicitly isolated Full/Reuse/Reindex diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from run_decoder_reference_parity import (
    LAYER_IDS,
    _holder,
    _identity_report,
    _initial_decoder_state,
    _metrics,
    _native_skeleton,
    _observe,
    _Options,
    _reference_runtime,
    _source_scope,
    _state_report,
)
from run_reference_parity import REFERENCE_HASHES, REVISION, _import_reference, _load_official, _native_source_manifest
from safetensors.torch import save_file

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41AttentionState
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    options = _Options(**vars(parser.parse_args()), attention_backend="tilelang")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one available GPU for this sequential component diagnostic")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.set_float32_matmul_precision("highest")
    config = DeepseekV41Config.from_pretrained(options.checkpoint, local_files_only=True)
    backend = BackendConfig(
        attn="tilelang", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch"
    )
    blocks, adapter = _native_skeleton(config, backend)
    reference = _import_reference(options.reference_dir)
    reference_args = _reference_runtime(reference, options)
    hidden, pre_mix, provenance = _initial_decoder_state(options, config)
    hidden, pre_mix = hidden.to(device), pre_mix.to(device)
    positions = torch.arange(options.sequence_length, device=device).unsqueeze(0)
    native_state = DeepseekV41AttentionState()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    saved_boundaries = {}
    source_before = _native_source_manifest()
    report = {
        "scope": "Same-input authentic attention components20–24; native KV/index ownership retained independently",
        "reference_revision": REVISION,
        "reference_source_hashes": REFERENCE_HASHES,
        "native_source_before": source_before,
        "input_provenance": provenance,
        "layers": {},
    }
    for layer_id in LAYER_IDS:
        logging.info("Loading official block and native attention %d", layer_id)
        with torch.device(device), reference.set_dtype(torch.bfloat16):
            official = reference.Block(layer_id, reference_args).eval().requires_grad_(False)
        native = blocks.pop(str(layer_id)).attn.to_empty(device=device).eval().requires_grad_(False)
        official_audit = _load_official(_holder(official, layer_id, native=False), options.checkpoint)
        native_audit = adapter.load_from_checkpoint(
            _holder(native, layer_id, native=True, attention_only=True), options.checkpoint
        )
        if set(native_audit.loaded_keys) != _source_scope(options.checkpoint, layer_id, attention_only=True):
            raise ValueError("Native attention checkpoint coverage does not match the exact source scope")
        observed, hooks = _observe(official)
        before_native = native_state
        before_official = tuple(
            getattr(reference.shared_attn, name) for name in ("compress_kv", "index_k", "topk_idxs", "candidates")
        )
        with torch.inference_mode(), torch.device(device), reference.set_dtype(torch.bfloat16):
            hidden, pre_mix = official(hidden, 0, pre_mix, None)
            same_input = observed["attn_norm"].to(device)
            actual = native(same_input, position_ids=positions, state=native_state)
            native_state = actual.state
            metrics = _metrics(observed["attn"], actual.hidden_states.cpu())
        for handle in hooks:
            handle.remove()
        report["layers"][str(layer_id)] = {
            "kind": "Full" if layer_id == 20 else "Reindex" if layer_id == 24 else "Reuse",
            "attention": metrics,
            "state": _state_report(reference, native_state, options.sequence_length),
            "state_identity": _identity_report(
                reference, before_official, before_native, native_state, layer_id=layer_id
            ),
            "source_state_audit": asdict(official_audit),
            "native_state_audit": asdict(native_audit),
        }
        saved_boundaries.update({f"layer.{layer_id}.{name}": value for name, value in observed.items()})
        logging.info("Same-input attention %d: %s", layer_id, metrics)
        options.output.write_text(json.dumps(report, indent=2) + "\n")
        del official, native, actual, same_input, observed, before_official, before_native
        gc.collect()
        torch.cuda.empty_cache()
    source_after = _native_source_manifest()
    changed = sorted(
        name for name in source_before.keys() | source_after.keys() if source_before.get(name) != source_after.get(name)
    )
    if changed:
        raise ValueError(f"Native source changed during the run: {changed}")
    boundary_path = options.output.with_suffix(".boundaries.safetensors")
    save_file(saved_boundaries, str(boundary_path))
    report["source_unchanged"] = True
    report["official_boundaries_artifact"] = str(boundary_path)
    report["all_attention_outputs_exact"] = all(value["attention"]["exact"] for value in report["layers"].values())
    report["all_shared_states_exact"] = all(
        item["exact"] for layer in report["layers"].values() for item in layer["state"].values()
    )
    report["completed"] = True
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_attention_outputs_exact"] and report["all_shared_states_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
