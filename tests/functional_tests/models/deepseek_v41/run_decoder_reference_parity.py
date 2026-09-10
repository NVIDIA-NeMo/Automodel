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

"""Compare released decoder blocks 20--24 against unchanged official inference.

This is a decoder-window diagnostic with saved official-prefix or seeded input,
not an end-to-end model or incremental KV-cache decoding test. Each complete block
uses all released experts. One official/native block pair resides on the GPU at
a time, while each implementation's own CSA2 state survives across the window.
A separate attention-only case changes candidate_topk_blocks from 2048 to 80;
this activates hierarchical pruning at sequence length 4096 while preserving
index_topk=512. Both cases retain official KV/index activation quantization.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType

import torch
from run_reference_parity import (
    REFERENCE_HASHES,
    REVISION,
    _CheckpointReader,
    _evaluate_gates,
    _import_reference,
    _load_official,
    _logit_metrics,
    _native_source_manifest,
    _tensor_metrics,
)
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionState,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41RMSNorm
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepseekV41StateDictAdapter

LOGGER = logging.getLogger(__name__)
LAYER_IDS = (20, 21, 22, 23, 24)


@dataclass(frozen=True)
class _Options:
    checkpoint: Path
    reference_dir: Path
    output: Path
    sequence_length: int = 4096
    seed: int = 113
    candidate_topk_blocks: int = 80
    attention_backend: str = "sdpa"
    check_layout_only: bool = False
    reference_artifact: Path | None = None
    diagnostic_moe_fp32_shared_add: bool = False
    expert_backend: str = "torch_mm"


def _diagnostic_late_shared_add(moe: nn.Module) -> list:
    """Replay native grouped accumulation and defer its cast past shared addition.

    This explicitly requested single-GPU precision experiment changes only the
    native diagnostic. The released oracle and production MoE remain untouched.
    Returning FP32 routed output lets the existing MoE perform shared addition
    in FP32; the second hook restores the input dtype at the complete MoE output.
    """

    def routed_sum(experts: nn.Module, inputs: tuple, _output: torch.Tensor) -> torch.Tensor:
        values, token_mask, weights, indices = inputs
        result = experts._forward_grouped_mm(
            values,
            token_mask,
            weights,
            indices,
            experts.gate_and_up_projs,
            experts.down_projs,
            None,
            None,
            experts.n_routed_experts,
            0,
        )
        assert result.dtype == torch.float32
        return result

    def complete_cast(_moe: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        return output.to(inputs[0].dtype)

    return [moe.experts.register_forward_hook(routed_sum), moe.register_forward_hook(complete_cast)]


def _initial_decoder_state(options: _Options, config: DeepseekV41Config) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Read authentic layer-19 outputs, or construct explicit seeded diagnostic inputs."""
    text = config.text_config
    shape = (1, options.sequence_length, text.hc_mult, text.hidden_size)
    if options.reference_artifact is None:
        hidden = torch.randn(shape).bfloat16()
        mix = torch.zeros(shape[:-1], dtype=torch.float32)
        mix[..., 0] = 1
        provenance = {"kind": "seeded independent hidden-state streams", "seed": options.seed}
    else:
        metadata_path = options.reference_artifact.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata["reference_revision"] != REVISION or metadata["reference_source_hashes"] != REFERENCE_HASHES:
            raise ValueError("Decoder input must come from the pinned, unchanged official reference")
        scope = metadata["configuration"]
        if scope["layers"] != LAYER_IDS[0] or scope["sequence_length"] != options.sequence_length:
            raise ValueError("Authentic decoder input requires the complete official layers 0--19 at the same length")
        if scope["reference_tp"] != 1 or scope["weight_compute_dtype"] != "bfloat16":
            raise ValueError("Authentic decoder input requires official TP1 with BF16 weight compute")
        with safe_open(str(options.reference_artifact), framework="pt", device="cpu") as artifact:
            hidden = artifact.get_tensor("final_streams")
            mix = artifact.get_tensor("final_pre_mix")
        provenance = {
            "kind": "authentic unchanged official TP1 prefix through layer 19",
            "artifact": str(options.reference_artifact),
            "metadata": str(metadata_path),
            "input_token_sha256": metadata["input_sha256"],
            "source_state_audits": metadata["state_audits"],
        }
    if hidden.shape != shape or hidden.dtype != torch.bfloat16:
        raise ValueError(f"Decoder streams must be BF16 with shape {shape}")
    if mix.shape != shape[:-1] or mix.dtype != torch.float32:
        raise ValueError(f"Decoder carried coefficients must be FP32 with shape {shape[:-1]}")
    return hidden, mix, provenance


def _holder(module: nn.Module, layer_id: int, *, native: bool, attention_only: bool = False) -> nn.Module:
    """Give a single module its original checkpoint path, without changing execution."""
    block = nn.Module() if attention_only else module
    if attention_only:
        block.attn = module
    root = nn.Module()
    root.layers = nn.ModuleDict({str(layer_id): block})
    if native:
        outer = nn.Module()
        outer.model = root
        return outer
    return root


def _native_skeleton(
    config: DeepseekV41Config, backend: BackendConfig, *, layer_ids: tuple[int, ...] = LAYER_IDS
) -> tuple[nn.ModuleDict, DeepseekV41StateDictAdapter]:
    """Construct real production blocks on meta and keep only the decoder window.

    Engram and vision lie outside this window and are omitted from the temporary
    skeleton. Every decoder shape and MoE setting comes from production model
    construction with the released configuration.
    """
    construction_config = copy.deepcopy(config)
    construction_config.text_config.engram_layer_ids = []
    construction_config.text_config.engram_num_embeddings = []
    construction_config.vision_config.num_hidden_layers = 0
    with torch.device("meta"):
        full = DeepseekV41ForCausalLM(construction_config, backend=backend)
    blocks = nn.ModuleDict({str(layer_id): full.model.layers[str(layer_id)] for layer_id in layer_ids})
    adapter = DeepseekV41StateDictAdapter(config, full.moe_config, backend, dtype=torch.bfloat16)
    return blocks, adapter


def _source_scope(checkpoint: Path, layer_id: int, *, attention_only: bool) -> set[str]:
    """Return every original tensor key in exactly the requested block scope."""
    reader = _CheckpointReader(checkpoint)
    try:
        prefix = f"layers.{layer_id}." + ("attn." if attention_only else "")
        return {name for name in reader.weight_map if name.startswith(prefix)}
    finally:
        reader.close()


def _load_pair(
    official: nn.Module,
    native: nn.Module,
    adapter: DeepseekV41StateDictAdapter,
    options: _Options,
    layer_id: int,
    *,
    attention_only: bool = False,
) -> dict:
    """Load both modules independently and audit complete source coverage.

    Official projection decoding is independent of the production adapter. The
    native path streams into grouped expert storage without retaining converted
    checkpoint dictionaries or gathering any model tensor.
    """
    official_holder = _holder(official, layer_id, native=False, attention_only=attention_only)
    native_holder = _holder(native, layer_id, native=True, attention_only=attention_only)
    source_scope = _source_scope(options.checkpoint, layer_id, attention_only=attention_only)
    official_audit = _load_official(official_holder, options.checkpoint)
    started = time.monotonic()
    native_audit = adapter.load_from_checkpoint(native_holder, options.checkpoint)
    torch.cuda.synchronize()
    if set(native_audit.loaded_keys) != source_scope:
        raise ValueError(
            f"Native layer {layer_id} source audit failed: "
            f"missing={sorted(source_scope - set(native_audit.loaded_keys))}, "
            f"unexpected={sorted(set(native_audit.loaded_keys) - source_scope)}"
        )
    # _load_official checks every destination and its shape. Require its source
    # count to cover the same full block, including scales and modality biases.
    if official_audit.checkpoint_source_tensors != len(source_scope):
        raise ValueError(f"Official layer {layer_id} source coverage differs from the complete checkpoint scope")
    native_state = {name: value for name, value in native_holder.state_dict().items() if torch.is_tensor(value)}
    report = {
        "official": asdict(official_audit),
        "native": {
            **asdict(native_audit),
            "expected_native_tensors": len(native_state),
            "expected_source_tensors": len(source_scope),
            "missing": [],
            "unexpected": [],
            "shape_mismatches": [],
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    official.eval().requires_grad_(False)
    native.eval().requires_grad_(False)
    return report


def _metrics(official: torch.Tensor, native: torch.Tensor) -> dict:
    """Compare equal-layout activations and reject nonfinite outputs."""
    if not torch.isfinite(official).all() or not torch.isfinite(native).all():
        raise ValueError("A decoder-window boundary contains nonfinite values")
    return {**asdict(_tensor_metrics(official, native)), "exact": torch.equal(official, native)}


def _observe(block: nn.Module) -> tuple[dict[str, torch.Tensor], list]:
    """Copy norm/attention/FFN outputs to CPU with observer-only hooks."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            value = output.hidden_states if hasattr(output, "hidden_states") else output
            captured[name] = value.detach().to("cpu", copy=True)

        return capture

    for name in ("attn_norm", "attn", "ffn_norm", "ffn"):
        handles.append(getattr(block, name).register_forward_hook(hook(name)))
    return captured, handles


@torch.inference_mode()
def _same_input_moe_report(
    official: nn.Module, native: nn.Module, observed: dict[str, torch.Tensor], device: torch.device
) -> dict:
    """Separate MoE arithmetic from preceding attention/cascade differences.

    Both existing routers and shared experts receive the exact official
    ffn_norm output. The native complete MoE also consumes that same tensor and
    is compared with the observed unchanged official MoE's original output.
    """
    hidden = observed["ffn_norm"].to(device)
    flattened = hidden.flatten(0, 1)
    token_mask = torch.ones(flattened.shape[0], dtype=torch.bool, device=device)
    reference_weights, reference_indices = official.ffn.gate(flattened, None)
    native.ffn.gate.set_routing_context(None, None)
    native_weights, native_indices, _ = native.ffn.gate(flattened, token_mask, None)
    return {
        "scope": "Identical official ffn_norm input; does not replace either cascaded hidden-state stream",
        "router_weights": _metrics(reference_weights, native_weights),
        "router_indices_exact": torch.equal(reference_indices, native_indices),
        "router_element_agreement": (reference_indices == native_indices).float().mean().item(),
        "shared_experts": _metrics(official.ffn.shared_experts(flattened), native.ffn.shared_experts(flattened)),
        "complete_moe": _metrics(observed["ffn"], native.ffn(hidden).cpu()),
    }


def _official_state(reference: ModuleType, sequence: int) -> dict[str, torch.Tensor]:
    """Normalize original cache views and absolute selection offsets for comparison."""
    state = reference.shared_attn
    return {
        "compressed_kv": state.compress_kv[:, :sequence],
        "index_keys": state.index_k[:, :sequence],
        "topk_indices": torch.where(state.topk_idxs >= 0, state.topk_idxs - sequence, -1).long(),
        "candidates": state.candidates,
    }


def _state_report(reference: ModuleType, native: DeepseekV41AttentionState, sequence: int) -> dict:
    """Compare shared KV/index/candidate contents, with index membership statistics."""
    official = _official_state(reference, sequence)
    result = {}
    for name in ("compressed_kv", "index_keys"):
        result[name] = _metrics(official[name], getattr(native, name))
    for name in ("topk_indices", "candidates"):
        expected, actual = official[name], getattr(native, name)
        if expected.shape != actual.shape:
            raise ValueError(f"Shared {name} layouts differ: {expected.shape} != {actual.shape}")
        result[name] = {
            "shape": list(expected.shape),
            "exact": torch.equal(expected, actual),
            "element_agreement": (expected == actual).float().mean().item(),
        }
    expected_indices, actual_indices = official["topk_indices"], native.topk_indices
    # Set overlap avoids exaggerating differences from one sorted index shifting
    # the rest of a row. Sentinel -1 occupies a dedicated column and is excluded.
    membership = []
    for indices in (expected_indices, actual_indices):
        mask = torch.zeros(*indices.shape[:2], sequence + 1, device=indices.device, dtype=torch.bool)
        membership.append(mask.scatter(-1, indices.masked_fill(indices < 0, sequence), True)[..., :sequence])
    intersection = (membership[0] & membership[1]).sum().item()
    union = (membership[0] | membership[1]).sum().item()
    result["topk_indices"]["selected_set_jaccard"] = intersection / max(union, 1)
    return result


def _identity_report(
    reference: ModuleType,
    before_official: tuple,
    before_native: DeepseekV41AttentionState,
    native: DeepseekV41AttentionState,
    *,
    layer_id: int,
) -> dict:
    """Verify Full publication, Reuse identity and Reindex-only selection replacement."""
    names = ("compress_kv", "index_k", "topk_idxs", "candidates")
    native_names = ("compressed_kv", "index_keys", "topk_indices", "candidates")
    result = {"official": {}, "native": {}}
    for name, native_name, old in zip(names, native_names, before_official):
        result["official"][native_name] = old is getattr(reference.shared_attn, name)
        result["native"][native_name] = getattr(before_native, native_name) is getattr(native, native_name)
    expected = {
        "compressed_kv": layer_id != 20,
        "index_keys": layer_id != 20,
        "topk_indices": layer_id not in (20, 24),
        "candidates": layer_id != 20,
    }
    if result["official"] != expected or result["native"] != expected:
        raise ValueError(f"Layer {layer_id} violates published CSA2 ownership: {result}, expected={expected}")
    return result


def _reference_runtime(reference: ModuleType, options: _Options):
    """Set the documented official BF16 runtime globals and preserve all 40 layer IDs."""
    args = reference.ModelArgs(**json.loads((options.reference_dir / "config.json").read_text()))
    args = replace(args, dtype="bf16", expert_dtype=None, max_batch_size=1, max_seq_len=options.sequence_length)
    reference.world_size = 1
    reference.rank = 0
    reference.default_dtype = torch.bfloat16
    reference.shared_attn = reference.SharedAttentionRuntime()
    return args


@torch.inference_mode()
def _full_vocabulary_readout(
    reference: ModuleType,
    config: DeepseekV41Config,
    adapter: DeepseekV41StateDictAdapter,
    options: _Options,
    reference_hidden: torch.Tensor,
    native_hidden: torch.Tensor,
) -> dict:
    """Read both decoder continuations through independently loaded final norms/heads.

    Args:
        reference: Unmodified pinned official module.
        config: Released architecture configuration.
        adapter: Production checkpoint adapter used by the native blocks.
        options: Scope and artifact destinations for this run.
        reference_hidden: Official collapsed [1, sequence, hidden] stream.
        native_hidden: Native collapsed stream with the same layout.

    Returns:
        Full-vocabulary/all-position metrics and strict readout state audits.
        This measures five-block continuations sharing the supplied prefix;
        it is not full native 25-layer end-to-end parity.
    """
    text = config.text_config
    device = reference_hidden.device
    with torch.device(device), reference.set_dtype(torch.bfloat16):
        official = nn.Module()
        official.norm = reference.RMSNorm(text.hidden_size, text.rms_norm_eps)
        official.head = reference.ParallelHead(text.vocab_size, text.hidden_size, text.rms_norm_eps, text.hc_eps)
        native = nn.Module()
        native.model = nn.Module()
        native.model.norm = DeepseekV41RMSNorm(text.hidden_size, text.rms_norm_eps, torch.bfloat16)
        native.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False, dtype=torch.float32)
    official_audit = _load_official(official, options.checkpoint)
    native_audit = adapter.load_from_checkpoint(native, options.checkpoint)
    if set(native_audit.loaded_keys) != {"norm.weight", "head.weight"}:
        raise ValueError("The decoder readout did not load exactly the final norm and complete head")
    reference_norm = official.norm(reference_hidden)
    native_norm = native.model.norm(native_hidden)
    reference_logits = official.head(reference_norm, full_logits=True)
    native_logits = native.lm_head(native_norm.float())
    metrics, positions = _logit_metrics(reference_logits, native_logits, chunk_size=32)
    save_file(positions, str(options.output.with_suffix(".positions.safetensors")))
    return {
        "scope": "Five-block decoder continuation with a shared supplied prefix; not full native 25-layer end-to-end parity",
        "all_positions_and_vocabulary_entries": True,
        "logits": metrics,
        "final_norm": _metrics(reference_norm, native_norm),
        "state_audit": {
            "official": asdict(official_audit),
            "native": {
                **asdict(native_audit),
                "expected": 2,
                "loaded": len(native_audit.loaded_keys),
                "missing": [],
                "unexpected": [],
                "shape_mismatches": [],
            },
        },
    }


def _pruning_diagnostic(
    reference: ModuleType,
    reference_args,
    config: DeepseekV41Config,
    backend: BackendConfig,
    adapter: DeepseekV41StateDictAdapter,
    options: _Options,
    inputs: dict[int, torch.Tensor],
    device: torch.device,
) -> dict:
    """Exercise real hierarchical pruning with a separately labeled candidate cap.

    Attention receives the saved official attn_norm inputs from blocks 20 and 24
    on both sides. Only the candidate cap differs from the released architecture;
    index_topk, all projection weights, and every forward operation are retained.
    """
    reference.shared_attn = reference.SharedAttentionRuntime()
    native_state = DeepseekV41AttentionState()
    candidate_args = replace(reference_args, candidate_topk_blocks=options.candidate_topk_blocks)
    candidate_config = copy.deepcopy(config.text_config)
    candidate_config.candidate_topk_blocks = options.candidate_topk_blocks
    positions = torch.arange(options.sequence_length, device=device).unsqueeze(0)
    report = {
        "scope": "Attention-only diagnostic with modified candidate_topk_blocks; not released-configuration parity",
        "released_candidate_topk_blocks": config.text_config.candidate_topk_blocks,
        "diagnostic_candidate_topk_blocks": options.candidate_topk_blocks,
        "index_topk": config.text_config.index_topk,
        "candidate_block_size": config.text_config.candidate_block_size,
        "inputs": "Identical saved official attn_norm inputs at layers 20 and 24",
        "layers": {},
    }
    for layer_id in (20, 24):
        with torch.device(device), reference.set_dtype(torch.bfloat16):
            official = reference.Attention(layer_id, candidate_args)
            native = DeepseekV41Attention(candidate_config, layer_id, backend)
        audit = _load_pair(official, native, adapter, options, layer_id, attention_only=True)
        hidden = inputs[layer_id].to(device)
        before_native = native_state
        before_official = tuple(
            getattr(reference.shared_attn, name) for name in ("compress_kv", "index_k", "topk_idxs", "candidates")
        )
        with torch.inference_mode(), torch.device(device), reference.set_dtype(torch.bfloat16):
            expected = official(hidden, 0)
            actual = native(hidden, position_ids=positions, state=native_state)
        native_state = actual.state
        candidates = native_state.candidates
        visible = positions.unsqueeze(-1) >= positions.unsqueeze(1)
        pruned_visible = (~candidates & visible).sum().item()
        if pruned_visible == 0:
            raise ValueError("The configured candidate diagnostic did not prune any visible positions")
        newest = torch.gather(candidates, -1, positions.unsqueeze(-1)).all().item()
        if not newest:
            raise ValueError("The candidate source did not retain every query's newest block")
        official_state = _official_state(reference, options.sequence_length)
        for indices, own_candidates in (
            (native_state.topk_indices, candidates),
            (official_state["topk_indices"], official_state["candidates"]),
        ):
            # The source publishes candidates but selects its own top-k from the
            # full scores; only later Reindex layers restrict to those blocks.
            if layer_id == 24:
                selected_candidates = own_candidates.gather(-1, indices.clamp_min(0))
                if not torch.all(selected_candidates | (indices < 0)):
                    raise ValueError("The Reindex diagnostic selected a position outside its retained blocks")
            if not torch.all((indices < 0) | (indices <= positions.unsqueeze(-1))):
                raise ValueError("The candidate diagnostic selected a future position")
        report["layers"][str(layer_id)] = {
            "state_audit": audit,
            "attention_output": _metrics(expected, actual.hidden_states),
            "shared_state": _state_report(reference, native_state, options.sequence_length),
            "identity_preserved": _identity_report(
                reference, before_official, before_native, native_state, layer_id=layer_id
            ),
            "visible_positions_pruned": pruned_visible,
            "newest_block_pinned": newest,
        }
        LOGGER.info("Candidate-cap layer %d: %s", layer_id, report["layers"][str(layer_id)]["attention_output"])
        del official, native, expected, actual, hidden, before_official, before_native
        gc.collect()
        torch.cuda.empty_cache()
    return report


def main() -> int:
    """Run or inspect the pinned released decoder-window diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--candidate-topk-blocks", type=int, default=80)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa", "tilelang"), default="sdpa")
    parser.add_argument("--expert-backend", choices=("torch_mm", "torch_linear"), default="torch_mm")
    parser.add_argument("--check-layout-only", action="store_true")
    parser.add_argument("--reference-artifact", type=Path)
    parser.add_argument(
        "--diagnostic-moe-fp32-shared-add",
        action="store_true",
        help="Native-only arithmetic diagnostic; results do not establish production parity",
    )
    options = _Options(**vars(parser.parse_args()))
    if options.diagnostic_moe_fp32_shared_add and options.expert_backend != "torch_mm":
        raise ValueError("The earlier grouped-MM late-add diagnostic requires --expert-backend=torch_mm")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    config = DeepseekV41Config.from_pretrained(options.checkpoint, local_files_only=True)
    text_config = config.text_config
    capacity = options.candidate_topk_blocks * text_config.candidate_block_size
    if capacity < text_config.index_topk + text_config.candidate_block_size:
        raise ValueError("Candidate cap must hold index_topk plus the newest partial block")
    if options.sequence_length <= capacity:
        raise ValueError("Sequence length must exceed the diagnostic candidate capacity to exercise pruning")
    if any(text_config.compress_ratios[layer] != 1 for layer in LAYER_IDS):
        raise ValueError("The pinned decoder window must use ratio-1 compressed attention")
    backend = BackendConfig(
        attn=options.attention_backend,
        linear="torch",
        rms_norm="torch_fp32",
        experts=options.expert_backend,
        dispatcher="torch",
        gate_precision="float32",
    )
    blocks, adapter = _native_skeleton(config, backend)
    reference = _import_reference(options.reference_dir)
    reference_args = _reference_runtime(reference, options)
    if options.check_layout_only:
        report = {}
        for layer_id in LAYER_IDS:
            block = blocks[str(layer_id)]
            report[str(layer_id)] = {
                "compress_ratio": block.attn.compress_ratio,
                "is_kv_source": block.attn.is_kv_source,
                "is_index_source": block.attn.is_index_source,
                "native_parameters": sum(parameter.numel() for parameter in block.parameters()),
                "original_source_tensors": len(_source_scope(options.checkpoint, layer_id, attention_only=False)),
                "all_meta": all(parameter.is_meta for parameter in block.parameters()),
            }
        print(json.dumps(report, indent=2))
        return 0
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one available GPU for this sequential decoder-window diagnostic")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(options.seed)
    torch.cuda.reset_peak_memory_stats()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    initial_hidden, initial_mix, input_provenance = _initial_decoder_state(options, config)
    input_file = options.output.with_suffix(".inputs.safetensors")
    save_file({"hidden_states": initial_hidden, "pre_mix": initial_mix}, str(input_file))
    reference_hidden, native_hidden = initial_hidden.to(device), initial_hidden.to(device)
    reference_mix, native_mix = initial_mix.to(device), initial_mix.to(device)
    positions = torch.arange(options.sequence_length, device=device).unsqueeze(0)
    native_state = DeepseekV41AttentionState()
    attention_inputs: dict[int, torch.Tensor] = {}
    report = {
        "reference_revision": REVISION,
        "reference_source_hashes": REFERENCE_HASHES,
        "native_source_hashes_before_forward": _native_source_manifest(),
        "checkpoint": str(options.checkpoint),
        "scope": "Released-weight decoder blocks 20--24 on saved hidden states; input provenance below distinguishes authentic prefix from seeded component inputs",
        "configuration": {
            "layer_ids": list(LAYER_IDS),
            "sequence_length": options.sequence_length,
            "seed": options.seed if options.reference_artifact is None else None,
            "weight_compute_dtype": "bfloat16",
            "official_dtype_argument": "bf16",
            "official_expert_dtype_argument": None,
            "official_forward_operations_modified": False,
            "native_attention_backend": options.attention_backend,
            "native_experts": options.expert_backend,
            "native_dispatcher": "torch (isolated single-GPU diagnostic)",
            "native_moe_fp32_shared_add_diagnostic": options.diagnostic_moe_fp32_shared_add,
            "production_parity_candidate": not options.diagnostic_moe_fp32_shared_add,
            "diagnostic_harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kv_index_activation_quantization": "Official FP8 window / NVFP4 compressed KV / MXFP4 index QK",
            "released_candidate_topk_blocks": text_config.candidate_topk_blocks,
            "released_index_topk": text_config.index_topk,
        },
        "inputs": {
            "path": str(input_file),
            "sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
            "provenance": input_provenance,
        },
        "layers": {},
    }
    for layer_id in LAYER_IDS:
        LOGGER.info("Loading released block pair %d", layer_id)
        with torch.device(device), reference.set_dtype(torch.bfloat16):
            official = reference.Block(layer_id, reference_args)
        native = blocks.pop(str(layer_id)).to_empty(device=device)
        audit = _load_pair(official, native, adapter, options, layer_id)
        diagnostic_hooks = _diagnostic_late_shared_add(native.ffn) if options.diagnostic_moe_fp32_shared_add else []
        official_observed, official_hooks = _observe(official)
        native_observed, native_hooks = _observe(native)
        before_native = native_state
        before_official = tuple(
            getattr(reference.shared_attn, name) for name in ("compress_kv", "index_k", "topk_idxs", "candidates")
        )
        started = time.monotonic()
        with torch.inference_mode(), torch.device(device), reference.set_dtype(torch.bfloat16):
            reference_hidden, reference_mix = official(reference_hidden, 0, reference_mix, None)
            torch.cuda.synchronize()
            reference_seconds = time.monotonic() - started
            started = time.monotonic()
            native_hidden, native_mix, native_state = native(
                native_hidden, native_mix, native_state, position_ids=positions
            )
            torch.cuda.synchronize()
            native_seconds = time.monotonic() - started
        for handle in official_hooks + native_hooks:
            handle.remove()
        boundaries = {name: _metrics(values, native_observed[name]) for name, values in official_observed.items()}
        boundaries["output"] = _metrics(reference_hidden, native_hidden)
        boundaries["next_pre_mix"] = _metrics(reference_mix, native_mix)
        report["layers"][str(layer_id)] = {
            "kind": "Full" if layer_id == 20 else "Reindex" if layer_id == 24 else "Reuse",
            "state_audit": audit,
            "boundaries": boundaries,
            "same_input_moe": _same_input_moe_report(official, native, official_observed, device),
            "shared_state": _state_report(reference, native_state, options.sequence_length),
            "identity_preserved": _identity_report(
                reference, before_official, before_native, native_state, layer_id=layer_id
            ),
            "official_forward_seconds": reference_seconds,
            "native_forward_seconds": native_seconds,
        }
        if layer_id in (20, 24):
            attention_inputs[layer_id] = official_observed["attn_norm"]
        if layer_id == LAYER_IDS[-1]:
            with torch.inference_mode():
                collapsed_reference = official.hc_pre(reference_hidden, reference_mix)
                collapsed_native = native.attn_hc.collapse(native_hidden, native_mix)
        options.output.write_text(json.dumps(report, indent=2) + "\n")
        LOGGER.info("Decoder layer %d output: %s", layer_id, boundaries["output"])
        for handle in diagnostic_hooks:
            handle.remove()
        del official, native, official_observed, native_observed, before_official, before_native
        gc.collect()
        torch.cuda.empty_cache()
    report["full_vocabulary_readout"] = _full_vocabulary_readout(
        reference, config, adapter, options, collapsed_reference, collapsed_native
    )
    gate_audits = dict(report["full_vocabulary_readout"]["state_audit"])
    for layer_id, layer_report in report["layers"].items():
        gate_audits[f"layer.{layer_id}.official"] = layer_report["state_audit"]["official"]
        native_audit = layer_report["state_audit"]["native"]
        gate_audits[f"layer.{layer_id}.native"] = {
            **native_audit,
            "expected": native_audit["expected_source_tensors"],
            "loaded": len(native_audit["loaded_keys"]),
        }
    report["gates"] = _evaluate_gates(report["full_vocabulary_readout"]["logits"], gate_audits)
    LOGGER.info("Decoder full-vocabulary readout: %s", report["full_vocabulary_readout"]["logits"])
    LOGGER.info("Predetermined decoder continuation gates: %s", report["gates"])
    save_file(
        {f"layer_{layer}_attention_input": hidden for layer, hidden in attention_inputs.items()},
        str(options.output.with_suffix(".attention_inputs.safetensors")),
    )
    report["candidate_pruning_diagnostic"] = _pruning_diagnostic(
        reference, reference_args, config, backend, adapter, options, attention_inputs, device
    )
    report["runtime"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "automodel_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    report["native_source_files_changed_during_forward"] = sorted(
        name
        for name, digest in _native_source_manifest().items()
        if report["native_source_hashes_before_forward"].get(name) != digest
    )
    if report["native_source_files_changed_during_forward"]:
        raise ValueError(
            "Native source changed during the run; keep production files fixed for a reviewable parity audit"
        )
    report["completed"] = True
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("Decoder-window and candidate-pruning evidence written to %s", options.output)
    return 0 if report["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
