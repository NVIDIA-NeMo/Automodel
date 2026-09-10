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

"""Continuous native backbone parity with one resident EP/FSDP Block at a time.

The oracle artifact comes from run_streaming_reference_parity.py --mode reference.
Native production Blocks execute unchanged with HybridEP, owner-sharded Engram,
and the shared dtype-aware FSDP parallelizer. Only their lifetime is shortened:
streams, mHC coefficients and immutable CSA2 state continue across all layers.
Shared embedding/norm/head parameters are replicated. This is a streamed forward
comparison, not a resident full-backbone FSDP training or checkpoint-resume test.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from run_reference_parity import (
    PARITY_THRESHOLDS,
    REFERENCE_HASHES,
    REVISION,
    _capture_outputs,
    _evaluate_gates,
    _logit_metrics,
    _native_source_manifest,
    _StateAudit,
    _tensor_metrics,
)
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


class _BlockContainer(nn.Module):
    """Retain the native block namespace and invoke its ordinary forward hook."""

    def __init__(self, block: nn.Module, index: int, config: Any, moe_config: Any) -> None:
        super().__init__()
        self.config = config.text_config
        self.moe_config = moe_config
        self.layers = nn.ModuleDict({str(index): block})
        self.norm = nn.Identity()  # Parameter-free observer interface; never executed.
        self.engram_hash = None
        self.index = str(index)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call the single unchanged production Block with its original arguments."""
        return self.layers[self.index](*args, **kwargs)


class _BlockSlice(nn.Module):
    """Namespace adapter for the public MoE parallelizer and checkpoint adapter."""

    def __init__(self, block: nn.Module, index: int, template: nn.Module) -> None:
        super().__init__()
        from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepseekV41StateDictAdapter

        self.config = template.config
        self.backend = template.backend
        self.moe_config = template.moe_config
        self._keep_in_fp32_modules_strict = template._keep_in_fp32_modules_strict
        self.model = _BlockContainer(block, index, self.config, self.moe_config)
        self.state_dict_adapter = DeepseekV41StateDictAdapter(
            self.config, self.moe_config, self.backend, dtype=torch.bfloat16
        )

    def _nemo_prepare_model_owned_dtensors(self, fsdp_mesh: Any) -> set[nn.Parameter]:
        """Reuse the production model's Engram ownership hook without changes."""
        from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM

        return DeepseekV41ForCausalLM._nemo_prepare_model_owned_dtensors(self, fsdp_mesh)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Enter both ordinary FSDP roots before calling the production Block."""
        return self.model(*args, **kwargs)


def _load_slice(model: nn.Module, checkpoint: Path, ep_mesh: Any) -> dict[str, Any]:
    """Load every materialized destination using the strict streaming adapter."""
    expected = len(model.state_dict())
    started = time.monotonic()
    loaded = model.state_dict_adapter.load_from_checkpoint(model, checkpoint, device_mesh=ep_mesh)
    torch.cuda.synchronize()
    return asdict(
        _StateAudit(
            expected,
            expected,
            (),
            (),
            (),
            len(loaded.loaded_keys),
            loaded.source_bytes,
            time.monotonic() - started,
            loaded.loaded_bytes,
            loaded.max_chunk_source_bytes,
            loaded.max_chunk_output_bytes,
        )
    )


def _storage_census(model: nn.Module, *, require_recipe_dtype: bool) -> dict[str, Any]:
    """Inspect actual DTensor local storage, including strict FP32 exceptions."""
    entries = {}
    strict = model._keep_in_fp32_modules_strict
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        expected = torch.float32 if any(keyword in name for keyword in strict) else torch.bfloat16
        matches = not parameter.is_floating_point() or parameter.dtype == local.dtype == expected
        entries[name] = {
            "global_dtype": str(parameter.dtype),
            "local_dtype": str(local.dtype),
            "global_shape": list(parameter.shape),
            "local_shape": list(local.shape),
            "local_tensor_bytes": local.numel() * local.element_size(),
            "local_storage_bytes": local.untyped_storage().nbytes(),
            "expected_dtype": str(expected),
            "matches_recipe": matches,
        }
        if require_recipe_dtype and not matches:
            raise RuntimeError(f"Streamed parameter dtype differs from recipe: {name}: {entries[name]}")
    return {
        "parameters": entries,
        "all_match_recipe": all(item["matches_recipe"] for item in entries.values()),
        "total_local_tensor_bytes": sum(item["local_tensor_bytes"] for item in entries.values()),
    }


def _capture_compute_dtypes(model: nn.Module) -> tuple[dict[str, Any], list[Any]]:
    """Observe production module inputs, outputs and weights without substitution."""
    observed, handles = {}, []
    selected = ("attn_hc", "ffn_hc", "attn_norm", "ffn_norm", "sinks_param", "experts")

    def make_hook(name: str) -> Any:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            inputs = [value for value in inputs if isinstance(value, torch.Tensor)]
            weights = {}
            for key, parameter in module.named_parameters(recurse=False):
                local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
                weights[key] = {"global_dtype": str(parameter.dtype), "local_dtype": str(local.dtype)}
            observed[name] = {
                "input_dtypes": [str(value.dtype) for value in inputs],
                "output_dtype": str(output.dtype) if isinstance(output, torch.Tensor) else None,
                "parameters": weights,
            }

        return hook

    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in selected:
            handles.append(module.register_forward_hook(make_hook(name)))
    return observed, handles


def _run(args: argparse.Namespace, device: torch.device, rank: int, world: int) -> int:
    """Carry the natural native activations through every requested source layer."""
    source_hashes = _native_source_manifest()
    from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
    from nemo_automodel.components.distributed.config import MoEParallelizerConfig
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.common.utils import cast_model_to_dtype
    from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41AttentionState
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection
    from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from nemo_automodel.components.moe.parallelizer import parallelize_model

    reference_metadata = json.loads(args.artifact.with_suffix(".json").read_text())
    if (
        reference_metadata["reference_revision"] != REVISION
        or reference_metadata["reference_source_hashes"] != REFERENCE_HASHES
        or reference_metadata["configuration"]["layers"] != args.num_layers
        or reference_metadata["configuration"]["sequence_length"] != args.sequence_length
    ):
        raise ValueError("Reference revision, source hashes, or backbone scope differs from the requested comparison")
    config = DeepseekV41Config.from_pretrained(args.checkpoint, local_files_only=True)
    if not 1 <= args.num_layers <= config.text_config.num_hidden_layers:
        raise ValueError("Requested layer count must be a nonempty prefix of the released backbone")
    config.text_config.num_hidden_layers = args.num_layers
    config.vision_config.num_hidden_layers = 0
    config.name_or_path = str(args.checkpoint)
    backend = BackendConfig(
        attn="tilelang",
        linear="torch",
        rms_norm="torch_fp32",
        experts=args.expert_backend,
        dispatcher="hybridep",
        gate_precision="float32",
    )
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
    )
    setup = DistributedSetup.build(
        strategy=FSDP2Config(),
        parallelism_sizes=ParallelismSizes(ep_size=world),
        moe_parallel_config=MoEParallelizerConfig(
            reshard_after_forward=True, lm_head_precision="float32", mp_policy=policy
        ),
        activation_checkpointing=False,
        world_size=world,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    # Construct the exact production hierarchy on meta once. Removing Blocks
    # changes only residency; no production module or forward is substituted.
    with torch.device("meta"):
        shared = DeepseekV41ForCausalLM(config, backend=backend, tokenizer=tokenizer)
    pending = {int(index): block for index, block in shared.model.layers.items()}
    shared.model.layers = nn.ModuleDict()
    shared.to_empty(device=device)
    storage_audits = {"native.shared": {"before_cast": _storage_census(shared, require_recipe_dtype=False)}}
    # Match initialize_weights' public cast at the materialized/pre-load stage.
    # Direct grouped-expert constructors otherwise retain default FP32 storage.
    cast_model_to_dtype(shared, torch.bfloat16)
    if shared.model.engram_hash is not None:
        shared.model.engram_hash.init_weights()
    shared.eval().requires_grad_(False)
    audits = {"native.shared": _load_slice(shared, args.checkpoint, setup.mesh_context.moe_mesh["ep"])}
    storage_audits["native.shared"]["after_load"] = _storage_census(shared, require_recipe_dtype=True)
    compute_audits = {}
    activations: dict[str, Any] = {}
    finite_layers: dict[str, bool] = {}
    layer_timings: dict[str, float] = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    # Keep newly materialized FSDP Parameters as ordinary tensors; constructing
    # them inside inference_mode would remove the version counters FSDP owns.
    with safe_open(args.artifact, framework="pt", device="cpu") as reference, torch.no_grad():
        input_ids_cpu = reference.get_tensor("input_ids")
        if input_ids_cpu.shape != (1, args.sequence_length):
            raise ValueError("The oracle must contain exactly one full-sequence sample")
        if hashlib.sha256(input_ids_cpu.numpy().tobytes()).hexdigest() != reference_metadata["input_sha256"]:
            raise ValueError("The saved oracle token checksum does not match its metadata")
        tokens = input_ids_cpu.to(device)
        mask = torch.ones_like(tokens) if rank == 0 else torch.zeros_like(tokens)
        positions = torch.arange(args.sequence_length, device=device).expand_as(tokens)
        hashes = (
            shared.model.engram_hash(tokens, token_mask=mask.bool(), position_ids=positions)
            if shared.model.engram_hash is not None
            else None
        )
        if rank == 0 and hashes is not None:
            if not torch.equal(hashes.cpu(), reference.get_tensor("engram_hash")):
                raise ValueError("Native full-backbone Engram hashes differ from the official oracle")
        hidden = shared.model.embed_tokens(tokens).unsqueeze(2).expand(-1, -1, config.text_config.hc_mult, -1)
        pre_mix = torch.zeros(*tokens.shape, config.text_config.hc_mult, device=device, dtype=torch.float32)
        pre_mix[..., 0] = 1
        state = DeepseekV41AttentionState()
        for index in range(args.num_layers):
            layer_started = time.monotonic()
            LOGGER.info("Preparing native EP%d/FSDP Block %d", world, index)
            block_slice = _BlockSlice(pending.pop(index), index, shared)
            parallelize_model(
                block_slice,
                setup.mesh_context.device_mesh,
                setup.mesh_context.moe_mesh,
                **setup.mesh_context.parallelize_axis_kwargs(),
                activation_checkpointing=False,
                reshard_after_forward=True,
                lm_head_precision="float32",
                mp_policy=policy,
            )
            block_slice.to_empty(device=device)
            storage_audits[f"native.layer.{index}"] = {
                "before_cast": _storage_census(block_slice, require_recipe_dtype=False)
            }
            cast_model_to_dtype(block_slice, torch.bfloat16)
            block = block_slice.model.layers[str(index)]
            if block.engram is not None:
                block.engram.embed.mark_sharding_contract()
            audits[f"native.layer.{index}"] = _load_slice(
                block_slice, args.checkpoint, setup.mesh_context.moe_mesh["ep"]
            )
            storage_audits[f"native.layer.{index}"]["after_load"] = _storage_census(
                block_slice, require_recipe_dtype=True
            )
            LOGGER.info(
                "Native Block %d loaded parameter storage: %s bytes; recipe dtype audit passed",
                index,
                storage_audits[f"native.layer.{index}"]["after_load"]["total_local_tensor_bytes"],
            )
            block_slice.eval().requires_grad_(False)
            captured, handles = _capture_outputs(block_slice, native=True) if rank == 0 else ({}, [])
            compute_audits[str(index)], dtype_handles = _capture_compute_dtypes(block_slice)
            hidden, pre_mix, state = block_slice(
                hidden,
                pre_mix,
                state,
                position_ids=positions,
                attention_mask=mask,
                engram_hash_ids=None if block.engram is None else hashes[:, :, block.engram.layer_hash_index],
            )
            torch.cuda.synchronize()
            finite_layers[str(index)] = bool(
                torch.isfinite(hidden).all()
                and torch.isfinite(pre_mix).all()
                and (state.compressed_kv is None or torch.isfinite(state.compressed_kv).all())
                and (state.index_keys is None or torch.isfinite(state.index_keys).all())
            )
            for handle in handles + dtype_handles:
                handle.remove()
            for name, value in captured.items():
                activations[name] = asdict(_tensor_metrics(reference.get_tensor(name), value))
            if rank == 0:
                LOGGER.info(
                    "Native Block %d boundary comparisons: %s",
                    index,
                    {name: activations[name]["max_absolute_difference"] for name in captured},
                )
            layer_timings[str(index)] = time.monotonic() - layer_started
            LOGGER.info(
                "Native Block %d complete: finite=%s; seconds=%.2f; allocated=%.2f GB",
                index,
                finite_layers[str(index)],
                layer_timings[str(index)],
                torch.cuda.memory_allocated() / 1e9,
            )
            del captured, handles, dtype_handles, block, block_slice
            gc.collect()
            torch.cuda.empty_cache()
        normalized = shared.model.norm(DeepseekV41HyperConnection.collapse(hidden, pre_mix))
        logits = shared.lm_head(normalized.float())
        torch.cuda.synchronize()
        finite_logits = bool(torch.isfinite(logits).all())
        metrics, per_position = None, None
        if rank == 0:
            activations["final_norm"] = asdict(_tensor_metrics(reference.get_tensor("final_norm"), normalized.cpu()))
            activations["final_streams"] = asdict(_tensor_metrics(reference.get_tensor("final_streams"), hidden.cpu()))
            activations["final_pre_mix"] = asdict(_tensor_metrics(reference.get_tensor("final_pre_mix"), pre_mix.cpu()))
            metrics, per_position = _logit_metrics(
                reference.get_tensor("logits"), logits, chunk_size=args.metric_chunk_size
            )

    all_audits = {**audits, **reference_metadata["state_audits"]}
    state_coverage = all(
        item["expected"] == item["loaded"]
        and not any(item[key] for key in ("missing", "unexpected", "shape_mismatches"))
        for item in all_audits.values()
    )
    finite = finite_logits and all(finite_layers.values())
    gates = (
        _evaluate_gates(metrics, all_audits)
        if rank == 0
        else {"passed": None, "checks": {"state_coverage": state_coverage, "all_positions_finite": finite}}
    )
    if rank == 0:
        gates["checks"]["all_streamed_layer_outputs_finite"] = finite
        gates["passed"] = bool(gates["passed"] and finite)
    final_source_hashes = _native_source_manifest()
    changed = [key for key, value in source_hashes.items() if final_source_hashes.get(key) != value]
    if changed:
        raise RuntimeError(f"Production source changed during streamed parity: {changed}")
    report = {
        "scope": "Continuous native streamed backbone forward; one actual EP/FSDP production Block resident at a time",
        "reference_artifact": str(args.artifact),
        "reference_metadata": reference_metadata,
        "rank": rank,
        "configuration": {
            "layers": args.num_layers,
            "sequence_length": args.sequence_length,
            "native_tp": 1,
            "native_ep": world,
            "native_engram_owners": world,
            "native_dispatcher": "hybridep",
            "expert_backend": args.expert_backend,
            "attention_backend": "tilelang",
            "shared_embedding_norm_head": "replicated",
            "block_fsdp": True,
            "native_global_valid_batch": 1,
            "input_distribution": "rank0_valid_other_ranks_all_masked",
            "numerical_comparison_performed": rank == 0,
            "activation_checkpointing": False,
            "training_or_resident_backbone_validation": False,
            "parameter_storage": "BF16 except production strict FP32 modules; audited global and local storage",
            "dtype_lifecycle": "public cast_model_to_dtype after materialization/FSDP and before strict weight loading",
        },
        "thresholds": PARITY_THRESHOLDS,
        "logits": metrics,
        "activations": activations,
        "state_audits": audits,
        "storage_dtype_audits": storage_audits,
        "compute_dtype_audits": compute_audits,
        "finite_layers": finite_layers,
        "gates": gates,
        "native_source_hashes_before_import": source_hashes,
        "native_source_files_changed_during_forward": changed,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "layer_seconds": layer_timings,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()},
    }
    rank_output = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    rank_output.write_text(json.dumps(report, indent=2) + "\n")
    if per_position is not None:
        save_file(per_position, str(rank_output.with_suffix(".positions.safetensors")))
    reports = [None] * world if rank == 0 else None
    dist.gather_object(report, reports, dst=0)
    passed = None
    if rank == 0:
        passed = bool(reports[0]["gates"]["passed"]) and all(
            all(item["gates"]["checks"].values()) for item in reports[1:]
        )
        report["all_rank_state_audits"] = {str(item["rank"]): item["state_audits"] for item in reports}
        report["all_rank_storage_dtype_audits"] = {str(item["rank"]): item["storage_dtype_audits"] for item in reports}
        report["all_rank_compute_dtype_audits"] = {str(item["rank"]): item["compute_dtype_audits"] for item in reports}
        report["all_rank_gates"] = {str(item["rank"]): item["gates"] for item in reports}
        report["all_rank_validation_passed"] = passed
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        LOGGER.info("Streamed backbone fixed parity gates: %s; logits: %s", passed, metrics)
    result = [passed]
    dist.broadcast_object_list(result, src=0)
    return 0 if result[0] else 1


def main() -> int:
    """Parse one-node EP4 comparison settings and execute the streamed backbone."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--metric-chunk-size", type=int, default=32)
    parser.add_argument("--expert-backend", choices=("torch_mm", "torch_linear"), default="torch_mm")
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 4 or int(os.environ.get("LOCAL_WORLD_SIZE", "4")) != world:
        raise ValueError("This residency-bounded diagnostic requires one node with EP4/Engram-owner4")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank={rank} %(levelname)s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    try:
        return _run(args, device, rank, world)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
