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

"""Four-rank CPU Checkpointer tests using explicit DTensor layouts."""

from __future__ import annotations

import gc
import json
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.ernie5_5.mtp_training import (
    Ernie5_5MTPForCausalLM,
)
from tests.unit_tests.models.ernie5_5 import _mtp_checkpoint_common as rc

WORLD = 4
HEADS = 4
EXPERT_NAMES = ("moe.experts.gate_and_up_projs", "moe.experts.down_projs")
PLAIN_NAMES = frozenset({"mtp.router_state.e_score_correction_bias"})
EMBED = "model.embed_tokens.weight"
MTP_BIAS = rc.MTP_PREFIX + rc.BIAS_SUFFIX
DECODER_BIAS = rc.BODY_PREFIX + rc.BIAS_SUFFIX
DENSE_ALIAS = rc.MTP_PREFIX + "input_layernorm.weight"
EXPERT_ALIAS = rc.MTP_PREFIX + "block_sparse_moe.experts.3.w2.weight"
EXPERT_BODY = rc.BODY_PREFIX + "block_sparse_moe.experts.3.w2.weight"
OWNER = rc.BODY_PREFIX + "self_attn.q_a_proj.weight"
OWNER_ALIAS = rc.MTP_PREFIX + "self_attn.q_a_proj.weight"
MISSING = rc.MTP_PREFIX + "e_proj.weight"
# Only this rank raises in its finisher; other ranks have clean local loads.
FINISHER_RANK = 2
FINISHER_MESSAGE = "finisher failed in checkpoint test"
PHASES = {
    "roundtrip": ("p1", "p2", "p3", "p4", "p9"),
    "value_mismatch": ("n_dense", "n_expert", "n_owner", "n_missing"),
    "failure_agreement": ("n_finisher", "n_comparison"),
    "diagnostic": ("p1", "n_expert"),
}
# Phase groups each layout runs, in order.
LAYOUT_GROUPS = {
    "ep_shard_ep": ("roundtrip", "value_mismatch", "failure_agreement"),
    "ep": ("roundtrip", "value_mismatch", "failure_agreement"),
    "embed2d": ("diagnostic",),
}
PHASE_NAMES = {
    "p1": "p1_clean_load",
    "p2": "p2_export_objects",
    "p3": "p3_save_and_resume_load",
    "p4": "p4_consolidated_export",
    "n_dense": "n_changed_dense_copy",
    "n_expert": "n_changed_expert_copy",
    "n_owner": "n_changed_body_owner",
    "n_missing": "n_missing_mtp_key",
    "n_finisher": "n_finisher_fails_on_one_rank",
    "n_comparison": "n_comparison_fails_on_one_rank",
    "p9": "p9_key_plan_after_sharding",
}


def local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def local_bytes(tensor: torch.Tensor) -> int:
    piece = local(tensor)
    return int(piece.numel() * piece.element_size())


def make_meshes(layout: str):
    dp_mesh = init_device_mesh("cpu", (WORLD,), mesh_dim_names=("dp",))
    if layout == "ep_shard_ep":
        ep_mesh = init_device_mesh("cpu", (WORLD // 2, 2), mesh_dim_names=("ep_shard", "ep"))
    else:
        ep_mesh = init_device_mesh("cpu", (WORLD,), mesh_dim_names=("ep",))
    embed_mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_a", "dp_b")) if layout == "embed2d" else None
    return dp_mesh, ep_mesh, embed_mesh


def shard(name: str, value: torch.Tensor, meshes) -> torch.Tensor:
    dp_mesh, ep_mesh, embed_mesh = meshes
    full = value.detach().clone().contiguous()
    if name in PLAIN_NAMES:
        return full
    if name == EMBED and embed_mesh is not None:
        return distribute_tensor(full, embed_mesh, [Shard(0), Shard(1)])
    if not name.endswith(EXPERT_NAMES):
        return distribute_tensor(full, dp_mesh, [Shard(0)])
    placements = [Shard(0)] if tuple(ep_mesh.mesh_dim_names) == ("ep",) else [Shard(1), Shard(0)]
    return distribute_tensor(full, ep_mesh, placements)


def shard_model(model: torch.nn.Module, values: dict[str, torch.Tensor], meshes) -> dict[str, torch.Tensor]:
    """Replace model parameters and buffers with NaN-filled shards; return expected local values."""
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    expected = {}
    for name, value in values.items():
        piece = shard(name, value, meshes)
        expected[name] = local(piece).clone()
        local(piece).fill_(float("nan"))
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        if name in parameters:
            setattr(
                parent,
                attribute,
                torch.nn.Parameter(piece, requires_grad=parameters[name].requires_grad),
            )
        else:
            if name not in buffers:
                raise KeyError(name)
            parent._buffers[attribute] = piece
    return expected


def instrument(adapter, stats: dict) -> None:
    """Observe the size of load sessions opened by this adapter without changing their behavior."""
    original = adapter.to_hf

    def to_hf(state_dict, **kwargs):
        result = original(state_dict, **kwargs)
        if kwargs.get("for_checkpoint_load"):
            pairs = getattr(result, "alias_pairs", {})
            same_layout = 0
            for scratch, destination in pairs.values():
                if isinstance(scratch, DTensor) and isinstance(destination, DTensor):
                    same_layout += (
                        tuple(scratch.placements) == tuple(destination.placements)
                        and scratch.device_mesh == destination.device_mesh
                        and scratch.shape == destination.shape
                        and local(scratch).shape == local(destination).shape
                        and scratch.dtype == destination.dtype
                    )
                elif not isinstance(scratch, DTensor) and not isinstance(destination, DTensor):
                    same_layout += scratch.shape == destination.shape and scratch.dtype == destination.dtype
            stats["load"] = {
                "destinations": len(result),
                "verification_pairs": len(pairs),
                "scratch_dtensors": sum(isinstance(scratch, DTensor) for scratch, _ in pairs.values()),
                "scratch_plain_tensors": sum(not isinstance(scratch, DTensor) for scratch, _ in pairs.values()),
                "scratch_with_the_layout_of_its_destination": int(same_layout),
                "scratch_bytes_on_this_rank": sum(local_bytes(scratch) for scratch, _ in pairs.values()),
                "body_destination_bytes_on_this_rank": sum(
                    local_bytes(destination) for _, destination in pairs.values()
                ),
                "all_destination_bytes_on_this_rank": sum(local_bytes(value) for value in result.values()),
                "holds_expert_3": EXPERT_BODY in result,
                "owner_local_elements": int(local(result[OWNER]).numel()) if OWNER in result else None,
                "consensus_groups": len(getattr(result, "groups", ())),
            }
        return result

    adapter.to_hf = to_hf


def build(source: torch.nn.Module, meshes):
    model = rc.attach(Ernie5_5MTPForCausalLM(rc.tiny_config("bfloat16", HEADS), rc.backend()))
    adapter = model.state_dict_adapter
    # Plan keys before sharding, matching the loader.
    model._pre_shard_hf_state_dict_keys = adapter.get_hf_state_dict_keys(model.state_dict())
    expected = shard_model(model, source.state_dict(), meshes)
    stats: dict = {}
    instrument(adapter, stats)
    return model, adapter, expected, stats


def local_difference(model: torch.nn.Module, expected: dict[str, torch.Tensor]) -> dict:
    state = model.state_dict()
    dtype_bad = [name for name in expected if name in state and local(state[name]).dtype != expected[name].dtype]
    value_bad = [name for name in expected if name in state and not torch.equal(local(state[name]), expected[name])]
    return {
        "same_names": set(state) == set(expected),
        "tensors": len(expected),
        "dtype_mismatches": len(dtype_bad),
        "value_mismatches": len(value_bad),
        "first_mismatch": (dtype_bad + value_bad)[:1],
    }


def base_load(model: torch.nn.Module, checkpoint: Path, root: Path, rank: int) -> None:
    checkpointer = CheckpointingConfig(
        enabled=False,
        save_consolidated=False,
        model_save_format="safetensors",
        model_repo_id="test/ernie5_5-mtp-tiny",
        checkpoint_dir=str(root / "run"),
    ).build(rank, 0, 0)
    try:
        checkpointer.load_base_model(model, torch.device("cpu"), str(root), str(checkpoint), load_base_model=True)
    finally:
        checkpointer.close()


def shard_files_report(directory: Path) -> dict:
    """On rank 0, check each completed shard file for exact alias equality with its decoder owner."""
    files = sorted(directory.glob("*.safetensors"))
    report = {
        "files": len(files),
        "alias_chunks": 0,
        "alias_chunks_unequal": 0,
        "alias_chunks_without_body": 0,
    }
    names: set[str] = set()
    holders = {"mtp_bias": 0, "hc_head_scale": 0, "hc_head_base": 0}
    for file in files:
        tensors = rc.load_file(str(file))
        names.update(tensors)
        for name, value in tensors.items():
            if name.startswith(rc.MTP_PREFIX) and name not in rc.OWNED_HF:
                report["alias_chunks"] += 1
                body = tensors.get(rc.body_name(name))
                if body is None:
                    report["alias_chunks_without_body"] += 1
                elif body.dtype != value.dtype or body.shape != value.shape or not torch.equal(body, value):
                    report["alias_chunks_unequal"] += 1
        holders["mtp_bias"] += MTP_BIAS in tensors
        holders["hc_head_scale"] += (rc.MTP_PREFIX + "hc_head_scale") in tensors
        holders["hc_head_base"] += (rc.MTP_PREFIX + "hc_head_base") in tensors
    report["distinct_names"] = len(names)
    report["files_holding"] = {key: int(value) for key, value in holders.items()}
    return report


def worker(rank: int, world_size: int, rendezvous: str, output_dir: str, layouts: tuple[str, ...]) -> None:
    torch.set_num_threads(1)
    result: dict = {"rank": rank, "layouts": {layout: {} for layout in layouts}}
    output = Path(output_dir) / f"rank{rank}.json"

    def phase(layout: str, name: str, function) -> None:
        records = result["layouts"][layout]
        try:
            records[name] = {"status": "ok", **function()}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001
            records[name] = {
                "status": "exception",
                "type": type(error).__name__,
                "message": str(error).strip()[-400:],
                "trace_tail": traceback.format_exc().splitlines()[-8:],
            }
        output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        print(f"RANK {rank} LAYOUT {layout} PHASE {name}: {records[name]['status']}", flush=True)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    try:
        root = Path(rendezvous).parent
        # Use the same seed and unsharded source model on every rank.
        source = Ernie5_5MTPForCausalLM(rc.tiny_config("bfloat16", HEADS), rc.backend())
        rc.fill(source, 71)
        reference = root / "ref"
        wanted = {short for layout in layouts for group in LAYOUT_GROUPS[layout] for short in PHASES[group]}
        if rank == 0:
            rc.write_reference(reference, source)
            if "n_dense" in wanted:
                rc.write_reference(root / "ref_dense", source, mutate=rc.change(DENSE_ALIAS))
            if "n_expert" in wanted:
                rc.write_reference(root / "ref_expert", source, mutate=rc.change(EXPERT_ALIAS))
            if "n_owner" in wanted:
                rc.write_reference(root / "ref_owner", source, mutate=rc.change(OWNER))
            if "n_missing" in wanted:
                rc.write_reference(root / "ref_missing", source, mutate=rc.drop(lambda name: name == MISSING))
        dist.barrier()
        # The phase functions below read ``meshes``, ``holder`` and ``work`` when
        # they are called; the loop at the end binds them for each layout. Every
        # phase builds its own model and adapter, or continues the clean load of
        # p1 in the same layout, so the layouts and their phase groups run back
        # to back in one process group.

        def p1_clean_load() -> dict:
            model, adapter, expected, stats = build(source, meshes)
            holder.update(model=model, adapter=adapter, expected=expected)
            state = model.state_dict()
            mtp_layout = {
                name: {
                    "kind": "DTensor" if isinstance(state[name], DTensor) else "plain",
                    "global_shape": list(state[name].shape),
                    "local_elements": int(local(state[name]).numel()),
                }
                for name in state
                if name.startswith("mtp.")
            }
            embedding_mesh = list(state[EMBED].device_mesh.mesh.shape)
            base_load(model, reference, root, rank)
            return {
                "candidate": rc.candidate_identity(),
                "hf_keys_planned_before_sharding": len(model._pre_shard_hf_state_dict_keys),
                "embedding_mesh_shape": embedding_mesh,
                "mtp_state_layout": mtp_layout,
                "load_session": stats.get("load"),
                "local_state": local_difference(model, expected),
            }

        def p2_export_objects() -> dict:
            model, adapter = holder["model"], holder["adapter"]
            exported = adapter.to_hf(model.state_dict())
            aliases = [name for name in exported if name.startswith(rc.MTP_PREFIX) and name not in rc.OWNED_HF]
            shared = [name for name in aliases if exported[name] is exported.get(rc.body_name(name))]
            head = exported[rc.MTP_PREFIX + "hc_head_fn"]
            return {
                "exported_keys": len(exported),
                "aliases": len(aliases),
                "aliases_same_object_as_body": len(shared),
                "alias_bytes_on_this_rank": sum(local_bytes(exported[name]) for name in aliases),
                "alias_bytes_not_shared_with_body": sum(
                    local_bytes(exported[name]) for name in aliases if name not in shared
                ),
                "own_expert_keys_including_copies": sum(rc.EXPERT_HF_MARK in name for name in exported),
                "hc_head_fn_shape": list(head.shape),
                "hc_head_fn_kind": "DTensor" if isinstance(head, DTensor) else "plain",
                "mtp_bias_kind": "DTensor" if isinstance(exported[MTP_BIAS], DTensor) else "plain",
                "mtp_bias_is_the_decoder_bias_object": exported[MTP_BIAS] is exported[DECODER_BIAS],
            }

        def p3_save_and_resume_load() -> dict:
            model, expected = holder["model"], holder["expected"]
            checkpointer = CheckpointingConfig(
                enabled=True,
                save_consolidated=False,
                is_async=False,
                model_save_format="safetensors",
                model_repo_id=str(reference),
                checkpoint_dir=str(work / "save_run"),
            ).build(rank, 0, 0)
            report: dict = {}
            try:
                checkpointer.save_model(model, str(work / "saved"), is_final_checkpoint=True)
                dist.barrier()
                with torch.no_grad():
                    for value in model.state_dict().values():
                        local(value).zero_()
                checkpointer.load_model(model, model_path=str(work / "saved" / "model"), is_init_step=False)
            finally:
                checkpointer.close()
            report["local_state_after_resume_load"] = local_difference(model, expected)
            if rank == 0:
                report["per_rank_files"] = shard_files_report(work / "saved" / "model")
            return report

        def p4_consolidated_export() -> dict:
            model = holder["model"]
            checkpointer = CheckpointingConfig(
                enabled=True,
                save_consolidated="final",
                is_async=False,
                model_save_format="safetensors",
                model_repo_id=str(reference),
                checkpoint_dir=str(work / "final_run"),
            ).build(rank, 0, 0)
            try:
                checkpointer.save_model(model, str(work / "saved_final"), is_final_checkpoint=True)
                dist.barrier()
            finally:
                checkpointer.close()
            if rank != 0:
                return {}
            consolidated_dir = work / "saved_final" / "model" / "consolidated"
            consolidated = rc.read_safetensors(consolidated_dir)
            reference_tensors = rc.read_safetensors(reference)
            config = json.loads((consolidated_dir / "config.json").read_text())
            return {
                "consolidated": rc.alias_report(consolidated),
                "consolidated_vs_reference": rc.state_difference(consolidated, reference_tensors),
                "reference_keys": len(reference_tensors),
                "config": {
                    key: config.get(key)
                    for key in (
                        "num_nextn_predict_layers",
                        "mtp_shared_last_layer",
                        "compress_ratios",
                    )
                },
            }

        def negative(directory: Path) -> dict:
            model, adapter, expected, stats = build(source, meshes)
            outcome = rc.raised(lambda: base_load(model, directory, root, rank))
            gc.collect()
            session = stats.get("load") or {}
            info = {
                "outcome": outcome,
                "holds_expert_3": session.get("holds_expert_3"),
                "owner_local_elements": session.get("owner_local_elements"),
                "consensus_groups": session.get("consensus_groups"),
                "decoder_after": rc.decoder_state_flags(adapter),
                # Read flags only: exporting through a still-healthy sharded adapter here would start collectives.
                "adapter_after": {
                    "failed_flag": bool(getattr(adapter, "_failed", False)),
                    "session_open": getattr(adapter, "_identity", None) is not None,
                },
            }
            if outcome["raised"]:
                info["adapter_after"]["load_to_hf"] = rc.raised(
                    lambda: adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
                )["raised"]
            else:
                info["local_state"] = local_difference(model, expected)
            return info

        def p9_key_plan_after_sharding() -> dict:
            model, adapter, _expected, _stats = build(source, meshes)
            return {"outcome": rc.raised(lambda: adapter.get_hf_state_dict_keys(model.state_dict()))}

        def n_finisher_fails_on_one_rank() -> dict:
            """Only FINISHER_RANK raises on a clean checkpoint; all other ranks must fail and identify it."""
            model, adapter, expected, _stats = build(source, meshes)
            opened = adapter.to_hf
            injected = {"done": False}

            def to_hf(state_dict, **kwargs):
                result = opened(state_dict, **kwargs)
                if kwargs.get("for_checkpoint_load") and rank == FINISHER_RANK:
                    inner = getattr(result, "decoder_destinations", None)
                    if inner is not None and hasattr(inner, "finishers"):

                        def boom() -> None:
                            raise RuntimeError(FINISHER_MESSAGE)

                        inner.finishers = (boom, *inner.finishers)
                        injected["done"] = True
                return result

            adapter.to_hf = to_hf
            outcome = rc.raised(lambda: base_load(model, reference, root, rank))
            gc.collect()
            info = {
                "outcome": outcome,
                "failing_finisher_installed_on_this_rank": injected["done"],
                "decoder_after": rc.decoder_state_flags(adapter),
                "adapter_after": {
                    "failed_flag": bool(getattr(adapter, "_failed", False)),
                    "session_open": getattr(adapter, "_identity", None) is not None,
                },
            }
            if outcome["raised"]:
                info["adapter_after"]["load_to_hf"] = rc.raised(
                    lambda: adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
                )["raised"]
            if rank != FINISHER_RANK:
                # These ranks finish decoder loading normally, but the outer adapter must still become unusable.
                info["local_state"] = local_difference(model, expected)
            return info

        def n_comparison_fails_on_one_rank() -> dict:
            """An unexpected local validation exception must reach both agreements."""
            model, adapter, _expected, _stats = build(source, meshes)
            opened = adapter.to_hf
            injected = {"done": False}

            def to_hf(state_dict, **kwargs):
                result = opened(state_dict, **kwargs)
                if kwargs.get("for_checkpoint_load") and rank == FINISHER_RANK:

                    def boom(_value):
                        raise RuntimeError("comparison failed in checkpoint regression")

                    adapter._local = boom
                    injected["done"] = True
                return result

            adapter.to_hf = to_hf
            outcome = rc.raised(lambda: base_load(model, reference, root, rank))
            gc.collect()
            return {
                "outcome": outcome,
                "injected": injected["done"],
                "decoder_after": rc.decoder_state_flags(adapter),
                "adapter_after": {"failed_flag": adapter._failed},
            }

        table = {
            "p1": p1_clean_load,
            "p2": p2_export_objects,
            "p3": p3_save_and_resume_load,
            "p4": p4_consolidated_export,
            "n_dense": lambda: negative(root / "ref_dense"),
            "n_expert": lambda: negative(root / "ref_expert"),
            "n_owner": lambda: negative(root / "ref_owner"),
            "n_missing": lambda: negative(root / "ref_missing"),
            "n_finisher": n_finisher_fails_on_one_rank,
            "n_comparison": n_comparison_fails_on_one_rank,
            "p9": p9_key_plan_after_sharding,
        }
        for layout in layouts:
            meshes = make_meshes(layout)
            holder: dict = {}
            work = root / layout
            for group in LAYOUT_GROUPS[layout]:
                for short in PHASES[group]:
                    phase(layout, PHASE_NAMES[short], table[short])
                    dist.barrier()
    finally:
        dist.destroy_process_group()


def evaluate(merged: dict, phase_group: str) -> list[str]:
    """Return differences from the expected outcome, or an empty list when all checks pass."""
    failures: list[str] = []
    ranks = {int(key): value for key, value in merged["ranks"].items()}

    def get(rank: int, phase: str) -> dict:
        return ranks[rank].get(phase) or {}

    names = [PHASE_NAMES[short] for short in PHASES[phase_group]]
    for rank in range(WORLD):
        for name in names:
            if get(rank, name).get("status") != "ok":
                failures.append(
                    f"rank {rank} {name}: status {get(rank, name).get('status')}: {get(rank, name).get('message')}"
                )
    if failures:
        return failures
    for rank in range(WORLD):
        if "p1_clean_load" in names:
            clean = get(rank, "p1_clean_load")
            state = clean["local_state"]
            if not (state["same_names"] and state["dtype_mismatches"] == 0 and state["value_mismatches"] == 0):
                failures.append(f"rank {rank} clean load: local state differs: {state}")
            if clean["hf_keys_planned_before_sharding"] != 147:
                failures.append(f"rank {rank}: key plan has {clean['hf_keys_planned_before_sharding']} keys")
            session = clean["load_session"] or {}
            if session.get("verification_pairs", 0) < 1 or session.get("verification_pairs") != session.get(
                "scratch_with_the_layout_of_its_destination"
            ):
                failures.append(f"rank {rank}: verification pairs and scratch layouts: {session}")
        if "p2_export_objects" in names:
            export = get(rank, "p2_export_objects")
            if export["aliases"] < 1 or export["aliases"] != export["aliases_same_object_as_body"]:
                failures.append(
                    f"rank {rank} export: {export['aliases_same_object_as_body']} of {export['aliases']} copies are "
                    f"the body object ({export['alias_bytes_not_shared_with_body']} bytes not shared)"
                )
            if export["hc_head_fn_shape"] != [32, 2]:
                failures.append(f"rank {rank} export: hc_head_fn shape {export['hc_head_fn_shape']}")
            if export["mtp_bias_is_the_decoder_bias_object"]:
                failures.append(f"rank {rank} export: the MTP bias is the decoder bias object")
        if "p3_save_and_resume_load" in names:
            state = get(rank, "p3_save_and_resume_load")["local_state_after_resume_load"]
            if not (state["same_names"] and state["dtype_mismatches"] == 0 and state["value_mismatches"] == 0):
                failures.append(f"rank {rank} resume load: local state differs: {state}")
    if "p3_save_and_resume_load" in names:
        files = get(0, "p3_save_and_resume_load")["per_rank_files"]
        if files["alias_chunks_unequal"] != 0 or files["distinct_names"] != 147:
            failures.append(f"per-rank files: {files}")
    if "p4_consolidated_export" in names:
        export = get(0, "p4_consolidated_export")
        difference = export["consolidated_vs_reference"]
        if not (
            difference["same_names"] and difference["dtype_mismatches"] == 0 and difference["value_mismatches"] == 0
        ):
            failures.append(f"consolidated export differs from the reference: {difference}")
        report = export["consolidated"]
        if (
            report["aliases"] != 49
            or report["aliases_unequal_to_body"] != 0
            or report["decoder_bias_copied_to_mtp_bias"]
        ):
            failures.append(f"consolidated export copies: {report}")
        if export["config"]["num_nextn_predict_layers"] != 1 or export["config"]["mtp_shared_last_layer"] is not True:
            failures.append(f"consolidated configuration: {export['config']}")

    def check_agreement(phase: str, detecting: list[int], alias: str) -> None:
        listed = str(detecting)
        for rank in range(WORLD):
            info = get(rank, phase)
            outcome = info["outcome"]
            if rank in detecting:
                if outcome["raised"] != "ValueError" or alias not in outcome.get("message", ""):
                    failures.append(f"{phase} rank {rank} (holds the changed shard): {outcome}")
                elif not any(listed in note for note in outcome.get("notes", [])):
                    failures.append(
                        f"{phase} rank {rank}: its error does not name the detecting ranks {listed}: {outcome}"
                    )
            else:
                if outcome["raised"] != "RuntimeError" or listed not in outcome.get("message", ""):
                    failures.append(
                        f"{phase} rank {rank} (does not hold it): expected RuntimeError naming {listed}: {outcome}"
                    )
            check_after(phase, rank, info)

    def check_after(phase: str, rank: int, info: dict) -> None:
        if not info["outcome"]["raised"]:
            return
        after = info["adapter_after"]
        if not (after["failed_flag"] or after["session_open"]) or after.get("load_to_hf") != "RuntimeError":
            failures.append(f"{phase} rank {rank}: adapter usable after the failure: {after}")
        decoder = info["decoder_after"]
        if not (decoder["decoder_poisoned"] or decoder["decoder_session_open"]):
            failures.append(f"{phase} rank {rank}: decoder adapter usable after the failure: {decoder}")

    if "n_changed_dense_copy" in names:
        check_agreement("n_changed_dense_copy", list(range(WORLD)), DENSE_ALIAS)
    if "n_changed_expert_copy" in names:
        holders = [rank for rank in range(WORLD) if get(rank, "n_changed_expert_copy").get("holds_expert_3")]
        if not holders or len(holders) == WORLD:
            failures.append(f"expert case: unexpected holders {holders}")
        check_agreement("n_changed_expert_copy", holders, EXPERT_ALIAS)
    if "n_changed_body_owner" in names:
        owners = [
            rank for rank in range(WORLD) if (get(rank, "n_changed_body_owner").get("owner_local_elements") or 0) > 0
        ]
        check_agreement("n_changed_body_owner", owners, OWNER_ALIAS)
    if "n_missing_mtp_key" in names:
        for rank in range(WORLD):
            info = get(rank, "n_missing_mtp_key")
            outcome = info["outcome"]
            if (
                outcome["raised"] != "CheckpointException"
                or "Missing key in checkpoint state_dict" not in outcome["message"]
            ):
                failures.append(f"missing key rank {rank}: {outcome}")
            check_after("n_missing_mtp_key", rank, info)
    if "n_finisher_fails_on_one_rank" in names:
        phase, listed = "n_finisher_fails_on_one_rank", str([FINISHER_RANK])
        if not get(FINISHER_RANK, phase).get("failing_finisher_installed_on_this_rank"):
            failures.append(f"{phase}: the failing finisher could not be installed on rank {FINISHER_RANK}")
        for rank in range(WORLD):
            info = get(rank, phase)
            outcome = info["outcome"]
            if rank == FINISHER_RANK:
                if outcome["raised"] != "RuntimeError" or FINISHER_MESSAGE not in outcome.get("message", ""):
                    failures.append(f"{phase} rank {rank} (its finisher fails): {outcome}")
                elif not any(listed in note for note in outcome.get("notes", [])):
                    failures.append(
                        f"{phase} rank {rank}: its error does not name the failing rank {listed}: {outcome}"
                    )
                decoder = info["decoder_after"]
                if not (decoder["decoder_poisoned"] or decoder["decoder_session_open"]):
                    failures.append(f"{phase} rank {rank}: decoder adapter usable after its finisher failed: {decoder}")
            elif outcome["raised"] != "RuntimeError" or f"Ranks {listed} rejected" not in outcome.get("message", ""):
                failures.append(
                    f"{phase} rank {rank} (its own load is clean): expected RuntimeError naming {listed}: {outcome}"
                )
            # Other ranks finish decoder sessions normally, so require only that their outer MTP adapter is unusable.
            after = info["adapter_after"]
            if outcome["raised"] and (
                not (after["failed_flag"] or after["session_open"]) or after.get("load_to_hf") != "RuntimeError"
            ):
                failures.append(f"{phase} rank {rank}: adapter usable after the failure: {after}")
    if "p9_key_plan_after_sharding" in names:
        for rank in range(WORLD):
            outcome = get(rank, "p9_key_plan_after_sharding")["outcome"]
            if outcome["raised"] != "ValueError" or "unsharded" not in outcome.get("message", ""):
                failures.append(f"key planning rank {rank}: missing explicit unsharded-state guard: {outcome}")
    if "n_comparison_fails_on_one_rank" in names:
        phase = "n_comparison_fails_on_one_rank"
        for rank in range(WORLD):
            info = get(rank, phase)
            outcome = info["outcome"]
            if outcome["raised"] != "RuntimeError":
                failures.append(f"{phase} rank {rank}: expected RuntimeError: {outcome}")
            if rank == FINISHER_RANK:
                if not info["injected"] or "comparison failed" not in outcome.get("message", ""):
                    failures.append(f"{phase} rank {rank}: local error missing: {outcome}")
                if not any(str([FINISHER_RANK]) in note for note in outcome.get("notes", [])):
                    failures.append(f"{phase} rank {rank}: detecting ranks missing: {outcome}")
            elif f"Ranks {[FINISHER_RANK]} rejected" not in outcome.get("message", ""):
                failures.append(f"{phase} rank {rank}: detecting ranks missing: {outcome}")
            if not info["adapter_after"]["failed_flag"]:
                failures.append(f"{phase} rank {rank}: adapter not failed")
            decoder = info["decoder_after"]
            if not (decoder["decoder_poisoned"] or decoder["decoder_session_open"]):
                failures.append(f"{phase} rank {rank}: decoder already finished before rejection")
    return failures


def run(root: Path, layouts: tuple[str, ...] = tuple(LAYOUT_GROUPS)) -> dict[str, dict]:
    """Run the phase groups of the given layouts on four fresh Gloo ranks.

    The two production-shaped layouts each run every positive, value-mismatch
    and error-agreement phase. The 2D embedding diagnostic needs only a clean
    load and subset-owner expert mismatch to exercise all-axis agreement.
    Starting the ranks costs far more than the phases, so one process group
    runs all layouts in order, each with its own meshes and save directory.

    Returns:
        For each layout, the per-rank phase records under ``ranks``, in the
        form ``evaluate`` reads.
    """
    if not layouts or any(layout not in LAYOUT_GROUPS for layout in layouts):
        raise ValueError("Unknown checkpoint layout")
    root.mkdir(parents=True, exist_ok=True)
    rank_dir = root / "ranks"
    rank_dir.mkdir()
    mp.spawn(
        worker,
        args=(WORLD, str(root / "rendezvous"), str(rank_dir), tuple(layouts)),
        nprocs=WORLD,
        join=True,
    )
    records = {rank: json.loads((rank_dir / f"rank{rank}.json").read_text()) for rank in range(WORLD)}
    merged = {
        layout: {"ranks": {str(rank): records[rank]["layouts"][layout] for rank in range(WORLD)}} for layout in layouts
    }
    failures = {
        layout: {group: evaluate(merged[layout], group) for group in LAYOUT_GROUPS[layout]} for layout in layouts
    }
    (root / "REPORT.json").write_text(json.dumps({"failures": failures, **merged}, indent=2) + "\n")
    return merged
