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

"""Drive the diffusion recipe on the toy custom-model MoE DiT and record per-step metrics.

Three modes:

* ``write-checkpoint DIR``: write the seeded toy checkpoint (diffusers layout), single process.
* ``train --model-dir DIR --ep-size N --out OUT.json``: under ``torchrun``, register the toy
  architecture, run ``TrainDiffusionRecipe`` with the mock dataloader and ``SimpleAdapter``
  and write per-step loss / grad-norm plus the expert-parameter layout to ``OUT.json``.
* ``compare REF.json EP.json``: assert loss / grad-norm parity and that EP was really applied.

The recipe is driven programmatically (instead of ``examples/diffusion/finetune/finetune.py``)
so the test-only architecture can be registered in every rank before the model is built.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.functional_tests.diffusion.toy_moe_dit import (  # noqa: E402
    register_toy_moe_dit,
    write_toy_moe_dit_checkpoint,
)

SEED = 1234
NUM_EXPERTS = 8
TEXT_EMBED_DIM = 32
IN_CHANNELS = 4
GATE_BIAS_UPDATE_FACTOR = 1e-3


def _recipe_config(
    model_dir: str,
    ep_size: int,
    max_steps: int,
    checkpoint_dir: str,
    world_size: int,
    config_overrides: dict | None,
    save_checkpoint: bool = False,
) -> dict:
    # Pure-PyTorch MoE backend (same knobs as ``toy_backend``) passed through ``model.backend``.
    backend = {
        "attn": "sdpa",
        "linear": "torch",
        "rms_norm": "torch",
        "rope_fusion": False,
        "experts": "torch",
        "dispatcher": "torch",
    }
    local_batch_size = 2
    return {
        "seed": SEED,
        "dist_env": {"backend": "nccl", "timeout_minutes": 2},
        "model": {
            "pretrained_model_name_or_path": model_dir,
            "mode": "finetune",
            # fp32 master weights, bf16 compute (FSDP MixedPrecisionPolicy casts parameters).
            "torch_dtype": "float32",
            "compute_dtype": "bfloat16",
            "backend": backend,
            "config_overrides": config_overrides or {},
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-3, "weight_decay": 0.0},
        "clip_grad_norm": {"max_norm": 1.0},
        "fsdp": {
            "tp_size": 1,
            "cp_size": 1,
            "pp_size": 1,
            "dp_replicate_size": 1,
            "ep_size": ep_size,
            "activation_checkpointing": False,
            "reduce_dtype": "float32",
            "enable_fsdp2_prefetch": False,
        },
        "flow_matching": {
            "adapter_type": "simple",
            "adapter_kwargs": {},
            "timestep_sampling": "uniform",
            "flow_shift": 3.0,
            "mix_uniform_ratio": 0.0,
            "i2v_prob": 0.0,
            "cfg_dropout_prob": 0.0,
            "use_loss_weighting": False,
            "log_interval": 1000,
            "summary_log_interval": 1000,
        },
        "step_scheduler": {
            "global_batch_size": local_batch_size * world_size,
            "local_batch_size": local_batch_size,
            "ckpt_every_steps": max_steps if save_checkpoint else 100000,
            "num_epochs": 1,
            "max_steps": max_steps,
            "log_remote_every_steps": 1,
            "save_checkpoint_every_epoch": False,
        },
        "data": {
            "dataloader": {
                "_target_": "nemo_automodel.components.datasets.diffusion.mock_dataloader.build_mock_dataloader",
                "length": 64,
                "num_channels": IN_CHANNELS,
                "num_frame_latents": 2,
                "spatial_h": 4,
                "spatial_w": 4,
                "text_seq_len": 8,
                "text_embed_dim": TEXT_EMBED_DIM,
                "num_workers": 0,
                "shuffle": False,
            }
        },
        "checkpoint": {
            "enabled": save_checkpoint,
            "checkpoint_dir": checkpoint_dir,
            "model_save_format": "safetensors",
            # Consolidated HF safetensors are written from the EP + FSDP shards of every rank.
            "save_consolidated": save_checkpoint,
        },
    }


def _expert_layout(model) -> list[dict]:
    """Describe every MoE ``experts`` parameter: DTensor mesh axes, placements, local/global shapes."""
    from torch.distributed.tensor import DTensor

    from nemo_automodel.components.moe.layers import MoE

    layout = []
    for name, module in model.named_modules():
        if not isinstance(module, MoE):
            continue
        for param_name, param in module.experts.named_parameters():
            entry = {
                "name": f"{name}.experts.{param_name}",
                "is_dtensor": isinstance(param, DTensor),
                "global_shape": list(param.shape),
            }
            if isinstance(param, DTensor):
                entry["mesh_dim_names"] = list(param.device_mesh.mesh_dim_names or ())
                entry["mesh_shape"] = list(param.device_mesh.shape)
                entry["placements"] = [
                    {"type": type(placement).__name__, "dim": getattr(placement, "dim", None)}
                    for placement in param.placements
                ]
                entry["local_shape"] = list(param.to_local().shape)
            else:
                entry["local_shape"] = list(param.shape)
            layout.append(entry)
    return layout


def _gate_correction_bias_abs_max(model) -> float | None:
    """Largest |e_score_correction_bias| over all MoE gates (``None`` when gates have no correction bias)."""
    from torch.distributed.tensor import DTensor

    from nemo_automodel.components.moe.layers import MoE

    values = []
    for module in model.modules():
        if isinstance(module, MoE) and getattr(module.gate, "e_score_correction_bias", None) is not None:
            bias = module.gate.e_score_correction_bias
            bias = bias.full_tensor() if isinstance(bias, DTensor) else bias
            values.append(float(bias.detach().abs().max()))
    return max(values) if values else None


def _model_hf_state_dict(model) -> dict:
    """Gather sharded parameters and buffers (collective) and convert them to HF checkpoint keys."""
    from torch.distributed.tensor import DTensor

    tensors = [*model.named_parameters(), *model.named_buffers()]
    full = {
        name: (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float().cpu()
        for name, tensor in tensors
    }
    return model.state_dict_adapter.to_hf(full)


def _max_abs_diff(hf_state_dict: dict, reference: dict) -> float:
    assert set(hf_state_dict) == set(reference), sorted(set(hf_state_dict) ^ set(reference))
    return max(float((hf_state_dict[key] - reference[key].float()).abs().max()) for key in reference)


def _max_abs_diff_vs_checkpoint(model, model_dir: str) -> float:
    """Diff the live (sharded) model against the HF safetensors it was loaded from."""
    from safetensors.torch import load_file

    reference = load_file(os.path.join(model_dir, "transformer", "model.safetensors"))
    return _max_abs_diff(_model_hf_state_dict(model), reference)


def _max_abs_diff_vs_saved_checkpoint(model, checkpoint_dir: str) -> tuple[float, list[str]]:
    """Diff the trained model against the checkpoint the recipe saved (all ranks' safetensors shards)."""
    import glob

    from safetensors.torch import load_file

    files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*_step_*", "model", "consolidated", "*.safetensors")))
    assert files, f"no model safetensors saved under {checkpoint_dir}"
    saved = {}
    for path in files:
        shard = load_file(path)
        assert not set(shard) & set(saved), sorted(set(shard) & set(saved))
        saved.update(shard)
    return _max_abs_diff(_model_hf_state_dict(model), saved), [os.path.relpath(p, checkpoint_dir) for p in files]


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.diffusion import train as diffusion_train

    register_toy_moe_dit()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config_overrides = json.loads(args.config_overrides) if args.config_overrides else None
    cfg = ConfigNode(
        _recipe_config(
            args.model_dir,
            args.ep_size,
            args.max_steps,
            args.checkpoint_dir,
            world_size,
            config_overrides,
            save_checkpoint=args.save_checkpoint,
        )
    )

    grad_norms: list[float] = []
    clip_calls: list[str] = []
    real_scale_and_clip = diffusion_train.scale_grads_and_clip_grad_norm
    real_clip = diffusion_train.clip_grad_norm

    def _record(name, fn):
        def wrapper(*fn_args, **fn_kwargs):
            clip_calls.append(name)
            if name == "scale_grads_and_clip_grad_norm":
                clip_calls.append(f"ep_axis_name={fn_kwargs.get('ep_axis_name')}")
            value = fn(*fn_args, **fn_kwargs)
            grad_norms.append(float(value))
            return value

        return wrapper

    diffusion_train.scale_grads_and_clip_grad_norm = _record("scale_grads_and_clip_grad_norm", real_scale_and_clip)
    diffusion_train.clip_grad_norm = _record("clip_grad_norm", real_clip)

    recipe = diffusion_train.TrainDiffusionRecipe(cfg)
    recipe.setup()
    for key, value in (config_overrides or {}).items():
        assert getattr(recipe.model.config, key) == value, f"model.config_overrides not applied: {key}"

    # Model construction consumes RNG differently with and without EP (rank-local expert
    # shapes differ), so re-seed before training: both legs then draw identical mock data,
    # timesteps and noise on each data-parallel rank.
    rank = dist.get_rank()
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed(SEED + rank)

    losses: list[float] = []
    real_step = recipe.flow_matching_pipeline.step

    def step_wrapper(*step_args, **step_kwargs):
        result = real_step(*step_args, **step_kwargs)
        losses.append(float(result[1].detach().float()))
        return result

    recipe.flow_matching_pipeline.step = step_wrapper

    gate_bias_updates = []
    real_update = getattr(recipe.model, "update_moe_gate_bias", None)
    if real_update is not None:

        def update_wrapper():
            gate_bias_updates.append(1)
            return real_update()

        recipe.model.update_moe_gate_bias = update_wrapper

    moe_mesh = recipe.moe_mesh
    layout = _expert_layout(recipe.model)
    checkpointer_moe_mesh = getattr(recipe.checkpointer, "moe_mesh", None)
    max_abs_diff_vs_checkpoint = _max_abs_diff_vs_checkpoint(recipe.model, args.model_dir)

    recipe.run_train_validation_loop()

    gate_bias_abs_max = _gate_correction_bias_abs_max(recipe.model)
    trained_vs_initial = _max_abs_diff_vs_checkpoint(recipe.model, args.model_dir)
    saved_vs_trained, saved_files = (
        _max_abs_diff_vs_saved_checkpoint(recipe.model, args.checkpoint_dir) if args.save_checkpoint else (None, None)
    )

    # Data-parallel mean of the per-rank losses (each rank sees a different shard).
    loss_tensor = torch.tensor(losses, dtype=torch.float64, device=recipe.device)
    dist.all_reduce(loss_tensor)
    loss_tensor /= dist.get_world_size()

    if rank == 0:
        result = {
            "ep_size": args.ep_size,
            "world_size": dist.get_world_size(),
            "losses": loss_tensor.tolist(),
            "rank0_losses": losses,
            "grad_norms": grad_norms,
            "clip_calls": sorted(set(clip_calls)),
            "gate_bias_update_calls": len(gate_bias_updates),
            "gate_correction_bias_abs_max": gate_bias_abs_max,
            "moe_mesh_dim_names": list(moe_mesh.mesh_dim_names) if moe_mesh is not None else None,
            "moe_mesh_shape": list(moe_mesh.shape) if moe_mesh is not None else None,
            "checkpointer_has_moe_mesh": checkpointer_moe_mesh is not None and checkpointer_moe_mesh is moe_mesh,
            "max_abs_diff_vs_checkpoint": max_abs_diff_vs_checkpoint,
            "trained_vs_initial_max_abs_diff": trained_vs_initial,
            "saved_checkpoint_files": saved_files,
            "saved_vs_trained_max_abs_diff": saved_vs_trained,
            "expert_layout": layout,
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != "expert_layout"}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


def _assert_close(name: str, ref: list[float], other: list[float], rtol: float) -> None:
    assert len(ref) == len(other) and ref, f"{name}: step counts differ ({len(ref)} vs {len(other)})"
    for step, (a, b) in enumerate(zip(ref, other), start=1):
        assert math.isfinite(a) and math.isfinite(b), f"{name} step {step}: non-finite ({a}, {b})"
        rel = abs(a - b) / max(abs(a), 1e-12)
        assert rel <= rtol, f"{name} step {step}: {a} vs {b} (rel diff {rel:.3e} > {rtol})"


def compare(args: argparse.Namespace) -> None:
    with open(args.reference) as f:
        ref = json.load(f)
    with open(args.candidate) as f:
        ep = json.load(f)

    print(f"{'step':>4} {'loss ep1':>12} {'loss ep2':>12} {'gnorm ep1':>12} {'gnorm ep2':>12}")
    for i, (la, lb, ga, gb) in enumerate(zip(ref["losses"], ep["losses"], ref["grad_norms"], ep["grad_norms"]), 1):
        print(f"{i:>4} {la:>12.6f} {lb:>12.6f} {ga:>12.6f} {gb:>12.6f}")

    # The reference leg is plain FSDP: no MoE mesh, dense clipping path.
    assert ref["moe_mesh_dim_names"] is None, ref["moe_mesh_dim_names"]
    assert ref["clip_calls"] == ["clip_grad_norm"], ref["clip_calls"]

    # The EP leg must really shard experts over an "ep" axis (guards against EP silently not applying).
    ep_size = ep["ep_size"]
    assert ep_size > 1
    assert "ep" in (ep["moe_mesh_dim_names"] or []), ep["moe_mesh_dim_names"]
    assert ep["checkpointer_has_moe_mesh"], "checkpointer did not receive the recipe moe_mesh"
    assert ep["clip_calls"] == ["ep_axis_name=ep", "scale_grads_and_clip_grad_norm"], ep["clip_calls"]
    assert ep["expert_layout"], "no MoE experts parameters found"
    for entry in ep["expert_layout"]:
        assert entry["is_dtensor"], entry
        assert "ep" in entry["mesh_dim_names"], entry
        ep_dim = entry["mesh_dim_names"].index("ep")
        assert entry["mesh_shape"][ep_dim] == ep_size, entry
        assert entry["placements"][ep_dim] == {"type": "Shard", "dim": 0}, entry
        assert entry["global_shape"][0] == NUM_EXPERTS, entry
        assert entry["local_shape"][0] == NUM_EXPERTS // ep_size, entry
    for entry in ref["expert_layout"]:
        assert "ep" not in entry.get("mesh_dim_names", []), entry

    for run in (ref, ep):
        # Both legs start from the checkpoint weights (loaded through the state_dict_adapter).
        assert run["max_abs_diff_vs_checkpoint"] == 0.0, run["max_abs_diff_vs_checkpoint"]
        assert run["gate_bias_update_calls"] == len(run["grad_norms"]), run["gate_bias_update_calls"]
        # The checkpoint enables router load-balancing bias: update_moe_gate_bias must move it.
        assert run["gate_correction_bias_abs_max"], run["gate_correction_bias_abs_max"]

    for run in (ref, ep):
        assert run["trained_vs_initial_max_abs_diff"] > 0.0, "optimizer did not update the model"
        if run["saved_checkpoint_files"] is not None:
            # The EP-sharded save (checkpointer built with the MoE mesh) round-trips the trained weights.
            assert run["saved_vs_trained_max_abs_diff"] == 0.0, run["saved_vs_trained_max_abs_diff"]

    _assert_close("loss", ref["losses"], ep["losses"], args.loss_rtol)
    _assert_close("grad_norm", ref["grad_norms"], ep["grad_norms"], args.grad_norm_rtol)
    # Training must actually move the model (the loss trajectory is not constant).
    assert len(set(round(x, 6) for x in ref["losses"])) > 1, ref["losses"]
    print("PASSED: EP parity on toy custom-model MoE DiT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p_write = sub.add_parser("write-checkpoint")
    p_write.add_argument("path")
    p_train = sub.add_parser("train")
    p_train.add_argument("--model-dir", required=True)
    p_train.add_argument("--ep-size", type=int, required=True)
    p_train.add_argument("--max-steps", type=int, default=5)
    p_train.add_argument("--checkpoint-dir", required=True)
    p_train.add_argument("--out", required=True)
    p_train.add_argument(
        "--save-checkpoint", action="store_true", help="Save a checkpoint after the last step (EP-sharded save)"
    )
    p_train.add_argument(
        "--config-overrides", default=None, help="JSON dict applied to the toy config (model.config_overrides)"
    )
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("reference")
    p_cmp.add_argument("candidate")
    p_cmp.add_argument("--loss-rtol", type=float, default=2e-2)
    p_cmp.add_argument("--grad-norm-rtol", type=float, default=5e-2)
    args = parser.parse_args()

    if args.mode == "write-checkpoint":
        write_toy_moe_dit_checkpoint(
            args.path,
            seed=SEED,
            in_channels=IN_CHANNELS,
            text_embed_dim=TEXT_EMBED_DIM,
            num_experts=NUM_EXPERTS,
            # DeepSeek-style sigmoid routing with load-balancing bias, so the recipe's
            # update_moe_gate_bias() call has a real effect.
            score_func="sigmoid",
            gate_bias_update_factor=GATE_BIAS_UPDATE_FACTOR,
        )
        print(f"wrote toy checkpoint to {args.path}")
    elif args.mode == "train":
        train(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
