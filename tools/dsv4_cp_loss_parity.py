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

"""Run a real-weight DeepSeek V4 EP8/CP loss-parity trace on 8 local GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from safetensors import safe_open

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.mesh_utils import create_device_mesh
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM
from nemo_automodel.components.moe.parallelizer import apply_cp, apply_ep
from nemo_automodel.components.moe.state_dict_utils import get_expert_range_for_rank_from_mesh

N_LAYERS = 4
EP_SIZE = 8
_EXPERT_KEY_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$")
_LAYER_KEY_RE = re.compile(r"^layers\.(\d+)\.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attn-backend", choices=("tilelang", "torch", "sdpa"), default="tilelang")
    parser.add_argument("--token-loss-output", type=Path, default=None)
    parser.add_argument("--activation-debug-output", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a DeepSeek V4 checkpoint with the real first-four-layer weights.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _log(message: str) -> None:
    if _rank() == 0:
        print(message, flush=True)


def _configure_real_4layer_config(checkpoint: Path) -> DeepseekV4Config:
    config = DeepseekV4Config.from_pretrained(str(checkpoint))
    config.num_hidden_layers = N_LAYERS
    config.compress_ratios = list(config.compress_ratios[:N_LAYERS])
    config.num_nextn_predict_layers = 0
    config.use_cache = False
    config._name_or_path = str(checkpoint)
    config.name_or_path = str(checkpoint)
    return config


def _is_selected_checkpoint_key(key: str, *, expert_start: int, expert_end: int) -> bool:
    if key.startswith(("embed.", "norm.", "head.", "hc_head_")):
        return True

    expert_match = _EXPERT_KEY_RE.match(key)
    if expert_match is not None:
        layer_idx = int(expert_match.group(1))
        expert_idx = int(expert_match.group(2))
        return layer_idx < N_LAYERS and expert_start <= expert_idx < expert_end

    layer_match = _LAYER_KEY_RE.match(key)
    if layer_match is None:
        return False
    return int(layer_match.group(1)) < N_LAYERS


def _load_real_4layer_checkpoint(
    checkpoint: Path,
    *,
    expert_start: int,
    expert_end: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")

    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        if _is_selected_checkpoint_key(key, expert_start=expert_start, expert_end=expert_end):
            by_file[filename].append(key)

    loaded: dict[str, torch.Tensor] = {}
    for filename, keys in sorted(by_file.items()):
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                loaded[key] = handle.get_tensor(key)

    stats = {
        "loaded_tensors": len(loaded),
        "loaded_shards": len(by_file),
        "expert_start": expert_start,
        "expert_end": expert_end,
    }
    return loaded, stats


def _build_and_load_model(args: argparse.Namespace, *, device, device_mesh, moe_mesh) -> DeepseekV4ForCausalLM:
    config = _configure_real_4layer_config(args.checkpoint)
    backend = BackendConfig(
        attn=args.attn_backend,
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch_mm",
        dispatcher="torch",
        disable_shared_expert_overlap=True,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=True,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.device(device):
        model = DeepseekV4ForCausalLM(config, backend=backend)
    cast_model_to_dtype(model, torch.bfloat16)
    model.train()

    if args.cp_size > 1:
        apply_cp(model, device_mesh["cp"])
    apply_ep(model, moe_mesh["ep"], moe_mesh=moe_mesh)

    ep_load_mesh = moe_mesh[tuple(dim for dim in moe_mesh.mesh_dim_names if dim != "pp")]
    expert_start, expert_end = get_expert_range_for_rank_from_mesh(ep_load_mesh, config.n_routed_experts)
    raw_state, load_stats = _load_real_4layer_checkpoint(
        args.checkpoint,
        expert_start=expert_start,
        expert_end=expert_end,
    )
    _log(
        "loaded raw checkpoint selection: "
        f"{load_stats['loaded_tensors']} tensors from {load_stats['loaded_shards']} shards; "
        f"rank0 experts {load_stats['expert_start']}:{load_stats['expert_end']}"
    )

    converted = model.state_dict_adapter.from_hf(raw_state, device_mesh=ep_load_mesh)
    converted = {
        key: value if isinstance(value, DTensor) or not torch.is_tensor(value) else value.to(device)
        for key, value in converted.items()
    }
    missing, unexpected = model.load_state_dict(converted, strict=False)
    allowed_missing = {name for name in missing if name.endswith("e_score_correction_bias")}
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(
            "Real-weight load did not cover the 4-layer model. "
            f"missing={list(missing)[:20]} unexpected={list(unexpected)[:20]}"
        )
    del raw_state, converted
    torch.cuda.empty_cache()
    return model


def _make_text_mock_batch(
    *,
    step: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    text = (
        "DeepSeek V4 context parallel loss parity mock text. "
        "Sparse attention gathers global key value states while local queries remain sharded. "
    )
    pattern = torch.tensor([ord(ch) for ch in text], device=device, dtype=torch.long)
    pattern = (pattern % (vocab_size - 2)) + 2
    repeats = (seq_len + pattern.numel() - 1) // pattern.numel()
    ids = pattern.repeat(repeats)[:seq_len]
    ids = ((ids + step * 17) % (vocab_size - 2)) + 2
    input_ids = ids.unsqueeze(0).expand(batch_size, -1).contiguous()
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _prepare_batch(
    batch: dict[str, torch.Tensor],
    *,
    device_mesh,
    cp_size: int,
):
    if cp_size > 1:
        cp_mesh = device_mesh["cp"]
        batch["_cp_manual_allgather"] = True
        batch["_cp_manual_pad_multiple"] = cp_size * 128
        batch["_dsv4_cp_group"] = cp_mesh.get_group()
    train_ctx, batch = make_cp_batch_and_ctx(device_mesh, batch, padding_token_id=0)
    labels = batch.pop("labels")
    return train_ctx, batch, labels


def _normalized_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    local_loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    local_count = (labels != -100).sum(dtype=torch.float32)
    global_loss_sum = local_loss_sum.detach().clone()
    global_count = local_count.detach().clone()
    dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    normalized = float((global_loss_sum / global_count.clamp_min(1)).item())
    count = int(global_count.item())
    # Backward traverses every rank while the logged scalar is normalized over
    # the same global token count for CP1 and CP2.
    return local_loss_sum / global_count.clamp_min(1), normalized, count


def _write_token_loss_debug(
    path: Path | None,
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    cp_group,
) -> None:
    if path is None:
        return

    local_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    local_labels = labels.detach().contiguous()
    local_token_loss = local_token_loss.detach().float().contiguous()

    if cp_group is not None and dist.get_world_size(group=cp_group) > 1:
        loss_parts = [torch.empty_like(local_token_loss) for _ in range(dist.get_world_size(group=cp_group))]
        label_parts = [torch.empty_like(local_labels) for _ in range(dist.get_world_size(group=cp_group))]
        dist.all_gather(loss_parts, local_token_loss, group=cp_group)
        dist.all_gather(label_parts, local_labels, group=cp_group)
        token_loss = torch.cat(loss_parts, dim=1)
        debug_labels = torch.cat(label_parts, dim=1)
    else:
        token_loss = local_token_loss
        debug_labels = local_labels

    if _rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_loss": token_loss.cpu().tolist(),
            "labels": debug_labels.cpu().tolist(),
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _register_activation_debug_hooks(model: DeepseekV4ForCausalLM) -> dict[str, dict[str, torch.Tensor]]:
    debug: dict[str, dict[str, torch.Tensor]] = {}

    def _make_hook(name: str):
        def _hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return
            out = output.detach().float()
            debug[name] = {
                "mean": out.mean(dim=tuple(range(2, out.ndim))).contiguous(),
                "rms": out.pow(2).mean(dim=tuple(range(2, out.ndim))).sqrt().contiguous(),
            }

        return _hook

    for layer_name, layer in model.model.layers.items():
        layer.register_forward_hook(_make_hook(f"layer_{layer_name}"))
    return debug


def _gather_cp_debug_tensor(tensor: torch.Tensor, cp_group) -> torch.Tensor:
    if cp_group is None or dist.get_world_size(group=cp_group) <= 1:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=cp_group))]
    dist.all_gather(parts, tensor.contiguous(), group=cp_group)
    return torch.cat(parts, dim=1)


def _write_activation_debug(
    path: Path | None,
    *,
    activation_debug: dict[str, dict[str, torch.Tensor]],
    cp_group,
) -> None:
    if path is None:
        return
    payload = {}
    for layer_name, stats in sorted(activation_debug.items()):
        payload[layer_name] = {
            name: _gather_cp_debug_tensor(value, cp_group).cpu().tolist() for name, value in stats.items()
        }
    if _rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.seq_len != 4096:
        raise ValueError(f"This parity harness is pinned to seqlen 4096, got {args.seq_len}.")
    if not torch.cuda.is_available():
        raise RuntimeError("This parity harness requires CUDA GPUs.")

    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    if world_size != EP_SIZE:
        raise RuntimeError(f"Expected world_size={EP_SIZE} for ep8 parity, got {world_size}.")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    distributed_config = FSDP2Config(backend="nccl")
    device_mesh, moe_mesh = create_device_mesh(
        distributed_config,
        dp_size=None,
        dp_replicate_size=1,
        tp_size=1,
        pp_size=1,
        cp_size=args.cp_size,
        ep_size=EP_SIZE,
        world_size=world_size,
    )
    if moe_mesh is None:
        raise RuntimeError("Expected a non-empty MoE mesh for ep8.")

    _log(
        f"building real DSV4 4-layer parity model: world={world_size} ep={EP_SIZE} "
        f"cp={args.cp_size} steps={args.steps} seq_len={args.seq_len} checkpoint={args.checkpoint}"
    )
    model = _build_and_load_model(args, device=device, device_mesh=device_mesh, moe_mesh=moe_mesh)
    activation_debug = _register_activation_debug_hooks(model) if args.activation_debug_output is not None else {}
    config = model.config
    dist.barrier()
    torch.cuda.synchronize()

    losses: list[float] = []
    counts: list[int] = []
    step_seconds: list[float] = []
    for step in range(args.steps):
        start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        batch = _make_text_mock_batch(
            step=step,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=config.vocab_size,
            device=device,
        )
        train_ctx, batch, labels = _prepare_batch(batch, device_mesh=device_mesh, cp_size=args.cp_size)
        with train_ctx():
            outputs = model(**batch)
            if step == 0:
                _write_activation_debug(
                    args.activation_debug_output,
                    activation_debug=activation_debug,
                    cp_group=batch.get("_dsv4_cp_group"),
                )
                _write_token_loss_debug(
                    args.token_loss_output,
                    logits=outputs.logits,
                    labels=labels,
                    cp_group=batch.get("_dsv4_cp_group"),
                )
            loss, loss_value, count = _normalized_loss(outputs.logits, labels)
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        losses.append(loss_value)
        counts.append(count)
        step_seconds.append(elapsed)
        _log(f"step={step:03d} cp={args.cp_size} loss={loss_value:.10f} tokens={count} seconds={elapsed:.3f}")

    dist.barrier()
    if _rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cp_size": args.cp_size,
            "ep_size": EP_SIZE,
            "world_size": world_size,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint),
            "num_hidden_layers": N_LAYERS,
            "num_hash_layers": int(config.num_hash_layers),
            "compress_ratios": list(config.compress_ratios),
            "losses": losses,
            "token_counts": counts,
            "step_seconds": step_seconds,
            "mean_step_seconds": sum(step_seconds) / len(step_seconds),
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
