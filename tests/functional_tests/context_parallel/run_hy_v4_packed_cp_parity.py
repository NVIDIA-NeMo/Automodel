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

"""Real-checkpoint HY V4 packed-THD CP1/CP2 forward/backward parity.

This test intentionally uses the production model loader, cuDNN DSA kernels,
``apply_cp``, and ``ContextParallelSharder``.  It compares a two-document
packed stream at CP=1 against contiguous-query CP=2, including the loss and
representative query, K/V, sink, and iHC parameter gradients.

Run::

    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_hy_v4_packed_cp_parity.py \
        --checkpoint /path/to/Hy4-preview-1l-reference-v2 \
        --output /path/to/hy-v4-packed-cp2.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.thd_utils import process_input_for_thd
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.parallelizer import apply_cp

_GRADIENT_PARAMETERS = (
    "model.layers.0.hc_attn_layer.hc_pre.hc_fn",
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.learnable_sink_param.weight",
    "model.hc_head.hc_head_fn",
)


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Return stable FP64 metrics for two equal-shaped tensors.

    Args:
        actual: CP2 result with arbitrary tensor layout.
        reference: CP1 result with the same semantic layout and shape.

    Returns:
        Scalar absolute-error, relative-L2, and cosine metrics.
    """
    if actual.shape != reference.shape:
        raise AssertionError(f"shape mismatch: actual={tuple(actual.shape)}, reference={tuple(reference.shape)}")
    actual64 = actual.detach().double().flatten().cpu()
    reference64 = reference.detach().double().flatten().cpu()
    difference = actual64 - reference64
    actual_norm = torch.linalg.vector_norm(actual64)
    reference_norm = torch.linalg.vector_norm(reference64)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-30)).item(),
        "cosine": (torch.dot(actual64, reference64) / (actual_norm * reference_norm).clamp_min(1e-30)).item(),
    }


def _model(checkpoint: Path) -> torch.nn.Module:
    """Load the exact proxy through AutoModel's production registry/adapter path."""
    backend = BackendConfig(
        attn="cudnn",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )
    model = NeMoAutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        backend=backend,
        use_liger_kernel=False,
        use_sdpa_patching=False,
    )
    model.train()
    if model.mtp_config.enabled:
        raise AssertionError("The one-layer CP parity proxy must disable MTP; MTP is covered by the training gate.")

    selected = set(_GRADIENT_PARAMETERS)
    available = {name for name, _ in model.named_parameters()}
    missing = selected - available
    if missing:
        raise AssertionError(f"Missing representative gradient parameters: {sorted(missing)}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    return model


def _packed_batch(device: torch.device) -> dict[str, torch.Tensor]:
    """Build two packed documents whose combined token axis is divisible by CP2.

    Args:
        device: CUDA device that owns the returned tensors.

    Returns:
        Fresh BSHD-style token fields ``[1, 16]`` and pack lengths ``[1, 2]``.
    """
    input_ids = torch.tensor(
        [[1, 42, 314, 1592, 2718, 4096, 8191, 17, 23, 71, 101, 509, 1021, 4093, 6553, 120000]],
        dtype=torch.long,
        device=device,
    )
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    # Never train across a packed-document boundary or past the final token.
    labels[:, [6, 15]] = -100
    position_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=torch.long,
        device=device,
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": torch.tensor([[7, 9]], dtype=torch.int32, device=device),
        "seq_lens_padded": torch.tensor([[7, 9]], dtype=torch.int32, device=device),
    }


def _cross_entropy_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return unnormalized next-token CE over non-ignored packed tokens.

    Args:
        logits: FP32/BF16 logits shaped ``[..., vocab]``.
        labels: Integer targets shaped like ``logits.shape[:-1]``.

    Returns:
        Newly computed scalar FP32 loss sum.
    """
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def run(checkpoint: Path, output: Path | None) -> dict[str, Any]:
    """Execute and assert real-checkpoint packed CP parity on exactly two GPUs."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"HY V4 packed parity requires exactly CP=2, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = _model(checkpoint)
    source_batch = _packed_batch(device)
    valid_token_count = int((source_batch["labels"] != -100).sum().item())

    reference_batch = process_input_for_thd(dict(source_batch), padding_token_id=int(model.config.pad_token_id))
    reference_output = model(
        input_ids=reference_batch["input_ids"].unsqueeze(0),
        position_ids=reference_batch["position_ids"].unsqueeze(0),
        padding_mask=reference_batch["padding_mask"].unsqueeze(0),
        qkv_format="thd",
        cu_seqlens=reference_batch["cu_seqlens"].unsqueeze(0),
        logits_to_keep=0,
    )
    reference_logits = reference_output.logits.detach()
    reference_loss = _cross_entropy_sum(reference_output.logits, reference_batch["labels"]) / valid_token_count
    reference_loss.backward()
    reference_gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(reference_gradients) != set(_GRADIENT_PARAMETERS):
        raise AssertionError(
            "Reference backward did not cover all requested parameters: "
            f"got={sorted(reference_gradients)}, expected={sorted(_GRADIENT_PARAMETERS)}"
        )
    model.zero_grad(set_to_none=True)
    del reference_output
    torch.cuda.empty_cache()

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    apply_cp(model, mesh["cp"])
    cp_batch = _packed_batch(device)
    sharder = ContextParallelSharder(
        model,
        mesh,
        cp_batch,
        padding_token_id=int(model.config.pad_token_id),
    )
    train_context, cp_batch = sharder.shard(cp_batch)
    local_labels = cp_batch.pop("labels")
    with train_context():
        cp_output = model(**cp_batch, logits_to_keep=0)
    local_loss_sum = _cross_entropy_sum(cp_output.logits, local_labels)
    (local_loss_sum / valid_token_count).backward()

    cp_loss = local_loss_sum.detach().float()
    dist.all_reduce(cp_loss, op=dist.ReduceOp.SUM)
    cp_loss /= valid_token_count
    gathered_logits = sharder.gather_token_tensor(cp_output.logits.detach(), seq_dim=1, trim=True)
    logit_metrics = _metrics(gathered_logits, reference_logits)

    gradient_metrics: dict[str, dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise AssertionError(f"CP backward did not produce a gradient for {name}")
        cp_gradient = parameter.grad.detach().clone()
        dist.all_reduce(cp_gradient, op=dist.ReduceOp.SUM)
        gradient_metrics[name] = _metrics(cp_gradient, reference_gradients[name])

    loss_abs = abs(cp_loss.item() - reference_loss.detach().float().item())
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "cp_size": world_size,
        "documents": [7, 9],
        "gradient_metrics": gradient_metrics,
        "logit_metrics": logit_metrics,
        "loss": {
            "absolute_difference": loss_abs,
            "cp1": reference_loss.detach().float().item(),
            "cp2": cp_loss.item(),
        },
        "sequence_length": 16,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    if loss_abs >= 5e-3:
        raise AssertionError(f"packed CP loss mismatch: {loss_abs:.6e}")
    if logit_metrics["relative_l2"] >= 1e-2 or logit_metrics["cosine"] <= 0.9999:
        raise AssertionError(f"packed CP logit parity failed: {logit_metrics}")
    bad_gradients = {
        name: metrics
        for name, metrics in gradient_metrics.items()
        if metrics["relative_l2"] >= 3e-2 or metrics["cosine"] <= 0.999
    }
    if bad_gradients:
        raise AssertionError(f"packed CP gradient parity failed: {bad_gradients}")

    if rank == 0:
        if output is not None:
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite parity report: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print("RESULT: PASS (HY V4 real-checkpoint packed THD CP1/CP2 forward/backward parity)", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
