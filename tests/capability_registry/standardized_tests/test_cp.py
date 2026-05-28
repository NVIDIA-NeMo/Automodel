# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Context-parallelism validation test.

Procedure on a world of ``world_size`` ranks (intended ``world_size=2``):

1. Rank 0 builds a reference (no-CP) model, trains ``num_steps - 1`` steps with
   SGD on the recipe dataset, then captures logits from a final forward pass.
2. Rank 0 broadcasts the captured logits to all ranks.
3. Every rank builds the same model, attaches CP forward-hooks (to strip the
   4-D causal attention mask that collides with sharded DTensor Q/K/V),
   trains the same number of steps wrapping each forward + backward in a
   ``context_parallel`` context over the input_ids/labels along the sequence
   dimension, and finally captures + unshards the logits.
4. KL divergence is computed against the rank-0 reference and reduced (MAX) across
   ranks. The capability "passes" iff the max KL is below the threshold.
"""

from __future__ import annotations

import gc

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental._attention import context_parallel_unshard

from transformers import AutoModelForCausalLM

from nemo_automodel.components.distributed.cp_utils import (
    attach_context_parallel_hooks,
    attach_cp_sdpa_hooks,
    make_cp_batch_and_ctx,
)
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from tests.capability_registry._dataset_utils import build_training_batches
from tests.capability_registry._distributed_utils import (
    broadcast_object_from_rank0,
    broadcast_tensor_from_rank0,
    init_distributed,
)
from tests.capability_registry._kl_utils import kl_divergence_from_logits
from tests.capability_registry.standardized_tests._base import CapabilityTestResult


_INIT_SEED = 1234
_OPTIMIZER_LR = 1e-5

# CP needs seq_len % (2 * cp_size) == 0; with world_size=2 that's a multiple of 4.
# Padding to 16 covers cp_size up to 8 safely.
_PAD_SEQ_LEN_DIVISIBLE = 16


def _log(rank: int, msg: str) -> None:
    """Print a progress line, prefixed with the rank for parallel-log clarity."""
    print(f"[cp:rank{rank}] {msg}", flush=True)


class CPTest:
    """CP=2 vs CP=1 parity validation via short training trajectory + KL on logits."""

    name: str = "cp"
    implemented: bool = True
    world_size: int = 2

    def run(
        self,
        *,
        model_id: str,
        dtype: torch.dtype,
        kl_threshold: float,
        num_steps: int,
        local_batch_size: int,
    ) -> CapabilityTestResult:
        """Train ref + variant for ``num_steps - 1`` steps, compare final-step logits."""
        device, device_type = init_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == self.world_size, f"CP test expects {self.world_size} ranks, got {world_size}"
        assert num_steps >= 1, "num_steps must be >= 1"
        assert device_type == "cuda", "CP test requires CUDA"

        _log(rank, f"distributed initialized, device={device}, world_size={world_size}")
        if rank == 0:
            _log(rank, f"building {num_steps} batch(es) on rank 0...")
            batches = build_training_batches(
                model_id=model_id,
                num_steps=num_steps,
                local_batch_size=local_batch_size,
                pad_seq_len_divisible=max(_PAD_SEQ_LEN_DIVISIBLE, 2 * world_size),
            )
            _log(rank, f"batches ready: shapes={[tuple(b['input_ids'].shape) for b in batches]}")
        else:
            batches = None
        batches = broadcast_object_from_rank0(batches, src=0)
        _log(rank, "batches broadcast complete")

        # ------------------------------------------------------------------
        # Reference run (rank 0 only).
        # ------------------------------------------------------------------
        ref_logits_full: torch.Tensor | None = None
        if rank == 0:
            _log(rank, "starting reference (no-CP) train+capture")
            torch.manual_seed(_INIT_SEED)
            torch.cuda.manual_seed_all(_INIT_SEED)
            ref_logits_full = _train_and_capture_reference(
                model_id=model_id,
                dtype=dtype,
                device=device,
                batches=batches,
                num_steps=num_steps,
                rank=rank,
            )
            _log(rank, f"reference run complete, logits shape={tuple(ref_logits_full.shape)}")
            torch.cuda.empty_cache()
            gc.collect()
        else:
            _log(rank, "waiting for rank 0 to finish reference run...")

        ref_logits = broadcast_tensor_from_rank0(ref_logits_full, device=device, src=0)
        del ref_logits_full
        dist.barrier()
        _log(rank, f"reference logits broadcast complete, shape={tuple(ref_logits.shape)}")

        # ------------------------------------------------------------------
        # Variant run (all ranks) — CP=world_size.
        # ------------------------------------------------------------------
        _log(rank, f"starting variant (CP={world_size}) train+capture")
        cp_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("cp",))
        torch.manual_seed(_INIT_SEED)
        torch.cuda.manual_seed_all(_INIT_SEED)
        cp_logits_full = _train_and_capture_cp(
            model_id=model_id,
            dtype=dtype,
            device=device,
            batches=batches,
            num_steps=num_steps,
            cp_mesh=cp_mesh,
        )
        _log(rank, f"variant run complete, logits shape={tuple(cp_logits_full.shape)}")

        # ------------------------------------------------------------------
        # KL divergence + reduction.
        # ------------------------------------------------------------------
        # Optional diagnostic: how far apart are the raw logits? Useful when
        # debugging CP setup (e.g. wrong RoPE positions, missing CP allgather).
        if rank == 0:
            diff = (ref_logits.float() - cp_logits_full.float()).abs()
            _log(
                rank,
                f"logits diff: max={diff.max().item():.4e}, mean={diff.mean().item():.4e}, "
                f"ref|max|={ref_logits.float().abs().max().item():.4e}",
            )
        kl = kl_divergence_from_logits(reference_logits=ref_logits, candidate_logits=cp_logits_full)
        max_kl = torch.tensor(float(kl.view(-1).max().item()), device=device)
        dist.all_reduce(max_kl, op=dist.ReduceOp.MAX)
        max_kl_val = float(max_kl.item())
        _log(rank, f"max_kl={max_kl_val:.3e}, threshold={kl_threshold:.2e}")

        return CapabilityTestResult(
            capability=self.name,
            passed=max_kl_val <= kl_threshold,
            skipped=False,
            max_kl=max_kl_val,
            threshold=kl_threshold,
            variant_label=f"CP={world_size}",
        )


def _train_and_capture_reference(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    batches: list[dict[str, torch.Tensor]],
    num_steps: int,
    rank: int = 0,
) -> torch.Tensor:
    """Build no-CP reference, train K-1 steps, return final-step logits."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    loss_fn = MaskedCrossEntropy(fp32_upcast=True, ignore_index=-100, reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=_OPTIMIZER_LR, momentum=0.0)

    model.train()
    for i in range(num_steps - 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batches[i].items()}
        labels = batch.get("labels")
        # Strip labels so HF forward returns plain logits instead of computing
        # its own loss (we want our external MaskedCrossEntropy + KL pipeline).
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        fwd_kwargs = filter_forward_kwargs(model, batch_no_labels)
        out = model(**fwd_kwargs)
        logits = getattr(out, "logits", out)
        loss = loss_fn(logits=logits, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _log(rank, f"ref train step {i + 1}/{num_steps - 1}: loss={loss.item():.4f}")

    model.eval()
    with torch.inference_mode():
        batch = {k: v.to(device, non_blocking=True) for k, v in batches[-1].items()}
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        fwd_kwargs = filter_forward_kwargs(model, batch_no_labels)
        out = model(**fwd_kwargs)
        logits = getattr(out, "logits", out)
        captured = logits.detach().to(dtype).contiguous()

    del model, optimizer, loss_fn
    gc.collect()
    return captured


def _train_and_capture_cp(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    batches: list[dict[str, torch.Tensor]],
    num_steps: int,
    cp_mesh,
) -> torch.Tensor:
    """Build CP variant, train K-1 steps under context_parallel ctx, return full-seq logits."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    # attach_context_parallel_hooks strips attention_mask + sets is_causal=True.
    # attach_cp_sdpa_hooks re-wraps Q/K/V as DTensors at the SDPA call site so
    # the CP allgather actually fires (otherwise each rank silently attends to
    # only its local seq shard and the logits diverge).
    attach_context_parallel_hooks(model)
    attach_cp_sdpa_hooks(model, cp_mesh["cp"])

    loss_fn = MaskedCrossEntropy(fp32_upcast=True, ignore_index=-100, reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=_OPTIMIZER_LR, momentum=0.0)

    rank = dist.get_rank()
    model.train()
    for i in range(num_steps - 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batches[i].items()}
        # make_cp_batch_and_ctx handles position_ids synthesis, attention_mask
        # stripping, padding-to-divisible, and shards input_ids / labels /
        # position_ids / padding_mask via context_parallel.
        train_ctx, batch = make_cp_batch_and_ctx(
            cp_mesh, batch, loss_mask=batch.get("loss_mask")
        )
        labels = batch.get("labels")

        with train_ctx():
            batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
            fwd_kwargs = filter_forward_kwargs(model, batch_no_labels)
            out = model(**fwd_kwargs)
            logits = getattr(out, "logits", out)
            loss = loss_fn(logits=logits, labels=labels)
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _log(rank, f"train step {i + 1}/{num_steps - 1}: loss={loss.item():.4f}")

    # Final inference step — wrap in context_parallel, unshard the logits.
    model.eval()
    batch = {k: v.to(device, non_blocking=True) for k, v in batches[-1].items()}
    train_ctx, batch = make_cp_batch_and_ctx(cp_mesh, batch, loss_mask=batch.get("loss_mask"))
    with train_ctx(), torch.no_grad():
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        fwd_kwargs = filter_forward_kwargs(model, batch_no_labels)
        out = model(**fwd_kwargs)
        logits = getattr(out, "logits", out)
        # logits is sharded along seq dim; unshard to recover the full sequence.
        (logits_full,) = context_parallel_unshard(cp_mesh["cp"], [logits.detach()], seq_dims=[1])
    captured = logits_full.to(dtype).contiguous()

    del model, optimizer, loss_fn
    gc.collect()
    return captured


