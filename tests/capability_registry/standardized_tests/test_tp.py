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

"""Tensor-parallelism validation test.

Procedure on a world of ``world_size`` ranks (intended ``world_size=2``):

1. All ranks call :meth:`NeMoAutoModelForCausalLM.from_pretrained` together to
   build the reference (no-TP) model — needed because the loader does internal
   cross-rank ``FirstRankPerNode`` barriers that would deadlock if only rank 0
   participated. Rank 0 then trains it ``num_steps - 1`` steps with SGD on the
   recipe dataset and captures logits from a final forward pass; other ranks
   keep the model parked.
2. Rank 0 broadcasts the captured logits to all ranks; all ranks free the
   reference model.
3. Every rank builds the variant via the same loader, this time passing a 1-D
   ``("tp",)`` device mesh and ``FSDP2Config(tp_size=world_size)`` so the
   framework applies the auto-selected TP plan. Same number of steps, same
   batches.
4. KL divergence is computed against the broadcast reference and reduced (MAX)
   across ranks. The capability "passes" iff the max KL is below threshold.
"""

from __future__ import annotations

import gc

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed.tensor.parallel import parallelize_module

from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.parallelizer import _get_parallel_plan
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from tests.capability_registry._dataset_utils import build_training_batches
from tests.capability_registry._distributed_utils import (
    broadcast_object_from_rank0,
    broadcast_tensor_from_rank0,
    init_distributed,
)
from tests.capability_registry._kl_utils import (
    kl_divergence_from_logits,
    maybe_gather_dtensor_to_replicated_local,
)
from tests.capability_registry.standardized_tests._base import CapabilityTestResult


_INIT_SEED = 1234
_OPTIMIZER_LR = 1e-5


def _log(rank: int, msg: str) -> None:
    """Print a progress line, prefixed with the rank for parallel-log clarity."""
    print(f"[tp:rank{rank}] {msg}", flush=True)


class TPTest:
    """TP=2 vs TP=1 parity validation via short training trajectory + KL on logits."""

    name: str = "tp"
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
        assert world_size == self.world_size, f"TP test expects {self.world_size} ranks, got {world_size}"
        assert num_steps >= 1, "num_steps must be >= 1"
        _log(rank, f"distributed initialized, device={device}, world_size={world_size}")

        # Build dataset on rank 0 only -> broadcast (avoid HF cache lock contention).
        if rank == 0:
            _log(rank, f"building {num_steps} batch(es) on rank 0...")
            batches = build_training_batches(
                model_id=model_id,
                num_steps=num_steps,
                local_batch_size=local_batch_size,
            )
            _log(rank, f"batches ready: shapes={[tuple(b['input_ids'].shape) for b in batches]}")
        else:
            batches = None
        batches = broadcast_object_from_rank0(batches, src=0)
        _log(rank, "batches broadcast complete")

        # ------------------------------------------------------------------
        # Reference run (rank 0 trains; other ranks load + park to satisfy
        # NeMoAutoModelForCausalLM's internal cross-rank barriers).
        # ------------------------------------------------------------------
        _log(rank, "loading reference model (no TP) on all ranks...")
        torch.manual_seed(_INIT_SEED)
        torch.cuda.manual_seed_all(_INIT_SEED)
        ref_model = NeMoAutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        _log(rank, "reference model loaded")
        dist.barrier()

        ref_logits_full: torch.Tensor | None = None
        if rank == 0:
            ref_logits_full = _train_and_capture(
                model=ref_model,
                device=device,
                batches=batches,
                num_steps=num_steps,
                rank=rank,
            )
            _log(rank, f"reference run complete, logits shape={tuple(ref_logits_full.shape)}")
        del ref_model
        torch.cuda.empty_cache()
        gc.collect()
        dist.barrier()

        # Broadcast reference logits to other ranks.
        ref_logits = broadcast_tensor_from_rank0(ref_logits_full, device=device, src=0)
        del ref_logits_full
        _log(rank, f"reference logits broadcast complete, shape={tuple(ref_logits.shape)}")

        # ------------------------------------------------------------------
        # Variant run (all ranks) — TP=world_size via NeMo's infrastructure.
        # ------------------------------------------------------------------
        _log(rank, f"loading variant model with TP={world_size}...")
        torch.manual_seed(_INIT_SEED)
        torch.cuda.manual_seed_all(_INIT_SEED)
        # Build the model on all ranks without a mesh (so weights load eagerly,
        # not as meta tensors), then apply the auto-selected TP plan via
        # parallelize_module. The NeMo loader gives us the NeMo Llama class with
        # combined qkv_proj / gate_up_proj, which is what the optimized plan keys
        # match.
        tp_model = NeMoAutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        tp_mesh = DeviceMesh(device_type, torch.arange(world_size, device="cpu"), mesh_dim_names=("tp",))
        plan = _get_parallel_plan(tp_model, sequence_parallel=False, tp_shard_plan=None, tp_size=world_size)
        parallelize_module(tp_model, tp_mesh, plan)
        _log(rank, f"variant model loaded; TP plan applied with {len(plan)} entries")

        tp_logits_full = _train_and_capture(
            model=tp_model,
            device=device,
            batches=batches,
            num_steps=num_steps,
            rank=rank,
            tp_mesh=tp_mesh,
        )
        _log(rank, f"variant run complete, logits shape={tuple(tp_logits_full.shape)}")

        # ------------------------------------------------------------------
        # KL divergence + reduction across ranks.
        # ------------------------------------------------------------------
        kl = kl_divergence_from_logits(reference_logits=ref_logits, candidate_logits=tp_logits_full)
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
            variant_label=f"TP={world_size}",
        )


def _train_and_capture(
    *,
    model: torch.nn.Module,
    device: torch.device,
    batches: list[dict[str, torch.Tensor]],
    num_steps: int,
    rank: int,
    tp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Train ``num_steps - 1`` steps then run a final inference step to capture logits."""
    loss_fn = MaskedCrossEntropy(fp32_upcast=True, ignore_index=-100, reduction="mean")
    # SGD with no momentum — keeps optimizer state at zero, avoids 8B-model OOM
    # while still validating that gradients flow correctly through the parallel
    # plan. Same optimizer for ref and variant -> trajectories are comparable.
    # foreach=False so the optimizer steps each param individually; the batched
    # foreach implementation does not handle a mix of DTensor (TP-sharded) and
    # plain torch.Tensor parameters that arise after partial parallelization.
    optimizer = torch.optim.SGD(model.parameters(), lr=_OPTIMIZER_LR, momentum=0.0, foreach=False)

    from torch.distributed.tensor import DTensor as _DT  # local import to keep top imports tidy

    model.train()
    for i in range(num_steps - 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batches[i].items()}
        labels = batch.get("labels")
        # Strip labels from the model call so the HF forward returns plain
        # logits instead of computing its own loss internally (which fails when
        # logits are TP-sharded DTensors).
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        fwd_kwargs = filter_forward_kwargs(model, batch_no_labels)
        out = model(**fwd_kwargs)
        logits = getattr(out, "logits", out)
        # TP wraps logits as DTensor; gather to a full replicated tensor so the
        # external loss + KL can run on plain tensors.
        if isinstance(logits, _DT):
            logits = logits.full_tensor()
        loss = loss_fn(logits=logits, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _log(rank, f"train step {i + 1}/{num_steps - 1}: loss={loss.item():.4f}")

    model.eval()
    _log(rank, "running final inference step to capture logits")
    with torch.inference_mode():
        batch = {k: v.to(device, non_blocking=True) for k, v in batches[-1].items()}
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        fwd_kwargs = filter_forward_kwargs(model, batch_no_labels)
        out = model(**fwd_kwargs)
        logits = getattr(out, "logits", out)
        if isinstance(logits, _DT):
            logits = logits.full_tensor()
        captured = logits.detach().contiguous().clone()

    del optimizer, loss_fn
    gc.collect()
    return captured
