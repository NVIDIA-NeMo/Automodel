#!/usr/bin/env python
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

"""TP logit parity test for models with custom (non-TP-plan) modules.

Validates that restoring a checkpoint with TP=2 produces the same logit
distribution as TP=1 when the model contains a custom replicated module
(``CustomEmbed``) whose weights are **not** part of the TP sharding plan
and **not** in the pretrained checkpoint.

Parity is measured via KL divergence between the TP=1 and TP=2 logit
distributions — a more robust metric than scalar-loss absolute tolerance
because it compares the full output distribution token-by-token.

This exercises the ``_broadcast_replicated_params_across_tp`` fix: without
the broadcast, each TP rank initialises ``CustomEmbed`` independently
with a different RNG state, producing divergent weights and an incorrect
logit distribution.

Test flow
---------
1. Build a base ``Qwen3ForCausalLM`` and save its weights (no
   ``CustomEmbed`` keys in the checkpoint).
2. **TP=1 reference**: build ``Qwen3WithCustomEmbed`` with
   ``torch.manual_seed(CUSTOM_SEED)``, load the base checkpoint
   (``strict=False``).  ``CustomEmbed`` retains its seed-based init.
   Compute logits.
3. **TP=2 with broadcast**: each rank builds ``Qwen3WithCustomEmbed``
   with ``torch.manual_seed(CUSTOM_SEED + rank)``  so rank >= 1 has
   divergent ``CustomEmbed`` weights.  Load base checkpoint, apply TP,
   then call ``_broadcast_replicated_params_across_tp``.  Rank 0's
   ``CustomEmbed`` is broadcast -> all ranks match the TP=1 reference.
   KL divergence should be ~0.
4. **TP=2 without broadcast** (negative test): same as above but skip
   the broadcast.  KL divergence should be large.

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/llm_pretrain_and_kd/run_tp_loss_parity_custom_module.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Sequence, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Replicate

from nemo_automodel._transformers.infrastructure import _broadcast_replicated_params_across_tp
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.distributed.parallelizer import _get_parallel_plan
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

# Seed used for the custom module; rank 0 uses this, rank k uses CUSTOM_SEED + k.
CUSTOM_SEED = 100


# ---------------------------------------------------------------------------
# Custom model: Qwen3 + a replicated EmbeddingBlock not in the TP plan
# ---------------------------------------------------------------------------
class CustomEmbed(nn.Module):
    """A simple learnable module that is NOT in any TP plan.

    It applies a learned linear projection to the hidden states.
    Because it is not registered in the HF TP plan, it remains replicated
    across TP ranks and must be explicitly synchronised.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


class Qwen3WithCustomEmbed(Qwen3ForCausalLM):
    """Tiny Qwen3 with an extra ``CustomEmbed`` layer after the transformer."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.custom_embed = CustomEmbed(config.hidden_size)

    def forward(self, **kwargs):
        outputs = self.model(**{k: v for k, v in kwargs.items() if k != "labels"})
        hidden = outputs[0]
        hidden = self.custom_embed(hidden)
        logits = self.lm_head(hidden)

        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(loss=None, logits=logits)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _init_distributed():
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    if torch.cuda.is_available():
        backend = "nccl"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    else:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        backend = "gloo"
    dist.init_process_group(backend=backend)


def _device() -> torch.device:
    if torch.cuda.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return torch.device("cpu")


def _device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_config() -> Qwen3Config:
    num_layers = 2
    return Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        use_cache=False,
        tie_word_embeddings=False,
        attention_bias=False,
        use_sliding_window=False,
        layer_types=["full_attention"] * num_layers,
    )


def _save_base_checkpoint(model: nn.Module, config: Qwen3Config, path: str) -> None:
    """Save model state_dict as safetensors + config.json."""
    os.makedirs(path, exist_ok=True)
    sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(path, "model.safetensors"))
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config.to_dict(), f)


def _gather_logits(logits: torch.Tensor, tp_mesh: DeviceMesh) -> torch.Tensor:
    """Gather TP-sharded DTensor logits to a full replicated local tensor."""
    if isinstance(logits, DTensor):
        logits = logits.redistribute(device_mesh=tp_mesh, placements=[Replicate()]).to_local()
    return cast(torch.Tensor, logits)


def _kl_divergence_from_logits(
    *, reference_logits: torch.Tensor, candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Return per-token KL(reference || candidate).

    Both inputs must be full (non-sharded) logits with shape ``[B, T, V]``.
    Returns a 1-D tensor of length ``B * T``.
    """
    assert reference_logits.shape == candidate_logits.shape
    vocab_size = reference_logits.shape[-1]
    ref_log_probs = F.log_softmax(reference_logits.float(), dim=-1).reshape(-1, vocab_size)
    cand_log_probs = F.log_softmax(candidate_logits.float(), dim=-1).reshape(-1, vocab_size)
    # F.kl_div expects input=log(q), target=log(p) when log_target=True -> KL(p || q)
    return F.kl_div(cand_log_probs, ref_log_probs, reduction="none", log_target=True).sum(-1)


def _compute_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits


def _build_custom_model_and_load(
    cfg: Qwen3Config,
    ckpt_path: str,
    device: torch.device,
    seed: int,
) -> Qwen3WithCustomEmbed:
    """Build Qwen3WithCustomEmbed, init with *seed*, load base checkpoint.

    The checkpoint does not contain ``custom_embed`` keys, so they keep
    the random init determined by *seed*.
    """
    torch.manual_seed(seed)
    if _device_type() == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Qwen3WithCustomEmbed(cfg).to(device=device, dtype=torch.float32)
    ckpt_sd = load_file(os.path.join(ckpt_path, "model.safetensors"), device=str(device))
    # strict=False: custom_embed keys are missing from the checkpoint
    model.load_state_dict(ckpt_sd, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    apply_cache_compatibility_patches()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kl_threshold", type=float, default=1e-6,
        help="Fail if max per-token KL divergence exceeds this threshold.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _init_distributed()
    device = _device()
    dt = _device_type()
    rank = _rank()
    world_size = _world_size()

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: requires world_size=2 (TP=2), got {world_size}", file=sys.stderr)
        return 1

    # Use a deterministic path so all ranks agree (tempfile.mkdtemp
    # creates different paths per process).
    tmpdir = os.path.join(os.getcwd(), ".tp_custom_embed_test_tmp")
    ckpt_path = os.path.join(tmpdir, "checkpoint")
    cfg = _build_config()

    # ------------------------------------------------------------------
    # 1. Save base Qwen3 checkpoint (NO custom_embed keys)
    # ------------------------------------------------------------------
    # Clean up any leftover from a previous run.
    if rank == 0:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
    dist.barrier()

    torch.manual_seed(42)
    if dt == "cuda":
        torch.cuda.manual_seed_all(42)
    base_model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.float32)
    if rank == 0:
        _save_base_checkpoint(base_model, cfg, ckpt_path)
    dist.barrier()
    del base_model

    # Deterministic inputs (same on all ranks).
    torch.manual_seed(999)
    if dt == "cuda":
        torch.cuda.manual_seed_all(999)
    input_ids = torch.randint(1, int(cfg.vocab_size), (2, 64), dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # 2. TP=1 reference logits
    #    Build Qwen3WithCustomEmbed with seed=CUSTOM_SEED so that
    #    custom_embed.proj.weight is deterministically initialised.
    #    The base Qwen3 weights are overwritten by the checkpoint.
    # ------------------------------------------------------------------
    ref_model = _build_custom_model_and_load(cfg, ckpt_path, device, seed=CUSTOM_SEED)
    ref_logits = _compute_logits(ref_model, input_ids)
    del ref_model

    # ------------------------------------------------------------------
    # 3. TP=2 WITH broadcast
    #    Rank 0 uses seed=CUSTOM_SEED (matches reference).
    #    Rank 1 uses seed=CUSTOM_SEED+1 (divergent custom_embed).
    #    After broadcast, rank 1 gets rank 0's custom_embed -> matches ref.
    # ------------------------------------------------------------------
    tp_model = _build_custom_model_and_load(cfg, ckpt_path, device, seed=CUSTOM_SEED + rank)
    tp_mesh = DeviceMesh(dt, torch.arange(world_size, device="cpu"), mesh_dim_names=("tp",))
    plan = _get_parallel_plan(tp_model, sequence_parallel=False)
    parallelize_module(tp_model, tp_mesh, plan)
    _broadcast_replicated_params_across_tp(tp_model, tp_mesh)

    tp_logits = _compute_logits(tp_model, input_ids)
    tp_logits_full = _gather_logits(tp_logits, tp_mesh)
    del tp_model

    # Compute KL divergence.
    kl = _kl_divergence_from_logits(reference_logits=ref_logits, candidate_logits=tp_logits_full)
    max_kl = kl.max().item()
    ok_bcast = max_kl <= args.kl_threshold

    ok_tensor = torch.tensor(1 if ok_bcast else 0, device=device, dtype=torch.int)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    all_ok_bcast = bool(ok_tensor.item())

    # Gather worst-case KL across ranks for reporting.
    kl_report = torch.tensor(max_kl, device=device, dtype=torch.float32)
    dist.all_reduce(kl_report, op=dist.ReduceOp.MAX)

    if rank == 0:
        status = "PASS" if all_ok_bcast else "FAIL"
        print(
            f"{status}: TP=2 (with broadcast) vs TP=1  "
            f"max_kl={kl_report.item():.6g}  "
            f"(threshold={args.kl_threshold:g})"
        )

    # ------------------------------------------------------------------
    # 4. TP=2 WITHOUT broadcast (negative test — KL should be large)
    # ------------------------------------------------------------------
    bad_model = _build_custom_model_and_load(cfg, ckpt_path, device, seed=CUSTOM_SEED + rank)
    tp_mesh2 = DeviceMesh(dt, torch.arange(world_size, device="cpu"), mesh_dim_names=("tp",))
    plan2 = _get_parallel_plan(bad_model, sequence_parallel=False)
    parallelize_module(bad_model, tp_mesh2, plan2)
    # *** deliberately skip broadcast ***

    bad_logits = _compute_logits(bad_model, input_ids)
    bad_logits_full = _gather_logits(bad_logits, tp_mesh2)
    del bad_model

    bad_kl = _kl_divergence_from_logits(reference_logits=ref_logits, candidate_logits=bad_logits_full)
    bad_max_kl = bad_kl.max().item()

    # The KL should exceed the threshold because custom_embed is
    # different on rank 0 vs rank 1 -> TP all-reduces mix wrong values.
    diverges = bad_max_kl > args.kl_threshold
    div_tensor = torch.tensor(1 if diverges else 0, device=device, dtype=torch.int)
    dist.all_reduce(div_tensor, op=dist.ReduceOp.MAX)
    any_diverges = bool(div_tensor.item())

    bad_kl_report = torch.tensor(bad_max_kl, device=device, dtype=torch.float32)
    dist.all_reduce(bad_kl_report, op=dist.ReduceOp.MAX)

    if rank == 0:
        status2 = "PASS" if any_diverges else "FAIL"
        print(
            f"{status2}: TP=2 (without broadcast) vs TP=1  "
            f"max_kl={bad_kl_report.item():.6g}  "
            f"(diverges as expected)"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    dist.barrier()
    if rank == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
    dist.destroy_process_group()

    return 0 if (all_ok_bcast and any_diverges) else 1


if __name__ == "__main__":
    raise SystemExit(main())
