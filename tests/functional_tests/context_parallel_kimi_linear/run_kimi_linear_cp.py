#!/usr/bin/env python
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

"""End-to-end Kimi Linear context-parallel test.

Validates that a hybrid Kimi Linear model (KDA linear attention + MLA full
attention + MoE) produces matching logits and parameter gradients between CP=1
and CP=2, across three document layouts:

  Case 1 (single_document): one document per row, batch 2
  Case 2 (packed_documents): two packed documents, the second straddling the
                             CP shard boundary
  Case 3 (padded_tail): two packed documents plus a padding tail

Unlike the load-balanced ``context_parallel`` path, Kimi Linear shards the batch
itself into contiguous slices via ``shard_batch_for_kimi_cp``, so the test drives
the real sharder for both the reference (cp_mesh=None) and the CP=2 run. That
covers the pieces a single-process test cannot reach:

  * ``KimiDeltaAttention._forward_with_cp`` and FLA's rank-to-rank recurrent
    state / conv-boundary handoff
  * ``KimiMLAAttention._forward_with_cp``, the ``_AllGatherSequence`` autograd
    Function, and ``document_causal_flex_attention`` against gathered keys
  * collective ordering across the two layer types within one forward/backward

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel_kimi_linear/run_kimi_linear_cp.py
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist

# The local shard is SEQ_LEN / 2 = 128 tokens: two FLA chunks per rank, so the
# KDA path exercises intra-shard chunking as well as the cross-rank state
# handoff, and one full FlexAttention query block against a two-block gathered
# KV. A shorter sequence loses both.
SEQ_LEN = 256
HIDDEN_SIZE = 64
VOCAB_SIZE = 128
# Relative tolerances against the reference tensor's own scale, matching the
# metric used by run_blockdiag_cp_2rank.py. The two paths run different kernels
# (eager softmax vs FlexAttention, batched KDA vs FLA's CP kernels) in bfloat16,
# so exact equality is not the bar; a CP path that is actually wrong lands at a
# relative difference of order 1, far above these.
LOGITS_TOLERANCE = 2e-2
GRAD_TOLERANCE = 5e-2
# The KDA gate parameters are the one exception: a padding tail changes how FLA
# partitions the sequence into chunks, so their gradients accumulate in a
# different order than the reference's single unpadded call (measured 5.8e-2 on
# A_log for the padded_tail case, against 2.1e-2 without a padding tail).
KDA_GATE_GRAD_TOLERANCE = 1e-1
KDA_GATE_PARAMS = "_fp32_params"


def init_distributed():
    """Initialize distributed environment from torchrun env vars."""
    if not (dist.is_available() and dist.is_initialized()):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _tiny_kimi_config():
    """Build a small Kimi Linear config with one KDA layer and one MLA layer."""
    from nemo_automodel.components.models.kimi_linear.config import KimiLinear48BConfig

    return KimiLinear48BConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=16,
        # q_head_dim is 12, which FlexAttention pads to 16 -- keep it non-power-of-two
        # so the CP path exercises the same head-dim padding the 48B checkpoint needs.
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=16,
        mla_use_nope=True,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=32,
        first_k_dense_replace=1,
        linear_attn_config={
            # 1-based layer indices: layer 0 is KDA, layer 1 is MLA.
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
        },
        torch_dtype="bfloat16",
    )


def _build_model(device):
    """Build a Kimi Linear model whose weights are identical on every rank."""
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.kimi_linear.model import KimiLinear48BForCausalLM

    backend = BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(42)
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=backend)
    model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    model.to(device)
    model.train()
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return model


def _doc_ids_for_case(case: str, device) -> torch.Tensor:
    """Return the [batch, sequence] document map describing one case."""
    if case == "single_document":
        # Batch 2 on purpose: KDA's CP path loops over rows and builds a separate
        # FLA CP context per row, which a single-row batch never reaches.
        return torch.ones(2, SEQ_LEN, dtype=torch.int32, device=device)
    if case == "packed_documents":
        # Document 1 sits entirely on rank 0; document 2 straddles the shard boundary.
        doc_ids = torch.ones(1, SEQ_LEN, dtype=torch.int32, device=device)
        doc_ids[:, 64:] = 2
        return doc_ids
    if case == "padded_tail":
        doc_ids = torch.ones(1, SEQ_LEN, dtype=torch.int32, device=device)
        doc_ids[:, 64:192] = 2
        doc_ids[:, 192:] = 0
        return doc_ids
    raise ValueError(f"Unknown case {case!r}")


def _weighted_loss(logits: torch.Tensor, upstream: torch.Tensor) -> torch.Tensor:
    """Sum the logits under a fixed per-token weighting.

    The weighting is zero on padding tokens, so the loss is the same function of
    the real tokens whether it is evaluated on the full sequence or summed over
    contiguous CP shards.
    """
    return (logits.float() * upstream).sum()


def _relative_max_diff(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Return the max absolute difference relative to the reference's own scale."""
    scale = max(reference.abs().max().item(), 1e-3)
    return (actual.float() - reference.float()).abs().max().item() / scale


def _grad_tolerance(name: str) -> float:
    """Return the gradient tolerance that applies to one parameter."""
    return KDA_GATE_GRAD_TOLERANCE if KDA_GATE_PARAMS in name else GRAD_TOLERANCE


def _run_case(case: str, model, cp_mesh, rank: int, world_size: int, device) -> int:
    from nemo_automodel.components.models.kimi_linear.cp import shard_batch_for_kimi_cp

    model.zero_grad(set_to_none=True)

    doc_ids = _doc_ids_for_case(case, device)
    batch_size = doc_ids.shape[0]

    torch.manual_seed(1234)
    input_ids = torch.randint(1, VOCAB_SIZE, (batch_size, SEQ_LEN), device=device)
    upstream = torch.randn(batch_size, SEQ_LEN, VOCAB_SIZE, device=device, dtype=torch.float32)
    # Padding tokens never contribute: the reference path drops them from the KDA
    # kernel entirely while the CP path keeps them as their own segment.
    upstream = upstream * (doc_ids > 0)[:, :, None].float()
    dist.broadcast(input_ids, src=0)
    dist.broadcast(upstream, src=0)

    # ---- reference: no context parallelism, packed documents via the 2D mask ----
    reference_logits = model(
        input_ids=input_ids,
        attention_mask=doc_ids,
        padding_mask=doc_ids <= 0,
    ).logits
    _weighted_loss(reference_logits, upstream).backward()
    reference_logits = reference_logits.detach()
    reference_grads = {
        name: param.grad.detach().float() for name, param in model.named_parameters() if param.grad is not None
    }
    model.zero_grad(set_to_none=True)
    dist.barrier()

    # ---- context parallelism: Kimi Linear's own contiguous sharder ----
    batch = {"input_ids": input_ids.clone(), "attention_mask": doc_ids.clone()}
    _, sharded, layout = shard_batch_for_kimi_cp(cp_mesh, None, batch)
    assert layout.original_seq_len == layout.padded_seq_len, "this test uses a CP-aligned sequence length"

    packed_context = sharded["kimi_packed_context"]
    seq_start = packed_context.seq_start

    cp_logits = model(
        input_ids=sharded["input_ids"],
        padding_mask=sharded["padding_mask"],
        kimi_packed_context=packed_context,
    ).logits
    _weighted_loss(cp_logits, upstream[:, seq_start : seq_start + cp_logits.shape[1]]).backward()

    gathered = [torch.empty_like(cp_logits) for _ in range(world_size)]
    dist.all_gather(gathered, cp_logits.detach().contiguous())
    # Contiguous layout: rank r owns [r * S / cp, (r + 1) * S / cp), so a plain
    # concatenation in rank order rebuilds the global sequence.
    cp_logits_full = torch.cat(gathered, dim=1)

    # Padding positions are outside the model's contract and the two paths treat
    # them differently on purpose: the reference drops them from the KDA kernel
    # entirely and pads the output back with zeros, while the CP path runs them as
    # their own segment. Their labels are -100, so only real tokens are compared;
    # the padding-only difference is still reported so the exclusion stays visible.
    valid = (doc_ids > 0)[:, :, None]
    logits_diff = _relative_max_diff(cp_logits_full * valid, reference_logits * valid)
    padding_logits_diff = _relative_max_diff(cp_logits_full * ~valid, reference_logits * ~valid)

    failures = []
    worst_grad_name, worst_grad_diff = "", 0.0
    for name, param in model.named_parameters():
        if name not in reference_grads:
            continue
        if param.grad is None:
            failures.append(f"{name} has no gradient under CP")
            continue
        grad = param.grad.detach().float()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        if not torch.isfinite(grad).all():
            failures.append(f"gradient of {name} is not finite")
            continue
        diff = _relative_max_diff(grad, reference_grads[name])
        if diff > _grad_tolerance(name):
            failures.append(f"gradient of {name} relative max diff {diff:.6f} > {_grad_tolerance(name)}")
        if diff > worst_grad_diff:
            worst_grad_name, worst_grad_diff = name, diff

    if logits_diff > LOGITS_TOLERANCE:
        failures.append(f"logits relative max diff {logits_diff:.6f} > {LOGITS_TOLERANCE}")

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Kimi Linear CP=1 vs CP={world_size} -- {case}")
        print(f"{'=' * 70}")
        print(f"Logits shape: CP={tuple(cp_logits_full.shape)}, reference={tuple(reference_logits.shape)}")
        print(f"Logits relative max diff: {logits_diff:.6f} (tolerance {LOGITS_TOLERANCE})")
        print(f"Padding-position logits relative max diff (not asserted): {padding_logits_diff:.6f}")
        print(f"Worst parameter gradient: {worst_grad_name} {worst_grad_diff:.6f}")
        if failures:
            for failure in failures:
                print(f"  FAILED: {failure}")
        else:
            print("  PASSED")
        print(f"{'=' * 70}")
    return 1 if failures else 0


def main():
    init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 GPUs, got {world_size}", file=sys.stderr)
        sys.exit(1)

    try:
        from fla.ops.cp import build_cp_context  # noqa: F401
    except ImportError as error:
        if rank == 0:
            print(f"ERROR: Kimi Linear CP requires fla with context-parallel support: {error}", file=sys.stderr)
        sys.exit(1)

    from torch.distributed.device_mesh import init_device_mesh

    # One model and one mesh for every case: the CP forward is selected by the
    # packed context, not by the attached mesh, so the reference passes stay on
    # the non-CP path even after the mesh is wired.
    model = _build_model(device)
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))["cp"]
    for block in model.model.layers.values():
        block.self_attn.setup_cp_attention(cp_mesh)

    results = {}
    for case in ("single_document", "packed_documents", "padded_tail"):
        dist.barrier()
        try:
            results[case] = _run_case(case, model, cp_mesh, rank, world_size, device)
        except Exception:  # noqa: BLE001 - report and keep the remaining cases running
            if rank == 0:
                print(f"  {case}: ERROR")
                traceback.print_exc()
            results[case] = 1

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("Summary - Kimi Linear CP tests")
        print(f"{'=' * 70}")
        for case, result in results.items():
            print(f"  {case}: {'PASSED' if result == 0 else 'FAILED'}")
        print(f"{'=' * 70}\n")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    sys.exit(1 if any(result != 0 for result in results.values()) else 0)


if __name__ == "__main__":
    main()
