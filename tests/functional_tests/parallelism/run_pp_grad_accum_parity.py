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

"""Per-parameter gradient regression for PP + gradient accumulation (PR #3530).

Stage re-initialization runs whenever the sequence length changes, and it used to
null every FSDP parameter gradient, so under accumulation only the final
micro-batch reached the optimizer. Loss cannot detect that -- it is computed
before the wipe -- and a global gradient norm can hide per-parameter error, so
this compares the gradients themselves.

The reference is the same pipeline running each accumulation window on its own
with gradients zeroed in between, summed afterwards. Accumulating N windows in
one window must equal the sum of the N windows run separately. Deriving the
reference from the same topology keeps FSDP sharding, dtype and reduction order
identical on both sides, so the comparison runs at a tight tolerance instead of
the loose bound a cross-topology reference would force.

Sequence length changes between windows, which is what triggers the stage reset
that caused the loss.

Usage:
    torchrun --nproc-per-node=2 run_pp_grad_accum_parity.py
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

VOCAB = 256
BATCH = 2
# Two accumulation windows with *different* sequence lengths. The change is the
# trigger: equal lengths would skip the stage reset and hide the regression.
WINDOW_SEQ_LENS = (32, 48)


def _build_model(device: torch.device) -> torch.nn.Module:
    """Build a tiny 4-layer Llama on *device* with real (non-zero) weights.

    Returns:
        A ``LlamaForCausalLM`` small enough for a 2-rank pipeline split.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        use_cache=False,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)


def _fsdp_parallelize(model, world_mesh, moe_mesh, *, dp_axis_names, **kwargs) -> None:
    """Shard *model* with FSDP2 so the stages hold ``FSDPModule`` parameters.

    The gradient wipe under test only touches ``FSDPModule`` submodules, so the
    regression does not reproduce on an unwrapped model.
    """
    from torch.distributed.fsdp import fully_shard

    dp_mesh = world_mesh[dp_axis_names[0]]
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        # After the pipeline split the stage holds its layers in a ModuleDict,
        # which iterates keys rather than modules.
        for layer in layers.values() if isinstance(layers, torch.nn.ModuleDict) else layers:
            fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)


def _batch(seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Build one deterministic batch.

    Args:
        seq_len: Sequence length for this window.
        device: Device to place the tensors on.

    Returns:
        Dict with ``input_ids`` and ``labels``, both of shape [BATCH, seq_len].
    """
    torch.manual_seed(seq_len)  # same window -> same data on every rank and phase
    ids = torch.randint(2, VOCAB, (BATCH, seq_len), device=device)
    dist.broadcast(ids, src=0)
    return {"input_ids": ids, "labels": ids.clone()}


def _run_window(pp, seq_len: int, device: torch.device) -> None:
    """Run one forward/backward window through the pipeline schedule."""
    batch = _batch(seq_len, device)
    labels = batch.pop("labels")
    model_input = batch.pop("input_ids")
    pp.update_seq_len(model_input.shape[1])
    losses = [] if pp.info.has_last_stage else None
    if pp.info.has_first_stage:
        pp.info.schedule.step(model_input, target=labels, losses=losses, **batch)
    else:
        pp.info.schedule.step(target=labels, losses=losses, **batch)


def _snapshot_grads(model) -> dict[str, torch.Tensor]:
    """Capture this rank's per-parameter gradients.

    Returns:
        Mapping of parameter name to a detached float32 copy of its gradient,
        with DTensors reduced to their local shard.
    """
    out = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        grad = grad.to_local() if hasattr(grad, "to_local") else grad
        out[name] = grad.detach().float().clone()
    return out


def _zero_grads(model) -> None:
    for param in model.parameters():
        param.grad = None


def main() -> None:
    """Compare accumulated gradients against the sum of per-window gradients."""
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())

    from nemo_automodel.components.distributed.pipelining import AutoPipeline

    mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("pp", "dp"))
    model = _build_model(device)

    def loss_fn(pred, target):
        logits = pred.logits if hasattr(pred, "logits") else pred
        return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), target.flatten(0, 1), reduction="sum")

    pp = AutoPipeline(
        world_mesh=mesh,
        moe_mesh=None,
        pp_axis_name="pp",
        dp_axis_names=("dp",),
        pp_schedule="1f1b",
        pp_microbatch_size=1,
        pp_batch_size=BATCH,
        device=device,
        dtype=torch.bfloat16,
        pp_seq_len=WINDOW_SEQ_LENS[0],
    ).build(model, loss_fn=loss_fn, parallelize_fn=_fsdp_parallelize)
    part = pp.parts[0]

    # Phase 1: accumulate both windows without an optimizer step in between.
    _zero_grads(part)
    for seq_len in WINDOW_SEQ_LENS:
        _run_window(pp, seq_len, device)
    accumulated = _snapshot_grads(part)

    # Phase 2: reference -- each window alone, then summed. Weights never change
    # (no optimizer step), so every window sees the same parameters as in phase 1.
    reference: dict[str, torch.Tensor] = {}
    for seq_len in WINDOW_SEQ_LENS:
        _zero_grads(part)
        _run_window(pp, seq_len, device)
        for name, grad in _snapshot_grads(part).items():
            reference[name] = grad if name not in reference else reference[name] + grad

    assert accumulated, f"[rank {rank}] no gradients captured; the test would be vacuous"
    missing = set(reference) ^ set(accumulated)
    assert not missing, f"[rank {rank}] gradient key mismatch: {sorted(missing)[:5]}"

    # A single window's gradients must not equal the accumulated ones, otherwise
    # the two phases are indistinguishable and the check proves nothing.
    _zero_grads(part)
    _run_window(pp, WINDOW_SEQ_LENS[-1], device)
    last_only = _snapshot_grads(part)
    differs = any(not torch.allclose(last_only[n], reference[n], rtol=1e-3, atol=1e-4) for n in reference)
    assert differs, f"[rank {rank}] windows produce identical gradients; test cannot detect a wipe"

    mismatches = []
    for name in sorted(reference):
        got, want = accumulated[name], reference[name]
        if not torch.allclose(got, want, rtol=2e-2, atol=2e-3):
            denom = want.abs().max().clamp_min(1e-12)
            mismatches.append(
                f"{name}: max|Δ|={(got - want).abs().max():.3e} rel={((got - want).abs().max() / denom):.3e}"
            )

    if mismatches:
        raise AssertionError(
            f"[rank {rank}] accumulated gradients != sum of per-window gradients over "
            f"{len(WINDOW_SEQ_LENS)} windows ({len(mismatches)}/{len(reference)} parameters differ). "
            f"Stage re-initialization is discarding earlier micro-batches.\n  " + "\n  ".join(mismatches[:8])
        )

    print(f"[rank {rank}] PP grad-accumulation parity OK over {len(reference)} parameters")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
