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

The second phase drives those same varying-length FSDP2 stages through Engine:
one complete ``forward_backward`` call is the reference for an explicit plan
split across two calls. This additionally verifies that a non-final pipeline
schedule completes FSDP post-backward cleanup without synchronizing gradients.

Usage:
    torchrun --nproc-per-node=2 run_pp_grad_accum_parity.py
    torchrun --nproc-per-node=4 run_pp_grad_accum_parity.py  # PP2 x DP2
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

VOCAB = 256
BATCH = 2
# Two accumulation windows with *different* sequence lengths. The change is the
# trigger: equal lengths would skip the stage reset and hide the regression.
WINDOW_SEQ_LENS = (32, 48)
LOCAL_WINDOW_WEIGHT_SUM = BATCH * sum(WINDOW_SEQ_LENS)


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
        Dict with ``input_ids``, ``attention_mask``, and ``labels``, all of
        shape [BATCH, seq_len]. The mask is explicitly all ones so direct
        schedule and Engine collation exercise identical model inputs.
    """
    torch.manual_seed(seq_len)  # same window -> same data on every rank and phase
    ids = torch.randint(2, VOCAB, (BATCH, seq_len), device=device)
    dist.broadcast(ids, src=0)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


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


def _scale_aware_grad_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    relative_max_error: float = 0.03,
    absolute_floor: float = 2e-3,
    relative_norm_error: float = 0.02,
) -> tuple[bool, float, float, float]:
    """Compare equal-shaped local gradient shards with a BF16-aware bound.

    Args:
        actual: Float32 local FSDP gradient shard being checked.
        expected: Equal-shaped float32 local FSDP reference shard.
        relative_max_error: Allowed max-element error relative to the largest
            absolute reference element.
        absolute_floor: Absolute max-element allowance for values near zero.
        relative_norm_error: Allowed relative error in the full shard norm.

    Returns:
        Whether both the max-error and norm checks pass, followed by the
        observed max error, its allowed bound, and ``||actual||/||expected||``.
    """
    if actual.shape != expected.shape:
        return False, float("inf"), 0.0, float("inf")
    if expected.numel() == 0:
        return True, 0.0, absolute_floor, 1.0

    max_error = float((actual - expected).abs().max())
    error_bound = relative_max_error * float(expected.abs().max()) + absolute_floor
    actual_norm = float(actual.norm())
    expected_norm = float(expected.norm())
    if expected_norm == 0.0:
        norm_ratio = 1.0 if actual_norm == 0.0 else float("inf")
        norm_close = actual_norm <= absolute_floor
    else:
        norm_ratio = actual_norm / expected_norm
        norm_close = abs(norm_ratio - 1.0) <= relative_norm_error
    return max_error <= error_bound and norm_close, max_error, error_bound, norm_ratio


def _engine_window(seq_len: int, weight_scale: float, device: torch.device):
    """Build one flat Datum window with a caller-specific token denominator.

    Args:
        seq_len: Token length of every Datum in the window.
        weight_scale: Constant value assigned to every token loss weight.
        device: Device holding the generated tensors.

    Returns:
        ``BATCH`` Datums whose ``input_ids``, ``attention_mask``, ``labels``,
        and ``weights`` each have shape ``[sequence]``. The Engine collates
        them to ``[BATCH, sequence]`` before PP splits the batch axis.
    """
    from nemo_automodel.components.datasets.datum import Datum

    batch = _batch(seq_len, device)
    return [
        Datum(
            model_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            loss_fn_inputs={
                "labels": labels,
                "weights": torch.full_like(labels, weight_scale, dtype=torch.float32),
            },
        )
        for input_ids, attention_mask, labels in zip(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
    ]


def _run_engine_planned_parity(
    pp,
    mesh,
    device: torch.device,
    rank: int,
    direct_normalized_grads: dict[str, torch.Tensor],
) -> None:
    """Compare one Engine window with the same FSDP2 PP work split across calls.

    Args:
        pp: Built PP2 AutoPipeline whose local stage is FSDP2-wrapped.
        mesh: ``[pp, dp]`` device mesh; DP may be one or two.
        device: CUDA device for this rank's token tensors.
        rank: Global rank used in actionable assertion messages.
        direct_normalized_grads: Per-parameter direct-schedule reference. Each
            value is the float32 local FSDP shard accumulated over both
            varying-length windows, with every microbatch loss divided by the
            complete local-window token denominator.
    """
    from nemo_automodel.components.distributed.mesh import MeshContext
    from nemo_automodel.engine import Engine

    def token_losses(pred, loss_inputs):
        """Return unweighted token cross entropy in the Engine loss layout.

        Args:
            pred: Pipeline output with logits shaped
                ``[pp_microbatch, sequence, vocab]``.
            loss_inputs: Mapping containing ``labels`` and ``weights`` shaped
                ``[pp_microbatch, sequence]``.

        Returns:
            Per-token cross entropy with the same shape as ``weights``. Engine
            applies the token weights and complete-window denominator.
        """
        logits = pred.logits if hasattr(pred, "logits") else pred
        return torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1),
            loss_inputs["labels"].flatten(0, 1),
            reduction="none",
        ).view_as(loss_inputs["weights"])

    window_a = _engine_window(WINDOW_SEQ_LENS[0], 1.0, device)
    window_b = _engine_window(WINDOW_SEQ_LENS[1], 1.0, device)
    part = pp.parts[0]
    mesh_context = MeshContext.from_meshes(mesh)

    _zero_grads(part)
    reference_engine = Engine(
        pp,
        device=device,
        mesh_context=mesh_context,
        microbatch_size=BATCH,
        optimizers=torch.optim.SGD(part.parameters(), lr=0.01),
        max_grad_norm=None,
    )
    reference_result = reference_engine.forward_backward(window_a + window_b, token_losses)
    reference_grads = _snapshot_grads(part)

    _zero_grads(part)
    planned_engine = Engine(
        pp,
        device=device,
        mesh_context=mesh_context,
        microbatch_size=BATCH,
        optimizers=torch.optim.SGD(part.parameters(), lr=0.01),
        max_grad_norm=None,
    )
    planned_engine.begin_accumulation([window_a, window_b])
    result_a = planned_engine.forward_backward(window_a, token_losses)
    result_b = planned_engine.forward_backward(window_b, token_losses)
    planned_grads = _snapshot_grads(part)

    assert result_a.weight_sum.item() != result_b.weight_sum.item(), (
        f"[rank {rank}] Engine fixture must use unequal call denominators"
    )
    torch.testing.assert_close(result_a.loss_sum + result_b.loss_sum, reference_result.loss_sum, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(
        result_a.weight_sum + result_b.weight_sum,
        reference_result.weight_sum,
        rtol=0,
        atol=0,
    )
    combined_loss = (result_a.loss_sum + result_b.loss_sum) / (result_a.weight_sum + result_b.weight_sum)
    torch.testing.assert_close(combined_loss, reference_result.loss, rtol=2e-2, atol=2e-3)

    missing = set(reference_grads) ^ set(planned_grads)
    assert not missing, f"[rank {rank}] Engine gradient key mismatch: {sorted(missing)[:5]}"
    direct_missing = set(direct_normalized_grads) ^ set(planned_grads)
    assert not direct_missing, f"[rank {rank}] direct/Engine gradient key mismatch: {sorted(direct_missing)[:5]}"
    mismatches = []
    oracle_mismatches = []
    for name in sorted(reference_grads):
        got, want = planned_grads[name], reference_grads[name]
        if not torch.allclose(got, want, rtol=2e-2, atol=2e-3):
            mismatches.append(f"{name}: max|delta|={(got - want).abs().max():.3e}")
        direct = direct_normalized_grads[name]
        single_close, max_error, error_bound, norm_ratio = _scale_aware_grad_close(want, direct)
        if not single_close:
            oracle_mismatches.append(
                f"{name}/single: max|delta|={max_error:.3e} bound={error_bound:.3e} norm_ratio={norm_ratio:.6f}"
            )
        planned_close, max_error, error_bound, norm_ratio = _scale_aware_grad_close(got, direct)
        if not planned_close:
            oracle_mismatches.append(
                f"{name}/planned: max|delta|={max_error:.3e} bound={error_bound:.3e} norm_ratio={norm_ratio:.6f}"
            )
    if mismatches:
        raise AssertionError(
            f"[rank {rank}] Engine planned PP gradients != one complete window "
            f"({len(mismatches)}/{len(reference_grads)} parameters differ).\n  " + "\n  ".join(mismatches[:8])
        )
    if oracle_mismatches:
        raise AssertionError(
            f"[rank {rank}] Engine normalized gradients disagree with the independent direct normalized oracle "
            f"({len(oracle_mismatches)} mismatches; global_weight_sum={reference_result.weight_sum.item():.1f}, "
            f"dp_size={mesh['dp'].size()}).\n  " + "\n  ".join(oracle_mismatches[:8])
        )

    print(f"[rank {rank}] Engine planned PP/FSDP2 parity OK over {len(reference_grads)} parameters")


def main() -> None:
    """Compare accumulated gradients against the sum of per-window gradients."""
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in {2, 4}:
        raise ValueError(f"PP grad-accumulation parity requires 2 or 4 ranks, got {world}")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())

    from nemo_automodel.components.distributed.pipelining import AutoPipeline

    mesh = init_device_mesh("cuda", (2, world // 2), mesh_dim_names=("pp", "dp"))
    model = _build_model(device)

    def loss_fn(pred, target):
        """Return one PP microbatch's CE normalized by the full local window.

        Args:
            pred: Pipeline output with logits shaped
                ``[pp_microbatch, sequence, vocab]``.
            target: Token labels shaped ``[pp_microbatch, sequence]``.

        Returns:
            Scalar summed cross entropy divided by the complete two-window
            local token denominator. This matches Engine's backward scale.
        """
        logits = pred.logits if hasattr(pred, "logits") else pred
        loss_sum = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1),
            target.flatten(0, 1),
            reduction="sum",
        )
        return loss_sum / LOCAL_WINDOW_WEIGHT_SUM

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
    catches_dropped_window = any(not _scale_aware_grad_close(last_only[name], reference[name])[0] for name in reference)
    assert catches_dropped_window, (
        f"[rank {rank}] last-window-only gradients pass the Engine oracle tolerance; test cannot detect a wipe"
    )

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
    _run_engine_planned_parity(pp, mesh, device, rank, reference)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
