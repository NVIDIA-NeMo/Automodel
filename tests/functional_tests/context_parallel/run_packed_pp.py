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

"""Engine/AutoPipeline parity for packed THD Llama.

The test runs the same two-microbatch, four-document update through PP=2 from
both raw THD metadata (``seq_lens``) and final THD metadata (``cu_seqlens``).
Each path runs training, forward-only evaluation, then training again on the
same pipeline. Evaluation must match a native eager Llama in summed loss and
weight statistics without creating gradients; both surrounding training calls
must match eager loss plus every local-stage gradient. Additional flat-Datum
runs cover explicit loss layouts and typed per-token callback outputs for raw
2+2 and final-THD 3+1 document splits. Under CP2 they run both forward and
forward/backward, use ragged internally padded sequences, and require every PP
and CP rank to receive the same restored Datum records. A final padded
two-Datum update verifies legacy callback mappings in logical input order.

Run with::

    torchrun --standalone --nproc-per-node=2 run_packed_pp.py

Set ``CP_SIZE=2`` and use four ranks to run the PP2 x CP2 output-restoration
matrix::

    CP_SIZE=2 torchrun --standalone --nproc-per-node=4 run_packed_pp.py
"""

from __future__ import annotations

import os
import warnings
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import LlamaConfig

from nemo_automodel.components.datasets.datum import (
    CollatedLossInputs,
    Datum,
    LossInputLayout,
    collate_datums,
)
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.context_parallel.utils import (
    attach_te_context_parallel,
    make_cp_batch_for_te,
)
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.llama.model import LlamaForCausalLM
from nemo_automodel.engine import Engine, collate_prebatched
from nemo_automodel.engine.outputs import LossFnOutputBatch, PerTokenOutput

VOCAB_SIZE = 64
SEQ_LEN = 8


def _build_model(device: torch.device) -> LlamaForCausalLM:
    """Build the deterministic two-layer model used by eager and PP paths."""
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    config._attn_implementation = "sdpa"
    backend = BackendConfig(attn="te", linear="torch", rms_norm="torch_fp32", rope_fusion=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = LlamaForCausalLM(config, backend=backend)
    return model.to(device=device, dtype=torch.bfloat16).train()


def _raw_batch(device: torch.device) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    """Return two packed rows containing two documents each.

    Token tensors use ``[B=2, S=8]``. The reset positions and lengths describe
    documents of lengths ``[3, 5]`` and ``[4, 4]``.
    """
    input_ids = torch.tensor(
        [[1, 2, 3, 11, 12, 13, 14, 15], [21, 22, 23, 24, 31, 32, 33, 34]],
        device=device,
    )
    labels = torch.tensor(
        [[2, 3, -100, 12, 13, 14, 15, -100], [22, 23, 24, -100, 32, 33, 34, -100]],
        device=device,
    )
    weights = torch.tensor(
        [[1.0, 2.0, 0.0, 1.5, 0.5, 2.5, 1.0, 0.0], [0.5, 1.0, 2.0, 0.0, 1.0, 3.0, 1.5, 0.0]],
        device=device,
    )
    model_inputs: dict[str, object] = {
        "input_ids": input_ids,
        "position_ids": torch.tensor(
            [[0, 1, 2, 0, 1, 2, 3, 4], [0, 1, 2, 3, 0, 1, 2, 3]],
            device=device,
        ),
        "seq_lens": torch.tensor([[3, 5], [4, 4]], device=device),
        "seq_lens_padded": torch.tensor([[3, 5], [4, 4]], device=device),
        "qkv_format": "thd",
    }
    return model_inputs, labels, weights


def _clone_mapping(values: dict[str, object]) -> dict[str, object]:
    return {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in values.items()}


def _token_losses(output, loss_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = getattr(output, "logits", output)
    return F.cross_entropy(
        logits.float().reshape(-1, VOCAB_SIZE),
        loss_inputs["labels"].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(loss_inputs["weights"])


def _eager_reference(
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, object],
    torch.Tensor,
    torch.Tensor,
]:
    """Build eager eval and training references for the shared packed batch.

    Args:
        device: CUDA device on which the reference model and batch run.

    Returns:
        The scalar eval loss sum, scalar eval weight sum, scalar normalized
        training loss, per-parameter gradient mapping, raw model-input mapping,
        labels of shape [batch, sequence], and weights of shape [batch, sequence].
        Raw token and position tensors have shape [batch, sequence].
    """
    raw_inputs, labels, weights = _raw_batch(device)
    prepared = make_cp_batch_for_te(
        None,
        {**_clone_mapping(raw_inputs), "labels": labels.clone()},
    )
    prepared_labels = prepared.pop("labels")

    model = _build_model(device)
    model.eval()
    with torch.no_grad():
        eval_output = model(**prepared)
        eval_token_losses = _token_losses(
            eval_output,
            {"labels": prepared_labels, "weights": weights.reshape(-1)},
        )
        eval_loss_sum = (eval_token_losses * weights.reshape(-1)).sum()
        eval_weight_sum = weights.sum()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("eager forward-only reference unexpectedly created parameter gradients")

    model.train()
    output = model(**prepared)
    token_losses = _token_losses(output, {"labels": prepared_labels, "weights": weights.reshape(-1)})
    loss = (token_losses * weights.reshape(-1)).sum() / weights.sum()
    loss.backward()
    grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    del model
    return eval_loss_sum.detach(), eval_weight_sum.detach(), loss.detach(), grads, raw_inputs, labels, weights


def _explicit_layout_datums(device: torch.device, lengths: list[int]) -> list[Datum]:
    """Build flat Datums carrying all three explicit loss-input layouts."""
    datums = []
    token_start = 1
    for datum_index, length in enumerate(lengths):
        input_ids = torch.arange(token_start, token_start + length, device=device)
        datums.append(
            Datum(
                model_inputs={"input_ids": input_ids},
                loss_fn_inputs={
                    "labels": (input_ids + 1) % VOCAB_SIZE,
                    "weights": torch.ones(length, dtype=torch.float32, device=device),
                    "advantages": input_ids.to(torch.float32) / 10,
                    "old_logprobs": -input_ids.to(torch.float32) / 20,
                    "sample_id": torch.tensor((datum_index + 1) * 11, device=device),
                    # Leading size equals the number of PP microbatches. It
                    # must remain a complete replicated vector in both.
                    "global_coefficients": torch.tensor([0.25, 0.75], device=device),
                },
                loss_fn_input_layouts={
                    "labels": LossInputLayout.PER_TOKEN,
                    "weights": LossInputLayout.PER_TOKEN,
                    "advantages": LossInputLayout.PER_TOKEN,
                    "old_logprobs": LossInputLayout.PER_TOKEN,
                    "sample_id": LossInputLayout.PER_DATUM,
                    "global_coefficients": LossInputLayout.REPLICATED,
                },
            )
        )
        token_start += length
    return datums


def _raw_explicit_layout_collate(datums: list[Datum]):
    return collate_datums(datums, packed=True)


def _final_explicit_layout_collate(datums: list[Datum]):
    """Produce a model-ready flat THD stream while retaining layout metadata."""
    model_inputs, loss_inputs = collate_datums(datums, packed=True)
    lengths = torch.tensor(
        [datum.seq_len for datum in datums], dtype=torch.int32, device=model_inputs["input_ids"].device
    )
    final_model_inputs = {
        "input_ids": model_inputs["input_ids"].reshape(-1),
        "position_ids": model_inputs["position_ids"].reshape(-1),
        "cu_seqlens": F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32),
        "max_seqlen": lengths.max(),
        "qkv_format": "thd",
    }
    final_loss_inputs = {
        name: value.reshape(-1) if loss_inputs.layouts[name] is LossInputLayout.PER_TOKEN else value
        for name, value in loss_inputs.items()
    }
    return final_model_inputs, CollatedLossInputs(
        final_loss_inputs,
        layouts=loss_inputs.layouts,
        item_to_datum=loss_inputs.item_to_datum,
    )


def _cp_padded_explicit_layout_collate(datums: list[Datum], *, final_thd: bool):
    """Pack Datums with per-sequence CP padding, retaining their real boundaries."""
    real_lengths = torch.tensor(
        [datum.seq_len for datum in datums], dtype=torch.int32, device=datums[0].input_ids.device
    )
    padded_lengths = ((real_lengths + 3) // 4) * 4  # TE CP2 requires every sequence slot to divide by 2*CP.
    total_tokens = int(padded_lengths.sum().item())
    device = datums[0].input_ids.device

    input_ids = torch.zeros((1, total_tokens), dtype=torch.long, device=device)
    position_ids = torch.zeros_like(input_ids)
    padding_mask = torch.ones_like(input_ids, dtype=torch.bool)
    token_fields = {
        "labels": torch.full((1, total_tokens), -100, dtype=torch.long, device=device),
        "weights": torch.zeros((1, total_tokens), dtype=torch.float32, device=device),
        "advantages": torch.zeros((1, total_tokens), dtype=torch.float32, device=device),
        "old_logprobs": torch.zeros((1, total_tokens), dtype=torch.float32, device=device),
    }
    offset = 0
    for datum, real_length, padded_length in zip(datums, real_lengths.tolist(), padded_lengths.tolist()):
        token_slice = slice(offset, offset + real_length)
        input_ids[0, token_slice] = datum.input_ids
        position_ids[0, token_slice] = torch.arange(real_length, device=device)
        padding_mask[0, token_slice] = False
        for name in token_fields:
            token_fields[name][0, token_slice] = datum.loss_fn_inputs[name]
        offset += padded_length

    model_inputs: dict[str, object]
    if final_thd:
        model_inputs = {
            "input_ids": input_ids.reshape(-1),
            "position_ids": position_ids.reshape(-1),
            "padding_mask": padding_mask.reshape(-1),
            "cu_seqlens": F.pad(real_lengths.cumsum(0), (1, 0)).to(torch.int32),
            "cu_seqlens_padded": F.pad(padded_lengths.cumsum(0), (1, 0)).to(torch.int32),
            "max_seqlen": real_lengths.max(),
            "qkv_format": "thd",
        }
        token_fields = {name: value.reshape(-1) for name, value in token_fields.items()}
    else:
        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "padding_mask": padding_mask,
            "seq_lens": real_lengths.unsqueeze(0),
            "seq_lens_padded": padded_lengths.unsqueeze(0),
            "qkv_format": "thd",
        }

    loss_inputs = CollatedLossInputs(
        {
            **token_fields,
            "sample_id": torch.stack([datum.loss_fn_inputs["sample_id"] for datum in datums]),
            "global_coefficients": datums[0].loss_fn_inputs["global_coefficients"],
        },
        layouts={
            "labels": LossInputLayout.PER_TOKEN,
            "weights": LossInputLayout.PER_TOKEN,
            "advantages": LossInputLayout.PER_TOKEN,
            "old_logprobs": LossInputLayout.PER_TOKEN,
            "sample_id": LossInputLayout.PER_DATUM,
            "global_coefficients": LossInputLayout.REPLICATED,
        },
        item_to_datum=tuple(range(len(datums))),
    )
    return model_inputs, loss_inputs


def _explicit_layout_losses(output, loss_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Use both RL-style token fields so their routing affects the numerator."""
    return _token_losses(output, loss_inputs) + 0.1 * loss_inputs["advantages"] + 0.05 * loss_inputs["old_logprobs"]


def _explicit_layout_eager_reference(
    device: torch.device,
    datums: list[Datum],
    collate_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute full-stream numerator and denominator for a layout-aware batch."""
    model_inputs, loss_inputs = collate_fn(datums)
    prepared = make_cp_batch_for_te(
        None,
        {**_clone_mapping(model_inputs), "labels": loss_inputs["labels"].clone()},
    )
    prepared = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in prepared.items()
    }
    prepared_labels = prepared.pop("labels")
    eager_loss_inputs = {
        "labels": prepared_labels,
        "weights": loss_inputs["weights"].reshape(-1).to(device),
        "advantages": loss_inputs["advantages"].reshape(-1).to(device),
        "old_logprobs": loss_inputs["old_logprobs"].reshape(-1).to(device),
    }

    model = _build_model(device).eval()
    with torch.no_grad():
        losses = _explicit_layout_losses(model(**prepared), eager_loss_inputs)
        loss_sum = (losses * eager_loss_inputs["weights"]).sum()
        weight_sum = eager_loss_inputs["weights"].sum()
    del model
    return loss_sum.detach(), weight_sum.detach()


def _build_pipeline(
    device: torch.device,
    mesh_context: MeshContext,
) -> AutoPipeline:
    model = _build_model(device)
    pipeline = AutoPipeline(
        world_mesh=mesh_context.device_mesh,
        moe_mesh=None,
        **mesh_context.pipeline_axis_kwargs(),
        pp_schedule="1f1b",
        pp_microbatch_size=1,
        pp_batch_size=2,
        device=device,
        dtype=torch.bfloat16,
        pp_seq_len=SEQ_LEN,
    ).build(model, loss_fn=_token_losses)
    del model
    if mesh_context.cp_size > 1:
        cp_mesh = mesh_context.device_mesh["cp"]
        tp_mesh = mesh_context.device_mesh["tp"]
        configured = sum(attach_te_context_parallel(part, cp_mesh, tp_mesh) for part in pipeline.parts)
        if configured == 0:
            raise AssertionError("PP2 x CP2 functional configured no Transformer Engine attention modules")
    return pipeline


def _assert_local_grad_parity(
    pipeline: AutoPipeline,
    reference_grads: dict[str, torch.Tensor],
) -> float:
    max_diff = 0.0
    checked = 0
    for part in pipeline.parts:
        for name, parameter in part.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"pipeline parameter {name} has no gradient")
            torch.testing.assert_close(
                parameter.grad.float(),
                reference_grads[name].float(),
                atol=5e-3,
                rtol=5e-2,
            )
            max_diff = max(max_diff, (parameter.grad.float() - reference_grads[name].float()).abs().max().item())
            checked += 1
    if checked == 0:
        raise AssertionError("pipeline rank owns no checked parameters")
    return max_diff


def _run_thd_layout(
    layout: str,
    device: torch.device,
    mesh_context: MeshContext,
    reference_eval_loss_sum: torch.Tensor,
    reference_eval_weight_sum: torch.Tensor,
    reference_loss: torch.Tensor,
    reference_grads: dict[str, torch.Tensor],
    raw_inputs: dict[str, object],
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> AutoPipeline:
    """Run training, forward-only, then training parity for one THD layout.

    Args:
        layout: ``raw`` for batch-major pre-THD tensors or ``final`` for the
            flattened model-ready THD stream.
        device: CUDA device on which this physical pipeline rank runs.
        mesh_context: Runtime mesh whose PP axis has size two.
        reference_eval_loss_sum: Scalar eager forward-only weighted numerator.
        reference_eval_weight_sum: Scalar eager full-sequence weight sum.
        reference_loss: Scalar eager normalized training loss.
        reference_grads: Mapping from parameter names to eager gradient tensors
            with each parameter's native shape.
        raw_inputs: Mapping whose token and position tensors have shape [batch,
            sequence] before THD flattening.
        labels: Target token IDs of shape [batch, sequence].
        weights: Token weights of shape [batch, sequence].

    Returns:
        The two-stage pipeline after the second training pass. Every local
        parameter has its native gradient shape.
    """
    pipeline = _build_pipeline(device, mesh_context)
    if layout == "raw":
        model_inputs = _clone_mapping(raw_inputs)
        loss_labels = labels.clone()
        loss_weights = weights.clone()
    elif layout == "final":
        model_inputs = make_cp_batch_for_te(
            None,
            {**_clone_mapping(raw_inputs), "labels": labels.clone()},
        )
        loss_labels = model_inputs.pop("labels")
        loss_weights = weights.reshape(-1).clone()
    else:
        raise ValueError(f"unknown THD layout: {layout}")

    datum = Datum(
        model_inputs=model_inputs,
        loss_fn_inputs={"labels": loss_labels, "weights": loss_weights},
    )
    engine = Engine(
        pipeline,
        device=device,
        mesh_context=mesh_context,
        collate_fn=collate_prebatched,
    )

    pre_eval_loss, pre_eval_outputs = engine.forward_backward([datum], _token_losses)
    torch.testing.assert_close(pre_eval_loss.float(), reference_loss.float(), atol=2e-3, rtol=2e-3)
    assert pre_eval_outputs == []
    pre_eval_grad_diff = _assert_local_grad_parity(pipeline, reference_grads)
    for part in pipeline.parts:
        part.zero_grad(set_to_none=True)

    forward_result = engine.forward([datum], _token_losses)

    torch.testing.assert_close(
        forward_result.loss_sum.float(),
        reference_eval_loss_sum.float(),
        atol=4e-2,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        forward_result.weight_sum.float(),
        reference_eval_weight_sum.float(),
        atol=0,
        rtol=0,
    )
    assert forward_result.loss_fn_outputs == []
    if any(parameter.grad is not None for part in pipeline.parts for parameter in part.parameters()):
        raise AssertionError(f"PP2 {layout} THD forward-only evaluation unexpectedly created parameter gradients")
    if any(part.training for part in pipeline.parts):
        raise AssertionError(f"PP2 {layout} THD forward-only evaluation did not keep every model part in eval mode")

    # Reuse the exact pipeline immediately. Together with the training call
    # above, this proves both train->eval and eval->train schedule transitions
    # restore temporary split/loss callbacks and backward state.
    loss, outputs = engine.forward_backward([datum], _token_losses)

    torch.testing.assert_close(loss.float(), reference_loss.float(), atol=2e-3, rtol=2e-3)
    assert outputs == []
    grad_diff = _assert_local_grad_parity(pipeline, reference_grads)
    if dist.get_rank() == 0:
        print(
            f"PP2 {layout} THD forward+backward parity passed "
            f"(eval loss-sum diff="
            f"{(forward_result.loss_sum.float() - reference_eval_loss_sum.float()).abs().item():.6f}, "
            f"train loss diff={(loss.float() - reference_loss.float()).abs().item():.6f}, "
            f"pre/post-eval grad max={pre_eval_grad_diff:.6f}/{grad_diff:.6f})"
        )
    return pipeline


def _run_explicit_loss_layout(
    pipeline: AutoPipeline,
    layout: str,
    execution: str,
    device: torch.device,
    mesh_context: MeshContext,
) -> None:
    """Validate loss routing plus typed token-output restoration for one layout."""
    if layout == "raw":
        lengths = [3, 5, 7, 3] if mesh_context.cp_size > 1 else [2, 2, 2, 2]
        collate_fn = (
            partial(_cp_padded_explicit_layout_collate, final_thd=False)
            if mesh_context.cp_size > 1
            else _raw_explicit_layout_collate
        )
        expected_microbatch_ids = {(11, 22), (33, 44)}
    elif layout == "final":
        lengths = [3, 3, 3, 9] if mesh_context.cp_size > 1 else [1, 1, 2, 4]
        collate_fn = (
            partial(_cp_padded_explicit_layout_collate, final_thd=True)
            if mesh_context.cp_size > 1
            else _final_explicit_layout_collate
        )
        expected_microbatch_ids = {(11, 22, 33), (44,)}
    else:
        raise ValueError(f"unknown explicit loss layout: {layout}")
    if execution not in {"forward", "forward_backward"}:
        raise ValueError(f"unknown Engine execution: {execution}")

    for part in pipeline.parts:
        part.zero_grad(set_to_none=True)

    datums = _explicit_layout_datums(device, lengths)
    reference_loss_sum, reference_weight_sum = _explicit_layout_eager_reference(device, datums, collate_fn)

    def loss_with_outputs(output, loss_inputs):
        sample_ids = tuple(int(value) for value in loss_inputs["sample_id"].tolist())
        if sample_ids not in expected_microbatch_ids:
            raise AssertionError(f"PP2 {layout} routed unexpected sample IDs {sample_ids}")
        valid_cu_seqlens = loss_inputs["cu_seqlens"].reshape(-1)
        valid_cu_seqlens = valid_cu_seqlens[valid_cu_seqlens >= 0]
        assert valid_cu_seqlens.numel() - 1 == len(sample_ids)
        if mesh_context.cp_size > 1:
            # Each full pipeline chunk reserves 12 padded slots and CP2 owns six.
            assert loss_inputs["weights"].numel() == 6
        torch.testing.assert_close(
            loss_inputs["global_coefficients"],
            torch.tensor([0.25, 0.75], device=device),
        )
        assert loss_inputs["global_coefficients"].shape == (2,)
        losses = _explicit_layout_losses(output, loss_inputs)
        rl_probe = torch.stack((loss_inputs["advantages"], loss_inputs["old_logprobs"]), dim=-1)
        return losses, LossFnOutputBatch(
            per_token={"rl_probe": PerTokenOutput(rl_probe)},
            per_datum=[{"sample_id": value} for value in loss_inputs["sample_id"]],
        )

    engine = Engine(
        pipeline,
        device=device,
        mesh_context=mesh_context,
        microbatch_size=4,
        collate_fn=collate_fn,
    )
    result = getattr(engine, execution)(datums, loss_with_outputs)

    if execution == "forward":
        torch.testing.assert_close(result.loss_sum.float(), reference_loss_sum.float(), atol=4e-2, rtol=2e-3)
        torch.testing.assert_close(result.weight_sum.float(), reference_weight_sum.float(), atol=0, rtol=0)
        outputs = result.loss_fn_outputs
        if any(parameter.grad is not None for part in pipeline.parts for parameter in part.parameters()):
            raise AssertionError(f"PP2 x CP{mesh_context.cp_size} {layout} forward unexpectedly created gradients")
    else:
        loss, outputs = result
        reference_loss = reference_loss_sum / reference_weight_sum
        torch.testing.assert_close(loss.float(), reference_loss.float(), atol=4e-2, rtol=2e-3)
        if not any(parameter.grad is not None for part in pipeline.parts for parameter in part.parameters()):
            raise AssertionError(f"PP2 x CP{mesh_context.cp_size} {layout} backward created no gradients")

    output_ids = torch.stack([item["sample_id"] for item in outputs]).to(torch.long)
    expected_ids = torch.tensor([11, 22, 33, 44], device=device)
    torch.testing.assert_close(output_ids, expected_ids)
    restored_probe = []
    for datum, item in zip(datums, outputs):
        expected_probe = torch.stack(
            (datum.loss_fn_inputs["advantages"], datum.loss_fn_inputs["old_logprobs"]),
            dim=-1,
        )
        torch.testing.assert_close(item["rl_probe"], expected_probe)
        assert not item["rl_probe"].requires_grad
        restored_probe.append(item["rl_probe"])
    flat_probe = torch.cat(restored_probe)
    gathered = [torch.empty_like(output_ids) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, output_ids)
    assert all(torch.equal(ids, expected_ids) for ids in gathered)
    gathered_probe = [torch.empty_like(flat_probe) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_probe, flat_probe)
    assert all(torch.equal(probe, flat_probe) for probe in gathered_probe)
    if dist.get_rank() == 0:
        datum_counts = "2+2" if layout == "raw" else "3+1"
        reported_loss = result.loss_sum.item() if execution == "forward" else result[0].item()
        print(
            f"PP2 x CP{mesh_context.cp_size} {layout} {execution} explicit loss/output routing passed "
            f"({datum_counts} Datums; loss={reported_loss:.6f})"
        )


def _run_padded_output_broadcast(pipeline: AutoPipeline, device: torch.device, mesh_context: MeshContext) -> None:
    for part in pipeline.parts:
        part.zero_grad(set_to_none=True)

    datums = []
    for sample_id, offset in ((17, 0), (29, 8)):
        input_ids = torch.arange(1 + offset, 1 + offset + SEQ_LEN, device=device) % VOCAB_SIZE
        labels = input_ids.roll(-1)
        labels[-1] = -100
        datums.append(
            Datum(
                model_inputs={"input_ids": input_ids},
                loss_fn_inputs={
                    "labels": labels,
                    "weights": (labels != -100).to(torch.float32),
                    "sample_id": torch.tensor(sample_id, device=device),
                },
            )
        )

    def loss_with_output(output, loss_inputs):
        logits = getattr(output, "logits", output)
        losses = _token_losses(output, loss_inputs)
        return losses, [{"sample_id": loss_inputs["sample_id"][0], "score": logits.float().mean()}]

    loss, outputs = Engine(
        pipeline,
        device=device,
        mesh_context=mesh_context,
        microbatch_size=2,
    ).forward_backward(datums, loss_with_output)

    expected_ids = torch.tensor([17, 29], device=device)
    output_ids = torch.stack([item["sample_id"] for item in outputs]).to(device=device, dtype=torch.long)
    torch.testing.assert_close(output_ids, expected_ids)
    gathered = [torch.empty_like(output_ids) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, output_ids)
    assert all(torch.equal(ids, expected_ids) for ids in gathered)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(item["score"]) for item in outputs)
    if dist.get_rank() == 0:
        print("PP2 padded per-Datum outputs passed (logical order [17, 29] synchronized on both ranks)")


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cp_size = int(os.environ.get("CP_SIZE", "1"))
    if cp_size not in {1, 2}:
        raise ValueError(f"packed PP functional supports CP_SIZE 1 or 2, got {cp_size}")
    expected_world_size = 2 * cp_size
    if dist.get_world_size() != expected_world_size:
        raise ValueError(
            f"packed PP functional with CP_SIZE={cp_size} requires {expected_world_size} ranks, "
            f"got {dist.get_world_size()}"
        )

    mesh_context = MeshContext.build(
        FSDP2Config(),
        ParallelismSizes(dp_size=1, pp_size=2, cp_size=cp_size),
        world_size=dist.get_world_size(),
    )
    try:
        if cp_size > 1:
            pipeline = _build_pipeline(device, mesh_context)
            for layout in ("raw", "final"):
                for execution in ("forward", "forward_backward"):
                    _run_explicit_loss_layout(pipeline, layout, execution, device, mesh_context)
                    dist.barrier()
            return

        (
            reference_eval_loss_sum,
            reference_eval_weight_sum,
            reference_loss,
            reference_grads,
            raw_inputs,
            labels,
            weights,
        ) = _eager_reference(device)
        _run_thd_layout(
            "raw",
            device,
            mesh_context,
            reference_eval_loss_sum,
            reference_eval_weight_sum,
            reference_loss,
            reference_grads,
            raw_inputs,
            labels,
            weights,
        )
        dist.barrier()
        final_pipeline = _run_thd_layout(
            "final",
            device,
            mesh_context,
            reference_eval_loss_sum,
            reference_eval_weight_sum,
            reference_loss,
            reference_grads,
            raw_inputs,
            labels,
            weights,
        )
        dist.barrier()
        _run_explicit_loss_layout(final_pipeline, "raw", "forward", device, mesh_context)
        dist.barrier()
        _run_explicit_loss_layout(final_pipeline, "final", "forward", device, mesh_context)
        dist.barrier()
        _run_padded_output_broadcast(final_pipeline, device, mesh_context)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
