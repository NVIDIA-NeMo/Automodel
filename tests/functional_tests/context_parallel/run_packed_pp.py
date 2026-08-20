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

"""Two-GPU Engine/AutoPipeline parity for packed THD Llama.

The test runs the same two-microbatch, four-document update through PP=2 from
both raw THD metadata (``seq_lens``) and final THD metadata (``cu_seqlens``).
Each path runs training, forward-only evaluation, then training again on the
same pipeline. Evaluation must match a native eager Llama in summed loss and
weight statistics without creating gradients; both surrounding training calls
must match eager loss plus every local-stage gradient. A final padded two-Datum
update verifies that Engine broadcasts the callback's per-Datum mappings to
both pipeline ranks in logical input order.

Run with::

    torchrun --standalone --nproc-per-node=2 run_packed_pp.py
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import LlamaConfig

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.context_parallel.utils import make_cp_batch_for_te
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.llama.model import LlamaForCausalLM
from nemo_automodel.engine import Engine, collate_prebatched

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
    if dist.get_world_size() != 2:
        raise ValueError("packed PP functional requires exactly two ranks")

    mesh_context = MeshContext.build(
        FSDP2Config(),
        ParallelismSizes(dp_size=1, pp_size=2),
        world_size=dist.get_world_size(),
    )
    try:
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
        _run_padded_output_broadcast(final_pipeline, device, mesh_context)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
