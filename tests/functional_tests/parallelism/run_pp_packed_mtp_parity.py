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

"""Packed Nemotron MTP: native PP2 against an unpartitioned reference.

Run from the repository root on two GPUs:
    PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/functional_tests/parallelism/run_pp_packed_mtp_parity.py
    PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/functional_tests/parallelism/run_pp_packed_mtp_parity.py --backend te

SDPA uses FP32; TE uses BF16. Two microbatches have different physical document
boundaries, internal padding, and rectangular boundary arrays with -1000 padding.
The reference derives MTP targets by slicing each document independently.
"""

import argparse
import copy
import json
import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mtp import PipelineCausalLMLoss
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

logger = logging.getLogger(__name__)
_SPANS = (((0, 64), (64, 128)), ((0, 32), (32, 64), (64, 128)))
_LENGTHS = ((63, 63), (31, 31, 63))


def _batch(device: torch.device) -> dict[str, torch.Tensor]:
    """Construct two distinct packed microbatches.

    Args:
        device: Device for all tensors.

    Returns:
        Mapping with input_ids, labels, position_ids [2, 128], int32 cu_seqlens
        and cu_seqlens_padded [2, 4] for real/physical cumulative lengths,
        max_seqlen [2], and a boolean
        attention_mask [2, 1, 128, 128] for the SDPA reference. Padding labels
        are -100; unused cumulative-length entries are -1000. Rows are
        independent packs, not a globally flattened token stream.
    """
    ids = torch.randint(1, 64, (2, 128), device=device)
    labels = torch.full_like(ids, -100)
    positions = torch.zeros_like(ids)
    doc_ids = torch.zeros_like(ids)
    for row, (spans, lengths) in enumerate(zip(_SPANS, _LENGTHS)):
        for doc, ((start, end), length) in enumerate(zip(spans, lengths)):
            labels[row, start : start + length] = ids[row, start : start + length]
            ids[row, start + length : end] = 0
            positions[row, start : start + length] = torch.arange(length, device=device)
            doc_ids[row, start:end] = doc
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
    causal = torch.ones(128, 128, dtype=torch.bool, device=device).tril()
    mask = (same_doc & causal & (ids != 0).unsqueeze(-2)).unsqueeze(1)
    return dict(
        input_ids=ids,
        labels=labels,
        position_ids=positions,
        attention_mask=mask,
        cu_seqlens=torch.tensor([[0, 63, 126, -1000], [0, 31, 62, 125]], dtype=torch.int32, device=device),
        cu_seqlens_padded=torch.tensor([[0, 64, 128, -1000], [0, 32, 64, 128]], dtype=torch.int32, device=device),
        max_seqlen=torch.tensor([64, 64], device=device),
    )


def _document_loss(
    predictions: tuple[torch.Tensor, ...], labels: torch.Tensor, spans: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    """Compute main and two-depth MTP CE without production target helpers.

    Args:
        predictions: Main and two MTP logits, each [1, sequence, vocab].
        labels: Original label IDs [1, sequence], with -100 for ignored tokens.
        spans: Half-open physical document intervals in the sequence axis.

    Returns:
        Scalar summed main loss plus 0.15 times each MTP depth's summed loss.
    """
    loss = torch.nn.functional.cross_entropy(predictions[0].float().flatten(0, 1), labels.flatten(), reduction="sum")
    for depth, logits in enumerate(predictions[1:], start=1):
        target = torch.full_like(labels, -100)
        for start, end in spans:
            if end - start > depth:
                target[:, start : end - depth] = labels[:, start + depth : end]
        loss = loss + 0.15 * torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), target.flatten(), reduction="sum"
        )
    return loss


def main() -> None:
    """Compare native pipeline loss, gradients, global norm, and one SGD update."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sdpa", "te"), default="sdpa")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 2, "This test requires exactly two ranks"
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        torch.manual_seed(4169)
        dtype = torch.float32 if args.backend == "sdpa" else torch.bfloat16
        config = MockNemotronV3Config(
            hidden_size=128,
            head_dim=32,
            num_hidden_layers=4,
            layers_block_type=["attention"] * 4,
            num_nextn_predict_layers=2,
            mtp_hybrid_override_pattern="*",
        )
        backend = BackendConfig(
            linear="torch",
            attn=args.backend,
            rms_norm="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        )
        model = NemotronHForCausalLM(config, backend=backend).to(device=device, dtype=dtype).train()
        reference = copy.deepcopy(model)
        batch = _batch(device)
        if args.backend == "te":
            batch.pop("attention_mask")
        # Full-model microbatches use the same inputs and weights as PP. The loss
        # oracle reads literal document spans, independently of seq_idx/cu arrays.
        reference_loss = torch.zeros((), device=device)
        for row, spans in enumerate(_SPANS):
            microbatch = {key: value[row : row + 1].clone() for key, value in batch.items() if key != "labels"}
            output = reference(**microbatch, qkv_format="thd")
            predictions = (output.logits, *(reference.lm_head(h) for h in output.mtp_per_depth_h))
            loss = _document_loss(predictions, batch["labels"][row : row + 1], spans)
            loss.backward()
            reference_loss += loss.detach()

        mesh = init_device_mesh("cuda", (2, 1), mesh_dim_names=("pp", "dp"))
        loss_fn = PipelineCausalLMLoss(MaskedCrossEntropy(), model, scaling_factor=0.3)
        loss_fn.is_packed = True
        pp = AutoPipeline(
            world_mesh=mesh,
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=device,
            dtype=dtype,
            pp_seq_len=128,
        ).build(model, loss_fn=loss_fn)
        part = pp.parts[0]
        loss_fn.model = part
        reference_parameters = dict(reference.named_parameters())
        before = {}
        for name, parameter in part.named_parameters():
            torch.testing.assert_close(parameter, reference_parameters[name], rtol=0, atol=0)
            before[name] = parameter.detach().float().clone()
        labels = batch.pop("labels")
        ids = batch.pop("input_ids")
        losses = [] if pp.info.has_last_stage else None
        kwargs = dict(target=labels if pp.info.has_last_stage else None, losses=losses, qkv_format="thd", **batch)
        if pp.info.has_first_stage:
            pp.info.schedule.step(ids, **kwargs)
        else:
            pp.info.schedule.step(**kwargs)

        rtol, atol = (2e-5, 2e-6) if dtype == torch.float32 else (0.02, 0.002)
        gradient_error = torch.zeros((), device=device, dtype=torch.float64)
        gradient_norm = torch.zeros_like(gradient_error)
        reference_local_norm = torch.zeros_like(gradient_error)
        for name, parameter in part.named_parameters():
            expected = reference_parameters[name].grad
            assert parameter.grad is not None and expected is not None, name
            assert torch.isfinite(parameter.grad).all() and torch.isfinite(expected).all(), name
            torch.testing.assert_close(parameter.grad, expected, rtol=rtol, atol=atol, msg=name)
            gradient_error += (parameter.grad.double() - expected.double()).square().sum()
            gradient_norm += parameter.grad.double().square().sum()
            reference_local_norm += expected.double().square().sum()
        dist.all_reduce(gradient_norm)
        reference_norm = sum(parameter.grad.double().square().sum() for parameter in reference.parameters())
        torch.testing.assert_close(gradient_norm.sqrt(), reference_norm.sqrt(), rtol=rtol, atol=atol)
        pipeline_loss = torch.stack(losses).sum() if losses is not None else torch.zeros_like(reference_loss)
        dist.broadcast(pipeline_loss, src=1)
        torch.testing.assert_close(pipeline_loss, reference_loss, rtol=rtol, atol=atol)

        torch.optim.SGD(part.parameters(), lr=0.001).step()
        torch.optim.SGD(reference.parameters(), lr=0.001).step()
        update_error = torch.zeros_like(gradient_error)
        update_norm = torch.zeros_like(gradient_error)
        for name, parameter in part.named_parameters():
            expected = reference_parameters[name]
            torch.testing.assert_close(parameter, expected, rtol=rtol, atol=atol, msg=name)
            update_error += (parameter.detach().double() - expected.detach().double()).square().sum()
            update_norm += (expected.detach().double() - before[name].double()).square().sum()
        assert update_norm > 0, "Optimizer update must be nonzero"
        logger.warning(
            json.dumps(
                dict(
                    rank=dist.get_rank(),
                    backend=args.backend,
                    dtype=str(dtype),
                    parameters=len(before),
                    pipeline_loss=float(pipeline_loss.detach()),
                    reference_loss=float(reference_loss),
                    global_gradient_norm=float(gradient_norm.sqrt()),
                    reference_gradient_norm=float(reference_norm.sqrt()),
                    gradient_relative_l2=float((gradient_error / reference_local_norm).sqrt()),
                    update_relative_l2=float((update_error / update_norm).sqrt()),
                )
            )
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
