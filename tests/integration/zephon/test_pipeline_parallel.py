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

"""Real CPU pipeline stages consume Zephon masks and preserve packed attention."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from transformers import LlamaForCausalLM

from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT
from nemo_automodel.components.datasets.zephon.attention import enable_xdoc_attention
from nemo_automodel.components.datasets.zephon.loader import convert_text_batch
from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.models.common.packing import configure_packing
from tests.utils.zephon_distributed import (
    distributed_environment as distributed_environment,
)
from tests.utils.zephon_distributed import run_ranks, tiny_llama_config


def _text_batch():
    from zephon.types import SampleBatch, SampleMeta, SampleRecord

    samples = SampleBatch(
        tuple(
            SampleRecord(
                meta=SampleMeta(sample_id=(0, 0, index), lane_id=0, chunk_id=0),
                payload={"input_ids": list(range(start, start + 7)), "positions": [0, 1, 2, 0, 1, 2, 3]},
            )
            for index, start in enumerate((2, 12))
        )
    )
    return convert_text_batch(
        samples,
        packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT,
        ignore_index=-100,
        mask_segment_starts=False,
        allow_cross_document_attention=False,
    )


def _loss(logits, target):
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum")


def _llama_pipeline_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        torch.manual_seed(39)
        config = tiny_llama_config()
        reference = LlamaForCausalLM(config).train()
        pp = AutoPipeline(
            world_mesh=DeviceMesh("cpu", [0, 1], mesh_dim_names=("pp",)),
            dp_axis_names=(),
            pp_schedule="gpipe",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).build(reference, loss_fn=_loss)
        configure_packing("sdpa", model=reference)
        for part in pp.parts:
            configure_packing("sdpa", model=part)
        batch = _text_batch()
        ids = batch["input_ids"]
        for xdoc in (False, True):
            mapped = enable_xdoc_attention(batch, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT) if xdoc else batch
            reference.zero_grad()
            for part in pp.parts:
                part.zero_grad()
            expected = reference(
                input_ids=ids,
                position_ids=mapped["position_ids"],
                attention_mask=mapped.get("attention_mask"),
                _packed_seq_ids=mapped.get("attention_document_ids"),
            ).logits
            expected_loss = _loss(expected, batch["labels"])
            expected_loss.backward()
            losses = []
            actual = pp.step(
                ids,
                target=batch["labels"] if rank == 1 else None,
                losses=losses if rank == 1 else None,
                attention_mask=mapped.get("attention_mask"),
                _packed_seq_ids=mapped.get("attention_document_ids"),
                position_ids=mapped["position_ids"],
            )
            if rank == 1:
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(sum(losses), expected_loss)
            for part in pp.parts:
                for name, parameter in part.named_parameters():
                    torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, atol=1e-5, rtol=1e-4)
            changed_ids = ids.clone()
            changed_ids[:, :3] = 20
            changed = pp.step(
                changed_ids,
                target=batch["labels"] if rank == 1 else None,
                losses=[] if rank == 1 else None,
                attention_mask=mapped.get("attention_mask"),
                _packed_seq_ids=mapped.get("attention_document_ids"),
                position_ids=mapped["position_ids"],
            )
            if rank == 1:
                if xdoc:
                    assert not torch.allclose(actual[:, 3:], changed[:, 3:])
                else:
                    torch.testing.assert_close(actual[:, 3:], changed[:, 3:], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_packed_llama_two_stage_forward_backward_and_xdoc(tmp_path, distributed_environment):
    run_ranks(_llama_pipeline_worker, (str(tmp_path),))


def _llama_tensor_worker(rank, directory):
    from copy import deepcopy

    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.parallel import parallelize_module

    from nemo_automodel.components.distributed.parallelizer import (
        _get_parallel_plan,
        _update_attention_head_counts_for_tp,
    )

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        torch.manual_seed(45)
        reference = LlamaForCausalLM(tiny_llama_config()).train()
        model = deepcopy(reference)
        mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("tp",))
        parallelize_module(model, mesh, _get_parallel_plan(model, tp_size=2))
        _update_attention_head_counts_for_tp(model, 2)
        configure_packing("sdpa", model=reference)
        configure_packing("sdpa", model=model)
        batch = _text_batch()
        ids = batch["input_ids"]
        for xdoc in (False, True):
            mapped = enable_xdoc_attention(batch, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT) if xdoc else batch
            reference.zero_grad()
            model.zero_grad()
            expected = reference(
                input_ids=ids,
                position_ids=mapped["position_ids"],
                attention_mask=mapped.get("attention_mask"),
                _packed_seq_ids=mapped.get("attention_document_ids"),
            ).logits
            _loss(expected, batch["labels"]).backward()
            actual = model(
                input_ids=ids,
                position_ids=mapped["position_ids"],
                attention_mask=mapped.get("attention_mask"),
                _packed_seq_ids=mapped.get("attention_document_ids"),
            ).logits
            actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
            torch.testing.assert_close(actual, expected)
            _loss(actual, batch["labels"]).backward()
            for name, parameter in model.named_parameters():
                gradient = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
                torch.testing.assert_close(gradient, reference.get_parameter(name).grad, atol=1e-5, rtol=1e-4)
    finally:
        dist.destroy_process_group()


def test_packed_llama_tensor_parallel_forward_backward(tmp_path, distributed_environment):
    run_ranks(_llama_tensor_worker, (str(tmp_path),))
