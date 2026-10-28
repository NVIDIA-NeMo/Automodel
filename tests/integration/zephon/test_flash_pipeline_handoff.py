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

"""Public PP metadata handoff with a CPU attention oracle, not Flash kernels."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from transformers import LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nemo_automodel.components.datasets.zephon.attention import (
    adapt_model_inputs,
    enable_xdoc_attention,
)
from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.models.common.packing import (
    PackingCapabilities,
    configure_packing,
)
from nemo_automodel.shared.packed_seq import (
    indexed_mask_to_4d_block_causal as block_causal_attention_mask,
)
from tests.utils.zephon_distributed import (
    distributed_environment as distributed_environment,
)
from tests.utils.zephon_distributed import run_ranks, tiny_llama_config


def _loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")


def _flash_handoff_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        torch.manual_seed(48)
        reference = LlamaForCausalLM(tiny_llama_config()).train()
        pp = AutoPipeline(
            world_mesh=DeviceMesh("cpu", [0, 1], mesh_dim_names=("pp",)),
            dp_axis_names=(),
            pp_schedule="gpipe",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).build(reference, loss_fn=_loss)
        for part in pp.parts:
            configure_packing("flash_attention_2", model=part)
        contract = PackingCapabilities("flash_varlen")
        ids = torch.tensor([[2, 3, 4, 5, 6, 7], [12, 13, 14, 15, 16, 17]])
        positions = torch.tensor([[0, 1, 0, 1, 2, 3], [0, 1, 2, 0, 1, 2]])
        batch = adapt_model_inputs(
            {
                "input_ids": ids,
                "position_ids": positions,
                "document_ids": torch.tensor([[1, 1, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2]]),
                "labels": ids + 1,
                "padding_mask": torch.zeros_like(ids, dtype=torch.bool),
            },
        )
        seen = []

        def cpu_varlen_attention(module, query, key, value, attention_mask, **kwargs):
            # Consume only metadata that reached the real HF attention call.
            if "cu_seq_lens_q" not in kwargs:
                # Xdoc without exclusions takes native causal attention, not
                # the packed-varlen hook; the CPU oracle preserves that route.
                seen.append(None)
                return original_attention(module, query, key, value, attention_mask, **kwargs)
            cu = kwargs.pop("cu_seq_lens_q")
            torch.testing.assert_close(cu, kwargs.pop("cu_seq_lens_k"))
            maximum = kwargs.pop("max_length_q")
            assert maximum == kwargs.pop("max_length_k") == int(cu.diff().max())
            assert cu[0] == 0 and cu[-1] == query.shape[0] * query.shape[2]
            assert cu.dtype == torch.int32
            seen.append(cu.tolist())
            attention_ids = torch.empty(int(cu[-1]), dtype=torch.long)
            for index, (start, end) in enumerate(zip(cu[:-1], cu[1:]), start=1):
                attention_ids[int(start) : int(end)] = index
            dense = block_causal_attention_mask(attention_ids.view(query.shape[0], query.shape[2]))
            kwargs.pop("is_causal", None)
            return original_attention(module, query, key, value, dense, is_causal=False, **kwargs)

        for xdoc in (False, True):
            mapped = enable_xdoc_attention(batch, packing_contract=contract) if xdoc else batch
            expected_positions = torch.arange(6).expand_as(positions) if xdoc else positions
            torch.testing.assert_close(mapped["position_ids"], expected_positions)
            assert mapped["document_ids"] is batch["document_ids"]
            reference.zero_grad()
            for part in pp.parts:
                part.zero_grad()
            ALL_ATTENTION_FUNCTIONS.register("sdpa", original_attention)
            expected = reference(
                input_ids=ids,
                position_ids=mapped["position_ids"],
                attention_mask=(None if xdoc else block_causal_attention_mask(mapped["attention_document_ids"])),
            ).logits
            expected_loss = _loss(expected, batch["labels"])
            expected_loss.backward()
            seen.clear()
            ALL_ATTENTION_FUNCTIONS.register("sdpa", cpu_varlen_attention)
            losses = []
            actual = pp.step(
                ids,
                target=batch["labels"] if rank == 1 else None,
                losses=losses if rank == 1 else None,
                position_ids=mapped["position_ids"],
                _packed_seq_ids=mapped.get("attention_document_ids"),
            )
            expected_offsets = [[0, 6], [0, 6]] if xdoc else [[0, 2, 6], [0, 3, 6]]
            assert seen[-2:] == expected_offsets
            # PyTorch may run the first microbatch once to infer stage metadata.
            assert seen[:-2] in ([], [expected_offsets[0]])
            if rank == 1:
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(sum(losses), expected_loss)
            for part in pp.parts:
                for name, parameter in part.named_parameters():
                    torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, atol=1e-5, rtol=1e-4)
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original_attention)
        dist.destroy_process_group()


def test_flash_metadata_after_public_pipeline_microbatch_split(tmp_path, distributed_environment):
    run_ranks(_flash_handoff_worker, (str(tmp_path),))
