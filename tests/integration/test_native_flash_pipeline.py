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

import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nemo_automodel.components.datasets.utils import neat_packed_collater
from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.models.common.packing import PackingCapabilities, configure_packing


def block_causal_attention_mask(document_ids):
    """Return additive [batch, 1, sequence, sequence] masks for IDs [batch, sequence]."""
    same = document_ids[:, :, None] == document_ids[:, None, :]
    valid = (document_ids[:, :, None] > 0) & (document_ids[:, None, :] > 0)
    allowed = (same & valid).tril().unsqueeze(1)
    return torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)


def _loss(logits, labels):
    """Sum next-token CE for logits [batch, sequence, vocab] and labels [batch, sequence]."""
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
        reference = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                attn_implementation="sdpa",
                use_cache=False,
            )
        ).train()
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
        documents = [
            {
                "input_ids": token_ids.tolist(),
                "labels": (token_ids + 1).tolist(),
                "position_ids": position_ids.tolist(),
                "attention_mask": mask,
            }
            for token_ids, position_ids, mask in zip(
                ids, positions, ([1, 1, 2, 2, 2, 0], [1, 1, 1, 2, 2, 2]), strict=True
            )
        ]
        documents[0]["labels"][-1] = -100
        batch = neat_packed_collater(documents, packing=contract, defer_flash_metadata=True)
        assert batch["_packed_seq_ids"].shape == ids.shape
        assert "cu_seq_lens_q" not in batch
        seen = []

        def cpu_varlen_attention(module, query, key, value, attention_mask, **kwargs):
            """Consume actual decoder offsets for Q/K/V [batch, heads, sequence, head_dim].

            Return attention output [batch, sequence, heads, head_dim] via SDPA.
            This verifies the Flash metadata contract on CPU, not a Flash kernel.
            """
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

        reference.zero_grad()
        for part in pp.parts:
            part.zero_grad()
        expected = reference(
            input_ids=ids,
            position_ids=positions,
            attention_mask=block_causal_attention_mask(batch["_packed_seq_ids"]),
        ).logits
        expected_loss = _loss(expected, batch["labels"])
        expected_loss.backward()
        ALL_ATTENTION_FUNCTIONS.register("sdpa", cpu_varlen_attention)
        losses = []
        actual = pp.step(
            ids,
            target=batch["labels"] if rank == 1 else None,
            losses=losses if rank == 1 else None,
            position_ids=batch["position_ids"],
            _packed_seq_ids=batch["_packed_seq_ids"],
        )
        expected_offsets = [[0, 2, 5, 6], [0, 3, 6]]
        assert seen[-2:] == expected_offsets
        # PyTorch may run the first microbatch once to infer stage metadata.
        assert seen[:-2] in ([], [expected_offsets[0]])
        if rank == 1:
            valid = batch["_packed_seq_ids"] > 0
            torch.testing.assert_close(actual[valid], expected[valid])
            torch.testing.assert_close(sum(losses), expected_loss)
        for part in pp.parts:
            for name, parameter in part.named_parameters():
                torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, atol=1e-5, rtol=1e-4)
        torch.optim.SGD(reference.parameters(), lr=0.1).step()
        for part in pp.parts:
            torch.optim.SGD(part.parameters(), lr=0.1).step()
            for name, parameter in part.named_parameters():
                torch.testing.assert_close(parameter, reference.get_parameter(name), atol=1e-5, rtol=1e-4)
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original_attention)
        dist.destroy_process_group()


def test_native_neat_collator_to_public_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_GIL", "1")
    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    context = mp.spawn(_flash_handoff_worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Native Flash PP handoff did not finish within 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
