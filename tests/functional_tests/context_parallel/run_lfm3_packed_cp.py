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

"""LFM3 packed context parallelism through the real process group.

One packed THD row is sharded over the CP ranks with the production sharder. Two checks run:

* The halo exchange is fed a tensor whose value is the global token index. Every rank must
  receive exactly the indices before its shard that belong to its first document, and the
  backward pass must return one gradient per reader to the rank that owns each token.
* A small LFM3 with convolution, full-attention and sliding-attention layers runs on its
  shard on every rank. Gathered logits and summed parameter gradients must match the same
  model on the whole row in one process, for SDPA in float32 and for FlashAttention in bfloat16.

Run::

    torchrun --standalone --nproc-per-node=2 tests/functional_tests/context_parallel/run_lfm3_packed_cp.py
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from nemo_automodel.components.models.lfm3.config import Lfm3Config
from nemo_automodel.components.models.lfm3.context_parallel import DocumentHalo, exchange_halo, shard_documents
from nemo_automodel.components.models.lfm3.model import Lfm3ForCausalLM

# One document longer than a shard, short ones around the shard boundaries, and a padding tail.
DOCUMENTS = (40, 3, 200, 9, 131, 60, 5)
SEQUENCE = 464
WINDOW = 16
# (attention backend, dtype, logits tolerance, gradient tolerance relative to the largest entry)
CASES = {
    "cpu": [("sdpa", torch.float32, 2e-5, 1e-4)],
    "cuda": [
        ("sdpa", torch.float32, 2e-5, 1e-4),
        ("flash_attention_2", torch.bfloat16, 2e-2, 5e-2),
        ("flash_attention_4", torch.bfloat16, 2e-2, 5e-2),
    ],
}


def _model(attn_implementation: str, dtype: torch.dtype, device: torch.device) -> Lfm3ForCausalLM:
    torch.manual_seed(0)
    config = Lfm3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=7,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        sliding_window=WINDOW,
        conv_bias=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=["conv", "conv", "full_attention", "conv", "sliding_attention", "full_attention", "conv"],
    )
    model = Lfm3ForCausalLM(config)
    model.set_attn_implementation(attn_implementation)
    return model.to(device=device, dtype=dtype).train()


def _batch(device: torch.device) -> dict[str, torch.Tensor | str]:
    generator = torch.Generator().manual_seed(1)
    ids = torch.zeros(1, SEQUENCE, dtype=torch.long)
    ids[0, : sum(DOCUMENTS)] = torch.randint(3, 97, (sum(DOCUMENTS),), generator=generator)
    spans = [*DOCUMENTS[:-1], SEQUENCE - sum(DOCUMENTS[:-1])]
    return {
        "input_ids": ids.to(device),
        "labels": ids.masked_fill(ids == 0, -100).to(device),
        "position_ids": torch.cat([torch.arange(span) for span in spans])[None].to(device),
        "seq_lens": torch.tensor([DOCUMENTS], device=device),
        "seq_lens_padded": torch.tensor([spans], device=device),
        "qkv_format": "thd",
    }


def _loss_and_gradients(
    model: Lfm3ForCausalLM, cp_mesh: DeviceMesh | None, device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _, batch, _ = shard_documents(cp_mesh, None, _batch(device))
    labels = batch.pop("labels")
    logits = model(**batch).logits.float()
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum").backward()
    return logits.detach(), {name: parameter.grad.float() for name, parameter in model.named_parameters()}


def _check_model(cp_mesh: DeviceMesh, device: torch.device) -> list[str]:
    failures = []
    for attn_implementation, dtype, logits_tolerance, gradient_tolerance in CASES[device.type]:
        logits, gradients = _loss_and_gradients(_model(attn_implementation, dtype, device), cp_mesh, device)
        shards = [torch.empty_like(logits) for _ in range(cp_mesh.size())]
        dist.all_gather(shards, logits.contiguous())
        for gradient in gradients.values():
            dist.all_reduce(gradient)
        if dist.get_rank() != 0:
            continue
        expected, expected_gradients = _loss_and_gradients(_model(attn_implementation, dtype, device), None, device)
        logits_error = (torch.cat(shards, dim=1)[:, :SEQUENCE] - expected).abs().max().item()
        gradient_error = max(
            ((gradients[name] - expected_gradients[name]).abs().max() / expected_gradients[name].abs().max()).item()
            for name in gradients
        )
        print(
            f"{attn_implementation} {dtype} CP{cp_mesh.size()}: max logit error {logits_error:.3e}, "
            f"max relative gradient error {gradient_error:.3e}",
            flush=True,
        )
        if logits_error > logits_tolerance or gradient_error > gradient_tolerance:
            failures.append(f"{attn_implementation}: logits {logits_error:.3e}, gradients {gradient_error:.3e}")
    return failures


def _check_halo(cp_mesh: DeviceMesh, device: torch.device) -> list[str]:
    """Send global token indices through the exchange and compare with the indices each rank must read."""
    size, rank = cp_mesh.size(), cp_mesh.get_local_rank()
    shard_length = SEQUENCE // size
    boundaries = [0, *torch.tensor(DOCUMENTS[:-1]).cumsum(0).tolist()]
    starts = [shard * shard_length for shard in range(size)]
    lengths = tuple(start - max(boundary for boundary in boundaries if boundary <= start) for start in starts)
    halo = DocumentHalo(cp_mesh.get_group(), lengths)
    failures = []
    for limit in (None, 6, WINDOW - 1):
        counts = [length if limit is None else min(length, limit) for length in lengths]
        tokens = torch.arange(starts[rank], starts[rank] + shard_length, device=device, dtype=torch.float32)
        tokens = tokens.view(1, 1, -1).requires_grad_()
        received = exchange_halo(tokens, halo, limit)
        received.sum().backward()
        expected = torch.arange(starts[rank] - counts[rank], starts[rank], device=device, dtype=torch.float32)
        readers = torch.zeros(SEQUENCE, device=device)
        for start, count in zip(starts, counts):
            readers[start - count : start] += 1
        if not torch.equal(received.flatten(), expected):
            failures.append(f"rank {rank} limit {limit}: received {received.flatten().tolist()}")
        if not torch.equal(tokens.grad.flatten(), readers[starts[rank] : starts[rank] + shard_length]):
            failures.append(f"rank {rank} limit {limit}: gradients do not match the readers of each token")
    return failures


def main() -> None:
    use_cuda = torch.cuda.is_available() and "--cpu" not in sys.argv
    # TF32 matmuls and convolutions differ by 3e-4 between shard shapes; float32 proper agrees to 1e-6.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl" if use_cuda else "gloo")
    device = torch.device(f"cuda:{dist.get_rank()}" if use_cuda else "cpu")
    cp_mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("cp",))["cp"]
    failures = _check_halo(cp_mesh, device) + _check_model(cp_mesh, device)
    # Agree on the verdict so a failing rank never leaves its peers waiting in a collective.
    passed = torch.tensor(int(not failures), device=device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    dist.destroy_process_group()
    if not passed.item():
        raise SystemExit("\n".join(failures) or "another rank failed")


if __name__ == "__main__":
    main()
