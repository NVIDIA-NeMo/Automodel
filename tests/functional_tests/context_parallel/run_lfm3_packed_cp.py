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

One packed THD row is sharded over the CP ranks with the production sharder. Three checks run:

* The halo exchange is fed a tensor whose value is the global token index. Every rank must
  receive exactly the indices before its shard that belong to its first document, and the
  backward pass must return one gradient per reader to the rank that owns each token.
* A small LFM3 with convolution, full-attention and sliding-attention layers runs on its
  shard on every rank. Gathered logits and summed parameter gradients must match the same
  model on the whole row in one process, for SDPA in float32 and for FlashAttention in bfloat16.
* With ``--fsdp2``, the recipe's composition: FSDP2 over the CP ranks through the model's
  parallelizer, the production sharder, the recipe's loss normalization and gradient clipping
  and one optimizer step, with whole-layer activation checkpointing off and on. Per-parameter
  gradients, the global gradient norm and the updated parameters must match one process.

Run::

    torchrun --standalone --nproc-per-node=2 tests/functional_tests/context_parallel/run_lfm3_packed_cp.py
    torchrun --standalone --nproc-per-node=2 tests/functional_tests/context_parallel/run_lfm3_packed_cp.py --fsdp2
"""

from __future__ import annotations

import sys
from dataclasses import replace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.models.lfm3.config import Lfm3Config
from nemo_automodel.components.models.lfm3.context_parallel import DocumentHalo, exchange_halo, shard_documents
from nemo_automodel.components.models.lfm3.model import Lfm3ForCausalLM
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

# One document longer than a shard, short ones around the shard boundaries, and a padding tail.
DOCUMENTS = (40, 3, 200, 9, 131, 60, 5)
SEQUENCE = 464
WINDOW = 16
MAX_GRAD_NORM = 1.0
LEARNING_RATE = 0.1
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
    """Build the seven-layer test model, identically on every call.

    Args:
        attn_implementation: Hugging Face attention backend of the attention layers.
        dtype: Parameter dtype.
        device: Device of the parameters.

    Returns:
        The model in training mode; every call returns the same initial parameters.
    """
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
    """Build one packed THD row of ``DOCUMENTS`` with a padding tail.

    Args:
        device: Device of the tensors.

    Returns:
        Batch with ``input_ids``, ``labels`` (-100 at padding) and ``position_ids`` of shape
        [1, SEQUENCE], ``seq_lens`` and ``seq_lens_padded`` of shape [1, documents], and the
        ``qkv_format`` tag.
    """
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
    """Run this rank's shard of the row and backpropagate the summed token loss.

    Args:
        model: Model whose parameters receive gradients.
        cp_mesh: CP mesh, or ``None`` to run the whole row.
        device: Device of the batch.

    Returns:
        Detached logits of shape [1, local sequence, vocab] and float32 gradients by parameter name.
    """
    _, batch, _ = shard_documents(cp_mesh, None, _batch(device))
    labels = batch.pop("labels")
    logits = model(**batch).logits.float()
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum").backward()
    return logits.detach(), {name: parameter.grad.float() for name, parameter in model.named_parameters()}


def _check_model(cp_mesh: DeviceMesh, device: torch.device) -> list[str]:
    """Compare the sharded model with one process for every attention backend of this device.

    Args:
        cp_mesh: CP mesh of all ranks.
        device: This rank's device.

    Returns:
        Descriptions of the cases whose logits or gradients are non-finite or outside tolerance;
        rank 0 compares, the other ranks return an empty list.
    """
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
        finite = all(
            torch.isfinite(tensor).all().item()
            for tensor in (*shards, expected, *gradients.values(), *expected_gradients.values())
        )
        logits_error = (torch.cat(shards, dim=1)[:, :SEQUENCE] - expected).abs().max().item()
        gradient_error = max(
            ((gradients[name] - expected_gradients[name]).abs().max() / expected_gradients[name].abs().max()).item()
            for name in gradients
        )
        print(
            f"{attn_implementation} {dtype} CP{cp_mesh.size()}: max logit error {logits_error:.3e}, "
            f"max relative gradient error {gradient_error:.3e}, finite {finite}",
            flush=True,
        )
        if not finite or logits_error > logits_tolerance or gradient_error > gradient_tolerance:
            failures.append(
                f"{attn_implementation}: logits {logits_error:.3e}, gradients {gradient_error:.3e}, finite {finite}"
            )
    return failures


def _check_halo(cp_mesh: DeviceMesh, device: torch.device) -> list[str]:
    """Send global token indices through the exchange and compare with the indices each rank must read.

    Args:
        cp_mesh: CP mesh of all ranks.
        device: This rank's device.

    Returns:
        Descriptions of the halo limits for which this rank received the wrong tokens or
        returned the wrong gradients.
    """
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


def _plain_name(name: str) -> str:
    """Return a parameter name without the activation-checkpoint wrapper segment."""
    return name.replace("._checkpoint_wrapped_module", "")


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """Gather an FSDP2-sharded DTensor into a plain Tensor; a plain Tensor is returned unchanged."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _check_fsdp2(mesh_context: MeshContext, device: torch.device) -> list[str]:
    """Train one step through the recipe's FSDP2 + CP composition and compare it with one process.

    The reference runs the whole row in one process with the recipe's objective, the summed
    token loss divided by the global supervised-token count, clips at ``MAX_GRAD_NORM`` and
    takes one SGD step. The composed side parallelizes the same model through its parallelizer,
    shards the row with the production sharder, scales the loss as the recipe does and clips
    through the recipe's helper.

    Args:
        mesh_context: Context for ``strategy: fsdp2``, ``dp_size: 1`` and ``cp_size`` equal to
            the world size, with or without activation checkpointing.
        device: This rank's CUDA device.

    Returns:
        Descriptions of the quantities that are non-finite or outside tolerance on this rank.
    """
    world = dist.get_world_size()
    label = f"fsdp2 cp{world} activation_checkpointing={bool(mesh_context.activation_checkpointing)}"
    batch = _batch(device)
    num_label_tokens = int((batch["labels"] != -100).sum().item())

    reference = _model("sdpa", torch.float32, device)
    _, full_row, _ = shard_documents(None, None, _batch(device))
    labels = full_row.pop("labels")
    loss = torch.nn.functional.cross_entropy(
        reference(**full_row).logits.flatten(0, 1), labels.flatten(), reduction="sum"
    )
    (loss / num_label_tokens).backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), MAX_GRAD_NORM)
    expected_gradients = {name: parameter.grad.clone() for name, parameter in reference.named_parameters()}
    torch.optim.SGD(reference.parameters(), lr=LEARNING_RATE).step()
    expected_parameters = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}

    model = parallelize_model(_model("sdpa", torch.float32, device), mesh_context)
    sharder = ContextParallelSharder(model, mesh_context.device_mesh, batch)
    _, local = sharder.shard(batch)
    labels = local.pop("labels")
    loss = torch.nn.functional.cross_entropy(model(**local).logits.flatten(0, 1), labels.flatten(), reduction="sum")
    # As in the recipe: normalize by the global label count, then undo FSDP's mean over the dp x cp group.
    (loss / num_label_tokens * world).backward()
    norm = scale_grads_and_clip_grad_norm(
        MAX_GRAD_NORM,
        [model],
        device_mesh=mesh_context.device_mesh,
        num_label_tokens=num_label_tokens,
        dp_group_size=world,
    )
    gradients = {_plain_name(name): _full(parameter.grad) for name, parameter in model.named_parameters()}
    torch.optim.SGD(model.parameters(), lr=LEARNING_RATE).step()
    parameters = {_plain_name(name): _full(parameter.detach()) for name, parameter in model.named_parameters()}

    failures = []
    if set(gradients) != set(expected_gradients):
        return [f"{label}: parameter names differ: {sorted(set(gradients) ^ set(expected_gradients))}"]
    finite = all(torch.isfinite(tensor).all().item() for tensor in (norm, *gradients.values(), *parameters.values()))
    gradient_error = max(
        ((gradients[name] - expected_gradients[name]).abs().max() / expected_gradients[name].abs().max()).item()
        for name in expected_gradients
    )
    parameter_error = max((parameters[name] - expected_parameters[name]).abs().max().item() for name in parameters)
    norm_error = ((norm.float() - expected_norm) / expected_norm).abs().item()
    if dist.get_rank() == 0:
        print(
            f"{label}: max relative gradient error {gradient_error:.3e}, relative norm error {norm_error:.3e} "
            f"(norm {expected_norm.item():.4f}), max parameter error {parameter_error:.3e}, finite {finite}",
            flush=True,
        )
    if not finite or gradient_error > 1e-4 or norm_error > 1e-4 or parameter_error > 1e-5:
        failures.append(
            f"{label}: gradients {gradient_error:.3e}, norm {norm_error:.3e}, parameters {parameter_error:.3e}, "
            f"finite {finite}"
        )
    return failures


def main() -> None:
    use_cuda = torch.cuda.is_available() and "--cpu" not in sys.argv
    # TF32 matmuls and convolutions differ by 3e-4 between shard shapes; float32 proper agrees to 1e-6.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl" if use_cuda else "gloo")
    device = torch.device(f"cuda:{dist.get_rank()}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if "--fsdp2" in sys.argv:
        if not use_cuda:
            raise SystemExit("--fsdp2 runs the recipe's FSDP2 composition and needs CUDA")
        precision = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        mesh_context = MeshContext.build(
            FSDP2Config(mp_policy=precision), ParallelismSizes(dp_size=1, cp_size=dist.get_world_size())
        )
        failures = []
        for activation_checkpointing in (False, True):
            failures += _check_fsdp2(replace(mesh_context, activation_checkpointing=activation_checkpointing), device)
    else:
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
