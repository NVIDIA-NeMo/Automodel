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

"""Engine/FSDP2 packed-THD CP and TP+CP parity for dense Llama, Qwen2, and Qwen3.

Run with::

    torchrun --nproc-per-node=2 tests/functional_tests/context_parallel/run_dense_packed_cp.py
    torchrun --nproc-per-node=2 tests/functional_tests/context_parallel/run_dense_packed_cp.py --tp-size 2
    torchrun --nproc-per-node=4 tests/functional_tests/context_parallel/run_dense_packed_cp.py --tp-size 2
"""

from __future__ import annotations

import argparse
import os
import warnings

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformer_engine.pytorch import DotProductAttention
from transformers import LlamaConfig, Qwen2Config, Qwen3Config

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.context_parallel.utils import (
    attach_te_context_parallel,
    make_cp_batch_for_te,
)
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.llama.model import LlamaForCausalLM
from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM
from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM
from nemo_automodel.engine import Engine, collate_prebatched

NUM_HIDDEN_LAYERS = 2


def _clone_batch(batch: dict[str, object]) -> dict[str, object]:
    """Clone a BSHD batch without sharing tensor storage.

    Args:
        batch: Mapping whose tensor values use ``[B, S]`` layout. ``B`` is
            batch and ``S`` is the packed slot width. Non-tensor metadata is
            copied by reference.

    Returns:
        Mapping with cloned tensors in the same layouts.
    """
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _build_model(
    model_kind: str,
    device: torch.device,
    *,
    sliding_window: int | None = None,
) -> torch.nn.Module:
    torch.manual_seed(1234)
    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    if model_kind == "llama":
        config = LlamaConfig(**common)
        model_cls = LlamaForCausalLM
    elif model_kind == "qwen2":
        config = Qwen2Config(**common)
        model_cls = Qwen2ForCausalLM
    else:
        config = Qwen3Config(**common, head_dim=8)
        model_cls = Qwen3ForCausalLM
    if sliding_window is not None:
        if not hasattr(config, "sliding_window") or not hasattr(config, "layer_types"):
            raise ValueError(f"{model_kind} config does not support sliding attention.")
        config.sliding_window = sliding_window
        config.layer_types = ["sliding_attention"] * config.num_hidden_layers
    config._attn_implementation = "sdpa"
    config.use_cache = False
    config.torch_dtype = torch.bfloat16
    backend = BackendConfig(attn="te", linear="torch", rms_norm="torch_fp32", rope_fusion=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = model_cls(config, backend=backend)
    assert len(model.model.layers) == NUM_HIDDEN_LAYERS
    assert all(isinstance(layer.self_attn.attn_module, DotProductAttention) for layer in model.model.layers)
    return model.to(device=device, dtype=torch.bfloat16).train()


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a replicated global tensor from a distributed tensor.

    Args:
        tensor: Plain tensor or DTensor sharded by FSDP, TP, or both.

    Returns:
        Plain tensor with the global shape, replicated over its device mesh.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _reconstruct_global_tokens(
    local: torch.Tensor,
    local_indices: torch.Tensor,
    *,
    total_tokens: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Restore TE's load-balanced local THD shards to global token order.

    Args:
        local: Local tensor ``[T/cp, ...]`` in TE partition order.
        local_indices: Global token indices ``[T/cp]`` held by this rank.
        total_tokens: Global packed token count ``T``.
        group: Context-parallel process group.

    Returns:
        Replicated tensor ``[T, ...]`` in original packed order.
    """
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    gathered_indices = [torch.empty_like(local_indices) for _ in range(world_size)]
    dist.all_gather(gathered, local, group=group)
    dist.all_gather(gathered_indices, local_indices, group=group)
    output = local.new_empty((total_tokens, *local.shape[1:]))
    for indices, shard in zip(gathered_indices, gathered):
        output.index_copy_(0, indices.to(torch.long), shard)
    return output


def _run_model(
    model_kind: str,
    device: torch.device,
    mesh_context: MeshContext,
    distributed_config: FSDP2Config,
) -> None:
    import transformer_engine_torch as tex

    assert mesh_context.device_mesh is not None
    cp_mesh = mesh_context.device_mesh["cp"]
    tp_mesh = mesh_context.device_mesh["tp"]
    cp_rank = cp_mesh.get_local_rank()
    cp_group = cp_mesh.get_group()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device),
        "labels": torch.tensor([[2, 3, 4, -100, 6, 7, 8, -100]], device=device),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=device),
        "seq_lens": torch.tensor([[4, 4]], device=device),
        "seq_lens_padded": torch.tensor([[4, 4]], device=device),
        "qkv_format": "thd",
    }

    baseline_model = _build_model(model_kind, device)
    baseline_batch = make_cp_batch_for_te(None, _clone_batch(batch))
    baseline_labels = baseline_batch.pop("labels")
    baseline_logits = baseline_model(**baseline_batch).logits.squeeze(0)
    baseline_loss = baseline_logits.float().square().mean()
    baseline_loss.backward()
    baseline_grads = [layer.self_attn.q_proj.weight.grad.detach().float() for layer in baseline_model.model.layers]
    assert baseline_labels.shape == (8,)

    cp_model = _build_model(model_kind, device)
    cp_model = FSDP2Manager(distributed_config, device_mesh=mesh_context.device_mesh).parallelize(cp_model)
    attention_cp_mesh = cp_mesh if cp_mesh.size() > 1 else None
    configured = attach_te_context_parallel(cp_model, attention_cp_mesh, tp_mesh)
    assert configured == NUM_HIDDEN_LAYERS

    raw_model_inputs = _clone_batch(batch)
    raw_labels = raw_model_inputs.pop("labels")
    datum = Datum(
        model_inputs=raw_model_inputs,
        loss_fn_inputs={
            "labels": raw_labels,
            "weights": torch.ones_like(raw_labels, dtype=torch.float32),
        },
    )
    assert "seq_lens" in datum.model_inputs and "cu_seqlens" not in datum.model_inputs

    observed: dict[str, torch.Tensor] = {}

    def loss_fn(output, loss_inputs):
        local_logits = _full_tensor(output.logits).squeeze(0)
        observed["logits"] = local_logits.detach()
        observed["cu_seqlens"] = loss_inputs["cu_seqlens"].detach()
        observed["labels"] = loss_inputs["labels"].detach()
        return local_logits.float().square().mean(dim=-1)

    forward_backward_result = Engine(
        cp_model,
        device=device,
        mesh_context=mesh_context,
        collate_fn=collate_prebatched,
        defer_fsdp_grad_sync=distributed_config.defer_fsdp_grad_sync,
    ).forward_backward([datum], loss_fn)

    local_logits = observed["logits"]
    assert observed["labels"].shape == (8 // cp_mesh.size(),)
    cu_seqlens = observed["cu_seqlens"]
    indices = tex.thd_get_partitioned_indices(cu_seqlens, 8, cp_mesh.size(), cp_rank).to(torch.int32)
    cp_logits = _reconstruct_global_tokens(
        local_logits,
        indices,
        total_tokens=8,
        group=cp_group,
    )
    cp_grads = [_full_tensor(layer.self_attn.q_proj.weight.grad).detach().float() for layer in cp_model.model.layers]

    cp_loss = cp_logits.float().square().mean()
    # The Engine must normalize the exact distributed logits tightly. The
    # baseline comparison is looser because independently executed BF16
    # attention paths can differ by roughly one ULP, which doubles under x**2.
    torch.testing.assert_close(forward_backward_result.loss.float(), cp_loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(forward_backward_result.loss.float(), baseline_loss.detach(), atol=1e-6, rtol=1e-2)
    torch.testing.assert_close(cp_logits, baseline_logits, atol=3e-2, rtol=3e-2)
    for cp_grad, baseline_grad in zip(cp_grads, baseline_grads):
        torch.testing.assert_close(cp_grad, baseline_grad, atol=1e-3, rtol=5e-2)
    assert torch.isfinite(cp_logits).all()
    assert all(torch.isfinite(cp_grad).all() for cp_grad in cp_grads)
    if dist.get_rank() == 0:
        loss_diff = (forward_backward_result.loss.float() - baseline_loss.detach()).abs().item()
        normalization_diff = (forward_backward_result.loss.float() - cp_loss).abs().item()
        output_diff = (cp_logits.float() - baseline_logits.float()).abs().max().item()
        grad_diff = max(
            (cp_grad - baseline_grad).abs().max().item() for cp_grad, baseline_grad in zip(cp_grads, baseline_grads)
        )
        print(
            f"{model_kind} full TP={tp_mesh.size()} CP={cp_mesh.size()}: "
            f"Engine packed parity passed "
            f"(loss={loss_diff:.6f}, normalization={normalization_diff:.6f}, "
            f"logits max={output_diff:.6f}, grad max={grad_diff:.6f})"
        )


def _run_packed_sliding_attention(model_kind: str, device: torch.device) -> None:
    """Compare packed TE sliding attention with per-document HuggingFace behavior."""
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device)
    position_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device)

    reference_model = _build_model(model_kind, device, sliding_window=3)
    reference_logits = torch.cat(
        [
            reference_model(
                input_ids[:4].unsqueeze(0),
                position_ids=position_ids[:4].unsqueeze(0),
                cache_position=torch.arange(4, device=device),
                use_cache=False,
            ).logits,
            reference_model(
                input_ids[4:].unsqueeze(0),
                position_ids=position_ids[4:].unsqueeze(0),
                cache_position=torch.arange(4, device=device),
                use_cache=False,
            ).logits,
        ],
        dim=1,
    )
    reference_logits.float().square().sum().backward()
    reference_grads = [layer.self_attn.q_proj.weight.grad.detach().float() for layer in reference_model.model.layers]

    packed_model = _build_model(model_kind, device, sliding_window=3)
    packed_model.load_state_dict(reference_model.state_dict())
    packed_logits = packed_model(
        input_ids,
        position_ids=position_ids,
        qkv_format="thd",
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        max_seqlen=4,
    ).logits
    packed_logits.float().square().sum().backward()
    packed_grads = [layer.self_attn.q_proj.weight.grad.detach().float() for layer in packed_model.model.layers]

    torch.testing.assert_close(packed_logits, reference_logits, atol=3e-2, rtol=3e-2)
    for packed_grad, reference_grad in zip(packed_grads, reference_grads):
        torch.testing.assert_close(packed_grad, reference_grad, atol=1e-3, rtol=5e-2)
    if dist.get_rank() == 0:
        output_diff = (packed_logits.float() - reference_logits.float()).abs().max().item()
        grad_diff = max(
            (packed_grad - reference_grad).abs().max().item()
            for packed_grad, reference_grad in zip(packed_grads, reference_grads)
        )
        print(f"{model_kind} packed sliding parity passed (logits max={output_diff:.6f}, grad max={grad_diff:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if dist.get_world_size() % args.tp_size != 0:
        raise ValueError(f"World size {dist.get_world_size()} must be divisible by TP size {args.tp_size}.")
    cp_size = dist.get_world_size() // args.tp_size
    distributed_config = FSDP2Config()
    mesh_context = MeshContext.build(
        distributed_config,
        ParallelismSizes(dp_size=1, cp_size=cp_size, tp_size=args.tp_size),
        world_size=dist.get_world_size(),
    )
    assert mesh_context.device_mesh is not None
    assert mesh_context.device_mesh["dp_shard_cp"].size() == cp_size
    try:
        for model_kind in ("llama", "qwen2", "qwen3"):
            _run_model(model_kind, device, mesh_context, distributed_config)
            dist.barrier()
        if args.tp_size == 1:
            for model_kind in ("qwen2", "qwen3"):
                _run_packed_sliding_attention(model_kind, device)
                dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
