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

"""Two-GPU packed-THD CP parity for two-layer dense Llama and Qwen2 models.

Run with::

    torchrun --nproc-per-node=2 tests/functional_tests/context_parallel/run_dense_packed_cp.py
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformer_engine.pytorch import DotProductAttention
from transformers import LlamaConfig, Qwen2Config

from nemo_automodel.components.distributed.cp_utils import attach_te_context_parallel, make_cp_batch_for_te
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.llama.model import LlamaForCausalLM
from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM

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


def _build_model(model_kind: str, device: torch.device) -> torch.nn.Module:
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
    else:
        config = Qwen2Config(**common)
        model_cls = Qwen2ForCausalLM
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


def _run_model(model_kind: str, device: torch.device, cp_mesh) -> None:
    import transformer_engine_torch as tex

    rank = dist.get_rank()
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
    baseline_logits.float().square().sum().backward()
    baseline_grads = [layer.self_attn.q_proj.weight.grad.detach().float() for layer in baseline_model.model.layers]
    assert baseline_labels.shape == (8,)

    cp_model = _build_model(model_kind, device)
    configured = attach_te_context_parallel(cp_model, cp_mesh)
    assert configured == NUM_HIDDEN_LAYERS
    cp_batch = make_cp_batch_for_te(cp_mesh, _clone_batch(batch))
    cp_batch.pop("labels")
    local_logits = cp_model(**cp_batch).logits.squeeze(0)
    local_logits.float().square().sum().backward()

    cu_seqlens = cp_batch["cu_seqlens"]
    indices = tex.thd_get_partitioned_indices(cu_seqlens, 8, cp_mesh.size(), rank).to(torch.int32)
    cp_logits = _reconstruct_global_tokens(
        local_logits,
        indices,
        total_tokens=8,
        group=cp_group,
    )
    cp_grads = [layer.self_attn.q_proj.weight.grad.detach().float() for layer in cp_model.model.layers]
    for cp_grad in cp_grads:
        dist.all_reduce(cp_grad, group=cp_group)

    torch.testing.assert_close(cp_logits, baseline_logits, atol=3e-2, rtol=3e-2)
    for cp_grad, baseline_grad in zip(cp_grads, baseline_grads):
        torch.testing.assert_close(cp_grad, baseline_grad, atol=5e-2, rtol=5e-2)
    assert torch.isfinite(cp_logits).all()
    assert all(torch.isfinite(cp_grad).all() for cp_grad in cp_grads)
    if rank == 0:
        output_diff = (cp_logits.float() - baseline_logits.float()).abs().max().item()
        grad_diff = max(
            (cp_grad - baseline_grad).abs().max().item() for cp_grad, baseline_grad in zip(cp_grads, baseline_grads)
        )
        print(f"{model_kind}: packed CP parity passed (logits max={output_diff:.6f}, grad max={grad_diff:.6f})")


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cp_mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))["cp"]
    try:
        for model_kind in ("llama", "qwen2"):
            _run_model(model_kind, device, cp_mesh)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
