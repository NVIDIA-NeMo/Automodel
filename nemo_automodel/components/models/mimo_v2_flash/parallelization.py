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

"""MiMo-specific distributed-parallelization registration."""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import safe_import_from


def _unwrap_checkpoint_module(module: torch.nn.Module) -> torch.nn.Module:
    """Return the attention module beneath any activation-checkpoint wrappers."""
    while hasattr(module, "_checkpoint_wrapped_module"):
        module = module._checkpoint_wrapped_module
    return module


def setup_mimo_te_context_parallel(model: torch.nn.Module, cp_mesh) -> int:
    """Configure every MiMo text attention layer for TE all-to-all CP.

    MiMo uses grouped-query attention, so TE's all-to-all transport must be able
    to split both query heads and KV groups evenly across the CP ranks.  Refuse
    any non-TE attention module instead of silently falling back to the older
    model-owned K/V gather path.

    Args:
        model: A complete MiMo model or one pipeline-local model part.
        cp_mesh: One-dimensional context-parallel device mesh.

    Returns:
        Number of local decoder attention modules configured.
    """
    cp_size = cp_mesh.size() if cp_mesh is not None else 1
    model.cp_mesh = cp_mesh if cp_size > 1 else None
    if cp_size <= 1:
        return 0

    has_te, dot_product_attention_cls = safe_import_from(
        "transformer_engine.pytorch.attention",
        "DotProductAttention",
    )
    if not has_te:
        raise ImportError("MiMo context parallelism requires Transformer Engine DotProductAttention.")

    cp_group = cp_mesh.get_group()
    cp_global_ranks = torch.distributed.get_process_group_ranks(cp_group)
    cp_stream = torch.cuda.Stream()
    configured = 0
    for name, wrapped_attention in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        self_attn = _unwrap_checkpoint_module(wrapped_attention)
        attn_module = getattr(self_attn, "attn_module", None)
        if not isinstance(attn_module, dot_product_attention_cls):
            backend_name = type(attn_module).__name__ if attn_module is not None else type(self_attn).__name__
            raise ValueError(
                "MiMo context parallelism only supports Transformer Engine DotProductAttention; "
                f"{name} uses {backend_name}. Set model.backend.attn='te'."
            )

        query_heads = int(getattr(self_attn, "num_attention_heads", attn_module.num_attention_heads))
        kv_heads = int(getattr(self_attn, "num_key_value_heads", attn_module.num_gqa_groups))
        if query_heads % cp_size != 0 or kv_heads % cp_size != 0:
            raise ValueError(
                "MiMo TE all-to-all context parallelism requires query and KV head counts "
                f"divisible by cp_size; {name} has query_heads={query_heads}, "
                f"kv_heads={kv_heads}, cp_size={cp_size}."
            )

        attn_module.set_context_parallel_group(
            cp_group,
            cp_global_ranks,
            cp_stream,
            cp_comm_type="a2a",
        )
        configured += 1

    return configured


def register_mimo_v2_parallel_strategies() -> None:
    """Register the MiMo FSDP2 strategy once for both checkpoint class names."""
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
        register_parallel_strategy,
    )

    for name in ("MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM"):
        if name in PARALLELIZATION_STRATEGIES:
            continue

        @register_parallel_strategy(name=name)
        class MiMoV2ParallelizationStrategy(DefaultParallelizationStrategy):
            """Apply dense sharding and enforce MiMo's TE all-to-all CP contract."""

            def parallelize(self, model, device_mesh, **kwargs):
                result = super().parallelize(model, device_mesh, **kwargs)
                cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
                cp_mesh = cp_mesh if cp_mesh is not None and cp_mesh.size() > 1 else None
                if cp_mesh is not None:
                    setup_mimo_te_context_parallel(model, cp_mesh)
                else:
                    model.cp_mesh = None
                return result
