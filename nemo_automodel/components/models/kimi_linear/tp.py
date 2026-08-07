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

"""Tensor- and sequence-parallel plan for Kimi Linear.

Kimi Linear interleaves KDA linear-attention layers with MLA full-attention
layers, and the two are treated differently:

* **MLA** follows the usual column/row pattern. ``q_proj`` and ``kv_b_proj``
  expand to per-head features and are column-sharded; ``o_proj`` contracts them
  and is row-sharded so its partial sums are reduced. ``kv_a_proj_with_mqa`` and
  ``kv_a_layernorm`` produce the head-shared compressed latent and stay
  replicated.
* **KDA stays replicated**, matching how Qwen3.5's GatedDeltaNet and Falcon-H1's
  Mamba2 mixer are handled. Its heads are arithmetically independent, but the
  layer runs FLA Triton kernels (short convolution, chunked delta rule) that take
  plain tensors, so head-sharding it would mean localizing every weight inside
  the forward rather than expressing it as a parallel style. That is a separate
  change.

Because the two flavours share module names (both define ``q_proj`` and
``o_proj``), the plan is emitted per layer from the model's own KDA /
full-attention split rather than through a wildcard.

Sequence parallelism shards only the normalization and residual path: each block
all-gathers the sequence before attention and reduce-scatters it back at the
output projection, so the KDA recurrence still sees whole documents. Routed
experts stay owned by expert parallelism, so the replicated output of a MoE (or
KDA) submodule is scattered back onto the sequence-parallel residual stream
explicitly.
"""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)

from nemo_automodel.components.distributed.optimized_tp_plans import (
    SequenceParallelAllGatherActivation,
    VocabParallelEmbedding,
)


def parallelize_kimi_linear(model: nn.Module, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Build the tensor-parallel plan for :class:`KimiLinearForCausalLM`.

    Args:
        model: The model being parallelized; its layer structure decides which
            blocks receive MLA sharding and which keep a dense MLP.
        sequence_parallel: Whether to additionally shard block boundaries along
            the sequence.

    Returns:
        Mapping of module paths to parallel styles.
    """
    plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Shard(1) if sequence_parallel else Replicate(),
        ),
        "lm_head": ColwiseParallel(
            input_layouts=Shard(1) if sequence_parallel else Replicate(),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
    }

    for layer_id, block in model.model.layers.items():
        prefix = f"model.layers.{layer_id}"
        if not block.is_linear_attn:
            plan[f"{prefix}.self_attn.q_proj"] = ColwiseParallel()
            plan[f"{prefix}.self_attn.kv_b_proj"] = ColwiseParallel()
            plan[f"{prefix}.self_attn.o_proj"] = RowwiseParallel(
                output_layouts=Shard(1) if sequence_parallel else Replicate()
            )
        if not block.is_moe_layer:
            # Dense blocks only: the MoE branch shares the ``mlp`` name but its
            # routed experts stay owned by expert parallelism.
            plan[f"{prefix}.mlp.gate_proj"] = ColwiseParallel()
            plan[f"{prefix}.mlp.up_proj"] = ColwiseParallel()
            plan[f"{prefix}.mlp.down_proj"] = RowwiseParallel(
                output_layouts=Shard(1) if sequence_parallel else Replicate()
            )

        if not sequence_parallel:
            continue

        # ``use_local_output=True`` everywhere on this boundary: the attention
        # layers run FLA Triton kernels and replicated helper projections, which
        # only accept plain tensors.
        plan[f"{prefix}.input_layernorm"] = SequenceParallelAllGatherActivation(use_local_output=True)
        plan[f"{prefix}.post_attention_layernorm"] = SequenceParallelAllGatherActivation(use_local_output=True)
        if block.is_linear_attn:
            # KDA is replicated, so it emits a full-sequence tensor that has to be
            # scattered back onto the sequence-parallel residual stream.
            plan[f"{prefix}.self_attn"] = PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
                use_local_output=True,
            )
        if block.is_moe_layer:
            # The MoE branch of the block is also named ``mlp``.
            plan[f"{prefix}.mlp"] = PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
                use_local_output=True,
            )

    if sequence_parallel:
        plan["model.norm"] = SequenceParallel(use_local_output=True)

    return plan
