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

"""Parallelization contract for the transformers ``FalconH1ForCausalLM``."""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate

from nemo_automodel.components.distributed.optimized_tp_plans import VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec


def falcon_h1_tp_plan(
    model: nn.Module | None,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelize Falcon-H1 (hybrid Transformer + Mamba2 SSM).

    Every Falcon-H1 decoder layer runs an attention branch (``self_attn``) and a
    Mamba2 branch (``mamba``) in parallel, followed by an MLP (``feed_forward``).
    Only the attention and MLP linears are tensor-parallel sharded; the Mamba2
    mixer stays replicated because its SSM scan / causal conv1d are not
    TP-shardable with stock kernels (same approach as Qwen3.5's GatedDeltaNet
    linear-attention branch).

    A dedicated plan is required because HuggingFace ships only
    ``_tp_plan = {"lm_head": "colwise_gather_output"}`` for FalconH1, and its MLP
    is named ``feed_forward`` (not ``mlp``). The generic llama-style fallback plan
    therefore matches neither the HF plan (the ``colwise_gather_output`` style is
    rejected) nor the MLP module names, leaving the dominant ``feed_forward``
    weights replicated across TP ranks — which OOMs large variants such as
    Falcon-H1-34B even under LoRA.

    ``sequence_parallel`` is accepted for signature compatibility but ignored: the
    parallel Mamba2 branch emits non-sequence-parallel activations that cannot be
    combined with sequence-parallel attention outputs.
    """
    return cast(
        dict[str, ParallelStyle],
        {
            "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.feed_forward.gate_proj": ColwiseParallel(),
            "model.layers.*.feed_forward.up_proj": ColwiseParallel(),
            "model.layers.*.feed_forward.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        },
    )


class FalconH1ForCausalLM:
    """Contract for the transformers ``FalconH1ForCausalLM``; HF ships only a minimal ``_tp_plan`` for it."""

    parallel_spec: ParallelSpec = ParallelSpec(tp_plan=falcon_h1_tp_plan)
