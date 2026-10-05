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

"""FSDP2 sharding for ``diffusion_gemma`` under pure FSDP (``ep_size=1``).

At ``ep_size=1`` there is no MoE mesh, so the model goes through the generic
:class:`~nemo_automodel.components.distributed.parallelizer.ModelParallelizer`
flow: one ``fully_shard`` unit per decoder layer plus the root. That would fold
each layer's grouped-expert tensors (``moe.experts.{gate_and_up_projs,down_projs}``,
the bulk of the 26B parameters) into the layer's single all-gather, gathering
every expert of a layer at once on each forward -- a large activation-memory
spike for a model that runs the shared stack twice (causal encode + bidirectional
decode, plus an optional self-conditioning pass).

:class:`DiffusionGemmaModelParallelizer` therefore shards ``moe.experts`` as its
**own** FSDP unit (``Shard(0)`` on the dp mesh) before the rest of each decoder
layer. Consequences:

* The grouped-expert parameters become global-``[n_experts]`` ``Shard(0)``
  DTensors on the dp mesh, so DCP sees the checkpoint's global expert shape and
  each rank reads only its shard (no ``[128] vs [16]`` size mismatch).
* During the experts' forward, FSDP all-gathers their parameters back to the
  full ``[n_experts, ...]`` tensor, so :class:`GroupedExperts` sees a plain
  (non-DTensor) tensor and runs with ``ep_size == 1`` -- all experts local, no
  expert-parallel token shuffle.  This is **pure FSDP, not EP**.
* Experts gather/reshard independently of the rest of the layer, bounding peak
  memory across the double (encode + decode) pass.
* ``moe.experts`` becomes a distinct ``FSDPModule`` that
  ``MoEFSDPSyncMixin._iter_fsdp_modules`` discovers (``block.moe.experts``).

No expert parallelism is introduced; this is the ``ep_size=1`` path only.
"""

from __future__ import annotations

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.diffusion_gemma.layers import DiffusionGemmaMoEDecoderLayer


class DiffusionGemmaModelParallelizer(ModelParallelizer):
    """Pure-FSDP2 strategy that shards grouped experts as their own units."""

    def _fully_shard_module(self, module: nn.Module, **kwargs) -> nn.Module:
        """Shard a decoder layer's ``moe.experts`` as one unit before the layer itself.

        Every other module (embeddings, final norm, self-conditioning, the root
        model) is a single unit. Both units go through the base primitive, so
        they keep the model's fp32 compute contract and ``fully_shard`` keywords.
        """
        layer = module._checkpoint_wrapped_module if isinstance(module, CheckpointWrapper) else module
        if isinstance(layer, DiffusionGemmaMoEDecoderLayer):
            super()._fully_shard_module(layer.moe.experts, **kwargs)
        return super()._fully_shard_module(module, **kwargs)


PARALLELIZER = DiffusionGemmaModelParallelizer()

__all__ = ["DiffusionGemmaModelParallelizer", "PARALLELIZER"]
