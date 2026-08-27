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

"""FSDP2 sharding for ``diffusion_gemma`` under pure FSDP (``ep_size=1``).

At ``ep_size=1`` there is no MoE mesh, so the model is sharded by the generic
:class:`~nemo_automodel.components.distributed.parallelizer.DefaultParallelizationStrategy`
(via ``FSDP2Manager.parallelize``), which applies ``fully_shard`` per decoder
layer and to the root.  The generic ``fully_shard`` flattens *all* of a decoder
layer's parameters into one FSDP unit, which folds each layer's grouped-expert
tensors (``moe.experts.{gate_and_up_projs,down_projs}``, the bulk of the 26B
parameters) into the layer's single all-gather.  That gathers every expert of a
layer at once on each forward — a large activation-memory spike for a model that
runs the shared stack twice (causal encode + bidirectional decode, plus an
optional self-conditioning pass).

``fully_shard_diffusion_gemma`` mirrors ``deepseek_v4``'s
``fully_shard_deepseek_v4``: it makes ``moe.experts`` its **own** FSDP unit
(sharded dim-0 on the dp mesh) *before* wrapping the rest of the decoder layer.
Consequences:

* The grouped-expert parameters become global-``[n_experts]`` ``Shard(0)``
  DTensors on the dp mesh, so DCP sees the checkpoint's global expert shape and
  each rank reads only its shard (no ``[128] vs [16]`` size mismatch).
* During the experts' forward, FSDP all-gathers their parameters back to the
  full ``[n_experts, ...]`` tensor, so :class:`GroupedExperts` sees a plain
  (non-DTensor) tensor and runs with ``ep_size == 1`` — all experts local, no
  expert-parallel token shuffle.  This is **pure FSDP, not EP**.
* Experts gather/reshard independently of the rest of the layer, bounding peak
  memory across the double (encode + decode) pass.
* ``moe.experts`` becomes a distinct ``FSDPModule`` that
  ``MoEFSDPSyncMixin._iter_fsdp_modules`` discovers (``block.moe.experts``).

No expert parallelism is introduced; this is the ``ep_size=1`` path only.
"""

from __future__ import annotations

from torch import nn
from torch.distributed.fsdp import fully_shard


def _has_fsdp_state(module: nn.Module) -> bool:
    """Return True if ``module`` has already been wrapped by ``fully_shard``."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    except ImportError:
        return False
    return _get_module_fsdp_state(module) is not None


def _fully_shard_once(module: nn.Module, *, mesh, mp_policy, offload_policy, **fsdp_kwargs) -> nn.Module:
    """Apply ``fully_shard`` to ``module`` unless it is already an FSDP unit."""
    if module is None or _has_fsdp_state(module):
        return module
    return fully_shard(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        **fsdp_kwargs,
    )


def fully_shard_diffusion_gemma(module: nn.Module, mesh, mp_policy, offload_policy=None, **fsdp_kwargs) -> nn.Module:
    """Apply FSDP2 to a ``diffusion_gemma`` decoder layer (or any other module).

    For a :class:`DiffusionGemmaMoEDecoderLayer`, shard its grouped experts
    (``moe.experts``) as a separate FSDP unit first, then shard the rest of the
    layer.  All other modules (embeddings, final norm, self-conditioning, the
    root model) are sharded as a single unit.

    Args:
        module: The module to shard (a decoder layer or the root model).
        mesh: The (1-D) data-parallel device mesh to shard across.
        mp_policy: FSDP2 mixed-precision policy.
        offload_policy: Optional FSDP2 CPU-offload policy.
        **fsdp_kwargs: Forwarded to ``fully_shard`` (e.g. ``reshard_after_forward``).

    Returns:
        The sharded module.
    """
    experts = getattr(getattr(module, "moe", None), "experts", None)
    if experts is not None:
        # Shard the grouped experts on their own so they gather/reshard
        # independently and DCP sees their global expert dimension.  The expert
        # parameters are then excluded from the parent layer's FSDP unit per
        # PyTorch FSDP2's nested-wrapping rules.
        _fully_shard_once(
            experts,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            **fsdp_kwargs,
        )

    return _fully_shard_once(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        **fsdp_kwargs,
    )


def register_diffusion_gemma_parallel_strategy() -> None:
    """Register the ``diffusion_gemma`` FSDP2 strategy (idempotent).

    Binds :func:`fully_shard_diffusion_gemma` as the per-module shard function
    of a :class:`DefaultParallelizationStrategy` subclass, keyed on the model
    class name so ``get_parallelization_strategy`` selects it at ``ep_size=1``.
    Invoked at import of ``model.py`` (a torch-enabled context), which always
    runs before the model is parallelized.
    """
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
        register_parallel_strategy,
    )

    name = "DiffusionGemmaForBlockDiffusion"
    if name in PARALLELIZATION_STRATEGIES:
        return

    @register_parallel_strategy(name=name)
    class DiffusionGemmaParallelizationStrategy(DefaultParallelizationStrategy):
        """Pure-FSDP2 strategy that shards grouped experts as their own units."""

        def parallelize(self, model, device_mesh, dp_shard_cp_mesh_name="dp_shard_cp", **kwargs):
            return super().parallelize(
                model,
                device_mesh,
                dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
                fully_shard_fn=fully_shard_diffusion_gemma,
                **kwargs,
            )
