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

"""NemotronH (hybrid Mamba2 / attention) parallelization contract.

Tensor parallelism is the plan below, per-block FSDP2 is the shared dtype-aware sharding
(``ParallelSpec.shard_by_dtype`` with the model's ``_keep_in_fp32_modules_strict``) and activation
checkpointing is the shared whole-layer wrapper (``ActivationCheckpointingSpec(granularity="layer")``).
The strategy adds only what no generic API expresses: the Mamba / Transformer Engine context-parallel
wiring and the block kinds activation checkpointing covers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torch.distributed.tensor.placement_types import Shard

import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.activation_checkpointing import ActivationCheckpointingSpec
from nemo_automodel.components.distributed.config import ActivationCheckpointingScope
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy, get_model_layer_groups

logger = logging.getLogger(__name__)

# Only the MLP blocks of the hybrid stack are tensor-parallel; Mamba and attention mixers stay replicated and
# the head keeps a vocab-sharded output. The HF model keeps its blocks at ``backbone.layers`` and the native port
# at ``model.layers``; entries that do not resolve on a given tree are ignored by ``parallelize_module``.
NEMOTRON_H_TP_PLAN: dict[str, ParallelStyle] = {
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    **{
        f"{prefix}.layers.*.mixer.{name}": style()
        for prefix in ("backbone", "model")
        for name, style in (("up_proj", ColwiseParallel), ("down_proj", RowwiseParallel))
    },
}

# Block kinds whose whole layer is activation-checkpointed; attention blocks are left unwrapped.
NEMOTRON_H_CHECKPOINTED_BLOCK_TYPES = ("mlp", "mamba")


def _setup_context_parallel(model: nn.Module, device_mesh: DeviceMesh) -> None:
    """Install the Mamba context-parallel wrapper and the TE attention CP group on every decoder block."""
    cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
    if cp_mesh is None or cp_mesh.size() <= 1:
        return
    cp_group = cp_mesh.get_group()
    cp_global_ranks = torch.distributed.get_process_group_ranks(cp_group)
    cp_layers = list(get_model_layer_groups(model)["language"])
    mtp_module = getattr(model, "mtp", None)
    mtp_layers = getattr(mtp_module, "layers", None)
    mtp_cp_enabled = model.supports.mtp_enabled
    parallelizer_utils.reject_unsupported_mtp_cp_pp(model)
    parallelizer_utils.reject_unsupported_mtp_cp(model)
    if mtp_cp_enabled and mtp_layers is None:
        raise RuntimeError(
            "MTP is enabled but model.mtp.layers is unavailable; cannot configure context parallelism for MTP"
        )
    if mtp_cp_enabled and mtp_layers is not None:
        # MTP blocks live outside the backbone container but execute
        # the same attention/Mamba CP collectives.
        cp_layers.extend(mtp_layers)

    for layer in cp_layers:
        if hasattr(layer, "block_type") and layer.block_type == "mamba":
            from nemo_automodel.components.distributed.context_parallel.mamba import MambaContextParallel

            mixer = layer.mixer
            mixer.cp = MambaContextParallel(
                cp_group=cp_group,
                num_heads=mixer.num_heads,
                head_dim=mixer.head_dim,
                n_groups=mixer.n_groups,
                d_state=mixer.ssm_state_size,
                mixer=mixer,
            )
        elif hasattr(layer, "block_type") and layer.block_type == "attention":
            from transformer_engine.pytorch.attention import DotProductAttention

            attn_module = layer.mixer.attn_module
            if isinstance(attn_module, DotProductAttention):
                attn_module.set_context_parallel_group(
                    cp_group,
                    cp_global_ranks,
                    torch.cuda.Stream(),
                    cp_comm_type="p2p",
                )


class NemotronHParallelizationStrategy(DefaultParallelizationStrategy):
    """The default flow with NemotronH's context-parallel wiring and its activation-checkpointing block rule."""

    def parallelize(
        self, model: nn.Module, device_mesh: DeviceMesh, sequence_parallel: bool = False, **kwargs
    ) -> nn.Module:
        """Wire context parallelism into the Mamba and attention mixers, then run the default flow."""
        assert not sequence_parallel, "Sequence parallelism is not supported for NemotronHForCausalLM"
        _setup_context_parallel(model, device_mesh)
        return super().parallelize(model, device_mesh, sequence_parallel=sequence_parallel, **kwargs)

    def select_activation_checkpointing_layers(
        self,
        model: nn.Module,
        layer_groups: Dict[str, List[nn.Module]],
        activation_checkpointing_scope: ActivationCheckpointingScope | None,
    ) -> Tuple[List[nn.Module], Tuple[str, ...]]:
        """The scope's layers restricted to the MLP and Mamba blocks."""
        layers, scopes = super().select_activation_checkpointing_layers(
            model, layer_groups, activation_checkpointing_scope
        )
        return [
            layer for layer in layers if getattr(layer, "block_type", None) in NEMOTRON_H_CHECKPOINTED_BLOCK_TYPES
        ], scopes


# HF NemotronH keeps decoder blocks under ``backbone.layers``; the native port under ``model.layers``.
NEMOTRON_H_PARALLEL_SPEC = ParallelSpec(
    tp_plan=NEMOTRON_H_TP_PLAN,
    layer_groups={"language": ("backbone.layers", "model.layers")},
    shard_by_dtype=True,
    strategy=NemotronHParallelizationStrategy(),
)
NEMOTRON_H_ACTIVATION_CHECKPOINTING_SPEC = ActivationCheckpointingSpec(granularity="layer")
