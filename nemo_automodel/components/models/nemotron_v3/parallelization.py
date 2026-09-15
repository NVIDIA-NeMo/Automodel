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

"""NemotronH (hybrid Mamba2 / attention) parallelization strategy."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Dict, Union

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, parallelize_module
from torch.distributed.tensor.placement_types import Shard

import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallel_styles import translate_to_lora
from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy

logger = logging.getLogger(__name__)


def _nemotronh_decoder_blocks(model: nn.Module) -> tuple[nn.Module, list[nn.Module]]:
    """Return ``(container, blocks)`` for a NemotronH model's decoder blocks.

    Two distinct classes share the name ``NemotronHForCausalLM``:

    * the HF model keeps its blocks in ``model.backbone.layers`` (an ``nn.ModuleList``), while
    * the native Nemotron-V3 model (``NemotronV3Model``) keeps them in ``model.model.layers``
      (an ``nn.ModuleDict`` keyed ``"0".."N-1"``).

    ``container`` is the underlying ``ModuleList``/``ModuleDict`` (so callers can write rewrapped
    blocks back into the model), and ``blocks`` is the ordered list of block modules.
    """
    inner = model.backbone if hasattr(model, "backbone") else model.model
    container = inner.layers
    blocks = list(container.values()) if isinstance(container, nn.ModuleDict) else list(container)
    return container, blocks


class NemotronHParallelizationStrategy(ParallelizationStrategy):
    """Specialized parallelization strategy for NemotronH models."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reshard_after_forward: bool | None = None,
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        """Apply NemotronH-specific parallelization."""
        assert not sequence_parallel, "Sequence parallelism is not supported for NemotronHForCausalLM"
        logger.info("Custom parallel plan is not supported for NemotronHForCausalLM. Using NemotronH-specific TP plan.")

        block_container, layers = _nemotronh_decoder_blocks(model)
        tp_mesh = device_mesh[tp_mesh_name]
        if tp_mesh.size() > 1:
            model_tp_plan: dict[str, ParallelStyle] = {
                "lm_head": translate_to_lora(ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)),
            }

            mlp_tp_plan: dict[str, ParallelStyle] = {
                "mixer.up_proj": translate_to_lora(ColwiseParallel()),
                "mixer.down_proj": translate_to_lora(RowwiseParallel()),
            }

            parallelize_module(model, tp_mesh, model_tp_plan)

            for layer in layers:
                if layer.block_type == "mlp":
                    parallelize_module(layer, tp_mesh, mlp_tp_plan)

        # Set up context parallel for Mamba and Attention layers
        cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
        if cp_mesh is not None and cp_mesh.size() > 1:
            cp_group = cp_mesh.get_group()
            cp_global_ranks = torch.distributed.get_process_group_ranks(cp_group)
            cp_layers = list(layers)
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

        if activation_checkpointing:
            # Write rewrapped blocks back into the real container (ModuleList -> int key,
            # ModuleDict -> str key) so the model, not just the local handle, is updated.
            block_items = (
                block_container.items() if isinstance(block_container, nn.ModuleDict) else enumerate(block_container)
            )
            for key, layer in list(block_items):
                if getattr(layer, "block_type", None) in ("mlp", "mamba"):
                    block_container[key] = checkpoint_wrapper(layer)
            # Refresh the local handle so the FSDP wrap below sees the wrapped blocks.
            _, layers = _nemotronh_decoder_blocks(model)

        if reapply_trainability is not None:
            reapply_trainability(model)

        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        fp32_compute_module_names = tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ())

        for layer in layers:
            parallelizer_utils.fully_shard_by_dtype(
                layer,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                fp32_compute_module_names=fp32_compute_module_names,
                reshard_after_forward=reshard_after_forward,
            )

        # do not reshard after forward for root model
        # because its parameters will be used in backward immediately
        return fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


# HF NemotronH keeps decoder blocks under ``backbone.layers``; the native port under ``model.layers``.
NEMOTRON_H_PARALLEL_SPEC = ParallelSpec(
    layer_groups={"language": ("backbone.layers", "model.layers")},
    strategy=NemotronHParallelizationStrategy(),
)
