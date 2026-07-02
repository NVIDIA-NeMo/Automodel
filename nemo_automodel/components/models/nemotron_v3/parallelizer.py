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

"""Nemotron-H-specific distributed parallelization."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor.parallel import ParallelStyle

import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed import parallelizer

logger = logging.getLogger(__name__)


def _nemotronh_decoder_blocks(model: nn.Module) -> tuple[nn.Module, list[nn.Module]]:
    """Return the Nemotron-H decoder-layer container and its ordered blocks.

    Two distinct classes share the name ``NemotronHForCausalLM``. The HF model
    keeps blocks in ``model.backbone.layers`` while the native Nemotron-V3
    model keeps them in ``model.model.layers``.
    """
    inner = model.backbone if hasattr(model, "backbone") else model.model
    container = inner.layers
    blocks = list(container.values()) if isinstance(container, nn.ModuleDict) else list(container)
    return container, blocks


class NemotronHParallelizationStrategy(parallelizer.ParallelizationStrategy):
    """Specialized parallelization strategy for Nemotron-H models."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: dict[str, ParallelStyle] | str | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reshard_after_forward: bool | None = None,
        **kwargs: object,
    ) -> nn.Module:
        """Apply Nemotron-H-specific parallelization."""
        del tp_shard_plan, kwargs
        assert not sequence_parallel, "Sequence parallelism is not supported for NemotronHForCausalLM"
        logger.info("Custom parallel plan is not supported for NemotronHForCausalLM. Using NemotronH-specific TP plan.")

        block_container, layers = _nemotronh_decoder_blocks(model)
        tp_mesh = device_mesh[tp_mesh_name]
        if tp_mesh.size() > 1:
            model_tp_plan: dict[str, ParallelStyle] = {
                "lm_head": parallelizer.translate_to_lora(
                    parallelizer.ColwiseParallel(output_layouts=parallelizer.Shard(-1), use_local_output=False)
                ),
            }

            mlp_tp_plan: dict[str, ParallelStyle] = {
                "mixer.up_proj": parallelizer.translate_to_lora(parallelizer.ColwiseParallel()),
                "mixer.down_proj": parallelizer.translate_to_lora(parallelizer.RowwiseParallel()),
            }

            parallelizer.parallelize_module(model, tp_mesh, model_tp_plan)

            for layer in layers:
                if layer.block_type == "mlp":
                    parallelizer.parallelize_module(layer, tp_mesh, mlp_tp_plan)

        cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
        if cp_mesh is not None and cp_mesh.size() > 1:
            cp_group = cp_mesh.get_group()

            for layer in layers:
                if hasattr(layer, "block_type") and layer.block_type == "mamba":
                    from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel

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
                            torch.distributed.get_process_group_ranks(cp_group),
                            torch.cuda.Stream(),
                            cp_comm_type="p2p",
                        )

        if activation_checkpointing:
            block_items = (
                block_container.items() if isinstance(block_container, nn.ModuleDict) else enumerate(block_container)
            )
            for key, layer in list(block_items):
                if getattr(layer, "block_type", None) in ("mlp", "mamba"):
                    block_container[key] = parallelizer.checkpoint_wrapper(layer)
            _, layers = _nemotronh_decoder_blocks(model)

        dp_mesh = parallelizer.get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)
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

        return parallelizer.fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


_STRATEGY = NemotronHParallelizationStrategy()


def get_parallelization_strategy() -> parallelizer.ParallelizationStrategy:
    """Return the lazily requested Nemotron-H strategy instance."""
    return _STRATEGY
