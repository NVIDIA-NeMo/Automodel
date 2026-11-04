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

"""Flint-owned FSDP precision boundaries for learned sinks and the output head."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from transformers.modeling_utils import PreTrainedModel

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.models.flint.layers import FlintDecoderLayer
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn


class FlintModelParallelizer(ModelParallelizer):
    """Preserve FP32 compute units within the existing generic FSDP2 flow."""

    _customizes_moe_fsdp = True

    def _apply(self, model: nn.Module, device_mesh: DeviceMesh, **kwargs: Any) -> nn.Module:
        """Apply inherited FSDP policies and retain this stage's CP token mesh.

        Args:
            model: Native full model or pruned PP stage; parameter tensor layouts
                remain [out,in] projections and grouped expert matrices.
            device_mesh: Resolved shared mesh, including its named CP axis.
            **kwargs: Existing shared FSDP/AC/compile policies, passed unchanged.

        Returns:
            Parallelized native module with its optional CP mesh. This mesh owns
            token indices for global [B,S] IDs and local [B,S/cp,H] residuals.
        """
        cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
        cp_size = cp_mesh.size() if cp_mesh is not None else 1
        for block in model.modules():
            if isinstance(block, FlintDecoderLayer):
                block.self_attn._te_cp_size = cp_size
                if block.self_attn.attn_module is not None:
                    block.self_attn.attn_module._flint_te_cp_size = cp_size
        result = super()._apply(model, device_mesh, **kwargs)
        model.cp_mesh = cp_mesh if cp_mesh is not None and cp_mesh.size() > 1 else None
        return result

    def _fully_shard_module(
        self,
        module: nn.Module,
        *,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        reshard_after_forward: bool | int | None = None,
        ignored_params: set[nn.Parameter] | None = None,
    ) -> nn.Module:
        """Apply the shared primitive with Flint's two protected dtype boundaries.

        Args:
            module: Decoder block, optional activation-checkpoint wrapper, input
                embedding, output head, or outer model. Block ordinary parameters
                use native [out,in] matrices and fused expert [experts,hidden,2*intermediate]
                / [experts,intermediate,hidden] layouts. Sink holders own [heads]
                parameters; the head owns [vocab,hidden].
            mesh: FSDP data-parallel mesh. Parameters become DTensors sharded by
                the shared FSDP primitive; callers must enter on all mesh ranks.
            mp_policy: Bulk compute/reduction policy; FP32 master storage remains
                unchanged. Sink holders and the independently wrapped head use FP32.
            offload_policy: Existing FSDP storage-offload policy.
            reshard_after_forward: Existing FSDP reshard policy, or its default.
            ignored_params: Runtime parameters owned by another FSDP unit, if any.
                They may have arbitrary global/native shapes or rank-local
                shards; their existing DTensor mesh/placements are preserved.

        Returns:
            The same module, mutated in place into FSDP units. Existing shapes
            and parameter ownership are preserved; protected child units use
            independent FP32 compute boundaries without changing bulk precision.
        """
        inner = unwrap_checkpoint_wrapper(module)
        if getattr(inner, "_flint_keep_sink_gathered", False):
            # The MoE dtype helper can visit this protected child before the
            # decoder block. Enforce the saved-offset lifetime on either route.
            result = fully_shard(
                module,
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                    output_dtype=None,
                    cast_forward_inputs=False,
                ),
                offload_policy=offload_policy,
                reshard_after_forward=False,
                ignored_params=ignored_params,
            )
            cp_group = getattr(inner, "cp_group", None)
            cp_size = cp_group.size() if cp_group is not None else inner._flint_te_cp_size
            if cp_size > 1:
                # TE A2A already gathers the complete sink adjoint on every CP
                # rank; remove the recipe's extra disjoint-token loss factor.
                result.set_gradient_divide_factor(float(mesh.size() * cp_size))
            return result
        if isinstance(inner, PreTrainedModel) and inner.config_class is FlintConfig:
            if inner._is_pipeline_parallel_stage():
                # The inherited PP builder has already copied and pruned this
                # stage before invoking shared FSDP. Record the exact destination
                # here, without creating a second parallelization entry point.
                inner.state_dict_adapter._pipeline_native_keys = frozenset(
                    canonical_parameter_fqn(name) for name in inner.state_dict()
                )
        if isinstance(inner, FlintDecoderLayer):
            if inner.residual_dtype == "fp32" and mp_policy is not None:
                mp_policy = replace(mp_policy, output_dtype=None)
            attn_module = inner.self_attn.attn_module
            if attn_module is not None and not isinstance(attn_module, FSDPModule):
                # TE's learned-sink backward retains the offset storage.
                # Keep this tiny FP32 unit gathered through backward while
                # the enclosing decoder retains its normal reshard policy.
                self._fully_shard_module(
                    attn_module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=False,
                )
            sink_module = inner.self_attn._fp32_params
            if isinstance(sink_module, FSDPModule):
                # The dtype helper may already own the FP32 sink unit for any
                # attention backend; keep it out of this block's dtype grouping.
                ignored_params = set(ignored_params or ()) | set(sink_module.parameters())
            # Do not pass this sidecar back into the helper: its primitive must
            # call bare fully_shard rather than recurse into this override.
            fully_shard_by_dtype(
                module,
                mesh,
                mp_policy,
                offload_policy,
                fp32_compute_module_names=("_fp32_params",),
                reshard_after_forward=reshard_after_forward,
                ignored_params=ignored_params,
            )
            return module
        if isinstance(inner, nn.Linear) and getattr(inner, "_flint_fp32_output_head", False):
            # Only the model-owned vocabulary head has this strict boundary.
            # Mixed-storage LoRA can also make projection/adapter linears
            # standalone FSDP units; they retain the caller's shared policy.
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
                cast_forward_inputs=False,
            )
        elif isinstance(inner, PreTrainedModel) and inner.config_class is FlintConfig and mp_policy is not None:
            head = inner.get_output_embeddings()
            if head is not None and not isinstance(head, FSDPModule):
                # The shared embedding helper skips a fully frozen head. It
                # still computes in FP32, so it needs its own FSDP unit before
                # the root can claim it under the bulk BF16 policy.
                fully_shard(
                    head,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.float32,
                        reduce_dtype=torch.float32,
                        output_dtype=torch.float32,
                        cast_forward_inputs=False,
                    ),
                    offload_policy=offload_policy,
                    reshard_after_forward=False,
                )
            # The outer root must preserve the head's FP32 logits in the
            # structured model output, even if bulk block outputs use BF16.
            # Its own bulk parameter/reduction precision is left unchanged.
            mp_policy = replace(mp_policy, output_dtype=None)
        elif getattr(inner, "_flint_preserve_output_dtype", False) and mp_policy is not None:
            # The MoE path also wraps the inner backbone. Preserve its natural
            # FP32 residual/final-normalization output before the outer head or
            # PP boundary, leaving its bulk parameter/reduction policy intact.
            mp_policy = replace(mp_policy, output_dtype=None)
        return fully_shard(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=True if reshard_after_forward is None else reshard_after_forward,
            ignored_params=ignored_params,
        )
