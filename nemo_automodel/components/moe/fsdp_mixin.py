# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.distributed.fsdp import FSDPModule, fully_shard

from nemo_automodel.components.moe.utils import FsdpOptimizationConfig


class MoEFSDPSyncMixin:
    """
    Mixin for managing FSDP synchronization state during MoE model training.

    Controls gradient sync, resharding, and backward hooks for FSDP-wrapped modules
    to optimize performance during gradient accumulation steps.

    Usage differs based on parallelism strategy:
    - Without pipeline parallelism (PP): set_fsdp_states_for_last_backward() is called only before
      the last microbatch/local batch backward to enable gradient sync and resharding. For earlier
      microbatches, FSDP's no_sync context handles deferring these operations. finalize_fsdp_states_post_backward()
      is not used since FSDP autograd hooks handle synchronization automatically.
    - With pipeline parallelism (PP): set_fsdp_states_for_first_forward() is called before the first
      forward, and finalize_fsdp_states_post_backward() is called once after all backwards complete
      to manually trigger FSDP post-backward hooks and gradient reduction synchronization.
    """

    @staticmethod
    def last_backward_for_fsdp_module(fsdp_module: FSDPModule) -> None:
        fsdp_module.set_is_last_backward(True)
        fsdp_module.set_reshard_after_backward(True)
        fsdp_module.set_requires_gradient_sync(True)

    @staticmethod
    def post_backward_for_fsdp_modules(fsdp_modules: list[FSDPModule]) -> None:
        for fsdp_module in fsdp_modules:
            fsdp_module.set_is_last_backward(True)
            fsdp_module.set_reshard_after_backward(True)
            fsdp_module.set_requires_gradient_sync(True)
            fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
            for state in fsdp_state._state_ctx.all_states:
                if state._fsdp_param_group:
                    state._fsdp_param_group.post_backward()

        for fsdp_module in fsdp_modules:
            fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
            fsdp_state._root_post_backward_final_callback()

    @staticmethod
    def first_forward_for_fsdp_module(
        fsdp_module: FSDPModule, requires_gradient_sync: bool = False, reshard_after_backward: bool = False
    ) -> None:
        fsdp_module.set_is_last_backward(False)
        fsdp_module.set_reshard_after_backward(reshard_after_backward)
        fsdp_module.set_requires_gradient_sync(requires_gradient_sync)

    def set_fsdp_states_for_last_backward(self) -> None:
        if self.backend.fsdp_optimization_config is None:
            return

        _model = self.model
        if isinstance(_model, FSDPModule):
            MoEFSDPSyncMixin.last_backward_for_fsdp_module(_model)

        if hasattr(self, "lm_head"):
            if isinstance(self.lm_head, FSDPModule):
                MoEFSDPSyncMixin.last_backward_for_fsdp_module(self.lm_head)

        if hasattr(_model, "layers"):
            for _, block in _model.layers.named_children():
                # Check if this block has MoE experts wrapped in FSDP
                if hasattr(block, "mlp") and hasattr(block.mlp, "experts"):
                    experts = block.mlp.experts
                    if isinstance(experts, FSDPModule):
                        MoEFSDPSyncMixin.last_backward_for_fsdp_module(experts)

    def finalize_fsdp_states_post_backward(self) -> None:
        if self.backend.fsdp_optimization_config is None:
            return

        fsdp_modules = []
        _model = self.model
        if isinstance(_model, FSDPModule):
            fsdp_modules.append(_model)

        if hasattr(self, "lm_head"):
            if isinstance(self.lm_head, FSDPModule):
                fsdp_modules.append(self.lm_head)

        if hasattr(_model, "layers"):
            for _, block in _model.layers.named_children():
                if hasattr(block, "mlp") and hasattr(block.mlp, "experts"):
                    experts = block.mlp.experts
                    if isinstance(experts, FSDPModule):
                        fsdp_modules.append(experts)

        MoEFSDPSyncMixin.post_backward_for_fsdp_modules(fsdp_modules)

    def set_fsdp_states_for_first_forward(self):
        fsdp_optimization_config: FsdpOptimizationConfig | None = self.backend.fsdp_optimization_config

        if fsdp_optimization_config is None:
            return

        _model = self.model

        # FSDP synchronization
        if isinstance(_model, FSDPModule):
            MoEFSDPSyncMixin.first_forward_for_fsdp_module(
                _model,
                requires_gradient_sync=not fsdp_optimization_config.defer_grad_sync_for_model,
                reshard_after_backward=not fsdp_optimization_config.defer_reshard_after_backward_for_model,
            )

        if hasattr(self, "lm_head"):
            if isinstance(self.lm_head, FSDPModule):
                MoEFSDPSyncMixin.first_forward_for_fsdp_module(
                    self.lm_head,
                    requires_gradient_sync=not fsdp_optimization_config.defer_grad_sync_for_lm_head,
                    reshard_after_backward=not fsdp_optimization_config.defer_reshard_after_backward_for_lm_head,
                )

        if hasattr(_model, "layers"):
            for _, block in _model.layers.named_children():
                # Check if this block has MoE experts wrapped in FSDP
                if hasattr(block, "mlp") and hasattr(block.mlp, "experts"):
                    experts = block.mlp.experts
                    if isinstance(experts, FSDPModule):
                        MoEFSDPSyncMixin.first_forward_for_fsdp_module(
                            experts,
                            requires_gradient_sync=not fsdp_optimization_config.defer_grad_sync_for_experts,
                            reshard_after_backward=not fsdp_optimization_config.defer_reshard_after_backward_for_experts,
                        )
