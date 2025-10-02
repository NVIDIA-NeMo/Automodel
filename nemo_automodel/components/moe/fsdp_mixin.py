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

from torch.distributed.fsdp import FSDPModule


class MoEFSDPSyncMixin:
    @staticmethod
    def last_backward_for_fsdp_module(fsdp_module: FSDPModule) -> None:
        fsdp_module.set_is_last_backward(True)
        fsdp_module.set_reshard_after_backward(True)
        fsdp_module.set_requires_gradient_sync(True)

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

        if hasattr(_model, "embed_tokens"):
            if isinstance(_model.embed_tokens, FSDPModule):
                MoEFSDPSyncMixin.last_backward_for_fsdp_module(_model.embed_tokens)

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

    def set_fsdp_states_for_first_forward(self):
        if self.backend.fsdp_optimization_config is None:
            return

        _model = self.model

        # FSDP synchronization
        if isinstance(_model, FSDPModule):
            MoEFSDPSyncMixin.first_forward_for_fsdp_module(
                _model,
                requires_gradient_sync=not self.backend.fsdp_optimization_config.disable_gradient_sync_for_model,
                reshard_after_backward=not self.backend.fsdp_optimization_config.disable_reshard_after_backward_for_model,
            )

        if hasattr(_model, "embed_tokens"):
            if isinstance(_model.embed_tokens, FSDPModule):
                MoEFSDPSyncMixin.first_forward_for_fsdp_module(
                    _model.embed_tokens,
                    requires_gradient_sync=not self.backend.fsdp_optimization_config.disable_gradient_sync_for_embed_tokens,
                    reshard_after_backward=not self.backend.fsdp_optimization_config.disable_reshard_after_backward_for_embed_tokens,
                )

        if hasattr(self, "lm_head"):
            if isinstance(self.lm_head, FSDPModule):
                MoEFSDPSyncMixin.first_forward_for_fsdp_module(
                    self.lm_head,
                    requires_gradient_sync=not self.backend.fsdp_optimization_config.disable_gradient_sync_for_lm_head,
                    reshard_after_backward=not self.backend.fsdp_optimization_config.disable_reshard_after_backward_for_lm_head,
                )

        if hasattr(_model, "layers"):
            for _, block in _model.layers.named_children():
                # Check if this block has MoE experts wrapped in FSDP
                if hasattr(block, "mlp") and hasattr(block.mlp, "experts"):
                    experts = block.mlp.experts
                    if isinstance(experts, FSDPModule):
                        MoEFSDPSyncMixin.first_forward_for_fsdp_module(
                            experts,
                            requires_gradient_sync=not self.backend.fsdp_optimization_config.disable_gradient_sync_for_experts,
                            reshard_after_backward=not self.backend.fsdp_optimization_config.disable_reshard_after_backward_for_experts,
                        )
