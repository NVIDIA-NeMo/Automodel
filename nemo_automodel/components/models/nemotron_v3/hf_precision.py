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

"""Hugging Face nemotron_h routing precision compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from nemo_automodel.components.models.common.precision import preserve_fp32_module_compute

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.mesh import MeshContext


def apply_hf_precision(model: nn.Module, *, mesh: MeshContext) -> nn.Module:
    """Keep the upstream HF router's FP32 projection and scoring outside autocast.

    Args:
        model: Loaded model, including HF modules retained by force_hf or fallback.
        mesh: Requested strategy and execution device mesh.

    Returns:
        Model with its HF router computation protected and original state intact.
    """
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHTopkRouter

    return preserve_fp32_module_compute(model, module_type=NemotronHTopkRouter, mesh_context=mesh)
