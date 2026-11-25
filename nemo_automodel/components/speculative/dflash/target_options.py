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

"""Typed inputs and outputs of a DFlash draft spec's ``build_target_kwargs``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from nemo_automodel.components.models.common import BackendConfig


@dataclass(frozen=True)
class DFlashTargetOptions:
    """How the recipe asks for the frozen target to be loaded.

    ``None`` means the recipe left the option unset; each target's
    ``build_target_kwargs`` applies its own default.

    Attributes:
        dispatcher: Expert-parallel token dispatcher for an MoE target.
        experts: Expert GEMM backend for an MoE target.
        enable_fsdp_optimizations: Whether the target's FSDP optimizations are on.
    """

    dispatcher: str | None = None
    experts: str | None = None
    enable_fsdp_optimizations: bool | None = None


class DFlashTargetKwargs(TypedDict, total=False):
    """Architecture-specific keyword arguments merged into the frozen target's ``from_pretrained`` call.

    Attributes:
        config: Overrides applied to the checkpoint config (e.g. ``architectures``).
        backend: Kernel and dispatcher selection for a custom-model target.
    """

    config: dict[str, list[str]]
    backend: BackendConfig


__all__ = ["DFlashTargetKwargs", "DFlashTargetOptions"]
