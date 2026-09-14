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

"""Pipeline-parallel building blocks.

``AutoPipeline`` is resolved lazily: importing the light ``pipelining.config`` (re-exported by
``components.distributed``) must not load ``torch.distributed.pipelining`` and, through it,
``torch._dynamo``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline

__all__ = ["AutoPipeline"]


def __getattr__(name: str) -> Any:
    if name == "AutoPipeline":
        from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline

        return AutoPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
