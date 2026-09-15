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

"""Parallelization contract for the diffusers ``HunyuanVideo15Transformer3DModel``.

Its ``transformer_blocks`` (declared in diffusers' ``_no_split_modules``) form the ``backbone`` layer group the
parallelizer derives; the model only asks for whole-block activation checkpointing.
"""

from __future__ import annotations

from nemo_automodel.components.distributed.activation_checkpointing import ActivationCheckpointingSpec


class HunyuanVideo15Transformer3DModel:
    """Contract for the diffusers ``HunyuanVideo15Transformer3DModel``; bound by the diffusion pipeline before sharding."""

    activation_checkpointing_spec: ActivationCheckpointingSpec = ActivationCheckpointingSpec(granularity="layer")
