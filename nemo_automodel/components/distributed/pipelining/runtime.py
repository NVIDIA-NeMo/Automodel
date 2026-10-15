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

from collections.abc import Hashable
from typing import Protocol

import torch
from torch import nn


class PipelineRuntimeInitializer(Protocol):
    """A model-owned runtime resource that must be ready before a PP step."""

    @property
    def resource_key(self) -> Hashable:
        """Identity of the process-local runtime resource this initializer owns."""
        ...

    @property
    def signature(self) -> Hashable:
        """Immutable configuration expected by ``resource_key``."""
        ...

    def prepare(
        self,
        *,
        num_tokens: int,
        device: torch.device,
    ) -> None:
        """Prepare the resource for the upcoming pipeline microbatch.

        Args:
            num_tokens: Maximum number of tokens in the upcoming pipeline microbatch.
            device: Device used by the pipeline stage.
        """
        ...


def collect_pipeline_runtime_initializers(model_parts: list[nn.Module]) -> list[PipelineRuntimeInitializer]:
    """Collect model-owned initializers, deduplicating compatible shared resources.

    The pipeline layer treats keys and signatures as opaque values.  A repeated
    resource key is initialized once when every provider reports the same
    signature; conflicting signatures are rejected before any pipeline traffic.

    Args:
        model_parts: Pipeline-local model partitions whose modules may provide runtime initializers.

    Returns:
        Initializers in module traversal order, with compatible shared resources deduplicated.

    Raises:
        TypeError: If an initializer's resource key or signature is not hashable.
        RuntimeError: If providers report incompatible signatures for the same resource key.
    """
    by_resource: dict[Hashable, PipelineRuntimeInitializer] = {}
    signatures: dict[Hashable, Hashable] = {}
    providers: dict[Hashable, str] = {}

    for part_index, model_part in enumerate(model_parts):
        for module_name, module in model_part.named_modules():
            provider = getattr(module, "get_pipeline_runtime_initializers", None)
            if provider is None:
                continue
            provider_name = f"model_parts[{part_index}].{module_name or '<root>'} ({type(module).__name__})"
            for initializer in provider():
                resource_key = initializer.resource_key
                signature = initializer.signature
                try:
                    hash(resource_key)
                    hash(signature)
                except TypeError as error:
                    raise TypeError("Pipeline runtime initializer keys and signatures must be hashable") from error

                previous_signature = signatures.setdefault(resource_key, signature)
                if previous_signature != signature:
                    raise RuntimeError(
                        f"Pipeline runtime resource {resource_key!r} has incompatible signatures from "
                        f"{providers[resource_key]} and {provider_name}"
                    )
                by_resource.setdefault(resource_key, initializer)
                providers.setdefault(resource_key, provider_name)

    return list(by_resource.values())
