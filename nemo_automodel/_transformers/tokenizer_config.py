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

"""Typed tokenizer and processor construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from transformers import AutoProcessor, PreTrainedTokenizerBase, ProcessorMixin

TokenizerLike = PreTrainedTokenizerBase | ProcessorMixin
TokenizerFactory = Callable[..., TokenizerLike]


@dataclass(frozen=True)
class TokenizerConfig:
    """Declarative tokenizer or processor factory configuration.

    ``factory=None`` represents an explicitly disabled tokenizer. Factory
    keyword arguments are copied at build time so the serializable config is
    never mutated or used to cache the runtime object.
    """

    factory: TokenizerFactory | None = None
    kwargs: Mapping[str, object] = field(default_factory=dict)

    def build(self) -> TokenizerLike | None:
        """Build the configured tokenizer or processor.

        Returns:
            A tokenizer or multimodal processor, or ``None`` when construction
            is disabled.
        """
        if self.factory is None:
            return None
        try:
            return self.factory(**dict(self.kwargs))
        except Exception as exc:
            message = str(exc)
            is_layer_types_validation = "num_hidden_layers" in message and (
                "layer_types" in message or "layer types" in message
            )
            if self.factory != AutoProcessor.from_pretrained or not is_layer_types_validation:
                raise
            from nemo_automodel._transformers.v4_patches.layer_types import relax_layer_types_validator

            relax_layer_types_validator()
            return self.factory(**dict(self.kwargs))


__all__ = ["TokenizerConfig", "TokenizerFactory", "TokenizerLike"]
