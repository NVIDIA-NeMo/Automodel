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

"""Typed Hugging Face processor construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transformers import AutoProcessor as TransformersAutoProcessor
from transformers import ProcessorMixin


class AutoProcessor:
    """Auto processor factory with implementation-owned configuration."""

    @dataclass(frozen=True, kw_only=True)
    class Config:
        """Declarative configuration for Hugging Face ``AutoProcessor``."""

        pretrained_model_name_or_path: str
        trust_remote_code: bool = False
        min_pixels: int | None = None
        max_pixels: int | None = None
        padding_side: Literal["left", "right"] | None = None

        def build(self) -> ProcessorMixin:
            """Build the configured processor without mutating config state."""
            processor_kwargs: dict[str, object] = {}
            if self.min_pixels is not None:
                processor_kwargs["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                processor_kwargs["max_pixels"] = self.max_pixels
            if self.padding_side is not None:
                processor_kwargs["padding_side"] = self.padding_side
            return TransformersAutoProcessor.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **processor_kwargs,
            )


__all__ = ["AutoProcessor"]
