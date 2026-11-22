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

"""Raw conversation records shared by VLM builders, tokenization, and packing."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

import torch
from PIL import Image

# Conversation content includes JSON media metadata and lazily decoded images.
MediaValue: TypeAlias = (
    str | int | float | bool | None | bytes | Image.Image | torch.Tensor | list["MediaValue"] | dict[str, "MediaValue"]
)


class ConversationMessage(TypedDict):
    """A role and its text or structured media content."""

    role: str
    content: str | list[dict[str, MediaValue]]


class ConversationSample(TypedDict, total=False):
    """Raw conversation with optional media metadata and token estimate."""

    conversation: list[ConversationMessage]
    mm_inputs_meta: dict[str, MediaValue]
    _text_tokens: int
