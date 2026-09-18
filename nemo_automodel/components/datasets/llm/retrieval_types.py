# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from enum import IntEnum


class PassageModality(IntEnum):
    """Identify whether a retrieval passage contains text, an image, or both."""

    TEXT_ONLY = 0
    IMAGE_ONLY = 1
    IMAGE_TEXT = 2


PASSAGE_MODALITY_NAMES = {
    PassageModality.TEXT_ONLY: "text_only",
    PassageModality.IMAGE_ONLY: "image_only",
    PassageModality.IMAGE_TEXT: "image_text",
}
