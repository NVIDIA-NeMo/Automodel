# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Biencoder models for embedding and retrieval tasks.

This module contains biencoder architectures and bidirectional models
optimized for information retrieval and semantic search tasks.
"""

from .biencoder_model import BiencoderModel, BiencoderOutput  # noqa: F401
from .llama_bidirectional_model import (  # noqa: F401
    LlamaBidirectionalConfig,
    LlamaBidirectionalModel,
    LlamaBidirectionalForSequenceClassification,
)

__all__ = [
    "BiencoderModel",
    "BiencoderOutput",
    "LlamaBidirectionalConfig",
    "LlamaBidirectionalModel",
    "LlamaBidirectionalForSequenceClassification",
]

