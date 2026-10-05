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

"""Bidirectional Ministral3 model for embedding and retrieval tasks."""

from nemo_automodel.components.models.ministral_bidirectional.model import (
    Ministral3BidirectionalConfig,
    Ministral3BidirectionalModel,
    Mistral3BidirectionalConfig,
    Mistral3BidirectionalModel,
    Mistral3VLBidirectionalForSequenceClassification,
)
from nemo_automodel.shared.import_utils import safe_import_from

_, Mistral3BiEncoderProcessor = safe_import_from(
    "nemo_automodel.components.models.ministral_bidirectional.processor",
    "Mistral3BiEncoderProcessor",
    msg="Mistral3BiEncoderProcessor requires the vision dependencies from the diffusion extra.",
)

__all__ = [
    "Ministral3BidirectionalModel",
    "Ministral3BidirectionalConfig",
    "Mistral3BidirectionalModel",
    "Mistral3BidirectionalConfig",
    "Mistral3VLBidirectionalForSequenceClassification",
    "Mistral3BiEncoderProcessor",
]
