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

"""Run the native packed-image/model parity cases on one GPU in functional CI."""

from collections.abc import Iterator

import pytest
import torch

# Reuse the same assertions; this module's fixture selects CUDA instead of CPU.
from tests.integration.test_gemma4_native_packing import (
    test_native_image_packing_matches_separate_logits_loss_and_gradients,
    test_pipeline_text_wrapper_preserves_model_prepared_masks_and_gradients,
    test_pipeline_vlm_wrapper_composes_native_image_policy_before_decoder,
    test_single_document_native_neat_retains_sliding_policy,
)

__all__ = [
    "test_native_image_packing_matches_separate_logits_loss_and_gradients",
    "test_pipeline_text_wrapper_preserves_model_prepared_masks_and_gradients",
    "test_pipeline_vlm_wrapper_composes_native_image_policy_before_decoder",
    "test_single_document_native_neat_retains_sliding_policy",
]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def device() -> Iterator[torch.device]:
    """Use FP32 CUDA arithmetic so parity isolates packed-attention semantics."""
    previous = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield torch.device("cuda")
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous
