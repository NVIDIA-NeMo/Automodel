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

"""Compatibility imports for shared physical packed-sequence metadata."""

from nemo_automodel.shared.packed_seq import (
    PackedSeqParams,
    packed_seq_params_from_doc_ids,
    to_flash_attention_kwargs,
)

__all__ = ["PackedSeqParams", "packed_seq_params_from_doc_ids", "to_flash_attention_kwargs"]
