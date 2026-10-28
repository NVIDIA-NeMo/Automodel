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

"""Stable passage identities shared by retrieval text and image collators."""

import hashlib


def document_id_to_int64(document_id: str) -> int:
    """Encode a corpus ID consistently across processes and input modalities.

    Args:
        document_id: Corpus document identifier encoded as UTF-8.

    Returns:
        Nonnegative 63-bit integer, preserving the existing text-collator encoding.
        This is a non-cryptographic identity hash, not a security checksum.
    """
    digest = hashlib.md5(document_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)
