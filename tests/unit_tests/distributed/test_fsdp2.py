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


def test_patch_is_packed_sequence_for_training_returns_false_when_helper_is_missing(monkeypatch):
    import transformers.modeling_flash_attention_utils as fa_utils

    from nemo_automodel.components.distributed.fsdp2 import _patch_is_packed_sequence_for_training

    monkeypatch.delattr(fa_utils, "_is_packed_sequence", raising=False)
    monkeypatch.delattr(fa_utils, "_is_packed_sequence_patched", raising=False)

    assert _patch_is_packed_sequence_for_training() is False
    assert not getattr(fa_utils, "_is_packed_sequence_patched", False)
