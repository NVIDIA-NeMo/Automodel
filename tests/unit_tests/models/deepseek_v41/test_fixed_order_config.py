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

"""Validate the opt-in backward contract without loading model weights."""

import pytest
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig

pytestmark = pytest.mark.cpu_only


def test_backward_default_and_roundtrip():
    assert DeepseekV41TextConfig().sparse_attention_backward == "atomic"
    cfg = DeepseekV41TextConfig(
        sparse_attention_backward="deterministic", sparse_attention_backward_workspace_bytes=1048576
    )
    restored = DeepseekV41TextConfig.from_dict(cfg.to_dict())
    assert restored.sparse_attention_backward == "deterministic"
    assert restored.sparse_attention_backward_workspace_bytes == 1048576
    assert restored.reuse_sparse_selection_on_recompute


@pytest.mark.parametrize(
    "options",
    [
        dict(sparse_attention_backward="unknown"),
        dict(sparse_attention_backward_workspace_bytes=0),
        dict(sparse_attention_backward_workspace_bytes=-1),
    ],
)
def test_invalid_backward_config(options):
    with pytest.raises(ValueError):
        DeepseekV41TextConfig(**options)


def test_fixed_order_rejects_non_tilelang_backend():
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention

    config = DeepseekV41TextConfig(sparse_attention_backward="deterministic")
    with pytest.raises(ValueError, match="requires TileLang"):
        DeepseekV41Attention(config, layer_idx=0, backend=BackendConfig(attn="cudnn", linear="torch", rms_norm="torch_fp32"))
