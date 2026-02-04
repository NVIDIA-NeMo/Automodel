# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import gc
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common.combined_projection.state_dict_adapter import (
    CombinedProjectionStateDictAdapter,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _skip_if_low_free_cuda_mem(required_bytes: int, safety_factor: float = 2.0) -> None:
    # mem_get_info returns (free, total) in bytes
    free, _ = torch.cuda.mem_get_info()
    if free < int(required_bytes * safety_factor):
        pytest.skip(
            f"Not enough free CUDA memory for this test. "
            f"Need ~{required_bytes * safety_factor / (1024**3):.2f} GiB, have {free / (1024**3):.2f} GiB."
        )


def _reset_cuda_mem_stats() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def test_combined_projection_from_hf_inplace_reduces_live_vram():
    """Regression: inplace=True should avoid keeping both split (q/k/v) and merged (qkv) tensors alive.

    We compare the delta in torch.cuda.memory_allocated() after conversion relative to the
    baseline allocated by the input HF tensors.
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.float16

    # Chosen to be large enough for a stable signal but small enough for modest GPUs.
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8  # GQA
    head_dim = hidden_size // num_attention_heads

    cfg = SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        tie_word_embeddings=False,
    )
    adapter = CombinedProjectionStateDictAdapter(cfg)

    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim

    def make_hf_state_dict():
        sd = {
            "model.layers.0.self_attn.q_proj.weight": torch.empty((q_size, hidden_size), device=device, dtype=dtype),
            "model.layers.0.self_attn.k_proj.weight": torch.empty((kv_size, hidden_size), device=device, dtype=dtype),
            "model.layers.0.self_attn.v_proj.weight": torch.empty((kv_size, hidden_size), device=device, dtype=dtype),
        }
        expected_saved = (
            _tensor_bytes(sd["model.layers.0.self_attn.q_proj.weight"])
            + _tensor_bytes(sd["model.layers.0.self_attn.k_proj.weight"])
            + _tensor_bytes(sd["model.layers.0.self_attn.v_proj.weight"])
        )
        return sd, expected_saved

    # Ensure we have enough headroom for baseline + merged output.
    tmp_sd, expected_saved_bytes = make_hf_state_dict()
    required = expected_saved_bytes * 2  # split + merged
    del tmp_sd
    _skip_if_low_free_cuda_mem(required_bytes=required)

    # Non-inplace: input dict retains q/k/v while output dict holds qkv => extra live VRAM ~ q+k+v bytes.
    _reset_cuda_mem_stats()
    hf_sd, expected_saved_bytes = make_hf_state_dict()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    out = adapter.from_hf(hf_sd, inplace=False)
    torch.cuda.synchronize()
    delta_non_inplace = torch.cuda.memory_allocated() - baseline
    del out, hf_sd

    # Inplace: q/k/v are popped and replaced by qkv => minimal extra live VRAM.
    _reset_cuda_mem_stats()
    hf_sd2, _ = make_hf_state_dict()
    torch.cuda.synchronize()
    baseline2 = torch.cuda.memory_allocated()
    out2 = adapter.from_hf(hf_sd2, inplace=True)
    torch.cuda.synchronize()
    delta_inplace = torch.cuda.memory_allocated() - baseline2
    del out2, hf_sd2

    # Allow allocator noise, but require a large savings relative to the merged tensor size.
    assert (delta_non_inplace - delta_inplace) >= int(expected_saved_bytes * 0.70), (
        f"Expected inplace conversion to save most of q/k/v live VRAM.\n"
        f"expected_saved_bytes={expected_saved_bytes}\n"
        f"delta_non_inplace={delta_non_inplace}\n"
        f"delta_inplace={delta_inplace}\n"
    )

