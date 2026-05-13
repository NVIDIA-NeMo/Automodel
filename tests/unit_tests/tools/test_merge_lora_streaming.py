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
"""CPU unit tests for tools/merge_lora_streaming.py.

Cover three things end-to-end without needing the actual 30B model or any GPU:

1. NEMO/HF-PEFT adapter-key translation (the ``base_model.model.`` strip).
2. The ``B @ A * (alpha / r)`` matmul + scale on tiny synthetic shards.
3. The error path when the adapter references a base key that does not exist
   in the base ``model.safetensors.index.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import safe_open, save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import merge_lora_streaming as mls  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_base(base_dir: Path, *, hidden: int = 8, out: int = 6) -> dict[str, torch.Tensor]:
    """Write a 2-shard synthetic base model with one mergeable q_proj.weight.

    Returns the dict of tensors written so the test can compare back.
    """
    torch.manual_seed(0)
    q_weight = torch.randn(out, hidden, dtype=torch.float32)
    norm_weight = torch.randn(hidden, dtype=torch.float32)
    other_weight = torch.randn(hidden, hidden, dtype=torch.float32)

    shard1 = {
        "thinker.model.layers.0.self_attn.q_proj.weight": q_weight,
        "thinker.model.layers.0.input_layernorm.weight": norm_weight,
    }
    shard2 = {
        "thinker.audio_tower.layers.0.self_attn.q_proj.weight": other_weight,
    }
    save_file(shard1, str(base_dir / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(shard2, str(base_dir / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": sum(t.numel() * 4 for t in {**shard1, **shard2}.values())},
        "weight_map": {
            "thinker.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "thinker.model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        },
    }
    with open(base_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    # Minimal HF metadata so copy_hf_metadata has something real to copy.
    with open(base_dir / "config.json", "w") as f:
        json.dump({"architectures": ["FakeModel"], "model_type": "fake"}, f)

    return {**shard1, **shard2}


def _write_tiny_adapter(
    adapter_dir: Path,
    *,
    inner_key: str,
    hidden: int = 8,
    out: int = 6,
    r: int = 2,
    alpha: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Write a NEMO-style adapter dir with one (lora_A, lora_B) pair.

    The NEMO/HF-PEFT adapter saves keys as
    ``thinker.base_model.model.<inner_key>.lora_{A,B}.weight``; after the
    ``base_model.model.`` strip the matching base key is
    ``thinker.<inner_key>.weight``.

    Returns (A, B, scale) so the test can recompute the expected delta.
    """
    torch.manual_seed(1)
    a = torch.randn(r, hidden, dtype=torch.float32)
    b = torch.randn(out, r, dtype=torch.float32)
    scale = alpha / r

    adapter_sd = {
        f"thinker.base_model.model.{inner_key}.lora_A.weight": a,
        f"thinker.base_model.model.{inner_key}.lora_B.weight": b,
    }
    save_file(adapter_sd, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": alpha,
        "bias": "none",
        "target_modules": [inner_key],
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f)
    return a, b, scale


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter_key, expected_base_key",
    [
        (
            "thinker.base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "thinker.model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "thinker.base_model.model.model.layers.5.self_attn.o_proj.lora_B.weight",
            "thinker.model.layers.5.self_attn.o_proj.weight",
        ),
        (
            "base_model.model.transformer.h.0.mlp.fc.lora_A.weight",
            "transformer.h.0.mlp.fc.weight",
        ),
    ],
)
def test_adapter_key_to_base_key_translates_correctly(adapter_key, expected_base_key):
    assert mls.adapter_key_to_base_key(adapter_key) == expected_base_key


def test_adapter_key_to_base_key_rejects_unknown_suffix():
    with pytest.raises(ValueError, match=r"\.lora_A\.weight"):
        mls.adapter_key_to_base_key("thinker.base_model.model.q_proj.weight")


def test_build_deltas_computes_scaled_matmul(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    inner_key = "model.layers.0.self_attn.q_proj"
    a, b, scale = _write_tiny_adapter(adapter_dir, inner_key=inner_key, r=2, alpha=4)

    deltas = mls.build_deltas(adapter_dir, dtype=torch.float32)

    base_key = f"thinker.{inner_key}.weight"
    assert set(deltas.keys()) == {base_key}
    expected = (b @ a) * scale
    assert torch.allclose(deltas[base_key], expected, atol=0.0, rtol=0.0)


def test_build_deltas_raises_on_unpaired_lora_b(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    save_file(
        {"thinker.base_model.model.q.lora_B.weight": torch.zeros(2, 2)},
        str(adapter_dir / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump({"r": 1, "lora_alpha": 1, "peft_type": "LORA"}, f)
    with pytest.raises(ValueError, match="missing lora_A pair"):
        mls.build_deltas(adapter_dir, dtype=torch.float32)


def test_stream_merge_shards_applies_delta_only_to_matching_key(tmp_path):
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    adapter_dir.mkdir()

    base_tensors = _write_tiny_base(base_dir, hidden=8, out=6)
    inner_key = "model.layers.0.self_attn.q_proj"
    base_key = f"thinker.{inner_key}.weight"
    a, b, scale = _write_tiny_adapter(adapter_dir, inner_key=inner_key, r=2, alpha=4)

    merged = mls.stream_merge_shards(base_dir, adapter_dir, out_dir, dtype=torch.float32)
    # Only the q_proj was patched; audio_tower q_proj must NOT have been touched.
    assert merged == {base_key}

    # Verify on-disk tensors.
    with safe_open(str(out_dir / "model-00001-of-00002.safetensors"), framework="pt") as f:
        merged_q = f.get_tensor(base_key)
        merged_norm = f.get_tensor("thinker.model.layers.0.input_layernorm.weight")
    with safe_open(str(out_dir / "model-00002-of-00002.safetensors"), framework="pt") as f:
        merged_audio_q = f.get_tensor("thinker.audio_tower.layers.0.self_attn.q_proj.weight")

    expected_q = base_tensors[base_key] + (b @ a) * scale
    assert torch.allclose(merged_q, expected_q, atol=0.0, rtol=0.0)
    # Other tensors in the same shard pass through unchanged.
    assert torch.equal(merged_norm, base_tensors["thinker.model.layers.0.input_layernorm.weight"])
    # Audio-tower q_proj is a sibling and must remain untouched.
    assert torch.equal(
        merged_audio_q,
        base_tensors["thinker.audio_tower.layers.0.self_attn.q_proj.weight"],
    )


def test_stream_merge_shards_raises_when_adapter_targets_missing_base_key(tmp_path):
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    adapter_dir.mkdir()
    _write_tiny_base(base_dir)
    # Adapter targets an inner key that does not exist in the (synthetic) base index.
    _write_tiny_adapter(
        adapter_dir,
        inner_key="does.not.exist.q_proj",
        r=2,
        alpha=4,
    )
    with pytest.raises(ValueError, match=r"base keys that do not exist"):
        mls.stream_merge_shards(base_dir, adapter_dir, out_dir, dtype=torch.float32)


def test_merge_streaming_end_to_end_copies_metadata(tmp_path):
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    adapter_dir.mkdir()
    _write_tiny_base(base_dir)
    _write_tiny_adapter(adapter_dir, inner_key="model.layers.0.self_attn.q_proj")

    mls.merge_streaming(base_dir, adapter_dir, out_dir, dtype=torch.float32)
    assert (out_dir / "config.json").exists()
    assert (out_dir / "model-00001-of-00002.safetensors").exists()
    assert (out_dir / "model-00002-of-00002.safetensors").exists()
    assert (out_dir / "model.safetensors.index.json").exists()


def test_merge_streaming_dtype_bfloat16(tmp_path):
    """Output shards must honor the requested dtype."""
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    adapter_dir.mkdir()
    _write_tiny_base(base_dir)
    inner_key = "model.layers.0.self_attn.q_proj"
    _write_tiny_adapter(adapter_dir, inner_key=inner_key)

    mls.merge_streaming(base_dir, adapter_dir, out_dir, dtype=torch.bfloat16)
    with safe_open(str(out_dir / "model-00001-of-00002.safetensors"), framework="pt") as f:
        merged_q = f.get_tensor(f"thinker.{inner_key}.weight")
    assert merged_q.dtype == torch.bfloat16
