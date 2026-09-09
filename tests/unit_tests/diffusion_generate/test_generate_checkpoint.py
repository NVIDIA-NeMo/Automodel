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

"""Unit tests for checkpoint loading in the diffusion generation script.

Covers ``load_checkpoint_into_pipeline`` (single- vs two-transformer routing and
mutual-exclusivity validation) and ``_load_checkpoint_into_attr`` (the dispatch
over the recognized on-disk checkpoint formats). The heavyweight loaders
(``from_pretrained``, FSDP / HF sharded readers) are patched out — these tests
exercise routing and validation logic, not GPU-bound deserialization.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import examples.diffusion.generate.generate as gen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cfg(model_kwargs=None, dtype="float32"):
    """Build a config node exposing ``cfg.model`` and ``cfg.inference``."""
    return SimpleNamespace(
        model=SimpleNamespace(**(model_kwargs or {})),
        inference=SimpleNamespace(dtype=dtype),
    )


class DummyTransformer:
    """Stand-in transformer with a classmethod ``from_pretrained`` so the
    consolidated-safetensors branch (``type(target).from_pretrained(...)``) can be
    exercised without a real diffusers module."""

    def __init__(self):
        self.loaded_state = None
        self.loaded_strict = None
        self.moved_to = None
        self.from_pretrained_args = None

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = state_dict
        self.loaded_strict = strict

    def to(self, *args, **kwargs):
        self.moved_to = (args, kwargs)
        return self

    @classmethod
    def from_pretrained(cls, path, torch_dtype=None):
        inst = cls()
        inst.from_pretrained_args = (path, torch_dtype)
        return inst


# ===========================================================================
# load_checkpoint_into_pipeline — routing + validation
# ===========================================================================
class TestLoadCheckpointIntoPipeline:
    def test_mutually_exclusive_single_and_two_stage_raises(self):
        pipe = SimpleNamespace(transformer=MagicMock())
        cfg = _make_cfg({"checkpoint": "/ckpt", "checkpoint_high_noise": "/ckpt_high"})
        with pytest.raises(ValueError, match="mutually exclusive"):
            gen.load_checkpoint_into_pipeline(pipe, cfg)

    def test_no_checkpoint_set_is_noop(self):
        pipe = SimpleNamespace(transformer=MagicMock())
        cfg = _make_cfg({})
        with patch.object(gen, "_load_checkpoint_into_attr") as mock_load:
            gen.load_checkpoint_into_pipeline(pipe, cfg)
        mock_load.assert_not_called()

    def test_single_checkpoint_loads_into_transformer(self):
        pipe = SimpleNamespace(transformer=MagicMock())
        cfg = _make_cfg({"checkpoint": "/ckpt"})
        with patch.object(gen, "_load_checkpoint_into_attr") as mock_load:
            gen.load_checkpoint_into_pipeline(pipe, cfg)
        mock_load.assert_called_once_with(pipe, "transformer", "/ckpt", torch.float32)

    def test_high_noise_loads_into_transformer(self):
        pipe = SimpleNamespace(transformer=MagicMock(), transformer_2=MagicMock())
        cfg = _make_cfg({"checkpoint_high_noise": "/ckpt_high"})
        with patch.object(gen, "_load_checkpoint_into_attr") as mock_load:
            gen.load_checkpoint_into_pipeline(pipe, cfg)
        mock_load.assert_called_once_with(pipe, "transformer", "/ckpt_high", torch.float32)

    def test_low_noise_loads_into_transformer_2(self):
        pipe = SimpleNamespace(transformer=MagicMock(), transformer_2=MagicMock())
        cfg = _make_cfg({"checkpoint_low_noise": "/ckpt_low"})
        with patch.object(gen, "_load_checkpoint_into_attr") as mock_load:
            gen.load_checkpoint_into_pipeline(pipe, cfg)
        mock_load.assert_called_once_with(pipe, "transformer_2", "/ckpt_low", torch.float32)

    def test_high_and_low_noise_load_into_both(self):
        pipe = SimpleNamespace(transformer=MagicMock(), transformer_2=MagicMock())
        cfg = _make_cfg({"checkpoint_high_noise": "/ckpt_high", "checkpoint_low_noise": "/ckpt_low"})
        with patch.object(gen, "_load_checkpoint_into_attr") as mock_load:
            gen.load_checkpoint_into_pipeline(pipe, cfg)
        assert mock_load.call_count == 2
        mock_load.assert_any_call(pipe, "transformer", "/ckpt_high", torch.float32)
        mock_load.assert_any_call(pipe, "transformer_2", "/ckpt_low", torch.float32)

    def test_low_noise_without_transformer_2_raises(self):
        # Single-transformer pipeline: transformer_2 attribute absent.
        pipe = SimpleNamespace(transformer=MagicMock())
        cfg = _make_cfg({"checkpoint_low_noise": "/ckpt_low"})
        with patch.object(gen, "_load_checkpoint_into_attr"):
            with pytest.raises(ValueError, match="transformer_2"):
                gen.load_checkpoint_into_pipeline(pipe, cfg)

    def test_low_noise_transformer_2_none_raises(self):
        # transformer_2 present but None (e.g. dropped for single-stage use).
        pipe = SimpleNamespace(transformer=MagicMock(), transformer_2=None)
        cfg = _make_cfg({"checkpoint_low_noise": "/ckpt_low"})
        with pytest.raises(ValueError, match="transformer_2"):
            gen.load_checkpoint_into_pipeline(pipe, cfg)

    def test_dtype_is_resolved_from_inference_cfg(self):
        pipe = SimpleNamespace(transformer=MagicMock())
        cfg = _make_cfg({"checkpoint": "/ckpt"}, dtype="bfloat16")
        with patch.object(gen, "_load_checkpoint_into_attr") as mock_load:
            gen.load_checkpoint_into_pipeline(pipe, cfg)
        assert mock_load.call_args[0][3] == torch.bfloat16


# ===========================================================================
# _load_checkpoint_into_attr — format dispatch
# ===========================================================================
class TestLoadCheckpointIntoAttr:
    def test_missing_directory_raises(self, tmp_path):
        pipe = SimpleNamespace(transformer=MagicMock())
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
            gen._load_checkpoint_into_attr(pipe, "transformer", str(missing), torch.float32)

    def test_none_target_raises(self, tmp_path):
        pipe = SimpleNamespace(transformer=None)
        with pytest.raises(AttributeError, match="transformer"):
            gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)

    def test_ema_checkpoint_loaded_strict(self, tmp_path):
        (tmp_path / "ema_shadow.pt").write_bytes(b"")
        target = MagicMock()
        pipe = SimpleNamespace(transformer=target)
        fake_state = {"w": torch.zeros(1)}
        with patch.object(gen.torch, "load", return_value=fake_state) as mock_torch_load:
            gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)
        mock_torch_load.assert_called_once()
        target.load_state_dict.assert_called_once_with(fake_state, strict=True)

    def test_consolidated_bin_unwraps_model_state_dict(self, tmp_path):
        (tmp_path / "consolidated_model.bin").write_bytes(b"")
        target = MagicMock()
        pipe = SimpleNamespace(transformer=target)
        inner = {"w": torch.zeros(1)}
        with patch.object(gen.torch, "load", return_value={"model_state_dict": inner}):
            gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)
        target.load_state_dict.assert_called_once_with(inner, strict=True)

    def test_consolidated_bin_raw_state_dict(self, tmp_path):
        (tmp_path / "consolidated_model.bin").write_bytes(b"")
        target = MagicMock()
        pipe = SimpleNamespace(transformer=target)
        raw = {"w": torch.zeros(1)}
        with patch.object(gen.torch, "load", return_value=raw):
            gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)
        target.load_state_dict.assert_called_once_with(raw, strict=True)

    def test_consolidated_safetensors_dir_uses_from_pretrained(self, tmp_path):
        st_dir = tmp_path / "model" / "consolidated"
        st_dir.mkdir(parents=True)
        (st_dir / "model.safetensors").write_bytes(b"")
        target = DummyTransformer()
        pipe = SimpleNamespace(transformer=target)

        gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)

        # The transformer attribute is replaced with the from_pretrained result.
        assert pipe.transformer is not target
        assert isinstance(pipe.transformer, DummyTransformer)
        assert pipe.transformer.from_pretrained_args == (str(st_dir), torch.float32)

    def test_sharded_fsdp_distcp_dispatch(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "__0_0.distcp").write_bytes(b"")
        target = MagicMock()
        pipe = SimpleNamespace(transformer=target)
        new_module = MagicMock()
        with patch.object(gen, "_load_sharded_fsdp_checkpoint", return_value=new_module) as mock_fsdp:
            gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)
        mock_fsdp.assert_called_once_with(target, str(model_dir), torch.float32)
        assert pipe.transformer is new_module

    def test_sharded_hf_safetensors_dispatch(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "shard-00000-model-00001-of-00001.safetensors").write_bytes(b"")
        target = MagicMock()
        pipe = SimpleNamespace(transformer=target)
        new_module = MagicMock()
        with patch.object(gen, "_load_sharded_hf_safetensors_checkpoint", return_value=new_module) as mock_hf:
            gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)
        mock_hf.assert_called_once_with(target, str(model_dir), torch.float32)
        assert pipe.transformer is new_module

    def test_unrecognized_format_leaves_weights_untouched(self, tmp_path):
        # Empty directory: no recognized checkpoint files present.
        target = MagicMock()
        pipe = SimpleNamespace(transformer=target)
        gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)
        target.load_state_dict.assert_not_called()
        assert pipe.transformer is target

    def test_ema_preferred_over_consolidated(self, tmp_path):
        # Both EMA and consolidated present: EMA wins.
        (tmp_path / "ema_shadow.pt").write_bytes(b"")
        (tmp_path / "consolidated_model.bin").write_bytes(b"")
        target = MagicMock()
        pipe = SimpleNamespace(transformer=target)
        ema_state = {"ema": torch.zeros(1)}
        with patch.object(gen.torch, "load", return_value=ema_state) as mock_torch_load:
            gen._load_checkpoint_into_attr(pipe, "transformer", str(tmp_path), torch.float32)
        # Only one load (the EMA one), and it was loaded into the target.
        mock_torch_load.assert_called_once()
        target.load_state_dict.assert_called_once_with(ema_state, strict=True)
