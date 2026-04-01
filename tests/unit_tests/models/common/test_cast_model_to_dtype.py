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

from unittest.mock import patch

import torch
import torch.nn as nn

from nemo_automodel.components.models.common.utils import (
    _get_fp32_module_keywords,
    _has_dtensor_params,
    _restore_fp32_buffers,
    _restore_fp32_modules,
    cast_model_to_dtype,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    """A small model for dtype testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.linear2 = nn.Linear(4, 2)


class ModelWithFp32Modules(nn.Module):
    """Model that declares modules to keep in fp32 (HF-style)."""

    _keep_in_fp32_modules = ["norm"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)


class ModelWithStrictFp32(nn.Module):
    """Model that declares strict fp32 modules."""

    _keep_in_fp32_modules_strict = ["head"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


class ModelWithBothFp32Attrs(nn.Module):
    """Model with both _keep_in_fp32_modules and _keep_in_fp32_modules_strict."""

    _keep_in_fp32_modules = ["norm"]
    _keep_in_fp32_modules_strict = ["head"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)


# ---------------------------------------------------------------------------
# Tests for _get_fp32_module_keywords()
# ---------------------------------------------------------------------------

class TestGetFp32ModuleKeywords:
    def test_no_attributes(self):
        model = SimpleModel()
        assert _get_fp32_module_keywords(model) == []

    def test_keep_in_fp32_modules(self):
        model = ModelWithFp32Modules()
        assert _get_fp32_module_keywords(model) == ["norm"]

    def test_keep_in_fp32_modules_strict(self):
        model = ModelWithStrictFp32()
        assert _get_fp32_module_keywords(model) == ["head"]

    def test_both_attributes_deduped(self):
        model = ModelWithBothFp32Attrs()
        keywords = _get_fp32_module_keywords(model)
        assert "head" in keywords
        assert "norm" in keywords
        assert len(keywords) == 2

    def test_duplicates_removed(self):
        class Model(nn.Module):
            _keep_in_fp32_modules = ["norm", "head"]
            _keep_in_fp32_modules_strict = ["head"]

            def __init__(self):
                super().__init__()

        model = Model()
        keywords = _get_fp32_module_keywords(model)
        assert keywords.count("head") == 1

    def test_none_attributes_ignored(self):
        class Model(nn.Module):
            _keep_in_fp32_modules = None

            def __init__(self):
                super().__init__()

        model = Model()
        assert _get_fp32_module_keywords(model) == []


# ---------------------------------------------------------------------------
# Tests for _restore_fp32_modules()
# ---------------------------------------------------------------------------

class TestRestoreFp32Modules:
    def test_matching_modules_restored(self):
        model = SimpleModel()
        model.to(torch.bfloat16)
        assert model.norm.weight.dtype == torch.bfloat16

        _restore_fp32_modules(model, ["norm"])
        assert model.norm.weight.dtype == torch.float32

    def test_non_matching_modules_unchanged(self):
        model = SimpleModel()
        model.to(torch.bfloat16)

        _restore_fp32_modules(model, ["norm"])
        assert model.linear1.weight.dtype == torch.bfloat16
        assert model.linear2.weight.dtype == torch.bfloat16

    def test_empty_keywords_noop(self):
        model = SimpleModel()
        model.to(torch.bfloat16)

        _restore_fp32_modules(model, [])
        # Everything stays bf16
        assert model.norm.weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Tests for cast_model_to_dtype()
# ---------------------------------------------------------------------------

class TestCastModelToDtype:
    def test_simple_model_cast_to_bf16(self):
        model = SimpleModel()
        assert model.linear1.weight.dtype == torch.float32

        cast_model_to_dtype(model, torch.bfloat16)
        assert model.linear1.weight.dtype == torch.bfloat16
        assert model.norm.weight.dtype == torch.bfloat16
        assert model.linear2.weight.dtype == torch.bfloat16

    def test_fp32_modules_preserved(self):
        model = ModelWithFp32Modules()
        cast_model_to_dtype(model, torch.bfloat16)

        assert model.norm.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16
        assert model.head.weight.dtype == torch.bfloat16

    def test_strict_fp32_modules_preserved(self):
        model = ModelWithStrictFp32()
        cast_model_to_dtype(model, torch.bfloat16)

        assert model.head.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16

    def test_both_fp32_attrs_preserved(self):
        model = ModelWithBothFp32Attrs()
        cast_model_to_dtype(model, torch.bfloat16)

        assert model.norm.weight.dtype == torch.float32
        assert model.head.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16

    def test_no_fp32_modules_all_cast(self):
        model = SimpleModel()
        cast_model_to_dtype(model, torch.bfloat16)

        for p in model.parameters():
            assert p.dtype == torch.bfloat16

    def test_fp16_dtype(self):
        model = SimpleModel()
        cast_model_to_dtype(model, torch.float16)

        for p in model.parameters():
            assert p.dtype == torch.float16


# ---------------------------------------------------------------------------
# Tests for DTensor-aware casting
# ---------------------------------------------------------------------------

class TestDTensorAwareCasting:
    def test_has_dtensor_params_false_for_plain_model(self):
        model = SimpleModel()
        assert not _has_dtensor_params(model)

    def test_dtensor_params_only_buffers_restored(self):
        """When model has DTensor params, only buffers of matching modules are restored to fp32."""
        model = ModelWithFp32Modules()

        with patch("nemo_automodel.components.models.common.utils._has_dtensor_params", return_value=True):
            cast_model_to_dtype(model, torch.bfloat16)

        # Parameters should be bf16 — FSDP2 requires uniform dtype
        for p in model.parameters():
            assert p.dtype == torch.bfloat16

    def test_dtensor_buffers_in_matching_modules_restored(self):
        """Buffers in fp32-keyword-matching modules are cast to fp32 even with DTensor params."""

        class ModelWithNormBuffer(nn.Module):
            _keep_in_fp32_modules = ["norm"]

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.norm = nn.LayerNorm(4)
                self.norm.register_buffer("e_score_bias", torch.zeros(4))

        model = ModelWithNormBuffer()

        with patch("nemo_automodel.components.models.common.utils._has_dtensor_params", return_value=True):
            cast_model_to_dtype(model, torch.bfloat16)

        # Parameters stay bf16
        assert model.norm.weight.dtype == torch.bfloat16
        assert model.linear.weight.dtype == torch.bfloat16
        # Buffer in matching module is restored to fp32
        assert model.norm.e_score_bias.dtype == torch.float32

    def test_fp32_restore_applied_for_plain_params(self):
        """When model has plain tensor params, fp32 restore works normally."""
        model = ModelWithFp32Modules()

        with patch("nemo_automodel.components.models.common.utils._has_dtensor_params", return_value=False):
            cast_model_to_dtype(model, torch.bfloat16)

        assert model.norm.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16


class TestRestoreFp32Buffers:
    def test_buffers_restored_params_untouched(self):
        """_restore_fp32_buffers casts buffers but not parameters."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(4)
                self.norm.register_buffer("bias_buf", torch.zeros(4))

        model = Model()
        model.to(torch.bfloat16)
        _restore_fp32_buffers(model, ["norm"])

        assert model.norm.bias_buf.dtype == torch.float32
        assert model.norm.weight.dtype == torch.bfloat16

    def test_non_matching_buffers_unchanged(self):
        """Buffers in non-matching modules stay in the cast dtype."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.linear.register_buffer("scale", torch.ones(4))
                self.norm = nn.LayerNorm(4)

        model = Model()
        model.to(torch.bfloat16)
        _restore_fp32_buffers(model, ["norm"])

        assert model.linear.scale.dtype == torch.bfloat16
