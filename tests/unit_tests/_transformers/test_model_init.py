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

import torch
from transformers import PreTrainedModel
from unittest.mock import patch

from nemo_automodel._transformers.model_init import (
    _filter_meta_device_from_init_context,
    _patched_get_init_context,
    no_hf_meta_device,
)


class TestFilterMetaDeviceFromInitContext:
    def test_removes_meta_device(self):
        contexts = [torch.device("meta"), torch.float32]
        result = _filter_meta_device_from_init_context(contexts)
        assert torch.device("meta") not in result
        assert torch.float32 in result

    def test_keeps_non_meta_devices(self):
        contexts = [torch.device("cpu"), torch.device("cuda")]
        result = _filter_meta_device_from_init_context(contexts)
        assert len(result) == 2

    def test_empty_list(self):
        assert _filter_meta_device_from_init_context([]) == []


class TestPatchedGetInitContext:
    def test_forwards_extra_args(self):
        """Verify _patched_get_init_context forwards *args/**kwargs (transformers v5.3.0 compat)."""
        received_args = {}

        def mock_original(cls, dtype, is_quantized, _is_ds_init_called, *args, **kwargs):
            received_args["args"] = args
            received_args["kwargs"] = kwargs
            return []

        with patch.object(_patched_get_init_context, "__wrapped__", mock_original):
            _patched_get_init_context(None, torch.float32, False, False, True, extra_kwarg="test")

        assert received_args["args"] == (True,)
        assert received_args["kwargs"] == {"extra_kwarg": "test"}

    def test_forwards_allow_all_kernels(self):
        """Simulate the exact transformers v5.3.0 call with allow_all_kernels param."""
        received_args = {}

        def mock_original(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels):
            received_args["allow_all_kernels"] = allow_all_kernels
            return []

        with patch.object(_patched_get_init_context, "__wrapped__", mock_original):
            _patched_get_init_context(None, torch.float32, False, False, None)

        assert received_args["allow_all_kernels"] is None

    def test_strips_meta_device_when_disabled(self):
        """When no_hf_meta_device context is active, meta devices are filtered out."""

        def mock_original(cls, dtype, is_quantized, _is_ds_init_called, *args, **kwargs):
            return [torch.device("meta"), torch.float32]

        with patch.object(_patched_get_init_context, "__wrapped__", mock_original):
            with no_hf_meta_device():
                result = _patched_get_init_context(None, torch.float32, False, False)
            assert torch.device("meta") not in result
            assert torch.float32 in result

    def test_keeps_meta_device_by_default(self):
        """Without no_hf_meta_device, meta devices are preserved."""

        def mock_original(cls, dtype, is_quantized, _is_ds_init_called, *args, **kwargs):
            return [torch.device("meta"), torch.float32]

        with patch.object(_patched_get_init_context, "__wrapped__", mock_original):
            result = _patched_get_init_context(None, torch.float32, False, False)
        assert torch.device("meta") in result

    def test_patch_installed_on_pretrained_model(self):
        """Verify the patch is actually installed on PreTrainedModel."""
        assert PreTrainedModel.get_init_context.__func__ is _patched_get_init_context


class TestNoHfMetaDevice:
    def test_context_manager_sets_and_restores(self):
        from nemo_automodel._transformers.model_init import _get_hf_meta_device_disabled

        assert not _get_hf_meta_device_disabled()
        with no_hf_meta_device():
            assert _get_hf_meta_device_disabled()
        assert not _get_hf_meta_device_disabled()

    def test_nested_context_managers(self):
        from nemo_automodel._transformers.model_init import _get_hf_meta_device_disabled

        with no_hf_meta_device():
            assert _get_hf_meta_device_disabled()
            with no_hf_meta_device():
                assert _get_hf_meta_device_disabled()
            assert _get_hf_meta_device_disabled()
        assert not _get_hf_meta_device_disabled()
