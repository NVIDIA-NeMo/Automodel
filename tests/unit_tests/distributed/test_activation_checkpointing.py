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

"""CPU coverage for the selective-AC save-set build (FFPA fold-in wiring)."""

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo_automodel.components.distributed import activation_checkpointing as ac


def test_unwrap_checkpoint_wrapper_returns_input_module_when_unwrapped():
    module = nn.Linear(2, 2)

    assert ac.unwrap_checkpoint_wrapper(module) is module


def test_unwrap_checkpoint_wrapper_returns_inner_module_when_wrapped():
    module = nn.Linear(2, 2)
    wrapped = checkpoint_wrapper(module)

    assert ac.unwrap_checkpoint_wrapper(wrapped) is module


def test_ffpa_forward_ops_folded_into_save_set(monkeypatch):
    """Ops returned by _ffpa_forward_ops() land in the save set (kernel-free wiring)."""
    dense, varlen = object(), object()
    monkeypatch.setattr(ac, "_ffpa_forward_ops", lambda: (dense, varlen))

    save_ops = ac._build_selective_ac_save_ops()
    assert dense in save_ops
    assert varlen in save_ops


def test_build_save_set_ok_when_ffpa_absent(monkeypatch):
    """CPU degrade path: _ffpa_forward_ops() -> () must not break the build."""
    monkeypatch.setattr(ac, "_ffpa_forward_ops", lambda: ())

    save_ops = ac._build_selective_ac_save_ops()
    assert isinstance(save_ops, frozenset) and len(save_ops) > 0


class _TextConfig:
    """Stand-in for a HF text sub-config that owns ``use_cache``."""

    def __init__(self, num_kv_shared_layers: int = 0):
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_cache = True


class _VisionConfig:
    """Stand-in for a HF vision sub-config without ``use_cache``."""

    def __init__(self):
        self.depth = 2


class _CompositeConfig:
    """Stand-in for a HF composite (VLM) config with declared sub-configs."""

    sub_configs = {"text_config": _TextConfig, "vision_config": _VisionConfig}

    def __init__(self, num_kv_shared_layers: int = 0):
        self.text_config = _TextConfig(num_kv_shared_layers)
        self.vision_config = _VisionConfig()
        self.use_cache = True


def _model_with_config(config) -> nn.Module:
    model = nn.Module()
    model.config = config
    return model


def test_detect_kv_sharing_disables_cache_on_composite_and_text_sub_config():
    """The text model reads ``text_config.use_cache``, so leaving it True keeps a
    DynamicCache alive under checkpointing and recompute double-appends K/V."""
    model = _model_with_config(_CompositeConfig())

    has_kv_sharing = ac.detect_kv_sharing_and_maybe_disable_cache(model)

    assert has_kv_sharing is False
    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False
    # Sub-configs without ``use_cache`` must not grow a phantom field.
    assert not hasattr(model.config.vision_config, "use_cache")


def test_detect_kv_sharing_disables_text_config_cache_without_sub_configs_attr():
    """Configs lacking the ``sub_configs`` class attribute fall back to ``text_config``."""

    class _LegacyComposite:
        def __init__(self):
            self.text_config = _TextConfig()
            self.use_cache = True

    model = _model_with_config(_LegacyComposite())

    assert ac.detect_kv_sharing_and_maybe_disable_cache(model) is False
    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False


def test_detect_kv_sharing_leaves_cache_enabled_for_kv_shared_models():
    """KV-shared models pass K/V between layers through the cache; it must stay on."""
    model = _model_with_config(_CompositeConfig(num_kv_shared_layers=2))

    has_kv_sharing = ac.detect_kv_sharing_and_maybe_disable_cache(model)

    assert has_kv_sharing is True
    assert model.config.use_cache is True
    assert model.config.text_config.use_cache is True
