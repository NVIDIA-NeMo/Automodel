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

"""CPU unit tests for the vLLM EAGLE-3 target backend surface.

The real vLLM forward needs a GPU and is validated on the server; here we pin the
parts that do not need vLLM: the engine-agnostic backend wiring, the dtype
mapping, the model shim, and ``serve_target``'s ``--engine vllm`` routing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.speculative.eagle.target_runner import RunnerEagle3TargetModel
from nemo_automodel.components.speculative.eagle.vllm_target import VLLMEagle3TargetModel


def test_engine_agnostic_backend_is_shared():
    """VLLMEagle3TargetModel reuses the shared RunnerEagle3TargetModel contract."""
    assert issubclass(VLLMEagle3TargetModel, RunnerEagle3TargetModel)


def test_vllm_dtype_str_mappings():
    """vLLM's ``dtype`` argument is a string; torch dtypes must map to it."""
    from nemo_automodel.components.speculative.eagle.vllm_runner import vllm_dtype_str

    assert vllm_dtype_str(None) == "auto"
    assert vllm_dtype_str(torch.float32) == "float32"
    assert vllm_dtype_str(torch.float16) == "float16"
    assert vllm_dtype_str(torch.bfloat16) == "bfloat16"


def test_vllm_dtype_str_rejects_unsupported():
    from nemo_automodel.components.speculative.eagle.vllm_runner import vllm_dtype_str

    with pytest.raises(ValueError, match="Unsupported vLLM target dtype"):
        vllm_dtype_str(torch.int8)


def test_vllm_model_shim_exposes_config_and_device():
    """The shim carries the HF config and a device-marker parameter."""
    from nemo_automodel.components.speculative.eagle.vllm_runner import _VLLMModelShim

    config = SimpleNamespace(num_hidden_layers=36, hidden_size=2560, vocab_size=151936)
    shim = _VLLMModelShim(config, torch.device("cpu"))
    assert shim.config is config
    params = list(shim.parameters())
    assert len(params) == 1 and params[0].device.type == "cpu"


def test_serve_target_arg_defaults_includes_vllm():
    from nemo_automodel.components.speculative import serve_target

    args = serve_target._parse_args(["--target", "x", "--engine", "vllm", "--tp-size", "4"])
    assert args.engine == "vllm" and args.tp_size == 4


def test_serve_target_routes_vllm(monkeypatch):
    from nemo_automodel.components.speculative import serve_target

    calls = {}
    monkeypatch.setattr(serve_target, "_build_vllm_target", lambda *_a, **_k: calls.setdefault("engine", "vllm"))
    monkeypatch.setattr(serve_target, "TargetModelServer", lambda *a, **k: object())
    monkeypatch.setattr(serve_target, "serve", lambda *a, **k: None)

    serve_target.main(["--target", "x", "--engine", "vllm"])
    assert calls["engine"] == "vllm"


def test_build_vllm_target_delegates(monkeypatch):
    """``_build_vllm_target`` forwards args to the backend's from_pretrained."""
    from nemo_automodel.components.speculative import serve_target
    from nemo_automodel.components.speculative.eagle import vllm_target

    captured = {}

    def _fake_from_pretrained(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return "wrapper"

    monkeypatch.setattr(vllm_target.VLLMEagle3TargetModel, "from_pretrained", staticmethod(_fake_from_pretrained))
    args = serve_target._parse_args(["--target", "org/m", "--engine", "vllm", "--tp-size", "2"])
    result = serve_target._build_vllm_target(args, torch.device("cpu"), torch.float32)
    assert result == "wrapper"
    assert captured["model_path"] == "org/m" and captured["tp_size"] == 2
