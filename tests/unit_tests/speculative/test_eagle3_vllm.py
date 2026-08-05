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


def _patch_runner_init_deps(monkeypatch, hf_config):
    """Stub out the CUDA + HF-config touch points in ``VLLMTargetRunner.__init__``.

    ``__init__`` reads the HF config, allocates a CUDA device marker via the model
    shim, and queries ``torch.cuda.current_device()`` -- none of which work on the
    CPU editing host. Patch all three so the pure-Python wiring can be exercised.
    Returns a dict the model-shim stub records its ``(config, device)`` into.
    """
    import transformers

    from nemo_automodel.components.speculative.eagle import vllm_runner

    recorded = {}

    class _StubShim:
        def __init__(self, config, device):
            recorded["config"] = config
            recorded["device"] = device

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: hf_config)
    monkeypatch.setattr(vllm_runner, "_VLLMModelShim", _StubShim)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return recorded


def test_vllm_target_runner_init_parses_config_and_defaults(monkeypatch):
    """``__init__`` reads dims off the HF config and picks safe lazy defaults."""
    from nemo_automodel.components.speculative.eagle import vllm_runner

    hf_config = SimpleNamespace(num_hidden_layers=28, hidden_size=3584, vocab_size=152064, rms_norm_eps=1e-5)
    recorded = _patch_runner_init_deps(monkeypatch, hf_config)
    monkeypatch.setenv("TMPDIR", "/scratch/tmp")

    runner = vllm_runner.VLLMTargetRunner("org/model", tp_size=2)

    assert (runner._num_layers, runner._hidden, runner._rms_eps) == (28, 3584, 1e-5)
    assert runner._tp_size == 2
    # The engine + capture layers + cached weights are all built lazily later.
    assert runner._llm is None and runner._aux_layer_ids is None
    assert runner._embed_w is None and runner._norm_w is None and runner._lm_head_w is None
    # No explicit storage path -> rooted under TMPDIR.
    assert runner._shared_storage_path == "/scratch/tmp/vllm_eagle3_hidden_states"
    # The shim carries the HF config and a CUDA device marker.
    assert recorded["config"] is hf_config and recorded["device"].type == "cuda"


def test_vllm_target_runner_init_honors_explicit_args(monkeypatch):
    """Explicit storage path / vllm_kwargs are kept; missing rms eps defaults to 1e-6."""
    from nemo_automodel.components.speculative.eagle import vllm_runner

    hf_config = SimpleNamespace(num_hidden_layers=32, hidden_size=4096, vocab_size=128000)
    _patch_runner_init_deps(monkeypatch, hf_config)

    runner = vllm_runner.VLLMTargetRunner(
        "org/model",
        shared_storage_path="/shared/hs",
        vllm_kwargs={"max_model_len": 2048},
    )

    assert runner._rms_eps == 1e-6  # getattr fallback when the config omits rms_norm_eps
    assert runner._shared_storage_path == "/shared/hs"
    assert runner._vllm_kwargs == {"max_model_len": 2048}


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
