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

import contextlib
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel import runtime as cp_runtime
from nemo_automodel.components.distributed.context_parallel.runtime import ContextParallelRuntime, CPForward


class _SubMesh:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class _DeviceMesh(dict):
    def __init__(self, cp_size: int):
        super().__init__(cp=_SubMesh(cp_size), tp=_SubMesh(1))
        self.mesh_dim_names = ("cp", "tp")


def _thd_batch() -> dict[str, torch.Tensor | str]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
        "qkv_format": "thd",
        "seq_lens": torch.tensor([[4]]),
        "seq_lens_padded": torch.tensor([[4]]),
    }


def test_build_resolves_backend_from_model_config(monkeypatch):
    model_config = SimpleNamespace(backend=SimpleNamespace(attn="magi"))
    mesh = _DeviceMesh(cp_size=2)
    magi = SimpleNamespace(enabled=True, hf_dispatch=True)
    calls = []

    monkeypatch.setattr(
        cp_runtime,
        "setup_magi",
        lambda config, device_mesh: calls.append((config, device_mesh)) or magi,
    )

    runtime = ContextParallelRuntime.build(model_config, device_mesh=mesh)

    assert calls == [(model_config, mesh)]
    assert runtime.requires_full_logits is True


def test_prepare_forward_derives_thd_from_batch(monkeypatch):
    seen = {}

    def fake_make_cp_batch_for_te(cp_mesh, batch, **kwargs):
        seen.update(cp_mesh=cp_mesh, batch=batch, kwargs=kwargs)
        return {"input_ids": torch.tensor([1, 2])}, torch.tensor([0, 2])

    monkeypatch.setattr(cp_runtime, "_prepare_thd_batch", fake_make_cp_batch_for_te)
    prepared = ContextParallelRuntime(device_mesh=_DeviceMesh(cp_size=2)).prepare_forward(None, _thd_batch())

    assert isinstance(prepared, CPForward)
    assert seen["kwargs"]["num_chunks"] == 1
    assert isinstance(prepared.context, contextlib.nullcontext)
    assert torch.equal(prepared.tokens.shard(torch.arange(4.0), seq_dim=0), torch.tensor([0.0, 2.0]))


def test_prepare_forward_requires_thd_sequence_metadata():
    batch = {"input_ids": torch.tensor([[1, 2]]), "qkv_format": "thd"}

    with pytest.raises(ValueError, match="seq_lens, seq_lens_padded"):
        ContextParallelRuntime().prepare_forward(None, batch)


def test_prepare_forward_returns_identity_token_layout_without_cp():
    tensor = torch.randn(1, 4, 3)
    batch = {"input_ids": torch.ones(1, 4, dtype=torch.long)}

    prepared = ContextParallelRuntime().prepare_forward(None, batch)

    assert prepared.batch is batch
    assert isinstance(prepared.context, contextlib.nullcontext)
    assert torch.equal(prepared.tokens.shard(tensor), tensor)
    assert torch.equal(prepared.tokens.gather(tensor), tensor)


def test_multimodal_model_rejects_thd_with_context_parallelism():
    runtime = ContextParallelRuntime(device_mesh=_DeviceMesh(cp_size=2))
    model = torch.nn.Module()
    model.supports = SimpleNamespace(is_multimodal=True)

    with pytest.raises(NotImplementedError, match="supports cp_size=1 only"):
        runtime.prepare_forward(model, _thd_batch())


def test_multimodal_detection_falls_back_to_model_config():
    runtime = ContextParallelRuntime(device_mesh=_DeviceMesh(cp_size=2))
    model = torch.nn.Module()
    model.supports = SimpleNamespace()
    model.config = SimpleNamespace(vision_config=SimpleNamespace())

    with pytest.raises(NotImplementedError, match="supports cp_size=1 only"):
        runtime.prepare_forward(model, _thd_batch())


def test_text_model_thd_does_not_depend_on_recipe_domain(monkeypatch):
    monkeypatch.setattr(
        cp_runtime,
        "_prepare_thd_batch",
        lambda cp_mesh, batch, **kwargs: (batch, torch.tensor([0, 2])),
    )
    model = torch.nn.Module()
    model.supports = SimpleNamespace(is_multimodal=False)

    prepared = ContextParallelRuntime(device_mesh=_DeviceMesh(cp_size=2)).prepare_forward(model, _thd_batch())

    assert prepared.batch["qkv_format"] == "thd"
