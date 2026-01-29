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
import types

import pytest

import nemo_automodel.components.models.biencoder.biencoder_model as bm


class DummyModel:
    def __init__(self):
        self.config = {}
        self.marker = []


def test_from_pretrained_happy_path(monkeypatch):
    calls = {"build": 0, "liger": 0, "sdpa": 0}
    last_kwargs = {}

    def fake_build(**kwargs):
        calls["build"] += 1
        nonlocal last_kwargs
        last_kwargs = kwargs
        return DummyModel()

    def fake_liger(model):
        calls["liger"] += 1
        model.marker.append("liger")
        return model

    def fake_sdpa(model, method):
        calls["sdpa"] += 1
        model.marker.append("sdpa")
        return model

    monkeypatch.setattr(bm.BiencoderModel, "build", staticmethod(fake_build))
    monkeypatch.setattr(bm, "_patch_liger_kernel", fake_liger)
    monkeypatch.setattr(bm, "_patch_attention", fake_sdpa)

    model = bm.NeMoAutoModelBiencoder.from_pretrained(
        pretrained_model_name_or_path="some/path",
        share_encoder=True,
        add_linear_pooler=False,
        out_dimension=None,
        do_gradient_checkpointing=True,
        train_n_passages=4,
        eval_negative_size=2,
        pooling="avg",
        l2_normalize=True,
        t=0.5,
        use_liger_kernel=True,
        use_sdpa_patching=True,
        sdpa_method=None,
        some_other_kwarg="x",
    )
    assert isinstance(model, DummyModel)
    # Patches applied
    assert "liger" in model.marker and "sdpa" in model.marker
    # Config annotated
    assert "nemo_version" in model.config
    # Ensure HF kwargs injected + passthrough of parameters to build
    assert last_kwargs["attn_implementation"] == "flash_attention_2"
    assert last_kwargs["train_n_passages"] == 4
    assert last_kwargs["eval_negative_size"] == 2
    assert last_kwargs["out_dimension"] == 768  # default when None
    assert last_kwargs["some_other_kwarg"] == "x"


def test_from_pretrained_retries_without_liger(monkeypatch):
    calls = {"build": 0, "liger": 0, "sdpa": 0}

    def fake_build(**kwargs):
        calls["build"] += 1
        return DummyModel()

    def fake_liger(_):
        calls["liger"] += 1
        raise RuntimeError("liger failed")

    def fake_sdpa(model, _):
        calls["sdpa"] += 1
        return model

    monkeypatch.setattr(bm.BiencoderModel, "build", staticmethod(fake_build))
    monkeypatch.setattr(bm, "_patch_liger_kernel", fake_liger)
    monkeypatch.setattr(bm, "_patch_attention", fake_sdpa)

    model = bm.NeMoAutoModelBiencoder.from_pretrained("x", use_liger_kernel=True, use_sdpa_patching=True)
    assert isinstance(model, DummyModel)
    # First attempt calls liger once, then retries without it (so only 1 liger call)
    assert calls["liger"] == 1
    # Build called twice (initial + retry)
    assert calls["build"] == 2
    # SDPA patch applied on retry
    assert calls["sdpa"] == 1


def test_from_pretrained_retries_without_sdpa(monkeypatch):
    calls = {"build": 0, "liger": 0, "sdpa": 0}

    def fake_build(**kwargs):
        calls["build"] += 1
        return DummyModel()

    def fake_liger(model):
        calls["liger"] += 1
        return model

    def fake_sdpa(_model, _method):
        calls["sdpa"] += 1
        raise Exception("sdpa failed")

    monkeypatch.setattr(bm.BiencoderModel, "build", staticmethod(fake_build))
    monkeypatch.setattr(bm, "_patch_liger_kernel", fake_liger)
    monkeypatch.setattr(bm, "_patch_attention", fake_sdpa)

    model = bm.NeMoAutoModelBiencoder.from_pretrained("x", use_liger_kernel=True, use_sdpa_patching=True)
    assert isinstance(model, DummyModel)
    # SDPA attempted once then retried without it (no second SDPA call)
    assert calls["sdpa"] == 1
    # Build twice (initial + retry)
    assert calls["build"] == 2
    # Liger called only on the first attempt of each build; second attempt still calls liger
    # but since use_liger_kernel remains True for this path, ensure it was called twice.
    assert calls["liger"] == 2
