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

"""CPU unit tests for the SGLang EAGLE-3 target backend (contract layer).

The SGLang forward itself needs a GPU + SGLang and is validated on the server;
here we fake the runner and verify the parts that must hold regardless of
engine:

1. The supervision contract matches the co-located HF backend bit-for-bit when
   the runner returns the same raw logits / aux states, so an SGLang run is
   numerically equivalent to a co-located one.
2. aux-layer defaulting / validation is shared with the HF backend.
3. ``get_input_embeddings`` / ``set_aux_layers`` / ``close`` are wired through.
4. ``serve_target`` routes ``--engine`` to the right builder.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.speculative.eagle.sglang_target import SGLangEagle3TargetModel
from nemo_automodel.components.speculative.eagle.target import (
    HFEagle3TargetModel,
    default_eagle3_aux_layer_ids,
    validate_eagle3_aux_layer_ids,
)
from nemo_automodel.components.speculative.eagle.target_runner import RunnerEagle3TargetModel

_VOCAB = 32
_HIDDEN = 16
_LAYERS = 4
_AUX = [0, 1, 3]


class _FakeCausalLM(nn.Module):
    """Deterministic causal-LM stand-in shared by the HF and fake-SGLang paths."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": _LAYERS, "hidden_size": _HIDDEN, "vocab_size": _VOCAB})
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.layers = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN) for _ in range(_LAYERS)])
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return CausalLMOutput(logits=self.lm_head(h))


class _FakeSGLangRunner:
    """Fake runner: returns the *unshifted* logits / aux a real runner would.

    It reuses ``_FakeCausalLM`` so the raw tensors are exactly what the HF
    backend captures, letting the test assert engine-equivalence.
    """

    def __init__(self, model: _FakeCausalLM):
        self.model = model
        self.aux_layer_ids = None
        self.closed = False

    def set_aux_layers(self, aux_layer_ids):
        self.aux_layer_ids = list(aux_layer_ids)

    def input_embedding_weight(self):
        return self.model.get_input_embeddings().weight

    def forward_eagle3(self, input_ids, attention_mask):
        captured = {}
        handles = [
            self.model.layers[i].register_forward_hook(
                lambda _m, _i, out, i=i: captured.__setitem__(i, out)
            )
            for i in self.aux_layer_ids
        ]
        try:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        finally:
            for h in handles:
                h.remove()
        aux = torch.cat([captured[i] for i in self.aux_layer_ids], dim=-1)
        return logits, aux

    def close(self):
        self.closed = True


def _inputs():
    torch.manual_seed(1)
    input_ids = torch.randint(0, _VOCAB, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, loss_mask


def test_generate_batch_matches_colocated_hf():
    """SGLang supervision is bit-for-bit identical to the co-located HF path."""
    torch.manual_seed(0)
    model = _FakeCausalLM().eval()
    hf = HFEagle3TargetModel(model, aux_layer_ids=_AUX)
    sgl = SGLangEagle3TargetModel(_FakeSGLangRunner(model), aux_layer_ids=_AUX)

    input_ids, attention_mask, loss_mask = _inputs()
    hf_batch = hf.generate_batch(input_ids, attention_mask, loss_mask)
    sgl_batch = sgl.generate_batch(input_ids, attention_mask, loss_mask)

    torch.testing.assert_close(sgl_batch.logits, hf_batch.logits)
    torch.testing.assert_close(sgl_batch.aux_hidden_states, hf_batch.aux_hidden_states)
    torch.testing.assert_close(sgl_batch.input_ids, hf_batch.input_ids)
    torch.testing.assert_close(sgl_batch.loss_mask, hf_batch.loss_mask)
    # The batch carries full logits (projection happens server-side), never the
    # precomputed encoding.
    assert sgl_batch.target_probs is None and sgl_batch.position_mask is None


def test_engine_agnostic_backend_is_shared():
    """The contract layer is engine-agnostic: the same backend works on any runner.

    ``SGLangEagle3TargetModel`` only adds SGLang construction; the supervision
    contract lives in ``RunnerEagle3TargetModel`` and a vLLM runner can reuse it
    by implementing the same ``TargetRunner`` surface.
    """
    assert issubclass(SGLangEagle3TargetModel, RunnerEagle3TargetModel)

    model = _FakeCausalLM().eval()
    runner = _FakeSGLangRunner(model)
    base = RunnerEagle3TargetModel(runner, aux_layer_ids=_AUX)
    sgl = SGLangEagle3TargetModel(_FakeSGLangRunner(model), aux_layer_ids=_AUX)

    input_ids, attention_mask, loss_mask = _inputs()
    base_batch = base.generate_batch(input_ids, attention_mask, loss_mask)
    sgl_batch = sgl.generate_batch(input_ids, attention_mask, loss_mask)
    torch.testing.assert_close(base_batch.logits, sgl_batch.logits)
    torch.testing.assert_close(base_batch.aux_hidden_states, sgl_batch.aux_hidden_states)


def test_shift_semantics():
    """logits / input_ids / loss_mask shift left by one; aux stays aligned."""
    model = _FakeCausalLM().eval()
    runner = _FakeSGLangRunner(model)
    sgl = SGLangEagle3TargetModel(runner, aux_layer_ids=_AUX)
    input_ids, attention_mask, loss_mask = _inputs()
    raw_logits, raw_aux = runner.forward_eagle3(input_ids, attention_mask)

    batch = sgl.generate_batch(input_ids, attention_mask, loss_mask)
    torch.testing.assert_close(batch.input_ids[:, :-1], input_ids[:, 1:])
    assert torch.all(batch.input_ids[:, -1] == 0)
    torch.testing.assert_close(batch.logits[:, :-1], raw_logits[:, 1:])
    # aux is position-aligned (not shifted).
    torch.testing.assert_close(batch.aux_hidden_states, raw_aux)


def test_default_aux_layer_ids_applied_and_forwarded():
    """When aux_layer_ids is None the shared default recipe is used and set on the runner."""

    class _Deep(_FakeCausalLM):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"num_hidden_layers": 32, "hidden_size": _HIDDEN, "vocab_size": _VOCAB})

    runner = _FakeSGLangRunner(_Deep())
    sgl = SGLangEagle3TargetModel(runner, aux_layer_ids=None)
    assert sgl.aux_layer_ids == default_eagle3_aux_layer_ids(32) == [1, 15, 28]
    assert runner.aux_layer_ids == sgl.aux_layer_ids


def test_invalid_aux_layer_ids_rejected():
    runner = _FakeSGLangRunner(_FakeCausalLM())
    with pytest.raises(ValueError, match="exactly 3"):
        SGLangEagle3TargetModel(runner, aux_layer_ids=[0, 1])
    with pytest.raises(ValueError, match="out of bounds"):
        SGLangEagle3TargetModel(runner, aux_layer_ids=[0, 1, 99])


def test_default_recipe_raises_on_shallow_target():
    # num_layers=4 -> [1, 1, 0] has a duplicate, so the default recipe must raise.
    runner = _FakeSGLangRunner(_FakeCausalLM())
    with pytest.raises(ValueError, match="too shallow"):
        SGLangEagle3TargetModel(runner, aux_layer_ids=None)


def test_input_embeddings_and_close():
    model = _FakeCausalLM().eval()
    runner = _FakeSGLangRunner(model)
    sgl = SGLangEagle3TargetModel(runner, aux_layer_ids=_AUX)
    torch.testing.assert_close(sgl.get_input_embeddings().weight, model.get_input_embeddings().weight)
    sgl.close()
    assert runner.closed


def test_shared_validate_helper_matches_hf():
    """The extracted helpers back both backends, so HF behavior is preserved."""
    model = _FakeCausalLM().eval()
    hf = HFEagle3TargetModel(model, aux_layer_ids=_AUX)
    assert hf.aux_layer_ids == validate_eagle3_aux_layer_ids(_AUX, _LAYERS) == _AUX


def test_serve_target_engine_routing(monkeypatch):
    from nemo_automodel.components.speculative import serve_target

    calls = {}

    def _record(engine):
        def _builder(*_a, **_k):
            calls["engine"] = engine
        return _builder

    monkeypatch.setattr(serve_target, "_build_hf_target", _record("hf"))
    monkeypatch.setattr(serve_target, "_build_sglang_target", _record("sglang"))
    monkeypatch.setattr(serve_target, "TargetModelServer", lambda *a, **k: object())
    monkeypatch.setattr(serve_target, "serve", lambda *a, **k: None)

    serve_target.main(["--target", "x", "--engine", "sglang", "--tp-size", "2"])
    assert calls["engine"] == "sglang"

    serve_target.main(["--target", "x"])
    assert calls["engine"] == "hf"


def test_serve_target_arg_defaults():
    from nemo_automodel.components.speculative import serve_target

    args = serve_target._parse_args(["--target", "x"])
    assert args.engine == "hf" and args.tp_size == 1
    args = serve_target._parse_args(["--target", "x", "--engine", "sglang", "--tp-size", "4"])
    assert args.engine == "sglang" and args.tp_size == 4


def test_build_sglang_target_delegates(monkeypatch):
    """``_build_sglang_target`` forwards args to the backend's from_pretrained."""
    from nemo_automodel.components.speculative import serve_target
    from nemo_automodel.components.speculative.eagle import sglang_target

    captured = {}

    def _fake_from_pretrained(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return "wrapper"

    monkeypatch.setattr(sglang_target.SGLangEagle3TargetModel, "from_pretrained", staticmethod(_fake_from_pretrained))
    args = serve_target._parse_args(["--target", "org/m", "--engine", "sglang", "--tp-size", "2"])
    result = serve_target._build_sglang_target(args, torch.device("cpu"), torch.float32)
    assert result == "wrapper"
    assert captured["model_path"] == "org/m" and captured["tp_size"] == 2


# ── SGLangTargetRunner: surface that does not need SGLang (the forward itself
#    needs a GPU and is covered on the server) ──────────────────────────────


class _FakeModelRunner:
    """Stands in for SGLang's ModelRunner for the non-forward runner methods."""

    def __init__(self):
        self.model = _FakeCausalLM().eval()
        self.model.set_eagle3_layers_to_capture = lambda ids: setattr(self, "captured_layers", list(ids))
        self.req_to_token_pool = types.SimpleNamespace(cleared=False)
        self.req_to_token_pool.clear = lambda: setattr(self.req_to_token_pool, "cleared", True)
        self.token_to_kv_pool_allocator = types.SimpleNamespace(cleared=False)
        self.token_to_kv_pool_allocator.clear = lambda: setattr(self.token_to_kv_pool_allocator, "cleared", True)


def test_runner_surface_without_sglang():
    from nemo_automodel.components.speculative.eagle.sglang_runner import SGLangTargetRunner

    mr = _FakeModelRunner()
    runner = SGLangTargetRunner(mr)
    assert runner.model is mr.model

    runner.set_aux_layers([0, 1, 3])
    assert mr.captured_layers == [0, 1, 3]
    torch.testing.assert_close(runner.input_embedding_weight(), mr.model.get_input_embeddings().weight)

    runner.close()
    assert mr.req_to_token_pool.cleared and mr.token_to_kv_pool_allocator.cleared
    runner.close()  # idempotent after the runner is released


def test_runner_forward_stacks_per_row(monkeypatch):
    """forward_eagle3 stacks the per-row extend outputs into batched tensors."""
    from nemo_automodel.components.speculative.eagle.sglang_runner import SGLangTargetRunner

    runner = SGLangTargetRunner(_FakeModelRunner())
    rows_logits = [torch.randn(5, _VOCAB), torch.randn(5, _VOCAB)]
    rows_aux = [torch.randn(5, 3 * _HIDDEN), torch.randn(5, 3 * _HIDDEN)]
    monkeypatch.setattr(runner, "_extend", lambda input_ids: (rows_logits, rows_aux))

    logits, aux = runner.forward_eagle3(torch.zeros(2, 5, dtype=torch.long), torch.ones(2, 5))
    assert logits.shape == (2, 5, _VOCAB) and aux.shape == (2, 5, 3 * _HIDDEN)
    torch.testing.assert_close(logits, torch.stack(rows_logits))
    torch.testing.assert_close(aux, torch.stack(rows_aux))
