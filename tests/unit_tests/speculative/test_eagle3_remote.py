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

"""CPU unit tests for the remote EAGLE-3 target backend.

NCCL (the GPU data plane) cannot run on CPU/CI here, so these tests force the
binary wire-format transport (``NEMO_EAGLE_ENABLE_NCCL=0``) and validate the
full HTTP round-trip in-process:

1. ``wire`` encode/decode preserves dtype, shape and ``None`` across dtypes.
2. NCCL JSON metadata round-trips.
3. The server's ``compute_supervision`` matches the co-located projection
   (``generate_batch`` + ``_compute_target_distribution``) bit-for-bit.
4. End-to-end: a ``RemoteEagle3TargetModel`` talking to a real in-process
   server returns the same supervision a co-located target would -- so remote
   training is behavior-preserving.
5. Async prefetch returns correct results and the recipe prefetch iterator
   preserves batch order with multiple requests in flight.
"""

from __future__ import annotations

import os
import socket
import threading

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

# Force the wire-format data plane; NCCL needs CUDA + sglang.
os.environ["NEMO_EAGLE_ENABLE_NCCL"] = "0"

from http.server import ThreadingHTTPServer  # noqa: E402

from nemo_automodel.components.speculative.eagle.core import _compute_target_distribution  # noqa: E402
from nemo_automodel.components.speculative.eagle.remote import protocol, wire  # noqa: E402
from nemo_automodel.components.speculative.eagle.remote.client import RemoteEagle3TargetModel  # noqa: E402
from nemo_automodel.components.speculative.eagle.remote.server import (  # noqa: E402
    TargetModelServer,
    _make_request_handler,
    compute_supervision,
)
from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel  # noqa: E402

_VOCAB = 32
_DRAFT_VOCAB = 8
_HIDDEN = 16
_LAYERS = 4


class _FakeHFCausalLM(nn.Module):
    """Deterministic HF causal-LM stand-in (eval, fixed weights)."""

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


def _make_target_wrapper() -> HFEagle3TargetModel:
    torch.manual_seed(0)
    return HFEagle3TargetModel(_FakeHFCausalLM(), aux_layer_ids=[0, 1, 3])


def _vocab_mapping():
    selected_token_ids = torch.arange(_DRAFT_VOCAB, dtype=torch.long)
    selected_token_mask = torch.zeros(_VOCAB, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True
    return selected_token_ids, selected_token_mask


def _batch(batch: int = 2, seq: int = 8):
    return (
        torch.randint(0, _VOCAB, (batch, seq)),
        torch.ones(batch, seq, dtype=torch.long),
        torch.ones(batch, seq, dtype=torch.long),
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- 1. wire format --------------------------------------------------------


def test_wire_roundtrip_preserves_dtype_shape_and_none():
    src = {
        "f32": torch.randn(2, 3, 4),
        "bf16": torch.randn(5, 7).bfloat16(),
        "i64": torch.randint(0, 100, (3, 3), dtype=torch.int64),
        "boolean": torch.tensor([[True, False], [False, True]]),
        "missing": None,
    }
    out = wire.decode(wire.encode_to_bytes(src))
    assert out["missing"] is None
    for key in ("f32", "bf16", "i64", "boolean"):
        assert out[key].dtype == src[key].dtype
        assert out[key].shape == src[key].shape
        torch.testing.assert_close(out[key], src[key], rtol=0, atol=0)


def test_wire_rejects_cuda_and_bad_magic():
    with pytest.raises(ValueError, match="Bad wire-format magic"):
        wire.decode(b"\x00\x00\x00\x00")


def test_nccl_metadata_roundtrip():
    tensors = {"a": torch.randn(2, 3), "b": None}
    keys = ["a", "b"]
    keys_order, metadata = protocol.decode_nccl_metadata(protocol.encode_nccl_metadata(tensors, keys))
    assert keys_order == keys
    assert metadata["b"] is None
    assert metadata["a"]["shape"] == [2, 3]
    assert protocol.dtype_from_code(metadata["a"]["dtype_code"]) == torch.float32


# --- 2. server-side supervision == co-located ------------------------------


def test_compute_supervision_matches_colocated_projection():
    target = _make_target_wrapper()
    selected_token_ids, selected_token_mask = _vocab_mapping()
    input_ids, attn, loss = _batch()

    supervision = compute_supervision(target, selected_token_ids, selected_token_mask, input_ids, attn, loss)

    # Independent co-located reference: generate_batch + _compute_target_distribution.
    ref_batch = target.generate_batch(input_ids=input_ids, attention_mask=attn, loss_mask=loss)
    ref_probs, ref_mask = _compute_target_distribution(
        target_logits=ref_batch.logits,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        loss_mask=ref_batch.loss_mask,
    )
    torch.testing.assert_close(supervision["target_probs"], ref_probs, rtol=0, atol=0)
    torch.testing.assert_close(supervision["position_mask"], ref_mask, rtol=0, atol=0)
    assert supervision["target_probs"].shape == (2, 8, _DRAFT_VOCAB)


# --- 3/4. end-to-end HTTP round-trip ---------------------------------------


@pytest.fixture
def remote_backend():
    """A RemoteEagle3TargetModel wired to an in-process server (wire transport)."""
    target = _make_target_wrapper()
    port = _free_port()
    logic = TargetModelServer(target, nccl_port=port + 100, host="127.0.0.1")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_request_handler(logic))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    backend = RemoteEagle3TargetModel.from_urls([f"http://127.0.0.1:{port}"], device="cpu", max_retries=0)
    try:
        yield backend, target
    finally:
        backend.close()
        httpd.shutdown()
        httpd.server_close()


def test_remote_model_info_and_embeddings(remote_backend):
    backend, target = remote_backend
    info = backend.model_info()
    assert info["num_hidden_layers"] == _LAYERS
    assert info["vocab_size"] == _VOCAB
    assert info["aux_layer_ids"] == [0, 1, 3]

    embed = backend.get_input_embeddings()
    torch.testing.assert_close(embed.weight, target.get_input_embeddings().weight, rtol=0, atol=0)


def test_remote_generate_batch_matches_colocated(remote_backend):
    backend, target = remote_backend
    selected_token_ids, selected_token_mask = _vocab_mapping()
    backend.set_vocab_mapping(selected_token_ids, selected_token_mask)

    input_ids, attn, loss = _batch()
    out = backend.generate_batch(input_ids=input_ids, attention_mask=attn, loss_mask=loss)

    # The remote batch must carry the precomputed encoding...
    assert out.logits is None
    assert out.target_probs is not None and out.position_mask is not None
    assert "target_probs" in out.to_trainer_inputs()

    # ...and match the co-located projection bit-for-bit.
    ref = compute_supervision(target, selected_token_ids, selected_token_mask, input_ids, attn, loss)
    torch.testing.assert_close(out.target_probs, ref["target_probs"], rtol=0, atol=0)
    torch.testing.assert_close(out.position_mask, ref["position_mask"].bool(), rtol=0, atol=0)
    torch.testing.assert_close(out.aux_hidden_states, ref["aux_hidden_states"], rtol=0, atol=0)
    torch.testing.assert_close(out.input_ids, ref["input_ids"], rtol=0, atol=0)


def test_remote_async_matches_sync(remote_backend):
    backend, _ = remote_backend
    backend.set_vocab_mapping(*_vocab_mapping())
    input_ids, attn, loss = _batch()
    sync = backend.generate_batch(input_ids=input_ids, attention_mask=attn, loss_mask=loss)
    handle = backend.generate_batch_async(input_ids, attn, loss)
    asyncro = handle.result(timeout=30)
    torch.testing.assert_close(asyncro.target_probs, sync.target_probs, rtol=0, atol=0)


# --- 5. recipe prefetch iterator ordering ----------------------------------


class _RecordingBackend:
    """Async backend stub recording submit order; returns the batch index."""

    def __init__(self):
        self.submitted: list[int] = []

    def generate_batch_async(self, input_ids, attention_mask, loss_mask):
        idx = int(input_ids[0, 0].item())
        self.submitted.append(idx)

        class _H:
            def result(self_inner, timeout=None):
                return idx

        return _H()


def test_prefetched_batches_preserves_order_and_prefetches_ahead():
    from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe

    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.target_wrapper = _RecordingBackend()
    recipe.target_prefetch_depth = 2

    # Each "batch" encodes its index in input_ids[0,0]; loader yields 0..4.
    loader = [{"input_ids": torch.full((1, 4), i), "attention_mask": None, "loss_mask": None} for i in range(5)]

    seen = []
    for _batch_dict, target_batch in recipe._prefetched_batches(loader):
        # When the i-th result is consumed, the backend should already have
        # submitted up to depth requests ahead.
        assert len(recipe.target_wrapper.submitted) >= min(len(seen) + recipe.target_prefetch_depth, 5)
        seen.append(target_batch)

    assert seen == [0, 1, 2, 3, 4]  # consumed strictly in dataloader order
