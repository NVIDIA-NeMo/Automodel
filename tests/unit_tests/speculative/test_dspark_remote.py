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

"""CPU unit tests for the remote DSpark target backend.

NCCL cannot run here, so the client and server use the binary wire format
(``NEMO_EAGLE_ENABLE_NCCL=0``) over a real in-process HTTP server. The core
contract: a remote batch equals what :class:`HFDSparkTargetModel` captures
co-located on the same target, bit for bit.
"""

from __future__ import annotations

import os
import socket
import threading
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

# Force the wire-format data plane; NCCL needs CUDA + sglang.
os.environ["NEMO_EAGLE_ENABLE_NCCL"] = "0"

from http.server import ThreadingHTTPServer  # noqa: E402

from nemo_automodel.components.speculative.dspark.remote import (  # noqa: E402
    DSparkTargetInfo,
    DSparkTargetModelServer,
    RemoteDSparkTargetModel,
)
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel  # noqa: E402
from nemo_automodel.components.speculative.eagle.remote import protocol, wire  # noqa: E402
from nemo_automodel.components.speculative.eagle.remote.server import (  # noqa: E402
    TargetModelServer,
    _make_request_handler,
)

VOCAB = 128
HIDDEN = 32
NUM_LAYERS = 4
LAYER_IDS = [0, 2, 3]


def _tiny_target() -> HFDSparkTargetModel:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32).eval()
    return HFDSparkTargetModel(model, target_layer_ids=LAYER_IDS)


def _info(**overrides) -> DSparkTargetInfo:
    fields = dict(num_hidden_layers=NUM_LAYERS, hidden_size=HIDDEN, vocab_size=VOCAB, target_layer_ids=tuple(LAYER_IDS))
    fields.update(overrides)
    return DSparkTargetInfo(**fields)


def _batch(batch: int = 2, seq: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(input_ids, attention_mask, loss_mask)``, each of shape [batch, sequence]."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, VOCAB, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    attention_mask[1, -2:] = 0  # right padding on the second row
    loss_mask = attention_mask.clone()
    return input_ids, attention_mask, loss_mask


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """An in-process DSpark target server on a free localhost port."""

    def __init__(self, target: HFDSparkTargetModel):
        self.port = _free_port()
        self.logic = DSparkTargetModelServer(target, target.model.config, nccl_port=self.port + 100, host="127.0.0.1")
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), _make_request_handler(self.logic))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def served_target():
    target = _tiny_target()
    server = _Server(target)
    try:
        yield server, target
    finally:
        server.stop()


# --- DSparkTargetInfo -------------------------------------------------------


def test_target_info_reports_mismatched_fields():
    assert _info().mismatched_fields(_info()) == []
    assert _info().mismatched_fields(_info(hidden_size=64, target_layer_ids=(1, 2, 3))) == [
        "hidden_size",
        "target_layer_ids",
    ]


# --- server logic -----------------------------------------------------------


def test_server_nccl_path_sends_the_supervision_in_dict_order():
    target = _tiny_target()
    logic = DSparkTargetModelServer(target, target.model.config, nccl_port=1)
    logic._nccl = SimpleNamespace(is_initialized=True)
    input_ids, attention_mask, loss_mask = _batch()

    body, used_nccl = logic.handle_generate(
        wire.encode_to_bytes({"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}),
        client_wants_nccl=True,
    )
    keys_order, metadata = protocol.decode_nccl_metadata(body)

    assert used_nccl
    assert keys_order == ["target_hidden_states", "target_last_hidden_states"]
    assert metadata["target_last_hidden_states"]["shape"] == [2, 8, HIDDEN]
    pending_tensors, pending_keys = logic._pending_nccl_send
    assert pending_keys == keys_order
    assert list(pending_tensors) == keys_order


def test_only_the_dspark_server_routes_output_embeddings():
    target = _tiny_target()
    dspark_routes = DSparkTargetModelServer(target, target.model.config, nccl_port=1).post_routes()
    assert protocol.EP_OUTPUT_EMBEDDINGS in dspark_routes
    assert protocol.EP_OUTPUT_EMBEDDINGS not in TargetModelServer(target, nccl_port=1).post_routes()


# --- end-to-end HTTP round trip --------------------------------------------


def test_remote_target_info_check_passes_for_matching_config(served_target):
    server, _ = served_target
    backend = RemoteDSparkTargetModel([server.url], device=torch.device("cpu"), max_retries=0)
    backend.check_target_info(_info())


def test_remote_target_info_check_names_server_and_fields(served_target):
    server, _ = served_target
    backend = RemoteDSparkTargetModel([server.url], device=torch.device("cpu"), max_retries=0)
    with pytest.raises(ValueError, match=rf"{server.url}.*\['num_hidden_layers', 'target_layer_ids'\]"):
        backend.check_target_info(_info(num_hidden_layers=28, target_layer_ids=(1, 7, 14)))


def test_remote_target_info_check_rejects_a_non_dspark_server(served_target, monkeypatch):
    server, _ = served_target
    backend = RemoteDSparkTargetModel([server.url], device=torch.device("cpu"), max_retries=0)
    eagle3_info = b'{"num_hidden_layers": 4, "hidden_size": 32, "vocab_size": 128, "aux_layer_ids": [0, 1, 3]}'
    monkeypatch.setattr(backend._pool.clients[0], "request", lambda *_args, **_kwargs: eagle3_info)

    with pytest.raises(ValueError, match="did not describe a DSpark target"):
        backend.check_target_info(_info())


def test_remote_embeddings_match_target_weights(served_target):
    server, target = served_target
    backend = RemoteDSparkTargetModel([server.url], device=torch.device("cpu"), max_retries=0)

    embed = backend.get_input_embeddings()
    head = backend.get_output_embeddings()

    torch.testing.assert_close(embed.weight, target.get_input_embeddings().weight, rtol=0, atol=0)
    torch.testing.assert_close(head.weight, target.get_output_embeddings().weight, rtol=0, atol=0)
    # Kept on CPU: the draft copies them once, so no GPU copy outlives setup.
    assert embed.weight.device.type == head.weight.device.type == "cpu"


def test_remote_generate_batch_equals_colocated(served_target):
    server, target = served_target
    backend = RemoteDSparkTargetModel([server.url], device=torch.device("cpu"), max_retries=0)
    input_ids, attention_mask, loss_mask = _batch()

    remote = backend.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)
    reference = target.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)

    torch.testing.assert_close(remote.target_hidden_states, reference.target_hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(remote.target_last_hidden_states, reference.target_last_hidden_states, rtol=0, atol=0)
    assert remote.input_ids is input_ids
    assert remote.loss_mask is loss_mask
    assert remote.position_ids is None and remote.seq_lens is None and remote.doc_remaining is None


def test_remote_requests_alternate_across_servers():
    targets = [_tiny_target(), _tiny_target()]
    servers = [_Server(target) for target in targets]
    calls = []
    for index, server in enumerate(servers):
        supervise = server.logic._supervise
        server.logic._supervise = lambda tensors, _i=index, _f=supervise: calls.append(_i) or _f(tensors)
    try:
        backend = RemoteDSparkTargetModel([s.url for s in servers], device=torch.device("cpu"), max_retries=0)
        for _ in range(3):
            backend.generate_batch(*_batch())
    finally:
        for server in servers:
            server.stop()
    assert calls == [0, 1, 0]


def test_close_disconnects_and_signals_server_shutdown(served_target):
    server, _ = served_target
    backend = RemoteDSparkTargetModel([server.url], device=torch.device("cpu"), max_retries=0)
    backend.close()
    assert server.logic.shutdown_event.is_set()
