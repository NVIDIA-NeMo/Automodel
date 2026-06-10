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

"""CPU unit tests for remote EAGLE-3 control-plane logic.

Complements ``test_eagle3_remote.py`` (the end-to-end wire round-trip) by
covering the pieces that the happy-path round-trip does not exercise: the
``serve_target`` CLI wiring, the client's URL/port parsing and HTTP
retry/close paths, and the server handlers that sit off the NCCL data plane.
The NCCL transport itself needs CUDA + a real rendezvous and is out of scope
for CPU tests, so NCCL is forced off here.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import pytest
import requests
import torch

# NCCL (the GPU data plane) cannot run on CPU; force the wire-format path so
# client/server construction never tries to stand up a transport.
os.environ["NEMO_EAGLE_ENABLE_NCCL"] = "0"

from nemo_automodel.components.speculative import serve_target  # noqa: E402
from nemo_automodel.components.speculative.eagle.remote import protocol, wire  # noqa: E402
from nemo_automodel.components.speculative.eagle.remote.client import (  # noqa: E402
    RemoteEagle3TargetModel,
    _AsyncHandle,
    _ServerClient,
)
from nemo_automodel.components.speculative.eagle.remote.server import (  # noqa: E402
    TargetModelServer,
    _make_request_handler,
    serve,
)
from nemo_automodel.components.speculative.eagle.remote.transport import NCCLTransport  # noqa: E402

# ── serve_target CLI ─────────────────────────────────────────────────────


def test_parse_args_defaults():
    args = serve_target._parse_args(["--target", "org/model"])
    assert args.target == "org/model"
    assert args.host == "0.0.0.0"
    assert args.port == 8001
    assert args.nccl_port is None
    assert args.aux_layer_ids is None
    assert args.trust_remote_code is False


def test_parse_args_explicit():
    args = serve_target._parse_args(
        [
            "--target",
            "m",
            "--host",
            "1.2.3.4",
            "--port",
            "9000",
            "--nccl-port",
            "9500",
            "--aux-layer-ids",
            "1",
            "2",
            "3",
            "--trust-remote-code",
        ]
    )
    assert (args.host, args.port, args.nccl_port) == ("1.2.3.4", 9000, 9500)
    assert args.aux_layer_ids == [1, 2, 3]
    assert args.trust_remote_code is True


@pytest.mark.parametrize(
    "argv, expected_nccl_port",
    [
        (["--target", "m", "--port", "8001"], 8101),  # default: http port + 100
        (["--target", "m", "--port", "8001", "--nccl-port", "7000"], 7000),  # explicit
    ],
)
def test_main_wires_server(monkeypatch, argv, expected_nccl_port):
    fake_model = mock.MagicMock()
    monkeypatch.setattr(
        serve_target.NeMoAutoModelForCausalLM, "from_pretrained", mock.MagicMock(return_value=fake_model)
    )
    monkeypatch.setattr(serve_target, "HFEagle3TargetModel", mock.MagicMock(return_value="wrapper"))
    captured = {}

    def _fake_server(wrapper, *, nccl_port, host):
        captured.update(wrapper=wrapper, nccl_port=nccl_port, host=host)
        return "server_logic"

    served = []
    monkeypatch.setattr(serve_target, "TargetModelServer", _fake_server)
    monkeypatch.setattr(serve_target, "serve", lambda logic, host, port: served.append((logic, host, port)))

    serve_target.main(argv)

    assert captured["nccl_port"] == expected_nccl_port
    assert captured["wrapper"] == "wrapper"
    assert served == [("server_logic", "0.0.0.0", 8001)]
    # The frozen target must be put in eval/no-grad mode.
    fake_model.requires_grad_.assert_called_once_with(False)


# ── _ServerClient: URL / port parsing ────────────────────────────────────


def test_host_and_default_nccl_port(monkeypatch):
    monkeypatch.delenv("NEMO_EAGLE_NCCL_PORT", raising=False)
    client = _ServerClient("http://host.local:8001/", timeout=1, max_retries=0)
    assert client._host() == "host.local"
    assert client._nccl_port() == 8101  # 8001 + 100 + offset 0


def test_nccl_port_env_override_with_offset(monkeypatch):
    monkeypatch.setenv("NEMO_EAGLE_NCCL_PORT", "5000")
    client = _ServerClient("http://h:8001", timeout=1, max_retries=0, nccl_rank_offset=3)
    assert client._nccl_port() == 5003


def test_nccl_port_nondigit_falls_back(monkeypatch):
    monkeypatch.delenv("NEMO_EAGLE_NCCL_PORT", raising=False)
    client = _ServerClient("http://hostname-no-port", timeout=1, max_retries=0)
    assert client._nccl_port() == 8100  # 8000 default + 100


# ── _ServerClient.request: retry / failure ───────────────────────────────


def test_request_retries_then_succeeds(monkeypatch):
    client = _ServerClient("http://h:1", timeout=1, max_retries=2)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ok = mock.MagicMock(content=b"payload")
    ok.raise_for_status.return_value = None
    calls = {"n": 0}

    def _post(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("refused")
        return ok

    monkeypatch.setattr(client._session, "post", _post)
    assert client.request("endpoint", b"x") == b"payload"
    assert calls["n"] == 2


def test_request_exhausts_retries(monkeypatch):
    client = _ServerClient("http://h:1", timeout=1, max_retries=1)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr(client._session, "post", mock.MagicMock(side_effect=requests.Timeout("slow")))
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        client.request("endpoint", b"x")


def test_close_swallows_request_errors(monkeypatch):
    client = _ServerClient("http://h:1", timeout=1, max_retries=0)
    monkeypatch.setattr(client, "request", mock.MagicMock(side_effect=Exception("server gone")))
    # No NCCL transport was created (disabled); close must not raise.
    client.close()


# ── _AsyncHandle / RemoteEagle3TargetModel surface ───────────────────────


def test_async_handle_cancel_delegates():
    future = mock.MagicMock()
    future.cancel.return_value = True
    handle = _AsyncHandle(future, convert=lambda r: r)
    assert handle.cancel() is True
    future.cancel.assert_called_once()


def test_remote_model_requires_at_least_one_url():
    with pytest.raises(ValueError, match="at least one server URL"):
        RemoteEagle3TargetModel([], device=torch.device("cpu"))


def test_remote_model_async_surface():
    model = RemoteEagle3TargetModel.from_urls(["http://a:1", "http://b:2"], device="cpu")
    assert model.supports_async is True
    assert model.num_remote_servers == 2


# ── TargetModelServer handlers (off the NCCL data plane) ──────────────────


class _FakeConfig:
    num_hidden_layers = 4
    hidden_size = 16
    vocab_size = 32


class _FakeModel:
    def __init__(self):
        self.config = _FakeConfig()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])


class _FakeTargetWrapper:
    def __init__(self):
        self.model = _FakeModel()
        self.aux_layer_ids = [1, 2, 3]

    def get_input_embeddings(self):
        return SimpleNamespace(weight=torch.randn(32, 16))


def _server() -> TargetModelServer:
    return TargetModelServer(_FakeTargetWrapper(), nccl_port=9100, host="127.0.0.1")


def test_handle_model_info_reports_config():
    info = json.loads(_server().handle_model_info(b"").decode())
    assert info["num_hidden_layers"] == 4
    assert info["hidden_size"] == 16
    assert info["vocab_size"] == 32
    assert info["aux_layer_ids"] == [1, 2, 3]


def test_handle_input_embeddings_roundtrips():
    out = wire.decode(_server().handle_input_embeddings(b""), map_location="cpu")
    assert out["weight"].shape == (32, 16)


def test_handle_init_nccl_disabled_returns_disabled():
    assert _server().handle_init_nccl(b"{}") == b'{"status": "disabled"}'


def test_handle_disconnect_sets_shutdown_event():
    server = _server()
    assert not server.shutdown_event.is_set()
    assert server.handle_disconnect(b"") == b'{"status": "ok"}'
    assert server.shutdown_event.is_set()


def test_handle_generate_requires_vocab_mapping():
    with pytest.raises(RuntimeError, match="set_vocab_mapping must be called before generate"):
        _server().handle_generate(b"", client_wants_nccl=False)


def test_infer_device_falls_back_to_cpu_without_params():
    class _NoParamWrapper:
        model = SimpleNamespace()  # no .parameters()
        aux_layer_ids = [1, 2, 3]

    server = TargetModelServer(_NoParamWrapper(), nccl_port=1)
    assert server._device == torch.device("cpu")


def test_close_without_nccl_is_noop():
    _server().close()  # no transport was created; must not raise


def test_handle_input_embeddings_gathers_dtensor():
    """The DTensor branch must gather to a full tensor before serializing."""

    class _FakeDTensor:
        def __init__(self, tensor):
            self._tensor = tensor

        def full_tensor(self):
            return self._tensor

    class _DTensorWrapper(_FakeTargetWrapper):
        def get_input_embeddings(self):
            return SimpleNamespace(weight=_FakeDTensor(torch.randn(8, 4)))

    server = TargetModelServer(_DTensorWrapper(), nccl_port=1)
    out = wire.decode(server.handle_input_embeddings(b""), map_location="cpu")
    assert out["weight"].shape == (8, 4)


# ── HTTP request handler routing (in-process server) ─────────────────────


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def running_server():
    """Start the real ThreadingHTTPServer with a fake logic object on a free port."""
    logic = _server()
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_request_handler(logic))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", logic
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_http_health_and_unknown_get(running_server):
    base, _ = running_server
    assert requests.get(f"{base}/{protocol.EP_HEALTH}", timeout=5).status_code == 200
    assert requests.get(f"{base}/does-not-exist", timeout=5).status_code == 404


def test_http_post_routing(running_server):
    base, _ = running_server
    # Known control-plane endpoints return 200.
    assert requests.post(f"{base}/{protocol.EP_HEARTBEAT}", data=b"", timeout=5).status_code == 200
    assert requests.post(f"{base}/{protocol.EP_MODEL_INFO}", data=b"", timeout=5).status_code == 200
    # Unknown POST path -> 404.
    assert requests.post(f"{base}/nope", data=b"", timeout=5).status_code == 404


def test_http_handler_surfaces_errors_as_500(running_server):
    base, _ = running_server
    # generate before set_vocab_mapping raises inside the handler -> 500.
    resp = requests.post(f"{base}/{protocol.EP_GENERATE}", data=b"", timeout=5)
    assert resp.status_code == 500


def test_serve_runs_until_disconnect():
    """``serve`` blocks until the shutdown event fires, then tears down cleanly."""
    logic = _server()
    port = _free_port()
    thread = threading.Thread(target=serve, args=(logic, "127.0.0.1", port), daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(50):  # wait for the server to bind
        try:
            if requests.get(f"{base}/{protocol.EP_HEALTH}", timeout=1).status_code == 200:
                break
        except requests.ConnectionError:
            threading.Event().wait(0.05)
    requests.post(f"{base}/{protocol.EP_DISCONNECT}", data=b"", timeout=5)
    thread.join(timeout=10)
    assert not thread.is_alive()


# ── NCCLTransport shallow paths (no CUDA / no rendezvous) ─────────────────


def test_transport_starts_uninitialized():
    transport = NCCLTransport(nccl_port=9100, host="127.0.0.1", is_server=True)
    assert transport.is_initialized is False
    assert transport._rank == 0  # server is rank 0
    assert NCCLTransport(nccl_port=9100, host="127.0.0.1", is_server=False)._rank == 1


def test_transport_initialize_without_sglang_returns_false():
    # sglang's custom process group is unavailable in this env, so initialize
    # takes the early wire-format fallback instead of a real NCCL rendezvous.
    transport = NCCLTransport(nccl_port=_free_port(), host="127.0.0.1", is_server=True)
    assert transport.initialize() is False
    assert transport.is_initialized is False


def test_transport_destroy_without_group_is_noop():
    NCCLTransport(nccl_port=9100, host="127.0.0.1", is_server=True).destroy()  # must not raise
