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

"""Remote EAGLE-3 target server.

Runs the frozen target model and, for each training request, produces the
draft-vocab supervision (aux hidden states, ``target_probs``, ``position_mask``)
and ships it back to the training client. The supervision computation reuses the
co-located building blocks verbatim -- ``HFEagle3TargetModel.generate_batch``
for the forward + aux capture and ``_compute_target_distribution`` for the
draft-vocab projection -- so a remote run is numerically identical to a
co-located one.

The HTTP request handling is split from the ``http.server`` plumbing
(:class:`TargetModelServer` holds the pure logic) so it can be unit-tested on
CPU with the NCCL data plane disabled (wire-format path).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import torch

from nemo_automodel.components.speculative.eagle.core import _compute_target_distribution
from nemo_automodel.components.speculative.eagle.remote import protocol, wire
from nemo_automodel.components.speculative.eagle.remote.transport import NCCLTransport

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_supervision(
    target_wrapper,
    selected_token_ids: torch.Tensor,
    selected_token_mask: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Produce the precomputed draft-vocab supervision for one batch.

    Mirrors the co-located path exactly: ``generate_batch`` runs the target and
    returns shifted logits / input_ids / loss_mask plus the aux hidden states;
    ``_compute_target_distribution`` then projects the shifted logits onto the
    draft vocab. Returns tensors keyed by :data:`protocol.SUPERVISION_KEYS`.
    """
    batch = target_wrapper.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)
    target_probs, position_mask = _compute_target_distribution(
        target_logits=batch.logits,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        loss_mask=batch.loss_mask,
    )
    return {
        "aux_hidden_states": batch.aux_hidden_states,
        "target_probs": target_probs,
        "position_mask": position_mask,
        "input_ids": batch.input_ids,
        "loss_mask": batch.loss_mask,
    }


class TargetModelServer:
    """Request-handling logic for the remote target server (HTTP-transport agnostic).

    Parameters
    ----------
    target_wrapper:
        A loaded ``HFEagle3TargetModel`` (or any object exposing the same
        ``generate_batch`` / ``get_input_embeddings`` surface).
    nccl_port:
        TCP rendezvous port for the NCCL data plane.
    host:
        Bind/advertise address (rendezvous master for NCCL).
    """

    def __init__(self, target_wrapper, *, nccl_port: int, host: str = "0.0.0.0"):
        self._target = target_wrapper
        self._host = host
        self._nccl_port = nccl_port
        self._selected_token_ids: Optional[torch.Tensor] = None
        self._selected_token_mask: Optional[torch.Tensor] = None
        self._nccl_enabled = os.environ.get("NEMO_EAGLE_ENABLE_NCCL", "1") == "1"
        self._nccl: Optional[NCCLTransport] = None
        self._pending_nccl_send: Optional[tuple[dict, list[str]]] = None
        # ``ThreadingHTTPServer`` handles requests concurrently, but /generate
        # is not reentrant: the supervision forward captures aux hidden states
        # through hooks registered on the shared model, and the NCCL path
        # stages tensors in the single ``_pending_nccl_send`` slot above.
        # Concurrent requests (multiple training ranks, or a client whose
        # prefetcher overlaps) would silently receive each other's tensors,
        # so the HTTP handler serializes the whole generate-and-flush sequence
        # on this lock. ``set_vocab_mapping`` shares it because it swaps the
        # tensors the generate forward reads.
        self._generate_lock = threading.Lock()
        # Lifecycle: a watchdog shuts the server down once the client goes away
        # so a finished training run does not leave a GPU-pinned server process.
        self._shutdown_event = threading.Event()
        self._device = self._infer_device()

    def _infer_device(self) -> torch.device:
        try:
            return next(self._target.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    @property
    def shutdown_event(self) -> threading.Event:
        return self._shutdown_event

    @property
    def generate_lock(self) -> threading.Lock:
        return self._generate_lock

    # -- handlers (return the HTTP response body as bytes) --

    def handle_model_info(self, _raw: bytes) -> bytes:
        config = self._target.model.config
        info = {
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(getattr(config, "hidden_size", 0)),
            "vocab_size": int(getattr(config, "vocab_size", 0)),
            "aux_layer_ids": list(self._target.aux_layer_ids),
        }
        return json.dumps(info).encode("utf-8")

    def handle_input_embeddings(self, _raw: bytes) -> bytes:
        """Return the target input-embedding weight (used once to seed the draft)."""
        weight = self._target.get_input_embeddings().weight
        if hasattr(weight, "full_tensor"):  # gather a sharded DTensor to full
            weight = weight.full_tensor()
        return wire.encode_to_bytes({"weight": weight.detach().cpu()})

    def handle_set_vocab_mapping(self, raw: bytes) -> bytes:
        tensors = wire.decode(raw, map_location="cpu")
        self._selected_token_ids = tensors["selected_token_ids"].to(self._device).long()
        self._selected_token_mask = tensors["selected_token_mask"].to(self._device).bool()
        return b'{"status": "ok"}'

    def handle_init_nccl(self, raw: bytes) -> bytes:
        if not self._nccl_enabled:
            return b'{"status": "disabled"}'
        nccl_port = json.loads(raw.decode("utf-8")).get("nccl_port", self._nccl_port)
        if self._nccl is not None and self._nccl.is_initialized:
            return b'{"status": "already_initialized"}'
        self._nccl = NCCLTransport(nccl_port=nccl_port, host=self._host, is_server=True)
        ok = self._nccl.initialize()
        if not ok:
            self._nccl = None
            return b'{"status": "failed"}'
        return b'{"status": "ok"}'

    def handle_generate(self, raw: bytes, *, client_wants_nccl: bool) -> tuple[bytes, bool]:
        """Run the target and serialize the supervision.

        Returns ``(body, used_nccl)``. When NCCL is used the body is the JSON
        metadata only and the tensors are queued for :meth:`flush_nccl_send`
        (sent *after* the HTTP response is flushed, to avoid a recv deadlock).
        """
        if self._selected_token_ids is None or self._selected_token_mask is None:
            raise RuntimeError("set_vocab_mapping must be called before generate")
        tensors = wire.decode(raw, map_location=str(self._device))
        supervision = compute_supervision(
            self._target,
            self._selected_token_ids,
            self._selected_token_mask,
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            loss_mask=tensors["loss_mask"],
        )
        use_nccl = client_wants_nccl and self._nccl is not None and self._nccl.is_initialized
        if use_nccl:
            for key, value in supervision.items():
                if value.is_cuda and not value.is_contiguous():
                    supervision[key] = value.contiguous()
            self._pending_nccl_send = (supervision, protocol.SUPERVISION_KEYS)
            return protocol.encode_nccl_metadata(supervision, protocol.SUPERVISION_KEYS), True
        cpu_supervision = {k: v.cpu() for k, v in supervision.items()}
        return wire.encode_to_bytes(cpu_supervision), False

    def flush_nccl_send(self) -> None:
        """Send the pending supervision tensors over NCCL (after the HTTP flush)."""
        pending = self._pending_nccl_send
        if pending is None:
            return
        self._pending_nccl_send = None
        data, keys_order = pending
        try:
            self._nccl.send_tensors(data, keys_order)
        except Exception:
            logger.exception("NCCL send failed; disabling NCCL for subsequent requests")
            transport, self._nccl = self._nccl, None
            if transport is not None:
                transport.destroy()
            raise

    def handle_disconnect(self, _raw: bytes) -> bytes:
        self._shutdown_event.set()
        return b'{"status": "ok"}'

    def close(self) -> None:
        if self._nccl is not None:
            self._nccl.destroy()
            self._nccl = None


def _make_request_handler(server_logic: TargetModelServer):
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):  # noqa: D401 - silence default stderr logging
            return

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def _send(self, body: bytes, *, content_type: str, extra_headers: Optional[dict] = None) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            if self.path.lstrip("/") == protocol.EP_HEALTH:
                self._send(b'{"status": "ok"}', content_type="application/json")
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802 - http.server API
            endpoint = self.path.lstrip("/")
            raw = self._read_body()
            try:
                if endpoint == protocol.EP_GENERATE:
                    wants_nccl = self.headers.get(protocol.NCCL_HEADER) == "1"
                    # Exclusive end-to-end: see the ``_generate_lock`` comment in
                    # ``TargetModelServer.__init__``.
                    with server_logic.generate_lock:
                        body, used_nccl = server_logic.handle_generate(raw, client_wants_nccl=wants_nccl)
                        headers = {protocol.NCCL_HEADER: "1"} if used_nccl else None
                        ctype = "application/json" if used_nccl else "application/octet-stream"
                        self._send(body, content_type=ctype, extra_headers=headers)
                        if used_nccl:
                            # Send tensors only after the HTTP response is flushed so
                            # the client has the metadata and can post its recv.
                            self.wfile.flush()
                            try:
                                server_logic.flush_nccl_send()
                            except Exception:
                                # The 200 response is already on the wire; writing a
                                # 500 now would be parsed as the response to the NEXT
                                # request on this keep-alive connection. Drop the
                                # connection instead (flush_nccl_send already logged
                                # and disabled NCCL for subsequent requests).
                                self.close_connection = True
                elif endpoint == protocol.EP_SET_VOCAB_MAPPING:
                    with server_logic.generate_lock:
                        body = server_logic.handle_set_vocab_mapping(raw)
                    self._send(body, content_type="application/json")
                elif endpoint == protocol.EP_MODEL_INFO:
                    self._send(server_logic.handle_model_info(raw), content_type="application/json")
                elif endpoint == protocol.EP_INPUT_EMBEDDINGS:
                    self._send(server_logic.handle_input_embeddings(raw), content_type="application/octet-stream")
                elif endpoint == protocol.EP_INIT_NCCL:
                    self._send(server_logic.handle_init_nccl(raw), content_type="application/json")
                elif endpoint == protocol.EP_HEARTBEAT:
                    self._send(b'{"status": "ok"}', content_type="application/json")
                elif endpoint == protocol.EP_DISCONNECT:
                    self._send(server_logic.handle_disconnect(raw), content_type="application/json")
                else:
                    self.send_error(404)
            except Exception as exc:  # surface as 500 rather than dropping the connection
                logger.exception("request to /%s failed", endpoint)
                self.send_error(500, str(exc))

    return _Handler


def serve(server_logic: TargetModelServer, host: str, port: int) -> None:
    """Run the blocking HTTP server until the client disconnects or Ctrl-C."""
    httpd = ThreadingHTTPServer((host, port), _make_request_handler(server_logic))
    httpd.daemon_threads = True
    watcher = threading.Thread(target=lambda: (server_logic.shutdown_event.wait(), httpd.shutdown()), daemon=True)
    watcher.start()
    logger.info("EAGLE-3 target server listening on %s:%d", host, port)
    try:
        httpd.serve_forever()
    finally:
        server_logic.close()
        httpd.server_close()
