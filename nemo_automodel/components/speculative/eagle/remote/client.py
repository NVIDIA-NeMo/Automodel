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

"""Training-side client for the remote EAGLE-3 target server.

:class:`RemoteEagle3TargetModel` implements the ``Eagle3TargetBackend`` contract
by delegating ``generate_batch`` to one or more remote target servers. It POSTs
``input_ids`` over HTTP and receives the supervision tensors either over NCCL
(GPU-direct, body carries only metadata) or as a binary wire blob (fallback).

Multiple server URLs are dispatched round-robin so the prefetch pipeline in the
training loop can keep several requests in flight (one per server) and overlap
target inference with draft training.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import logging
import os
import threading
import time
from types import SimpleNamespace
from typing import Optional

import requests
import torch

from nemo_automodel.components.speculative.eagle.backend import Eagle3TargetBackend
from nemo_automodel.components.speculative.eagle.remote import protocol, wire
from nemo_automodel.components.speculative.eagle.remote.transport import NCCLTransport, nccl_transport_available
from nemo_automodel.components.speculative.eagle.target import Eagle3TargetBatch

logger = logging.getLogger(__name__)


class _ServerClient:
    """HTTP + NCCL connection to a single remote target server."""

    def __init__(self, url: str, timeout: int, max_retries: int, nccl_rank_offset: int = 0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        # The target server is a direct in-cluster peer; never route requests
        # through an ambient HTTP(S) proxy (e.g. a corporate ``http_proxy``),
        # which cannot reach a pod-local or intra-cluster address. trust_env
        # also disables ``no_proxy`` parsing, so this works without per-host
        # ``no_proxy`` configuration on every node.
        self._session.trust_env = False
        self._nccl_rank_offset = nccl_rank_offset
        self._nccl_enabled = os.environ.get("NEMO_EAGLE_ENABLE_NCCL", "1") == "1"
        self._nccl: Optional[NCCLTransport] = None
        self._nccl_attempted = False
        self._nccl_lock = threading.Lock()

    def _host(self) -> str:
        return self.url.split("://", 1)[-1].split(":", 1)[0].split("/", 1)[0]

    def _nccl_port(self) -> int:
        env = os.environ.get("NEMO_EAGLE_NCCL_PORT")
        if env is not None:
            base = int(env)
        else:
            http_port = self.url.rsplit(":", 1)[-1].split("/", 1)[0]
            base = (int(http_port) if http_port.isdigit() else 8000) + 100
        return base + self._nccl_rank_offset

    def request(self, endpoint: str, payload: bytes, *, content_type: str = "application/octet-stream") -> bytes:
        """POST ``payload`` to ``endpoint`` with exponential-backoff retry."""
        url = f"{self.url}/{endpoint}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, data=payload, timeout=self.timeout, headers={"Content-Type": content_type}
                )
                resp.raise_for_status()
                return resp.content
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"remote target request to {url} failed after {self.max_retries + 1} attempts") from last_exc

    def _init_nccl(self) -> bool:
        with self._nccl_lock:
            if self._nccl is not None and self._nccl.is_initialized:
                return True
            if self._nccl_attempted:
                return False
            self._nccl_attempted = True

            # Only ask the server to bring up NCCL if this process can actually
            # join the group. A client without sglang (the common disaggregated
            # case: sglang target server + sglang-free training client) cannot,
            # and contacting the server anyway leaves it blocked on a rendezvous
            # that never completes until it times out. Fall back to wire instead.
            if not nccl_transport_available():
                logger.info("NCCL unavailable in this process (sglang not importable); using wire format")
                self._nccl = None
                return False

            port = self._nccl_port()
            self._nccl = NCCLTransport(nccl_port=port, host=self._host(), is_server=False)

            # Client init blocks on the TCP store rendezvous; run it in a thread
            # while we tell the server to initialize its side over HTTP.
            result: list[bool] = [False]

            def _client_init():
                result[0] = self._nccl.initialize()

            init_thread = threading.Thread(target=_client_init, daemon=True)
            init_thread.start()
            try:
                resp = self.request(protocol.EP_INIT_NCCL, json.dumps({"nccl_port": port}).encode())
                status = json.loads(resp.decode()).get("status")
            except Exception as exc:
                logger.warning("server NCCL init request failed: %s", exc)
                status = None
            init_thread.join(timeout=150)
            if status not in ("ok", "already_initialized") or not result[0]:
                logger.warning("NCCL init failed (server status=%s); using wire format", status)
                self._nccl = None
                return False
            return True

    def generate(self, payload: bytes) -> dict[str, Optional[torch.Tensor]]:
        """POST /generate and return the supervision tensors (NCCL or wire)."""
        if self._nccl_enabled and not self._nccl_attempted:
            self._init_nccl()
        headers = {"Content-Type": "application/octet-stream"}
        use_nccl = self._nccl is not None and self._nccl.is_initialized
        if use_nccl:
            headers[protocol.NCCL_HEADER] = "1"
        url = f"{self.url}/{protocol.EP_GENERATE}"
        resp = self._session.post(url, data=payload, timeout=self.timeout, headers=headers)
        resp.raise_for_status()
        if resp.headers.get(protocol.NCCL_HEADER) == "1" and use_nccl:
            keys_order, metadata = protocol.decode_nccl_metadata(resp.content)
            return self._nccl.recv_tensors(metadata, keys_order)
        return wire.decode(resp.content, map_location="cpu")

    def close(self) -> None:
        try:
            self.request(protocol.EP_DISCONNECT, b"{}", content_type="application/json")
        except Exception:
            pass
        if self._nccl is not None:
            self._nccl.destroy()
            self._nccl = None
        self._session.close()


class _AsyncHandle:
    """Future-like wrapper that converts a worker-thread result into a batch."""

    def __init__(self, future, convert):
        self._future = future
        self._convert = convert

    def result(self, timeout: Optional[float] = None) -> Eagle3TargetBatch:
        return self._convert(self._future.result(timeout=timeout))

    def cancel(self) -> bool:
        return self._future.cancel()


class RemoteEagle3TargetModel(Eagle3TargetBackend):
    """EAGLE-3 target backend that delegates forward passes to remote servers."""

    def __init__(self, urls: list[str], *, device: torch.device, timeout: int = 120, max_retries: int = 3):
        if not urls:
            raise ValueError("RemoteEagle3TargetModel requires at least one server URL")
        # A distinct NCCL rank offset per server lets a multi-rank training job
        # pair each rank with its own server (1:1) without port collisions.
        self._clients = [_ServerClient(u, timeout, max_retries, nccl_rank_offset=i) for i, u in enumerate(urls)]
        self._next = itertools.cycle(range(len(self._clients)))
        self._device = device
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._embeddings: Optional[SimpleNamespace] = None

    @property
    def num_remote_servers(self) -> int:
        return len(self._clients)

    @property
    def supports_async(self) -> bool:
        return True

    @classmethod
    def from_urls(cls, urls: list[str], *, device, **kwargs) -> "RemoteEagle3TargetModel":
        return cls(urls=urls, device=torch.device(device), **kwargs)

    def model_info(self) -> dict:
        return json.loads(self._clients[0].request(protocol.EP_MODEL_INFO, b"").decode())

    def get_input_embeddings(self):
        """Fetch the target input-embedding weight once and cache it.

        Returns an object exposing ``.weight`` (the only attribute the draft's
        ``copy_embeddings_from_target`` reads), matching the offline-cache path.
        """
        if self._embeddings is None:
            raw = self._clients[0].request(protocol.EP_INPUT_EMBEDDINGS, b"")
            weight = wire.decode(raw, map_location="cpu")["weight"]
            self._embeddings = SimpleNamespace(weight=weight.to(self._device))
        return self._embeddings

    def set_vocab_mapping(self, selected_token_ids: torch.Tensor, selected_token_mask: torch.Tensor) -> None:
        payload = wire.encode_to_bytes(
            {
                "selected_token_ids": selected_token_ids.cpu(),
                "selected_token_mask": selected_token_mask.cpu(),
            }
        )
        for client in self._clients:
            client.request(protocol.EP_SET_VOCAB_MAPPING, payload)

    @staticmethod
    def _build_payload(input_ids, attention_mask, loss_mask) -> bytes:
        return wire.encode_to_bytes(
            {
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
                "loss_mask": loss_mask.cpu(),
            }
        )

    def _to_batch(self, result: dict, attention_mask: torch.Tensor) -> Eagle3TargetBatch:
        dev = self._device
        return Eagle3TargetBatch(
            aux_hidden_states=result["aux_hidden_states"].to(dev),
            input_ids=result["input_ids"].to(dev),
            attention_mask=attention_mask.to(dev),
            loss_mask=result["loss_mask"].to(dev),
            target_probs=result["target_probs"].to(dev),
            position_mask=result["position_mask"].to(dev).bool(),
        )

    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, loss_mask) -> Eagle3TargetBatch:
        payload = self._build_payload(input_ids, attention_mask, loss_mask)
        client = self._clients[next(self._next)]
        result = client.generate(payload)
        return self._to_batch(result, attention_mask)

    def generate_batch_async(self, input_ids, attention_mask, loss_mask) -> _AsyncHandle:
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(2, self.num_remote_servers), thread_name_prefix="eagle-target"
            )
        payload = self._build_payload(input_ids, attention_mask, loss_mask)
        client = self._clients[next(self._next)]
        future = self._executor.submit(client.generate, payload)
        return _AsyncHandle(future, lambda result: self._to_batch(result, attention_mask))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        for client in self._clients:
            client.close()
