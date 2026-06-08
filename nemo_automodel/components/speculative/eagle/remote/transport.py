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

"""Dedicated NCCL transport for GPU-to-GPU supervision-tensor transfer.

A 2-process NCCL group connects the target server (rank 0) to the training
client (rank 1). HTTP stays the control plane (input_ids up, tensor metadata
down); this group is the data plane for the large supervision tensors, working
over NVLink intra-node and RDMA/RoCE inter-node.

The group is created from an explicit ``TCPStore`` so it is independent of the
training job's default process group. We delegate the actual group creation to
SGLang's ``init_custom_process_group`` (the proven path; it builds a *non*
default group from a provided store). SGLang is an optional, non-bundled
dependency -- when it is absent :meth:`NCCLTransport.initialize` returns False
and the caller falls back to the binary wire format.

Environment variables:

- ``NEMO_EAGLE_ENABLE_NCCL`` -- ``"1"`` (default) to attempt NCCL, ``"0"`` to
  force the wire-format fallback.
- ``NEMO_EAGLE_NCCL_PORT`` -- TCP rendezvous port (default: HTTP port + 100).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import torch
import torch.distributed as dist

from nemo_automodel.shared.import_utils import safe_import_from

logger = logging.getLogger(__name__)

_HAS_SGLANG_PG, _init_custom_process_group = safe_import_from("sglang.srt.utils.common", "init_custom_process_group")

# dtypes NCCL P2P does not support; transmitted as raw uint8 views.
_NCCL_UNSUPPORTED_DTYPES = {torch.int16, torch.int8, torch.bool}
_ELEMENT_SIZE = {torch.int16: 2, torch.int8: 1, torch.bool: 1}


class NCCLTransport:
    """A dedicated 2-process NCCL group between server (rank 0) and client (rank 1).

    Parameters
    ----------
    nccl_port:
        TCP port for the rendezvous store.
    host:
        Hostname/IP of the server (rendezvous master).
    is_server:
        True on the server side (rank 0), False on the client side (rank 1).
    """

    def __init__(self, nccl_port: int, host: str, is_server: bool):
        self._nccl_port = nccl_port
        self._host = host
        self._is_server = is_server
        self._rank = 0 if is_server else 1
        self._pg: Optional[dist.ProcessGroup] = None
        self._initialized = False
        self._init_lock = threading.Lock()
        self._group_name = f"nemo_eagle_target_transfer_{nccl_port}"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, timeout_seconds: int = 120) -> bool:
        """Establish the NCCL group via TCP rendezvous; blocks until both peers connect.

        Returns True on success, False on any failure (caller falls back to wire).
        """
        with self._init_lock:
            if self._initialized:
                return True
            if not _HAS_SGLANG_PG:
                logger.warning(
                    "NCCL transport unavailable: sglang.init_custom_process_group not importable; "
                    "falling back to wire format. Install sglang to enable GPU-direct transfer."
                )
                return False
            try:
                from torch.distributed import TCPStore
                from torch.distributed.distributed_c10d import timedelta

                timeout = timedelta(seconds=timeout_seconds)
                # Rank 0 (server) is the store master; rank 1 (client) connects.
                # An explicit store keeps this group independent of the training
                # job's default process group.
                store = TCPStore(
                    host_name=self._host,
                    port=self._nccl_port,
                    world_size=2,
                    is_master=self._rank == 0,
                    timeout=timeout,
                    multi_tenant=True,
                )
                self._pg = _init_custom_process_group(
                    backend="nccl",
                    store=store,
                    world_size=2,
                    rank=self._rank,
                    group_name=self._group_name,
                    timeout=timeout,
                )
                self._initialized = True
                logger.info("NCCL transport initialized (rank=%d, port=%d)", self._rank, self._nccl_port)
                return True
            except Exception as exc:
                logger.error("NCCL transport init failed (rank=%d): %s", self._rank, exc)
                self._pg = None
                self._initialized = False
                return False

    def send_tensors(self, tensor_dict: dict[str, Optional[torch.Tensor]], keys_order: list[str]) -> None:
        """Send tensors (server side) in ``keys_order``; skips ``None`` entries."""
        assert self._initialized and self._pg is not None, "NCCL transport not initialized"
        assert self._is_server, "only the server sends tensors"
        for key in keys_order:
            tensor = tensor_dict.get(key)
            if tensor is None:
                continue
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if tensor.dtype in _NCCL_UNSUPPORTED_DTYPES:
                tensor = tensor.view(torch.uint8)
            dist.send(tensor, dst=1, group=self._pg)

    def recv_tensors(
        self, metadata: dict[str, Optional[dict]], keys_order: list[str]
    ) -> dict[str, Optional[torch.Tensor]]:
        """Receive tensors (client side) per ``metadata`` in ``keys_order``."""
        from nemo_automodel.components.speculative.eagle.remote.protocol import dtype_from_code

        assert self._initialized and self._pg is not None, "NCCL transport not initialized"
        assert not self._is_server, "only the client receives tensors"

        result: dict[str, Optional[torch.Tensor]] = {}
        for key in keys_order:
            meta = metadata.get(key)
            if meta is None:
                result[key] = None
                continue
            dtype = dtype_from_code(meta["dtype_code"])
            shape = tuple(meta["shape"])
            if dtype in _NCCL_UNSUPPORTED_DTYPES:
                numel = 1
                for dim in shape:
                    numel *= dim
                buf = torch.empty(numel * _ELEMENT_SIZE[dtype], dtype=torch.uint8, device="cuda")
                dist.recv(buf, src=0, group=self._pg)
                result[key] = buf.view(dtype).reshape(shape)
            else:
                buf = torch.empty(shape, dtype=dtype, device="cuda")
                dist.recv(buf, src=0, group=self._pg)
                result[key] = buf
        return result

    def destroy(self) -> None:
        """Abort and unregister the group.

        The group is asymmetric: the client can finish before the long-lived
        server, so a blocking ``destroy_process_group`` (which expects both
        peers) would hang. Abort the local communicator and scrub it from
        PyTorch's global registry so the later default-group teardown does not
        try to shut it down again.
        """
        pg = self._pg
        self._pg = None
        self._initialized = False
        if pg is None:
            return
        try:
            pg.abort()
        except Exception:
            pass
        try:
            from torch.distributed.distributed_c10d import _unregister_process_group, _world

            group_name = _world.pg_names.get(pg) or getattr(pg, "group_name", None)
            _world.pg_map.pop(pg, None)
            _world.pg_names.pop(pg, None)
            _world.pg_group_ranks.pop(pg, None)
            _world.pg_backend_config.pop(pg, None)
            _world.pg_coalesce_state.pop(pg, None)
            _world.pg_to_tag.pop(pg, None)
            for groups in _world.tags_to_pg.values():
                while pg in groups:
                    groups.remove(pg)
            if group_name is not None:
                _unregister_process_group(group_name)
        except Exception:
            pass
