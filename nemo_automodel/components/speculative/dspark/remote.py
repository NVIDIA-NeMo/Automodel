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

"""Train-inference disaggregation for DSpark: a remote target server and its client.

The frozen DSpark target runs in its own process, on other GPUs or nodes, and
serves the two hidden-state tensors the draft trains on. The transport is the
EAGLE-3 one, unchanged (:mod:`nemo_automodel.components.speculative.eagle.remote`):
HTTP is the control plane and NCCL or the binary wire format is the data plane.
Only the payload differs. The server captures ``target_hidden_states`` and
``target_last_hidden_states`` with :class:`HFDSparkTargetModel`, the same code the
colocated path runs, and both data planes transfer them losslessly, so a remote
batch equals the colocated one for the same target forward.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from nemo_automodel.components.speculative.dspark.target import DSparkTargetBatch, HFDSparkTargetModel
from nemo_automodel.components.speculative.eagle.remote import protocol
from nemo_automodel.components.speculative.eagle.remote.client import RemoteTargetPool
from nemo_automodel.components.speculative.eagle.remote.server import TargetModelServer


@dataclasses.dataclass(frozen=True)
class DSparkTargetInfo:
    """Target properties the draft is built against.

    The server reports them and the training client compares them with the values
    it derived from its own config, so a server running a different target, depth,
    or capture layers fails at setup instead of training on mismatched features.
    """

    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    target_layer_ids: tuple[int, ...]

    @classmethod
    def from_text_config(cls, text_config: PretrainedConfig, target_layer_ids: Sequence[int]) -> DSparkTargetInfo:
        """Build the info from the target's text config and the captured layers."""
        return cls(
            num_hidden_layers=int(text_config.num_hidden_layers),
            hidden_size=int(text_config.hidden_size),
            vocab_size=int(text_config.vocab_size),
            target_layer_ids=tuple(int(layer_id) for layer_id in target_layer_ids),
        )

    def mismatched_fields(self, other: DSparkTargetInfo) -> list[str]:
        """Return the names of the fields whose values differ from ``other``."""
        return [
            field.name for field in dataclasses.fields(self) if getattr(self, field.name) != getattr(other, field.name)
        ]


class DSparkTargetModelServer(TargetModelServer):
    """Request-handling logic for a remote DSpark target.

    Args:
        target_wrapper: The frozen target wrapped for DSpark capture.
        target_text_config: The config holding the target text stack's
            ``num_hidden_layers``, ``hidden_size`` and ``vocab_size``.
        nccl_port: TCP rendezvous port for the NCCL data plane.
        host: Bind/advertise address (rendezvous master for NCCL).
    """

    def __init__(
        self,
        target_wrapper: HFDSparkTargetModel,
        target_text_config: PretrainedConfig,
        *,
        nccl_port: int,
        host: str = "0.0.0.0",
    ):
        super().__init__(target_wrapper, nccl_port=nccl_port, host=host)
        self.target_info = DSparkTargetInfo.from_text_config(target_text_config, target_wrapper.target_layer_ids)

    def post_routes(self) -> dict[str, tuple[Callable[[bytes], bytes], str]]:
        """Add the output-embedding endpoint the DSpark draft head is seeded from."""
        return {
            **super().post_routes(),
            protocol.EP_OUTPUT_EMBEDDINGS: (self.handle_output_embeddings, "application/octet-stream"),
        }

    def handle_model_info(self, _raw: bytes) -> bytes:
        """Return the served :class:`DSparkTargetInfo` as JSON."""
        return json.dumps(dataclasses.asdict(self.target_info)).encode("utf-8")

    def handle_output_embeddings(self, _raw: bytes) -> bytes:
        """Return the target ``lm_head`` weight (used once to seed the draft head)."""
        return self._encode_weight(self._target.get_output_embeddings().weight)

    def _supervise(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Capture the DSpark features for one decoded request.

        Args:
            tensors: Request tensors on the server device: ``input_ids``,
                ``attention_mask`` and ``loss_mask``, each of shape [batch, sequence].

        Returns:
            ``target_hidden_states`` of shape [batch, sequence, len(target_layer_ids)
            * hidden] and ``target_last_hidden_states`` of shape [batch, sequence,
            hidden], both in the target's compute dtype.
        """
        batch = self._target.generate_batch(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            loss_mask=tensors["loss_mask"],
        )
        return {
            "target_hidden_states": batch.target_hidden_states,
            "target_last_hidden_states": batch.target_last_hidden_states,
        }


class RemoteDSparkTargetModel:
    """DSpark target backend that runs the frozen target on remote servers.

    It exposes the surface ``train_dspark`` calls on :class:`HFDSparkTargetModel`
    (``generate_batch``, ``get_input_embeddings``, ``get_output_embeddings``).
    Requests go to the servers round-robin, one at a time per server.

    Args:
        urls: One ``http://host:port`` URL per target server.
        device: Device the returned features are moved to.
        timeout: Per-request HTTP timeout in seconds.
        max_retries: Retries for idempotent requests on connection errors.
        enable_nccl: Whether to use the NCCL data plane when available. A server
            pairs NCCL with one client only, so pass False when several training
            ranks share a server.
    """

    def __init__(
        self,
        urls: list[str],
        *,
        device: torch.device,
        timeout: int = 120,
        max_retries: int = 3,
        enable_nccl: bool = True,
    ):
        self._pool = RemoteTargetPool(urls, timeout=timeout, max_retries=max_retries, enable_nccl=enable_nccl)
        self._device = device

    def check_target_info(self, expected: DSparkTargetInfo) -> None:
        """Raise if any server serves a target that differs from ``expected``.

        Args:
            expected: The target properties derived from the training config.

        Raises:
            ValueError: If a server is not a DSpark target server, or reports
                different properties (naming the server and the mismatched fields).
        """
        for client in self._pool.clients:
            served = json.loads(client.request(protocol.EP_MODEL_INFO, b"").decode())
            try:
                served_info = DSparkTargetInfo(**{**served, "target_layer_ids": tuple(served["target_layer_ids"])})
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"{client.url} did not describe a DSpark target (got model info {served}); "
                    "point recipe_args.remote_urls at serve_dspark_target servers."
                ) from exc
            mismatched = expected.mismatched_fields(served_info)
            if mismatched:
                raise ValueError(
                    f"The DSpark target served at {client.url} does not match the training config "
                    f"(mismatched fields: {mismatched}; served {served_info}, expected {expected})."
                )

    def get_input_embeddings(self) -> SimpleNamespace:
        """Fetch the target input-embedding weight, of shape [vocab, hidden] on CPU, as ``.weight``."""
        return SimpleNamespace(weight=self._pool.fetch_weight(protocol.EP_INPUT_EMBEDDINGS))

    def get_output_embeddings(self) -> SimpleNamespace:
        """Fetch the target ``lm_head`` weight, of shape [vocab, hidden] on CPU, as ``.weight``."""
        return SimpleNamespace(weight=self._pool.fetch_weight(protocol.EP_OUTPUT_EMBEDDINGS))

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DSparkTargetBatch:
        """Run the remote target once and return the DSpark features.

        Args:
            input_ids: Long tensor of shape [batch, sequence] containing target tokens.
            attention_mask: Binary tensor of shape [batch, sequence], with one for
                valid tokens and zero for padding.
            loss_mask: Tensor of shape [batch, sequence] selecting supervised tokens.

        Returns:
            DSparkTargetBatch with target_hidden_states of shape [batch, sequence,
            len(target_layer_ids) * hidden] and target_last_hidden_states of shape
            [batch, sequence, hidden] on this backend's device, plus the input_ids and
            loss_mask arguments returned as given, as the colocated wrapper does.
        """
        payload = self._pool.encode_inputs(input_ids, attention_mask, loss_mask)
        result = self._pool.next_client().generate(payload)
        return DSparkTargetBatch(
            target_hidden_states=result["target_hidden_states"].to(self._device),
            target_last_hidden_states=result["target_last_hidden_states"].to(self._device),
            input_ids=input_ids,
            loss_mask=loss_mask,
        )

    def close(self) -> None:
        """Disconnect from every server; a server shuts down when a client disconnects."""
        self._pool.close()


__all__ = ["DSparkTargetInfo", "DSparkTargetModelServer", "RemoteDSparkTargetModel"]
