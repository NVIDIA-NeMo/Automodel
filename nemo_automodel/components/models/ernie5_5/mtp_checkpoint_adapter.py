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

"""Model-owned extension of the existing whole-state MTP adapter.

Reuses the decoder's public two-call load session, layouts and distributed
export. Shared-body checkpoint copies are independent read destinations and
must agree with the decoder destinations before its finishers run. A failed,
abandoned or exchanged session makes this adapter unusable. Read copies are
compared after conversion to the destination dtype, using value equality:
signed zero agrees, NaN does not. This is not an audit of stored file bytes.
Ordinary validation and decoder-finisher failures are agreed across rank groups.
Control interrupts propagate immediately and require launcher-driven peer termination.
"""

from __future__ import annotations

import copy
import weakref
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.mtp_state_dict_adapter import (
    _OWNED,
    Ernie5_5MTPStateDictAdapter,
    _mtp_body_suffix,
)


class _SessionIdentity:
    """Unforgeable identity used only to distinguish active local load sessions."""


class _CheckpointDestinations(dict[str, torch.Tensor]):
    """HF destinations and local runtime state owned by one checkpoint read.

    Tensor values use released global layouts (DTensors hold only local shards).
    The decoder mapping stays strongly referenced and unchanged. Owned values
    retain their native layouts; the MTP head matrix is [streams, streams*hidden].
    Alias pairs hold (independent scratch, decoder destination), both in the same
    released layout. Their plain local shards are compared after storage casts.
    """

    original: dict[str, torch.Tensor]
    decoder_destinations: dict[str, torch.Tensor]
    owned_native: dict[str, torch.Tensor]
    alias_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]
    groups: tuple[dist.ProcessGroup, ...]
    flag_device: torch.device
    created: tuple[weakref.ReferenceType[torch.Tensor], ...]


class Ernie5_5MTPCheckpointAdapter(Ernie5_5MTPStateDictAdapter):
    """Load in place and save sharded MTP state without duplicate native owners.

    The admitted distributed topology covers all job ranks. Error agreement uses
    every axis group of the embedding's mesh, which covers both flattened and
    two-dimensional diagnostic checkpoint layouts. The production ERNIE
    parallelizer admits a one-dimensional FSDP mesh. This requires symmetric
    calls on ranks; KeyboardInterrupt and SystemExit propagate immediately.
    """

    _supports_low_memory_dcp_load = True

    def __init__(self, config: Ernie5_5Config) -> None:
        """Bind the decoder layouts and initialize an empty session lifecycle."""
        super().__init__(config)
        self.config = config
        self._active_session: weakref.ReferenceType[_CheckpointDestinations] | None = None
        self._identity: _SessionIdentity | None = None
        self._failed = False
        self._ended_destinations: list[weakref.ReferenceType[torch.Tensor]] = []

    def __deepcopy__(self, memo: dict[int, Any]) -> Ernie5_5MTPCheckpointAdapter:
        """Copy the adapter without sharing a live session with a copied model.

        Args:
            memo: Python ``copy.deepcopy`` memo, whose graph nodes have arbitrary
                library-owned types. The opaque value type is confined to this
                standard-library protocol boundary, as in the decoder adapter.

        Returns:
            An independent idle adapter, or an unusable adapter when the original
            has an open/failed session and may have been partially overwritten.
        """
        duplicate = copy.copy(self)
        duplicate.config = copy.deepcopy(self.config, memo)
        duplicate.decoder = copy.deepcopy(self.decoder, memo)
        duplicate._active_session = None
        duplicate._identity = None
        duplicate._failed = self._failed or self._identity is not None
        duplicate._ended_destinations = list(self._ended_destinations)
        return duplicate

    def _split_state(self, state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Split decoder tensors and the nine native MTP values without copying.

        Args:
            state: Native names mapped to registered tensors in native layouts;
                DTensor shapes are global. The owned head is [streams, streams*hidden].

        Returns:
            Decoder mapping and owned mapping (without ``mtp.``), both aliasing inputs.

        Raises:
            ValueError: If the native state does not contain exactly eight owned parameters and one router bias.
        """
        owned = {name[4:]: value for name, value in state.items() if name.startswith("mtp.")}
        if owned.keys() != _OWNED.keys():
            raise ValueError("MTP native state must contain exactly eight owned parameters and one bias")
        return {name: value for name, value in state.items() if not name.startswith("mtp.")}, owned

    def _owned_hf(self, owned: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return released views of the nine owned tensors.

        Args:
            owned: Native owned tensors. The head matrix has shape [streams, streams*hidden].

        Returns:
            Released names and views; head shape [streams*hidden, streams], others unchanged.
        """
        return {
            self.mtp_prefix + suffix: value.T if name == "hyper_head.hc_fn" else value
            for name, suffix in _OWNED.items()
            for value in [owned[name]]
        }

    def _body_aliases(self, decoder: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return MTP body names pointing at the decoder's released tensors.

        Args:
            decoder: Released decoder destinations/export views, in their released layouts.

        Returns:
            Alias names and the same tensor objects, excluding the decoder's router bias.
        """
        return {
            self.mtp_prefix + _mtp_body_suffix(name[len(self.body_prefix) :]): value
            for name, value in decoder.items()
            if name.startswith(self.body_prefix) and name != self.body_prefix + self.bias_suffix
        }

    def _open(self, session: _CheckpointDestinations) -> None:
        """Hold the tensor-bearing destination mapping weakly until its paired close."""
        identity = _SessionIdentity()
        owner = weakref.ref(self)

        def abandon(_reference: weakref.ReferenceType[_CheckpointDestinations]) -> None:
            adapter = owner()
            if adapter is not None and adapter._identity is identity:
                adapter._failed = True
                adapter._identity = None
                adapter._active_session = None

        self._identity = identity
        self._active_session = weakref.ref(session, abandon)

    @staticmethod
    def _local(value: torch.Tensor) -> torch.Tensor:
        """Return the local shard of a global DTensor, or the original plain tensor.

        Args:
            value: Tensor in any layout; a DTensor has its declared global shape and placements.

        Returns:
            Plain local tensor, without gathering or copying its storage.
        """
        return value.to_local() if isinstance(value, DTensor) else value

    @staticmethod
    def _raise_together(session: _CheckpointDestinations, error: BaseException | None) -> None:
        """Agree on failure across all mesh axes before ranks return or raise.

        Args:
            session: Released-layout destination mapping and explicit runtime groups/device.
            error: Local failure, or None. No tensor values are exchanged.

        Raises:
            RuntimeError: If another rank rejects the load; the message identifies detecting ranks.
            BaseException: The original local validation or finisher error is re-raised on its detecting rank.
        """
        if not session.groups:
            if error is not None:
                raise error
            return
        # Identify every detecting rank, including expert holders. Explicit
        # mesh-axis reductions propagate the same rank list over either layout.
        flag = torch.zeros(dist.get_world_size(), dtype=torch.int32, device=session.flag_device)
        if error is not None:
            flag[dist.get_rank()] = 1
        for group in session.groups:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        detecting_ranks = flag.nonzero().flatten().cpu().tolist()
        if detecting_ranks:
            if error is not None:
                error.add_note(f"MTP checkpoint load rejected on ranks {detecting_ranks}")
                raise error
            raise RuntimeError(
                f"Ranks {detecting_ranks} rejected the MTP checkpoint load; discard this model and adapter"
            )

    def get_hf_state_dict_keys(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        """Plan released names before sharding, without starting a load session.

        Raises:
            RuntimeError: If a load is active, failed or abandoned.
            ValueError: If key planning is requested after tensors have been sharded.
        """
        if self._failed or self._identity is not None:
            raise RuntimeError("Abandoned or active MTP load: discard this model and adapter")
        if any(isinstance(value, DTensor) for value in state_dict.values()):
            raise ValueError(
                "Key planning needs the unsharded state: request the Hugging Face keys before the model is sharded"
            )
        return super().get_hf_state_dict_keys(state_dict)

    def to_hf(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        quantization: bool = False,
        for_checkpoint_load: bool = False,
        v4_compatible: bool = False,
        **kwargs: str | bool | DeviceMesh | None,
    ) -> dict[str, torch.Tensor]:
        """Return release-layout export views or one opaque in-place load session.

        Args:
            state_dict: Native tensor mapping with decoder ``model.*`` and nine ``mtp.*``
                entries. DTensor global shapes/placements follow the decoder adapter;
                owned head [streams, streams*hidden], bias [experts]. Bias may be a
                replicated plain FP32 tensor while the parameters are DTensors.
            quantization: Must be False.
            for_checkpoint_load: Open a load session requiring an unchanged paired mapping.
            v4_compatible: Must be False.
            **kwargs: Public decoder-checkpointer compatibility options.

        Returns:
            Released tensors, head [streams*hidden, streams]. Export aliases share decoder
            objects. Load aliases use independent scratch in the decoder destination's
            dtype/mesh/placement; comparisons therefore occur after destination casting.
            Load destinations overwrite the model before ``from_hf`` verifies/finishes.

        Raises:
            NotImplementedError: If quantization or v4 metadata export is requested.
            RuntimeError: If a load is active, failed or abandoned.
            ValueError: If the native ownership or full-job embedding mesh is invalid.
        """
        if quantization or v4_compatible:
            raise NotImplementedError("Quantization and v4 metadata export are unsupported")
        if self._failed or self._identity is not None:
            raise RuntimeError("Abandoned or active MTP load: discard this model and adapter")
        decoder_state, owned = self._split_state(state_dict)
        if not for_checkpoint_load:
            if not any(isinstance(value, DTensor) for value in state_dict.values()):
                return super().to_hf(state_dict, **kwargs)
            result = self.decoder.to_hf(decoder_state, **kwargs)
            result.update(self._body_aliases(result))
            result.update(self._owned_hf(owned))
            return result
        try:
            decoder = self.decoder.to_hf(decoder_state, for_checkpoint_load=True, **kwargs)
            session = _CheckpointDestinations(decoder)
            session.decoder_destinations = decoder
            session.owned_native = owned
            session.alias_pairs = {}
            for name, destination in self._body_aliases(decoder).items():
                scratch = torch.empty_like(destination)
                session[name] = scratch
                session.alias_pairs[name] = (scratch, destination)
            session.update(self._owned_hf(owned))
            session.groups = ()
            embedding = decoder_state["model.embed_tokens.weight"]
            if any(isinstance(value, DTensor) for value in state_dict.values()):
                if not isinstance(embedding, DTensor):
                    raise ValueError("Sharded MTP load requires a DTensor embedding over all job ranks")
                if embedding.device_mesh.size() != dist.get_world_size():
                    raise ValueError("MTP checkpoint load requires an embedding mesh covering all job ranks")
                session.groups = tuple(embedding.device_mesh.get_all_groups())
            session.flag_device = self._local(embedding).device
            session.original = dict(session)
            given = {id(value) for value in state_dict.values()}
            session.created = tuple(weakref.ref(value) for value in session.values() if id(value) not in given)
            self._open(session)
            return session
        except BaseException:
            self._failed = True
            raise

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        **kwargs: str | bool | None,
    ) -> dict[str, torch.Tensor]:
        """Verify shared copies, finish the decoder session and return native owners.

        Args:
            hf_state_dict: The exact live mapping returned by the load-opening ``to_hf``,
                with unchanged keys and tensor objects in released layouts. Without
                a session, a complete unsharded mapping follows the parent contract.
            device_mesh: Decoder load compatibility mesh. A plain-state load requires None.
            **kwargs: Decoder compatibility options.

        Returns:
            Native tensors with unique decoder and MTP owners; session-owned head matrix
            has shape [streams, streams*hidden]. Failed sessions are never reusable.

        Raises:
            RuntimeError: If session identity or lifecycle is invalid, or another rank rejects the load.
            ValueError: If keys, destination identities or shared values violate the load contract.
            BaseException: Unexpected validation and decoder-finisher errors propagate after rank agreement; process-control interrupts propagate immediately.
        """
        session = None if self._active_session is None else self._active_session()
        if self._failed:
            raise RuntimeError("Failed MTP load: discard this model and adapter")
        if session is None:
            ended = {id(value): value for reference in self._ended_destinations if (value := reference()) is not None}
            if isinstance(hf_state_dict, _CheckpointDestinations) or any(
                ended.get(id(value)) is value for value in hf_state_dict.values()
            ):
                self._failed = True
                raise RuntimeError("Missing or replaced MTP load session: discard this model and adapter")
            return super().from_hf(hf_state_dict, device_mesh=device_mesh, **kwargs)
        try:
            problem: BaseException | None = None
            try:
                if hf_state_dict is not session:
                    problem = RuntimeError("MTP load destinations were replaced")
                elif hf_state_dict.keys() != session.original.keys():
                    problem = ValueError("MTP load destination keys changed")
                elif any(value is not session.original[name] for name, value in hf_state_dict.items()):
                    problem = ValueError("MTP load destination tensor objects changed")
                else:
                    for name, (scratch, destination) in session.alias_pairs.items():
                        if not torch.equal(self._local(scratch), self._local(destination)):
                            problem = ValueError(f"MTP shared tensor differs from its decoder owner: {name}")
                            break
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                # Every rank must join agreement even if validation itself fails.
                # CheckpointException is a BaseException, not an Exception.
                problem = error
            self._raise_together(session, problem)
            result = {}
            try:
                result = self.decoder.from_hf(session.decoder_destinations, device_mesh=device_mesh, **kwargs)
                result.update({"mtp." + name: value for name, value in session.owned_native.items()})
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                problem = error
            self._raise_together(session, problem)
            return result
        except BaseException:
            self._failed = True
            raise
        finally:
            self._identity = None
            self._active_session = None
            self._ended_destinations = [reference for reference in self._ended_destinations if reference() is not None]
            self._ended_destinations.extend(session.created)
