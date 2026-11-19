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

"""State-dict adapter between the native ERNIE 5.5 model and the released Hugging Face checkpoint layout.

``Ernie5_5StateDictAdapter`` is the object the checkpointer talks to. It serves an unsharded model with the
conversions of ``checkpoint_layout``, whose released tensors are views of the native ones, and a sharded model
with the destinations and export views of ``checkpoint_io``.

A checkpoint load is a session of two calls: ``to_hf(..., for_checkpoint_load=True)`` returns the destinations
that the reader fills, and ``from_hf`` receives the same mapping back, installs what could not be read in place
and returns the native state. The adapter keeps one session at a time and fails closed: after a session that was
abandoned, replaced or failed, every later load or save raises, because the model may be partially overwritten.

The tensors of a LoRA adapter keep their native names. ``to_hf`` and ``from_hf`` return a state of adapter
tensors alone as it is, without a session, and ``to_hf`` appends the adapter tensors of a state that also holds
base tensors to the converted base tensors. In a load, ``from_hf`` raises ``ValueError`` unless the tensors it
receives have the keys and shapes of the adapter tensors that ``to_hf(..., for_checkpoint_load=True)`` was given.

Limitations:
    * Decoder-only: the configuration must have ``num_nextn_predict_layers == 0`` and untied embedding and
      output-head weights. MTP weights are neither loaded nor exported.
    * No quantized checkpoints, no ``v4_compatible`` metadata export and no streaming export of single tensors.
    * A sharded load or save needs the layouts listed in ``checkpoint_io``. All ranks must make the same calls
      in the same order.
    * A load is not transactional: after a failure the model and the adapter must be discarded. Concurrent
      sessions and use from several threads are not supported.
    * An unsharded load through the checkpointer is supported only without a device mesh. Like a sharded
      load it reads into model storage; the fused K/V projection of each DSA layer is its only temporary.
    * A conversion from the released layout without a load session (``from_hf`` on plain tensors) packs every
      routed expert into new tensors and so holds a second copy of the experts; it suits small checkpoints.
    * Supported stored dtypes of a checkpoint are BF16, FP16 and FP32. The adapter never sees the stored dtype,
      so it cannot reject another one: the reader would convert it silently (FP64 would be rounded).
"""

import copy
import functools
import weakref
from collections.abc import Callable, Mapping
from typing import Any, Self

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.ernie5_5.checkpoint_io import (
    model_hf_dcp_load_destinations,
    model_hf_distributed_export,
)
from nemo_automodel.components.models.ernie5_5.checkpoint_layout import (
    dsa_kv_layout,
    install_fused_kv,
    map_model_from_hf,
    map_model_to_hf,
)
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config, Ernie5_5ModelPlan

# The registered model holds every parameter and buffer under its ``model`` attribute.
_MODEL_PREFIX = "model."


def _is_lora_key(name: str) -> bool:
    """Return whether a state key names a tensor of a LoRA adapter."""
    return ".lora_" in name


class _SessionToken:
    """Identity of one load session; two tokens are never the same object."""


class _LoadDestinations(dict[str, torch.Tensor]):
    """Destinations of one checkpoint load, keyed by Hugging Face name.

    The mapping owns the temporaries of the load and the callbacks that install them. The adapter holds it
    weakly, so dropping the mapping releases the temporaries.

    Attributes:
        original: The destinations as created, to detect a mapping whose keys or tensors were exchanged.
        native: The native tensors that the load fills, named relative to the inner model: the tensors of the
            state that ``to_hf`` was given.
        unsharded: Whether the state is unsharded (plain tensors) rather than sharded (DTensors).
        finishers: Callbacks that install the temporaries into the native tensors after the read, in order.
        created: Weak references to the destination tensors that are not tensors of the given state: the
            views and temporaries that the adapter created for this load.
    """

    original: dict[str, torch.Tensor]
    native: dict[str, torch.Tensor]
    unsharded: bool
    finishers: tuple[Callable[[], None], ...]
    created: tuple[weakref.ReferenceType[torch.Tensor], ...]


class Ernie5_5StateDictAdapter(StateDictAdapter):
    """Convert between the native state of ``Ernie5_5_MoeForCausalLM`` and the released checkpoint layout.

    Native names carry the ``model.`` prefix of the registered class. For a sharded and for an unsharded state,
    ``to_hf`` returns destinations or export views that alias model storage, so that a checkpoint read writes
    the model in place and a distributed read or save moves each rank's shard only.

    Key planning (``get_hf_state_dict_keys``) is supported for the unsharded state only, which is how the
    framework uses it: once, before the model is sharded. It converts a shape-only copy of the state with
    ``to_hf``. A sharded state is rejected, and like a save it raises while a load session is open and after a
    session was abandoned, replaced or failed.

    Loading into an unsharded model is supported without a device mesh, which is what a run in one process
    passes. An unsharded state together with a device mesh is not supported: ``from_hf`` raises, and inside a
    load session the adapter then refuses all further work.

    ``_supports_low_memory_dcp_load`` is set because a load allocates only small temporaries, also in one
    process: the fused K/V projection of each DSA layer and, for a sharded model, the three alphas of each mHC
    unit. Every other destination, the routed experts included, is model storage.

    Attributes:
        config: Decoder-only model configuration.
    """

    _supports_low_memory_dcp_load = True

    def __init__(self, config: Ernie5_5Config) -> None:
        """Bind the adapter to a decoder-only configuration.

        The dtypes of a checkpoint read are described in the module docstring of ``checkpoint_io``.

        Args:
            config: Model configuration with ``num_nextn_predict_layers == 0`` and untied embedding and
                output-head weights.

        Raises:
            ValueError: If the configuration has MTP layers or tied embedding and output-head weights.
        """
        if config.num_nextn_predict_layers != 0 or config.tie_word_embeddings:
            raise ValueError("Adapter requires an untied decoder-only config with explicit MTP depth zero")
        self.config = config
        self._active: weakref.ReferenceType[_LoadDestinations] | None = None
        self._token: _SessionToken | None = None
        self._poisoned = False
        # Shapes of the adapter tensors of the model, between the two calls of a LoRA load.
        self._lora_shapes: dict[str, torch.Size] | None = None
        # Tensors that the adapter created as destinations of sessions that have ended.
        self._ended: list[weakref.ReferenceType[torch.Tensor]] = []

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Return an adapter for a deep copy of the configuration that shares no load session with this one.

        A model is copied together with its adapter. A copy made while a load session is open, or after one was
        abandoned, replaced or failed, refuses all further work: the copied model may be partially overwritten,
        and the open session belongs to the original.

        Args:
            memo: The memo dictionary of ``copy.deepcopy``.

        Returns:
            A new adapter, usable when this adapter is idle. It refuses the destinations of this adapter's
            ended sessions as this adapter does.
        """
        duplicate = copy.copy(self)
        duplicate.config = copy.deepcopy(self.config, memo)
        duplicate._active = None
        duplicate._token = None
        duplicate._poisoned = self._poisoned or self._token is not None
        duplicate._lora_shapes = None
        duplicate._ended = list(self._ended)
        return duplicate

    def _holds_ended_destinations(self, mapping: Mapping[str, torch.Tensor]) -> bool:
        """Return whether a mapping holds a tensor that was created as a destination of an ended session.

        Such a mapping is a copy of finished destinations. Its tensors hold what an earlier load left in them,
        and the entries that alias model storage hold the current model values.

        Args:
            mapping: Hugging Face names mapped to plain tensors of any shape.

        Returns:
            True when at least one tensor object of the mapping is a view or a temporary of an ended session.
        """
        ended = {id(tensor) for weak in self._ended if (tensor := weak()) is not None}
        return any(id(value) in ended for value in mapping.values())

    @staticmethod
    def _strip_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the state named relative to the inner model.

        Args:
            state: Native names with the ``model.`` prefix mapped to tensors of any shape.

        Returns:
            The same tensor objects, in the same order, without the prefix.

        Raises:
            ValueError: If a name does not start with ``model.``.
        """
        if any(not name.startswith(_MODEL_PREFIX) for name in state):
            raise ValueError("Every native model tensor must use the model. prefix")
        return {name.removeprefix(_MODEL_PREFIX): value for name, value in state.items()}

    @staticmethod
    def _add_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the state under the names of the registered class.

        Args:
            state: Names relative to the inner model mapped to tensors of any shape.

        Returns:
            The same tensor objects, in the same order, with the ``model.`` prefix.
        """
        return {_MODEL_PREFIX + name: value for name, value in state.items()}

    def _rename_feed_forward(
        self, state: Mapping[str, torch.Tensor], *, to_module_names: bool
    ) -> dict[str, torch.Tensor]:
        """Rename the feed-forward tensors of every decoder layer between the two native spellings.

        The eager conversions name the feed-forward ``feed_forward``; the model registers it as ``mlp`` (dense)
        or ``moe`` (MoE).

        Args:
            state: Names relative to the inner model mapped to tensors of any shape.
            to_module_names: Rename ``layers.{index}.feed_forward.*`` to the module names when true, the
                module names to ``feed_forward`` when false.

        Returns:
            The same tensor objects. The renamed tensors of each layer follow all other entries, layer by
            layer.

        Raises:
            ValueError: If a layer has no feed-forward tensor under the source name or already has one under
                the target name.
        """
        result = dict(state)
        for layer in Ernie5_5ModelPlan.from_config(self.config).decoder_layers:
            prefix = f"layers.{layer.layer_index}."
            module = "moe." if layer.feed_forward == "moe" else "mlp."
            source, target = ("feed_forward.", module) if to_module_names else (module, "feed_forward.")
            source_prefix, target_prefix = prefix + source, prefix + target
            keys = [key for key in result if key.startswith(source_prefix)]
            if not keys or any(key.startswith(target_prefix) for key in result):
                raise ValueError(f"Missing or ambiguous decoder FFN names for layer {layer.layer_index}")
            for key in keys:
                result[target_prefix + key[len(source_prefix) :]] = result.pop(key)
        return result

    def _unsharded_to_hf(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert a complete unsharded native state to the released layout.

        Args:
            state_dict: Exactly the native tensors of the model under their ``model.`` names, in the shapes of
                the model's parameters and buffers; plain tensors on any device, the meta device included.

        Returns:
            Released Hugging Face names mapped to tensors in the released layouts, in the order of
            ``map_model_to_hf``. Every tensor is an input object or a view of one, and so aliases the given
            storage: the per-expert matrices, the mHC projection and the hyper-head matrix are transposed
            views, and the three mHC alphas are slices of the scale vector. The one exception is the fused
            K/V projection of each DSA layer, which is a new tensor.

        Raises:
            ValueError: If a name lacks the ``model.`` prefix, a tensor is missing or unexpected, or a
                component conversion rejects its tensors.
        """
        native = self._strip_prefix(state_dict)
        return map_model_to_hf(self._rename_feed_forward(native, to_module_names=False), self.config)

    def _unsharded_finishers(
        self, native: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
    ) -> list[Callable[[], None]]:
        """Return the callbacks that complete a checkpoint read into an unsharded model.

        A read fills the destinations in place, and all of them alias model storage except the fused K/V
        projection of each DSA layer. Each callback splits one filled fused projection into the native key and
        value projections.

        Args:
            native: Native tensors named relative to the inner model, in the shapes of the model's parameters
                and buffers. The callbacks overwrite ``layers.{index}.attention.k_b_proj``
                [heads * latent, key_channels] and ``layers.{index}.attention.v_b_proj``
                [heads * value_channels, latent] of every DSA layer in place.
            targets: Released Hugging Face names mapped to the destinations of the load. The callbacks read
                ``model.layers.{index}.self_attn.kv_b_proj.weight`` of shape
                [heads * (key_channels + value_channels), latent] of every DSA layer.

        Returns:
            One callback per DSA layer, in layer order.
        """
        finishers: list[Callable[[], None]] = []
        for layer in Ernie5_5ModelPlan.from_config(self.config).decoder_layers:
            if layer.attention == "dsa":
                attention = f"layers.{layer.layer_index}.attention."
                fused = targets[f"model.layers.{layer.layer_index}.self_attn.kv_b_proj.weight"]
                keys, values = native[attention + "k_b_proj"], native[attention + "v_b_proj"]
                finishers.append(functools.partial(install_fused_kv, fused, keys, values, dsa_kv_layout(self.config)))
        return finishers

    def _unsharded_from_hf(
        self, hf_state_dict: Mapping[str, torch.Tensor], device_mesh: DeviceMesh | None
    ) -> dict[str, torch.Tensor]:
        """Convert a complete unsharded state in the released layout to the native state.

        Args:
            hf_state_dict: Exactly the released tensors of the decoder-only model, in the released layouts;
                plain tensors.
            device_mesh: Mesh of a distributed load; must be ``None``. Loading into an unsharded model with a
                device mesh is not supported, because this conversion does not shard.

        Returns:
            Native ``model.`` names mapped to tensors in the shapes of the model's parameters and buffers.
            Every tensor is an input object or a view of one (the mHC projection is a transposed view), except
            the grouped expert tensors, the split DSA K/V projections and the mHC scale vectors, which are new
            tensors, and the hyper-head matrix ``hc_fn``, which is copied when its transpose is not contiguous.

        Raises:
            ValueError: If a device mesh is given, a tensor is missing or unexpected, or a component
                conversion rejects its tensors.
        """
        if device_mesh is not None:
            raise ValueError("Loading into an unsharded model with a device mesh is not supported")
        native = map_model_from_hf(hf_state_dict, self.config)
        return self._add_prefix(self._rename_feed_forward(native, to_module_names=True))

    def _open_session(self, destinations: _LoadDestinations) -> None:
        """Make ``destinations`` the one active load session.

        The adapter keeps only a weak link to the mapping. If the mapping is dropped before ``from_hf``
        receives it, the session counts as abandoned and the adapter refuses all further work.

        Args:
            destinations: Hugging Face names mapped to the destination tensors of the load, of any shape.
        """
        token = _SessionToken()
        owner = weakref.ref(self)

        def abandon(_: weakref.ReferenceType[_LoadDestinations]) -> None:
            """Mark the adapter as unusable if this session is still the active one."""
            adapter = owner()
            if adapter is not None and adapter._token is token:
                adapter._poisoned = True
                adapter._token = None
                adapter._active = None

        self._token = token
        self._active = weakref.ref(destinations, abandon)

    def get_hf_state_dict_keys(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        """Return the Hugging Face keys of the unsharded model, in the order of an unsharded save.

        The framework plans the keys once, before the model is sharded. A sharded state is rejected instead of
        planned: its export names only the experts that one rank holds and orders the keys differently, so it
        cannot stand in for the plan of the whole model.

        Args:
            state_dict: Exactly the native tensors of the unsharded model under their ``model.`` names, in the
                shapes of the model's parameters and buffers; plain tensors on any device. Only shapes and
                dtypes are read.

        Returns:
            The released Hugging Face names, in the order in which ``to_hf`` returns them for an unsharded
            state.

        Raises:
            ValueError: If a tensor of the state is a DTensor, or the unsharded conversion rejects the state.
            RuntimeError: If a load session is open, or an earlier session was abandoned, replaced or failed.
            NotImplementedError: If the configuration has been changed to MTP layers or tied weights.
        """
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
        """Return the released layout of the native state, as load destinations or as tensors to save.

        The state is treated as sharded when its tensors are DTensors. With ``for_checkpoint_load`` the result
        opens a load session that ``from_hf`` must close; without it the result is what a save writes.

        Args:
            state_dict: Exactly the native tensors of the model under their ``model.`` names, in the shapes of
                the model's parameters and buffers. Either all plain tensors, or all DTensors in the layouts
                required by ``model_hf_dcp_load_destinations`` (load) and ``model_hf_distributed_export``
                (save): for a DTensor the shapes are global and each rank holds its shard.
            quantization: Must be false; quantized checkpoints are not supported.
            for_checkpoint_load: Return destinations for a checkpoint read instead of tensors to save.
            v4_compatible: Must be false; the legacy metadata export is not supported.
            **kwargs: Other options of the checkpointer (``exclude_key_regex``, ``device_mesh``,
                ``legacy_paramwrapper_layout``). They do not apply to this model and are not read.

        Returns:
            Released Hugging Face names mapped to tensors in the released layouts.

            For a load, the mapping must be passed to ``from_hf`` unchanged after the read. The destinations
            alias model storage, so the read overwrites the model in place before ``from_hf`` runs. The
            exceptions are temporaries that ``from_hf`` installs: the fused K/V projection of each DSA layer
            and, for a sharded state, the mHC alphas. Sharded: the destinations of
            ``model_hf_dcp_load_destinations``. Unsharded: the views of ``map_model_to_hf``.

            For a save, sharded: the views of ``model_hf_distributed_export``, with the model-level tensors
            first; the per-expert names cover only the experts held by this rank. Unsharded: the views of
            ``map_model_to_hf`` in its order, with the model-level tensors last; they alias the state except
            for the fused K/V projection of each DSA layer, and the per-expert views are not contiguous.

        Raises:
            NotImplementedError: If ``quantization`` or ``v4_compatible`` is set, or the configuration has
                been changed to MTP layers or tied weights.
            RuntimeError: If a load session is active, or an earlier session was abandoned, replaced or
                failed.
            ValueError: If the state is empty (save), mixes plain tensors and DTensors (save), has a name
                without the ``model.`` prefix, or a conversion rejects its tensors.
            TypeError: If a sharded state holds a plain tensor where a DTensor is required.
        """
        if quantization or v4_compatible:
            raise NotImplementedError("Quantization and v4 metadata export are unsupported")
        if self.config.num_nextn_predict_layers != 0 or self.config.tie_word_embeddings:
            raise NotImplementedError("Only untied decoder-only export is supported")
        adapters = {name: value for name, value in state_dict.items() if _is_lora_key(name)}
        if adapters and len(adapters) == len(state_dict):
            shapes = {name: value.shape for name, value in adapters.items()}
            self._lora_shapes = shapes if for_checkpoint_load else None
            return dict(state_dict)
        if adapters and not for_checkpoint_load:
            base = {name: value for name, value in state_dict.items() if name not in adapters}
            converted = self.to_hf(base, quantization=quantization, v4_compatible=v4_compatible, **kwargs)
            converted.update(adapters)
            return converted
        tensors = [value for value in state_dict.values() if isinstance(value, torch.Tensor)]
        sharded = [isinstance(value, DTensor) for value in tensors]
        if for_checkpoint_load:
            if self._poisoned or self._token is not None:
                raise RuntimeError("Abandoned or concurrent load: discard this model and adapter")
            native = self._strip_prefix(state_dict)
            unsharded = bool(tensors) and not any(sharded)
            if unsharded:
                targets = self._unsharded_to_hf(state_dict)
                finishers = self._unsharded_finishers(native, targets)
            else:
                targets, finishers = model_hf_dcp_load_destinations(native, self.config)
            destinations = _LoadDestinations(targets)
            destinations.native = native
            destinations.unsharded = unsharded
            destinations.finishers = tuple(finishers)
            destinations.original = dict(destinations)
            given = {id(value) for value in tensors}
            destinations.created = tuple(
                weakref.ref(value) for value in destinations.values() if id(value) not in given
            )
            self._open_session(destinations)
            return destinations
        if self._poisoned or self._token is not None:
            raise RuntimeError("Abandoned or active load: discard this model and adapter")
        if not tensors:
            raise ValueError("Expected model tensors")
        if not any(sharded):
            return self._unsharded_to_hf(state_dict)
        if not all(sharded):
            raise ValueError("Mixed local and distributed tensor export is unsupported")
        return model_hf_distributed_export(self._strip_prefix(state_dict), self.config)

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        **kwargs: str | bool | None,
    ) -> dict[str, torch.Tensor]:
        """Return the native state for tensors in the released layout, closing a load session if one is open.

        Without a session, a complete mapping of plain tensors is converted on its own. Otherwise the mapping
        must be the object that ``to_hf(..., for_checkpoint_load=True)`` returned for the open session, with the
        same keys and the same tensor objects. Anything else makes the adapter refuse all further work: a copy
        of the destinations, made during the session or after it, the destinations of another adapter or of a
        session that has already ended, a mapping of DTensors or an empty mapping without a session.

        Args:
            hf_state_dict: Released Hugging Face names mapped to tensors in the released layouts: either the
                destinations of the active load session after the read, or, when no session is open, exactly
                the released tensors of the decoder-only model as plain tensors, none of which was a
                destination of an earlier session.
            device_mesh: Mesh of a distributed load. A sharded session does not need it. An unsharded session
                and a conversion without a session require ``None``: loading into an unsharded model with a
                device mesh is not supported.
            **kwargs: Other options of the caller. None applies to this model; they are not read.

        Returns:
            Native ``model.`` names mapped to tensors in the shapes of the model's parameters and buffers. For
            a session these are the tensors of the state that ``to_hf`` was given, in its order: the model's
            own tensors, now holding the checkpoint values (for a sharded state on every rank). For a
            conversion without a session they are converted from the given tensors: the grouped expert
            tensors, the split DSA K/V projections and the mHC scale vectors are new tensors, the hyper-head
            matrix is copied when its transpose is not contiguous, and every other tensor is a given object
            or a view of one.

        Raises:
            RuntimeError: If the mapping is not the destinations of the active session, or an earlier session
                was abandoned, replaced or failed.
            ValueError: If the keys or tensors of the destinations were exchanged, a device mesh is given for
                an unsharded conversion, or a conversion rejects its tensors.
        """
        if hf_state_dict and all(_is_lora_key(name) for name in hf_state_dict):
            # The framework copies the adapter tensors of an expert-parallel model into its parameters without
            # comparing them, so a tensor that is missing or broadcasts would be installed silently.
            expected, self._lora_shapes = self._lora_shapes, None
            found = {name: value.shape for name, value in hf_state_dict.items()}
            if expected is not None and found != expected:
                names = sorted(expected.keys() ^ found.keys()) or sorted(
                    name for name in expected if found[name] != expected[name]
                )
                raise ValueError(
                    f"LoRA checkpoint and model differ in the keys or shapes of {len(names)} adapter tensors, "
                    f"the first is {names[0]}"
                )
            return dict(hf_state_dict)
        idle = self._active is None and self._token is None and not self._poisoned
        if (
            idle
            and not isinstance(hf_state_dict, _LoadDestinations)
            and hf_state_dict
            and all(
                isinstance(value, torch.Tensor) and not isinstance(value, DTensor) for value in hf_state_dict.values()
            )
            and not self._holds_ended_destinations(hf_state_dict)
        ):
            return self._unsharded_from_hf(hf_state_dict, device_mesh)
        session = None if self._active is None else self._active()
        if session is None or session is not hf_state_dict:
            # Not the destinations of the open session (a copy, the destinations of another adapter or of a
            # session that has ended), or no session is open: the state of the model is unknown from here on.
            self._poisoned = True
        if self._poisoned or session is None:
            raise RuntimeError("Missing or replaced load session: discard this model and adapter")
        try:
            if session.keys() != session.original.keys():
                raise ValueError("Loaded keys changed")
            if any(value is not session.original[key] for key, value in session.items()):
                raise ValueError("Load destinations were replaced")
            if session.unsharded and device_mesh is not None:
                raise ValueError("Loading into an unsharded model with a device mesh is not supported")
            for finish in session.finishers:
                finish()
            return self._add_prefix(session.native)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._token = None
            self._active = None
            # Remember the tensors that were created for this session, so that a later copy of the mapping is
            # not taken for a fresh checkpoint.
            self._ended = [weak for weak in self._ended if weak() is not None]
            self._ended.extend(session.created)

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: torch.Tensor, **kwargs: str | bool | DeviceMesh | None
    ) -> list[tuple[str, torch.Tensor]]:
        """Reject the conversion of a single tensor.

        The released layout fuses the key and value projections of a DSA layer and splits the grouped expert
        tensors, so one native tensor cannot be converted without the others.

        Args:
            fqn: Native name of the tensor.
            tensor: Tensor of any shape.
            **kwargs: Options of the caller; not read.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("streaming export is not implemented")
