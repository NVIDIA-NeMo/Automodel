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

"""Distributed-checkpoint destinations and export views of a sharded ERNIE 5.5 model.

Load destinations expose the storage of an already sharded, materialized model under the released Hugging Face
names and layouts, so that a distributed checkpoint (DCP) read writes every rank's shard in place. Tensors whose
released layout cannot alias model storage (the three mHC alphas and the fused DSA K/V projection) are read into
temporaries and installed by a ``finish`` callback.

Export views present the same sharded storage under the released names and layouts for a DCP save. Only the
fused DSA K/V projection is copied; the three mHC alphas are slices of the gathered scale vector.

Dtypes. The DCP reader converts the dtype stored in the checkpoint while it fills a destination; nothing here
reads or detects the stored dtype. A destination that aliases model storage has the dtype of the native tensor.
The only destination whose dtype is free is the temporary target of the fused DSA K/V projection:
    * its temporary target dtype is fixed to FP32. FP32 represents every BF16, FP16 and FP32 value exactly, so
      the read does not round for a checkpoint that stores one of these three dtypes;
    * ``finish`` then casts to the dtype of the native key and value projections (BF16, FP16 or FP32);
    * BF16, FP16 and FP32 are the supported stored dtypes. A checkpoint that stores another dtype (FP64,
      integer or quantized formats) is not supported: the reader would convert it silently, and FP64 values
      would be rounded.

Limitations:
    * Storage must be materialized (not on the meta device). mHC parameters and the DSA K/V projections must be
      one-dimensional ``Shard(0)`` DTensors.
    * The K/V shards of a DSA layer must hold whole attention heads: the number of data-parallel ranks must
      divide ``hybrid_mla_num_attention_heads``. With the 64 heads of the released configuration a load or a
      save works on 1, 2, 4, 8, 16, 32 or 64 ranks; on any other number of ranks both raise ``ValueError``.
    * Routed experts must have one of two layouts, without empty shards: ``(Shard(1), Shard(0))`` DTensors on
      an ``(ep_shard, ep)`` mesh, or ``(Shard(0),)`` DTensors on an ``(ep,)`` mesh, where every rank holds
      whole experts (one data-parallel rank per expert group). Tied embedding and output-head weights and MTP
      layers are rejected.
    * Load destinations alias model storage and are written in place. They are meant for a checkpoint load
      between optimizer steps; a failed load leaves the model partially overwritten.
    * All ranks must call the export functions in the same order: gathering the mHC scale is a collective.
      The per-expert keys of a rank cover only the experts that the rank holds.
    * Optimizer state, PEFT adapters, quantized checkpoints and streaming export are not handled here.
"""

from collections.abc import Callable, Mapping
from typing import Literal

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.models.ernie5_5.checkpoint_layout import (
    COMMON_KEYS,
    DSA_HF_TO_NATIVE,
    HCA_HF_TO_NATIVE,
    MOE_HF_TO_NATIVE,
    common_tensor_specs,
    split_fused_kv,
)
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config, Ernie5_5LayerPlan, Ernie5_5ModelPlan

# Temporary target dtype of the fused DSA K/V projection: the dtype of the temporary that a checkpoint read fills.
# It is a fixed choice, not the dtype stored in the checkpoint, which the reader converts. FP32 represents every
# BF16, FP16 and FP32 value exactly, so the read does not round for these stored dtypes.
_FUSED_KV_TEMPORARY_DTYPE = torch.float32

# The two accepted layouts of the grouped expert tensors: mesh dimension names and placements.
_EXPERT_LAYOUTS = ((("ep_shard", "ep"), (Shard(1), Shard(0))), (("ep",), (Shard(0),)))

# Hugging Face names of the two hyper-connection units of a decoder layer, by native name.
_MHC_UNITS = (("attn_hc", "self_attention_hyper_connection"), ("ffn_hc", "mlp_hyper_connection"))


def _shard_range(size: int, shards: int, rank: int) -> tuple[int, int]:
    """Return the contiguous partition used by torch.chunk.

    Args:
        size: Length of the sharded dimension.
        shards: Number of shards, i.e. the size of the mesh dimension that shards it.
        rank: Index of the shard, in ``[0, shards)``.

    Returns:
        Half-open ``(start, end)`` index range of shard ``rank``. Trailing shards are empty when ``size``
        does not fill them.

    Raises:
        ValueError: If an argument is not an integer, ``size`` or ``shards`` is not positive, or ``rank``
            is outside ``[0, shards)``.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (size, shards, rank)):
        raise ValueError("Shard arguments must be integers")
    if size < 1 or shards < 1 or not 0 <= rank < shards:
        raise ValueError("Invalid shard range")
    chunk = (size + shards - 1) // shards
    return min(size, rank * chunk), min(size, (rank + 1) * chunk)


# Load destinations: sharded model storage exposed under Hugging Face names and layouts for a distributed read.


def common_hf_dcp_views(
    state: Mapping[str, torch.Tensor], *, vocab: int, hidden: int, streams: int
) -> dict[str, torch.Tensor]:
    """Return detached model-storage views in the eight released HF layouts.

    Args:
        state: Mapping that contains the eight native model-level tensors, named relative to the inner
            model: ``embed_tokens.weight`` [vocab, hidden], ``lm_head.weight`` [vocab, hidden],
            ``norm.weight`` [hidden], ``multimax.multimax_ranges`` [4], ``multimax.multimax_ts`` [4],
            ``hyper_head.hc_fn`` [streams, streams * hidden], ``hyper_head.hc_base`` [streams] and
            ``hyper_head.hc_scale`` [1]. For a DTensor these are the global shapes; placements are not
            inspected. Every tensor must be materialized and BF16, FP16 or FP32, and the hyper-head
            tensors FP32. Other entries are ignored.
        vocab: Vocabulary size.
        hidden: Hidden size.
        streams: Number of residual streams.

    Returns:
        Hugging Face names mapped to detached views that alias the given storage, so writing a view
        writes the model. Layouts equal the native ones except ``model.mhc_contract.hc_head_fn``, which
        is the transposed view of shape [streams * hidden, streams].

    Raises:
        ValueError: If a dimension is not a positive integer, a tensor is missing, not materialized, of
            another shape or of an unsupported dtype.
    """
    for name, value in (("vocab", vocab), ("hidden", hidden), ("streams", streams)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid {name}: {value}")
    specs = common_tensor_specs(vocab=vocab, hidden=hidden, streams=streams)
    missing = specs.keys() - state.keys()
    if missing:
        raise ValueError(f"Missing common target keys: {sorted(missing)}")
    result = {}
    allowed = (torch.bfloat16, torch.float16, torch.float32)
    for native, (hf, shape) in specs.items():
        value = state[native]
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise ValueError(f"{native} must be a materialized tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{native} has shape {tuple(value.shape)}, expected {shape}")
        if value.dtype not in allowed:
            raise ValueError(f"Unsupported dtype for {native}: {value.dtype}")
        if native.startswith("hyper_head.") and value.dtype != torch.float32:
            raise ValueError(f"{native} must be FP32")
        view = value.detach()
        result[hf] = view.T if native == "hyper_head.hc_fn" else view
    return result


def _validate_mhc(fn: DTensor, base: DTensor, scale: DTensor, *, streams: int, hidden: int, hf_prefix: str) -> None:
    """Check the three parameters of one mHC unit before a distributed load or save.

    Args:
        fn: Materialized FP32 DTensor of global shape [mix, streams * hidden] with placements ``(Shard(0),)``,
            where ``mix`` is ``(2 + streams) * streams``.
        base: Materialized FP32 DTensor of global shape [mix] on the same mesh and device, with the same
            placements.
        scale: Materialized FP32 DTensor of global shape [3] on the same mesh and device, with the same
            placements. Ranks beyond the third hold an empty shard.
        streams: Number of residual streams.
        hidden: Hidden size.
        hf_prefix: Hugging Face prefix of the unit, ending with a dot.

    Raises:
        ValueError: If a dimension or the prefix is invalid, the parameters do not share one mesh that this rank
            belongs to, or placements, shapes, dtype or device are not as described.
        TypeError: If a parameter is not a DTensor.
    """
    for name, value in (("streams", streams), ("hidden", hidden)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid {name}: {value}")
    if not isinstance(hf_prefix, str) or not hf_prefix or not hf_prefix.endswith("."):
        raise ValueError(f"hf_prefix must be a nonempty dotted prefix; got {hf_prefix!r}")
    parameters = (fn, base, scale)
    if any(not isinstance(value, DTensor) for value in parameters):
        raise TypeError("mHC parameters must be DTensors")
    mesh = fn.device_mesh
    if base.device_mesh != mesh or scale.device_mesh != mesh or mesh.get_coordinate() is None:
        raise ValueError(f"The mHC parameters of {hf_prefix} must share one mesh that this rank belongs to")
    placements = [tuple(value.placements) for value in parameters]
    if any(placement != (Shard(0),) for placement in placements):
        raise ValueError(f"The mHC parameters of {hf_prefix} must have placements (Shard(0),); got {placements}")
    mix = (2 + streams) * streams
    shapes = [tuple(value.shape) for value in parameters]
    expected = [(mix, streams * hidden), (mix,), (3,)]
    if shapes != expected:
        raise ValueError(f"The mHC parameters of {hf_prefix} have global shapes {shapes}; expected {expected}")
    if any(
        value.dtype != torch.float32 or value.device != fn.device or value.device.type == "meta" for value in parameters
    ):
        raise ValueError(
            f"The mHC parameters of {hf_prefix} must be materialized FP32 tensors on one device; got dtypes "
            f"{[str(value.dtype) for value in parameters]} on devices {[str(value.device) for value in parameters]}"
        )


class MhcDcpLoadPart:
    """Load the HF projection and bias plus three one-element scale tensors.

    The projection and bias destinations alias model storage. The checkpoint stores the scale as three
    one-element tensors, which cannot alias a sharded vector of three, so they are loaded into replicated
    temporaries and copied into the rank-local scale shard by ``finish``.

    Only one-dimensional ``Shard(0)`` parameters are supported.

    Attributes:
        targets: Hugging Face names mapped to DCP destinations: ``mapping_proj.weight``, the transposed
            view of ``fn`` with global shape [streams * hidden, mix]; ``bias`` with global shape [mix];
            and ``alpha_pre``, ``alpha_post`` and ``alpha_res``, replicated FP32 temporaries of shape [1].
    """

    def __init__(
        self, fn: DTensor, base: DTensor, scale: DTensor, *, streams: int, hidden: int, hf_prefix: str
    ) -> None:
        """Validate the parameters of one mHC unit and build its destinations.

        Args:
            fn: Materialized FP32 DTensor of global shape [mix, streams * hidden] with placements
                ``(Shard(0),)``, where ``mix`` is ``(2 + streams) * streams``.
            base: Materialized FP32 DTensor of global shape [mix] on the same mesh and device, with the
                same placements.
            scale: Materialized FP32 DTensor of global shape [3] on the same mesh and device, with the
                same placements. Ranks beyond the third hold an empty shard.
            streams: Number of residual streams.
            hidden: Hidden size.
            hf_prefix: Hugging Face prefix of the unit, ending with a dot.

        Raises:
            ValueError: If a dimension or the prefix is invalid, the parameters do not share one mesh that
                this rank belongs to, placements, shapes, dtype or device are not as described, or the
                local scale shard has an unexpected size.
            TypeError: If a parameter is not a DTensor.
        """
        _validate_mhc(fn, base, scale, streams=streams, hidden=hidden, hf_prefix=hf_prefix)
        mesh = fn.device_mesh
        self._scale = scale.to_local().detach()
        rank = mesh.get_coordinate()[0]
        self._start, end = _shard_range(3, mesh.size(), rank)
        if self._scale.numel() != end - self._start:
            raise ValueError(
                f"The local shard of the mHC scale of {hf_prefix} has {self._scale.numel()} elements; "
                f"expected {end - self._start}"
            )
        self._alpha_names = ("alpha_pre", "alpha_post", "alpha_res")
        self.targets = {
            hf_prefix + "mapping_proj.weight": fn.detach().T,
            hf_prefix + "bias": base.detach(),
        }
        self._alphas = []
        for name in self._alpha_names:
            local = torch.empty((1,), dtype=torch.float32, device=fn.device)
            target = DTensor.from_local(
                local, mesh, (Replicate(),), shape=torch.Size((1,)), stride=(1,), run_check=False
            )
            self._alphas.append(target)
            self.targets[hf_prefix + name] = target
        self._finished = False

    @torch.no_grad()
    def finish(self) -> None:
        """Install the loaded alpha values in the local scale shard once.

        Raises:
            RuntimeError: If the part has already been finished.
        """
        if self._finished:
            raise RuntimeError("mHC load part has already been finished")
        for offset in range(self._scale.numel()):
            self._scale[offset].copy_(self._alphas[self._start + offset].to_local()[0])
        self._finished = True


def _validate_kv(
    key: DTensor, value: DTensor, *, heads: int, latent: int, key_channels: int, value_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check the split K/V projections of one DSA layer and return this rank's shards.

    The projections are sharded along their first axis over the data-parallel ranks, and every rank must hold
    whole attention heads: the number of ranks of the mesh must divide the head count.

    Args:
        key: Materialized DTensor of global shape [heads * latent, key_channels] with placements
            ``(Shard(0),)``; BF16, FP16 or FP32.
        value: Materialized DTensor of global shape [heads * value_channels, latent] on the same mesh and
            device, with the same placements; BF16, FP16 or FP32.
        heads: Global number of attention heads, ``hybrid_mla_num_attention_heads`` of the configuration; a
            multiple of the mesh size.
        latent: Latent width of the K/V projection.
        key_channels: Key rows per head in the fused matrix.
        value_channels: Value rows per head in the fused matrix.

    Returns:
        The detached local shards, which alias the given storage: keys of shape
        [local_heads * latent, key_channels] and values of shape [local_heads * value_channels, latent], where
        ``local_heads`` is ``heads`` divided by the mesh size.

    Raises:
        ValueError: If a dimension is invalid, the projections do not share one mesh that this rank belongs
            to, the number of ranks of the mesh does not divide the head count, or placements, shapes, dtypes
            or devices are not as described.
        TypeError: If a projection is not a DTensor.
    """
    for name, size in (
        ("heads", heads),
        ("latent", latent),
        ("key_channels", key_channels),
        ("value_channels", value_channels),
    ):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"Invalid {name}: {size}")
    if not isinstance(key, DTensor) or not isinstance(value, DTensor):
        raise TypeError("K and V must be DTensors")
    mesh = key.device_mesh
    if value.device_mesh != mesh or mesh.get_coordinate() is None:
        raise ValueError("The K and V projections must share one mesh that this rank belongs to")
    if key.placements != (Shard(0),) or value.placements != (Shard(0),):
        raise ValueError(
            "The K and V projections must have placements (Shard(0),); got "
            f"{tuple(key.placements)} and {tuple(value.placements)}"
        )
    ranks = mesh.size()
    if heads % ranks != 0:
        raise ValueError(
            f"The K and V projections of a DSA layer are sharded over {ranks} data-parallel ranks, and every rank "
            "must hold whole attention heads: the number of data-parallel ranks must divide "
            f"hybrid_mla_num_attention_heads={heads}"
        )
    shapes = [tuple(key.shape), tuple(value.shape)]
    expected = [(heads * latent, key_channels), (heads * value_channels, latent)]
    if shapes != expected:
        raise ValueError(f"The K and V projections have global shapes {shapes}; expected {expected}")
    allowed = (torch.bfloat16, torch.float16, torch.float32)
    if key.dtype not in allowed or value.dtype not in allowed:
        raise ValueError(f"The K and V projections must be BF16, FP16 or FP32; got {key.dtype} and {value.dtype}")
    if key.device != value.device or key.device.type == "meta":
        raise ValueError(
            f"The K and V projections must be materialized on one device; got {key.device} and {value.device}"
        )
    local_heads = heads // ranks
    local_key, local_value = key.to_local().detach(), value.to_local().detach()
    local_shapes = [tuple(local_key.shape), tuple(local_value.shape)]
    local_expected = [(local_heads * latent, key_channels), (local_heads * value_channels, latent)]
    if local_shapes != local_expected:
        raise ValueError(f"The local K and V shards have shapes {local_shapes}; expected {local_expected}")
    return local_key, local_value


class DsaKVDcpLoadPart:
    """Load a fused HF KV tensor into a temporary, then install split weights.

    Only one-dimensional ``Shard(0)`` projections whose shards hold whole attention heads are supported:
    the number of ranks of the mesh must divide the head count.

    The temporary target dtype is fixed to FP32. It is not the dtype stored in the checkpoint, which this
    class never sees: the DCP reader converts the stored values while it fills the temporary. That conversion
    is exact for the supported stored dtypes BF16, FP16 and FP32; another stored dtype is not supported.
    ``finish`` then casts to the dtype of the native projections.

    Attributes:
        target: Temporary FP32 DTensor of global shape [heads * (key_channels + value_channels), latent] with
            placements ``(Shard(0),)``; the DCP destination of the released ``kv_b_proj.weight``. Each rank
            holds the rows of its own heads.
    """

    def __init__(
        self,
        key_weight: DTensor,
        value_weight: DTensor,
        *,
        heads: int,
        latent: int,
        key_channels: int,
        value_channels: int,
    ) -> None:
        """Validate the split projections and allocate the rank-local fused temporary.

        Args:
            key_weight: Materialized DTensor of global shape [heads * latent, key_channels] with
                placements ``(Shard(0),)``; BF16, FP16 or FP32.
            value_weight: Materialized DTensor of global shape [heads * value_channels, latent] on the
                same mesh and device, with the same placements; BF16, FP16 or FP32. It must not share
                storage with ``key_weight``.
            heads: Global number of attention heads, ``hybrid_mla_num_attention_heads`` of the
                configuration; a multiple of the mesh size.
            latent: Latent width of the K/V projection.
            key_channels: Key rows per head in the fused matrix.
            value_channels: Value rows per head in the fused matrix.

        Raises:
            ValueError: If a dimension is invalid, the projections do not share one mesh that this rank
                belongs to, the number of ranks of the mesh does not divide the head count, placements,
                shapes, dtypes or devices are not as described, or the projections share storage.
            TypeError: If a projection is not a DTensor.
        """
        self._key, self._value = _validate_kv(
            key_weight,
            value_weight,
            heads=heads,
            latent=latent,
            key_channels=key_channels,
            value_channels=value_channels,
        )
        if self._key.untyped_storage().data_ptr() == self._value.untyped_storage().data_ptr():
            raise ValueError("The K and V projections must not share storage")
        mesh = key_weight.device_mesh
        local_heads = heads // mesh.size()
        self._heads, self._latent = local_heads, latent
        self._key_channels, self._value_channels = key_channels, value_channels
        local = torch.empty(
            (local_heads * (key_channels + value_channels), latent),
            device=self._key.device,
            dtype=_FUSED_KV_TEMPORARY_DTYPE,
        )
        self.target = DTensor.from_local(
            local,
            mesh,
            (Shard(0),),
            shape=torch.Size((heads * (key_channels + value_channels), latent)),
            stride=(latent, 1),
            run_check=False,
        )
        self._finished = False

    @torch.no_grad()
    def finish(self) -> None:
        """Install a fully loaded temporary into the detached model targets once.

        The rank-local rows of the fused temporary are split per head and copied, with a cast to the
        dtype of the native projections, into the local shards of the key and value projections.

        Raises:
            RuntimeError: If the part has already been finished.
        """
        if self._finished:
            raise RuntimeError("KV load part has already been finished")
        key, value = split_fused_kv(
            self.target.to_local().detach(),
            heads=self._heads,
            latent=self._latent,
            key_channels=self._key_channels,
            value_channels=self._value_channels,
        )
        self._key.copy_(key)
        self._value.copy_(value)
        self._finished = True


def expert_hf_dcp_views(
    gate_up: DTensor,
    down: DTensor,
    *,
    layer: int,
    experts: int,
    latent: int,
    intermediate: int,
) -> dict[str, torch.Tensor]:
    """Return rank-local HF w1/w2/w3 destinations aliasing expert storage.

    Two layouts of the grouped expert tensors are accepted:

    * ``(Shard(1), Shard(0))`` over ``(ep_shard, ep)``: ``ep`` shards the expert axis and ``ep_shard`` the
      next axis. The resulting per-expert checkpoint tensors retain the ``ep_shard`` mesh and shard their HF
      column dimension.
    * ``(Shard(0),)`` over ``(ep,)``: every rank holds whole experts, as in a run with one data-parallel rank
      per expert group. The resulting per-expert checkpoint tensors are plain tensors of the rank that holds
      the expert.

    A checkpoint load writes through the views and must run between optimizer steps; an export only reads
    them. No caller may assume that all keys are present on every rank.

    Args:
        gate_up: Materialized DTensor of global shape [experts, latent, 2 * intermediate] in one of the two
            layouts. ``ep`` shards the expert axis; ``ep_shard``, when present, shards the latent axis. The
            last axis holds the transposed gate projection followed by the transposed up projection.
        down: Materialized DTensor of global shape [experts, intermediate, latent] on the same mesh and
            device, with the same placements and dtype. ``ep`` shards the expert axis; ``ep_shard``, when
            present, shards the intermediate axis. It must not share storage with ``gate_up``.
        layer: Index of the decoder layer, used in the Hugging Face names.
        experts: Global number of routed experts.
        latent: Latent width of the expert projections.
        intermediate: Intermediate width of the expert projections.

    Returns:
        For every expert held by this rank, ``model.layers.{layer}.block_sparse_moe.experts.{id}.w1.weight``
        and ``...w3.weight`` of global shape [intermediate, latent] and ``...w2.weight`` of global shape
        [latent, intermediate]. On an ``(ep_shard, ep)`` mesh each is a DTensor on the ``ep_shard`` mesh
        with placements ``(Shard(1),)`` whose local tensor is a transposed view of this rank's expert
        storage. On an ``(ep,)`` mesh each is a plain tensor, the complete matrix as a transposed view of
        this rank's expert storage.

    Raises:
        ValueError: If a dimension is invalid, the layout is neither of the two accepted ones, the two
            tensors differ in mesh or placements, shapes, dtype or device are not as described, this rank
            is outside the mesh, a shard is empty or the tensors share storage.
        TypeError: If a parameter is not a DTensor.
    """
    for name, value in (("layer", layer), ("experts", experts), ("latent", latent), ("intermediate", intermediate)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "layer" else 1):
            raise ValueError(f"Invalid {name}: {value}")
    if not isinstance(gate_up, DTensor) or not isinstance(down, DTensor):
        raise TypeError("Expert parameters must be DTensors")
    mesh = gate_up.device_mesh
    mesh_names = tuple(mesh.mesh_dim_names or ())
    if (
        (mesh_names, tuple(gate_up.placements)) not in _EXPERT_LAYOUTS
        or down.placements != gate_up.placements
        or down.device_mesh != mesh
    ):
        # The names of a framework mesh are members of a string enumeration; joining them prints their values.
        raise ValueError(
            "Expert parameters must share one mesh and be (Shard(1), Shard(0)) DTensors on an (ep_shard, ep) mesh "
            f"or (Shard(0),) DTensors on an (ep,) mesh; got placements {tuple(gate_up.placements)} and "
            f"{tuple(down.placements)} on mesh dimensions ({', '.join(mesh_names)})"
        )
    shapes = [tuple(gate_up.shape), tuple(down.shape)]
    expected = [(experts, latent, 2 * intermediate), (experts, intermediate, latent)]
    if shapes != expected:
        raise ValueError(
            f"The grouped expert tensors of layer {layer} have global shapes {shapes}; expected {expected}"
        )
    if gate_up.dtype != down.dtype or gate_up.device != down.device or gate_up.device.type == "meta":
        raise ValueError(
            f"The grouped expert tensors of layer {layer} must share one dtype and one materialized device; got "
            f"{gate_up.dtype} on {gate_up.device} and {down.dtype} on {down.device}"
        )
    coord = mesh.get_coordinate()
    if coord is None:
        raise ValueError("Rank is outside the expert mesh")
    if len(mesh_names) == 2:
        shard_rank, ep_rank = coord
        shard_mesh, ep_mesh = mesh["ep_shard"], mesh["ep"]
        e0, e1 = _shard_range(experts, ep_mesh.size(), ep_rank)
        l0, l1 = _shard_range(latent, shard_mesh.size(), shard_rank)
        i0, i1 = _shard_range(intermediate, shard_mesh.size(), shard_rank)
    else:
        # Every rank holds whole experts: only the expert axis is partitioned.
        shard_mesh = None
        e0, e1 = _shard_range(experts, mesh.size(), coord[0])
        l0, l1, i0, i1 = 0, latent, 0, intermediate
    if e0 == e1 or l0 == l1 or i0 == i1:
        raise ValueError("Empty expert or inner-dimension shards are unsupported")
    gate_local, down_local = gate_up.to_local().detach(), down.to_local().detach()
    if gate_local.untyped_storage().data_ptr() == down_local.untyped_storage().data_ptr():
        raise ValueError("Gate/up and down tensors must not share storage")
    local_shapes = [tuple(gate_local.shape), tuple(down_local.shape)]
    local_expected = [(e1 - e0, l1 - l0, 2 * intermediate), (e1 - e0, i1 - i0, latent)]
    if local_shapes != local_expected:
        raise ValueError(
            f"The local shards of the grouped expert tensors of layer {layer} have shapes {local_shapes}; "
            f"expected {local_expected}"
        )
    result: dict[str, torch.Tensor] = {}
    for local_id, expert_id in enumerate(range(e0, e1)):
        specs = (
            (1, gate_local[local_id, :, :intermediate].T, (intermediate, latent)),
            (3, gate_local[local_id, :, intermediate:].T, (intermediate, latent)),
            (2, down_local[local_id].T, (latent, intermediate)),
        )
        for projection, view, shape in specs:
            key = f"model.layers.{layer}.block_sparse_moe.experts.{expert_id}.w{projection}.weight"
            if shard_mesh is None:
                result[key] = view
            else:
                result[key] = DTensor.from_local(
                    view, shard_mesh, (Shard(1),), shape=torch.Size(shape), stride=(shape[1], 1), run_check=False
                )
    return result


def _decoder_layer_names(
    attention: Literal["hca", "dsa"], ffn: Literal["dense", "moe"]
) -> tuple[dict[str, str], set[str]]:
    """Return the native tensor names of one decoder layer, split by how they reach the released layout.

    Args:
        attention: Attention kind of the layer.
        ffn: Feed-forward kind of the layer.

    Returns:
        Two collections of names relative to the decoder block. The first maps every tensor whose released
        layout equals the native one to its released name relative to ``model.layers.{index}``: the two norm
        weights, the attention tensors other than the split K/V projections, and the dense projections or the
        non-expert MoE tensors. The second holds the names whose layout is converted: the six mHC parameters,
        the two grouped expert tensors of a MoE layer and the split K/V projections of a DSA layer.
    """
    direct = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    }
    for hf, native in (HCA_HF_TO_NATIVE if attention == "hca" else DSA_HF_TO_NATIVE).items():
        direct["attention." + native] = "self_attn." + hf
    if ffn == "dense":
        for projection in ("gate_proj", "up_proj", "down_proj"):
            direct[f"mlp.{projection}.weight"] = f"mlp.{projection}.weight"
    else:
        for hf, native in MOE_HF_TO_NATIVE.items():
            direct["moe." + native] = "block_sparse_moe." + hf
    converted = {f"{unit}.{field}" for unit, _hf in _MHC_UNITS for field in ("fn", "base", "scale")}
    if ffn == "moe":
        converted |= {"moe.experts.gate_and_up_projs", "moe.experts.down_projs"}
    if attention == "dsa":
        converted |= {"attention.k_b_proj", "attention.v_b_proj"}
    return direct, converted


def model_hf_dcp_load_destinations(
    state: Mapping[str, torch.Tensor], config: Ernie5_5Config
) -> tuple[dict[str, torch.Tensor], list[Callable[[], None]]]:
    """Return the DCP destinations of a whole sharded model and the callbacks that complete the load.

    Every layer's destinations are created before the read starts, and every layer owns its temporaries. The
    caller must run all callbacks, in order, after the read and before the model is used, and must discard the
    model after any failure.

    Args:
        state: Exactly the materialized native tensors of the model, named relative to the inner model: the
            eight model-level tensors in the layouts of ``common_hf_dcp_views`` and, for every decoder layer,
            ``layers.{index}.*`` with the module names ``mlp`` (dense) or ``moe`` (MoE) for the feed-forward.
            The two norm weights, the attention tensors other than the split K/V projections, the dense
            projections and the non-expert MoE tensors may have any shape and placement and are aliased as
            they are. ``attn_hc.{fn,base,scale}`` and ``ffn_hc.{fn,base,scale}`` follow ``MhcDcpLoadPart``;
            ``moe.experts.gate_and_up_projs`` and ``moe.experts.down_projs`` follow ``expert_hf_dcp_views``;
            ``attention.k_b_proj`` and ``attention.v_b_proj`` follow ``DsaKVDcpLoadPart``.
        config: Decoder-only model configuration with untied embedding and output head; supplies the layer
            plan and the dimensions.

    Returns:
        The destinations and the callbacks. The destinations map released Hugging Face names to tensors in the
        released layouts: the model-level tensors first, then one decoder layer after another. Every destination
        aliases model storage except the six mHC alphas of a layer (replicated FP32 temporaries of shape [1])
        and the fused K/V projection of a DSA layer (a temporary of the fixed temporary target dtype FP32, see
        ``DsaKVDcpLoadPart``). The per-expert names cover only the experts held by this rank. The callbacks
        install the temporaries: two per layer for the mHC units and one per DSA layer for the K/V projection,
        in layer order.

    Raises:
        ValueError: If the configuration has MTP layers or tied embeddings, a model-level tensor is missing, a
            layer has a missing or unexpected name, a name is outside the model-level tensors and the decoder
            layers, a tensor is not materialized, two destinations share a Hugging Face name or a component
            rejects its tensors, for example because the number of data-parallel ranks does not divide the
            number of DSA attention heads.
        TypeError: If an mHC, expert or K/V parameter is not a DTensor.
    """
    if config.num_nextn_predict_layers != 0 or config.tie_word_embeddings:
        raise ValueError("Only explicit MTP=0 with untied embeddings is supported")
    plan = Ernie5_5ModelPlan.from_config(config)
    targets: dict[str, torch.Tensor] = {}
    finishers: list[Callable[[], None]] = []
    covered: set[str] = set()

    def add(mapping: Mapping[str, torch.Tensor]) -> None:
        """Register destinations, rejecting a Hugging Face name that is already taken.

        Args:
            mapping: Hugging Face names mapped to destination tensors of any shape.

        Raises:
            ValueError: If a name is already registered.
        """
        overlap = targets.keys() & mapping.keys()
        if overlap:
            raise ValueError(f"Duplicate checkpoint targets: {sorted(overlap)}")
        targets.update(mapping)

    if not COMMON_KEYS <= state.keys():
        raise ValueError("Missing common model tensors")
    add(common_hf_dcp_views(state, vocab=config.vocab_size, hidden=config.hidden_size, streams=config.hc_mult))
    covered.update(COMMON_KEYS)
    for layer in plan.decoder_layers:
        native_prefix = f"layers.{layer.layer_index}."
        hf_prefix = f"model.layers.{layer.layer_index}."
        relative = {key[len(native_prefix) :]: value for key, value in state.items() if key.startswith(native_prefix)}
        direct, converted = _decoder_layer_names(layer.attention, layer.feed_forward)
        required = set(direct) | converted
        if relative.keys() != required:
            raise ValueError(
                f"Invalid layer {layer.layer_index}: missing={sorted(required - relative.keys())}, "
                f"unexpected={sorted(relative.keys() - required)}"
            )
        covered.update(native_prefix + name for name in required)
        for native, hf in direct.items():
            value = relative[native]
            if value.device.type == "meta":
                raise ValueError(f"{native_prefix}{native} must be materialized")
            add({hf_prefix + hf: value.detach()})
        for native, hf in _MHC_UNITS:
            mhc_part = MhcDcpLoadPart(
                relative[native + ".fn"],
                relative[native + ".base"],
                relative[native + ".scale"],
                streams=config.hc_mult,
                hidden=config.hidden_size,
                hf_prefix=hf_prefix + hf + ".",
            )
            add(mhc_part.targets)
            finishers.append(mhc_part.finish)
        if layer.feed_forward == "moe":
            add(
                expert_hf_dcp_views(
                    relative["moe.experts.gate_and_up_projs"],
                    relative["moe.experts.down_projs"],
                    layer=layer.layer_index,
                    experts=config.n_routed_experts,
                    latent=config.moe_latent_size,
                    intermediate=config.moe_intermediate_size,
                )
            )
        if layer.attention == "dsa":
            kv_part = DsaKVDcpLoadPart(
                relative["attention.k_b_proj"],
                relative["attention.v_b_proj"],
                heads=config.hybrid_mla_num_attention_heads,
                latent=config.hybrid_mla_kv_lora_rank,
                key_channels=config.hybrid_mla_qk_nope_head_dim,
                value_channels=config.hybrid_mla_v_head_dim,
            )
            add({hf_prefix + "self_attn.kv_b_proj.weight": kv_part.target})
            finishers.append(kv_part.finish)
    if covered != set(state):
        raise ValueError(f"Unexpected native model keys: {sorted(set(state) - covered)}")
    return targets, finishers


# Export views: sharded model storage presented under Hugging Face names and layouts for a distributed save.


def mhc_hf_export_views(
    fn: DTensor, base: DTensor, scale: DTensor, *, streams: int, hidden: int, hf_prefix: str
) -> dict[str, DTensor]:
    """Return the five released tensors of one mHC unit for a distributed save.

    All ranks of the mesh must call this function in the same order: replicating the scale is a collective.

    Args:
        fn: Materialized FP32 DTensor of global shape [mix, streams * hidden] with placements ``(Shard(0),)``,
            where ``mix`` is ``(2 + streams) * streams``.
        base: Materialized FP32 DTensor of global shape [mix] on the same mesh and device, with the same
            placements.
        scale: Materialized FP32 DTensor of global shape [3] on the same mesh and device, with the same
            placements. Ranks beyond the third hold an empty shard.
        streams: Number of residual streams.
        hidden: Hidden size.
        hf_prefix: Hugging Face prefix of the unit, ending with a dot.

    Returns:
        Hugging Face names mapped to DTensors on the same mesh: ``mapping_proj.weight``, the transposed view of
        ``fn`` with global shape [streams * hidden, mix]; ``bias``, a view of ``base`` with global shape [mix];
        and ``alpha_pre``, ``alpha_post`` and ``alpha_res``, replicated FP32 tensors of shape [1] sliced from a
        gathered copy of ``scale`` (elements 0, 1 and 2). The projection and the bias alias model storage.

    Raises:
        ValueError: If a dimension or the prefix is invalid, the parameters do not share one mesh that this
            rank belongs to, or placements, shapes, dtype or device are not as described.
        TypeError: If a parameter is not a DTensor.
    """
    _validate_mhc(fn, base, scale, streams=streams, hidden=hidden, hf_prefix=hf_prefix)
    replicated_scale = scale.detach().redistribute(fn.device_mesh, (Replicate(),))
    result = {hf_prefix + "mapping_proj.weight": fn.detach().T, hf_prefix + "bias": base.detach()}
    for index, name in enumerate(("alpha_pre", "alpha_post", "alpha_res")):
        result[hf_prefix + name] = replicated_scale[index : index + 1]
    return result


def merge_dsa_kv_for_hf(
    key: DTensor, value: DTensor, *, heads: int, latent: int, key_channels: int, value_channels: int
) -> DTensor:
    """Merge the split key and value projections of one DSA layer into the released fused layout.

    Only one-dimensional ``Shard(0)`` projections whose shards hold whole attention heads are supported: the
    number of ranks of the mesh must divide the head count. Each rank merges its own heads; no data moves
    between ranks.

    Args:
        key: Materialized DTensor of global shape [heads * latent, key_channels] with placements
            ``(Shard(0),)``; BF16, FP16 or FP32.
        value: Materialized DTensor of global shape [heads * value_channels, latent] on the same mesh and
            device, with the same placements and dtype.
        heads: Global number of attention heads, ``hybrid_mla_num_attention_heads`` of the configuration; a
            multiple of the mesh size.
        latent: Latent width of the K/V projection.
        key_channels: Key rows per head in the fused matrix.
        value_channels: Value rows per head in the fused matrix.

    Returns:
        A newly allocated DTensor of global shape [heads * (key_channels + value_channels), latent] with
        placements ``(Shard(0),)`` and the device and dtype of the inputs. It stores, for each head in turn,
        the transposed key block followed by the value rows. Each rank holds the rows of its own heads.

    Raises:
        ValueError: If a dimension is invalid, the projections do not share one mesh that this rank belongs
            to, the number of ranks of the mesh does not divide the head count, or placements, shapes, dtypes
            or devices are not as described.
        TypeError: If a projection is not a DTensor.
    """
    local_key, local_value = _validate_kv(
        key, value, heads=heads, latent=latent, key_channels=key_channels, value_channels=value_channels
    )
    if key.dtype != value.dtype:
        raise ValueError(
            f"The K and V projections must have matching dtypes to be merged; got {key.dtype} and {value.dtype}"
        )
    mesh = key.device_mesh
    local_heads = heads // mesh.size()
    local = (
        torch.cat(
            (
                local_key.reshape(local_heads, latent, key_channels).transpose(1, 2),
                local_value.reshape(local_heads, value_channels, latent),
            ),
            dim=1,
        )
        .reshape(local_heads * (key_channels + value_channels), latent)
        .contiguous()
    )
    shape = (heads * (key_channels + value_channels), latent)
    return DTensor.from_local(local, mesh, (Shard(0),), shape=torch.Size(shape), stride=(latent, 1), run_check=False)


def _decoder_export_part(
    state: Mapping[str, torch.Tensor], config: Ernie5_5Config, layer: Ernie5_5LayerPlan
) -> dict[str, torch.Tensor]:
    """Return the released tensors of one sharded decoder layer for a distributed save.

    All ranks of the meshes must call this function in the same order. The per-expert names differ between
    the ranks of the ``ep`` mesh dimension.

    Args:
        state: Exactly the materialized tensors of one decoder block, named relative to it with the module
            names ``mlp`` (dense) or ``moe`` (MoE) for the feed-forward. The two norm weights, the attention
            tensors other than the split K/V projections, the dense projections and the non-expert MoE
            tensors may have any shape and placement and are exported as they are.
            ``attn_hc.{fn,base,scale}`` and ``ffn_hc.{fn,base,scale}`` follow ``mhc_hf_export_views``;
            ``moe.experts.gate_and_up_projs`` and ``moe.experts.down_projs`` follow ``expert_hf_dcp_views``;
            ``attention.k_b_proj`` and ``attention.v_b_proj`` follow ``merge_dsa_kv_for_hf``.
        config: Model configuration; supplies the stream count, the hidden size, the number and widths of the
            routed experts and the DSA K/V dimensions.
        layer: Index and attention and feed-forward kinds of the layer.

    Returns:
        ``model.layers.{index}.*`` names mapped to tensors in the released layouts: the tensors exported as
        they are, then the five tensors of each mHC unit, then the per-expert matrices held by this rank and
        the fused K/V projection. Every tensor aliases model storage except the mHC alphas and the fused K/V
        projection, which are new.

    Raises:
        ValueError: If a state name is missing or unexpected, a tensor is not materialized, two results share
            a Hugging Face name or a component rejects its tensors, for example because the number of
            data-parallel ranks does not divide the number of DSA attention heads.
        TypeError: If an mHC, expert or K/V parameter is not a DTensor.
    """
    prefix = f"model.layers.{layer.layer_index}."
    direct, converted = _decoder_layer_names(layer.attention, layer.feed_forward)
    required = set(direct) | converted
    if set(state) != required:
        raise ValueError(
            f"Invalid native keys: missing={sorted(required - state.keys())}, "
            f"unexpected={sorted(state.keys() - required)}"
        )
    result: dict[str, torch.Tensor] = {}

    def add(mapping: Mapping[str, torch.Tensor]) -> None:
        """Register exported tensors, rejecting a Hugging Face name that is already taken.

        Args:
            mapping: Hugging Face names mapped to tensors of any shape.

        Raises:
            ValueError: If a name is already registered.
        """
        if result.keys() & mapping.keys():
            raise ValueError("Duplicate HF tensor name")
        result.update(mapping)

    for native, hf in direct.items():
        value = state[native]
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise ValueError(f"{native} must be materialized")
        add({prefix + hf: value.detach()})
    for native, hf in _MHC_UNITS:
        add(
            mhc_hf_export_views(
                state[native + ".fn"],
                state[native + ".base"],
                state[native + ".scale"],
                streams=config.hc_mult,
                hidden=config.hidden_size,
                hf_prefix=prefix + hf + ".",
            )
        )
    if layer.feed_forward == "moe":
        add(
            expert_hf_dcp_views(
                state["moe.experts.gate_and_up_projs"],
                state["moe.experts.down_projs"],
                layer=layer.layer_index,
                experts=config.n_routed_experts,
                latent=config.moe_latent_size,
                intermediate=config.moe_intermediate_size,
            )
        )
    if layer.attention == "dsa":
        add(
            {
                prefix + "self_attn.kv_b_proj.weight": merge_dsa_kv_for_hf(
                    state["attention.k_b_proj"],
                    state["attention.v_b_proj"],
                    heads=config.hybrid_mla_num_attention_heads,
                    latent=config.hybrid_mla_kv_lora_rank,
                    key_channels=config.hybrid_mla_qk_nope_head_dim,
                    value_channels=config.hybrid_mla_v_head_dim,
                )
            }
        )
    return result


def model_hf_distributed_export(state: Mapping[str, torch.Tensor], config: Ernie5_5Config) -> dict[str, torch.Tensor]:
    """Return the released tensors of a whole sharded model for a distributed save.

    All ranks must call this function in the same order and must discard the result after any failure. It does
    not write files.

    Args:
        state: Exactly the materialized native tensors of the model, named relative to the inner model: the
            eight model-level tensors in the layouts of ``common_hf_dcp_views`` and, for every decoder layer,
            ``layers.{index}.*`` with the module names ``mlp`` (dense) or ``moe`` (MoE) for the feed-forward.
            The two norm weights, the attention tensors other than the split K/V projections, the dense
            projections and the non-expert MoE tensors may have any shape and placement and are exported as
            they are. ``attn_hc.{fn,base,scale}`` and ``ffn_hc.{fn,base,scale}`` follow
            ``mhc_hf_export_views``; ``moe.experts.gate_and_up_projs`` and ``moe.experts.down_projs`` follow
            ``expert_hf_dcp_views``; ``attention.k_b_proj`` and ``attention.v_b_proj`` follow
            ``merge_dsa_kv_for_hf``.
        config: Model configuration without MTP layers; supplies the layer plan and the dimensions.

    Returns:
        Released Hugging Face names mapped to tensors in the released layouts: the eight model-level tensors
        first, then one decoder layer after another. Every tensor aliases model storage except the mHC alphas
        and the fused K/V projection of a DSA layer, which are new. The per-expert names cover only the
        experts held by this rank.

    Raises:
        NotImplementedError: If the configuration has MTP layers.
        ValueError: If a model-level tensor is missing, a layer has a missing or unexpected name, a name is
            outside the model-level tensors and the decoder layers, a tensor is not materialized, two results
            share a Hugging Face name or a component rejects its tensors, for example because the number of
            data-parallel ranks does not divide the number of DSA attention heads.
        TypeError: If an mHC, expert or K/V parameter is not a DTensor.
    """
    plan = Ernie5_5ModelPlan.from_config(config)
    if plan.mtp_attention:
        raise NotImplementedError("MTP export is unsupported")
    if not COMMON_KEYS <= state.keys():
        raise ValueError("Missing common model tensors")
    covered = set(COMMON_KEYS)
    result = common_hf_dcp_views(state, vocab=config.vocab_size, hidden=config.hidden_size, streams=config.hc_mult)
    for layer in plan.decoder_layers:
        prefix = f"layers.{layer.layer_index}."
        local = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        converted = _decoder_export_part(local, config, layer)
        if result.keys() & converted.keys():
            raise ValueError("Duplicate HF output keys")
        result.update(converted)
        covered.update(prefix + key for key in local)
    if covered != set(state):
        raise ValueError(f"Unexpected native model tensors: {sorted(set(state) - covered)[:5]}")
    return result
