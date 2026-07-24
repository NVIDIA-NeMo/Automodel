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

"""Tensor-free control-plane contracts for the speculative-training stream.

Per the train-inference disaggregation RFC, every produced sample
is split into a tensor-free reference (this module) and the actual supervision
tensors, which live in a pluggable :mod:`store`. References hop between the
target-side producer and the draft-side consumer over queues, HTTP, and
checkpoint metadata -- places where serializing a :class:`torch.Tensor` would be
either expensive or wrong.

Three guarantees back this module:

1. :func:`assert_no_tensors` enforces "no tensors" recursively on every public
   object exposed from this package, so a producer that slipped a tensor into a
   queue or ref trips a validation error before it lands on a wire.
2. :class:`SampleRef` is a frozen dataclass. Once placed on a queue, a ref
   cannot be mutated under the holder, so the consumer is guaranteed to see the
   same feature keys the producer promised.
3. :class:`FeatureSpec` carries ``dtype`` + ``shape`` (and nothing else), the
   minimum metadata a consumer needs to preallocate the receive buffer *before*
   materializing the tensor -- mirroring how
   :func:`nemo_automodel.components.speculative.eagle.remote.protocol.encode_nccl_metadata`
   already ships dtype + shape ahead of an NCCL recv so the client can allocate.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, NoReturn

import torch

from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

# NumPy is optional. A numpy.ndarray is treated the same as a torch.Tensor for
# the no-tensors invariant -- we reject it.
_np_ok, _np_module = safe_import("numpy", alt=None)


class FeatureAlgorithm(str, Enum):
    """Which speculative-decoding draft family produced this sample.

    The on-wire string value (``eagle3`` / ``dflash`` / ``dspark``) is the
    stable schema key the consumer matches against. New algorithms land by
    adding a member here plus a schema row in the RFC's "Feature schema per
    algorithm" table; both moves are part of the same change.
    """

    EAGLE3 = "eagle3"
    DFLASH = "dflash"
    DSPARK = "dspark"


def _algorithm_required_features(algo: FeatureAlgorithm) -> frozenset[str]:
    """Required feature-name set per algorithm (RFC "Feature schema per algorithm").

    Mirrors the "Required feature keys" column of the RFC's schema table.
    Algorithm-specific schema modules (for example :mod:`eagle3`) add per-key
    dtype and shape validation on top of this name set.
    """
    if algo is FeatureAlgorithm.EAGLE3:
        return frozenset({"aux_hidden_states", "input_ids", "attention_mask", "loss_mask"})
    if algo is FeatureAlgorithm.DFLASH:
        return frozenset({"hidden_states", "input_ids", "attention_mask", "loss_mask"})
    if algo is FeatureAlgorithm.DSPARK:
        return frozenset({"target_hidden_states", "target_last_hidden_states", "input_ids", "loss_mask"})
    raise ValueError(f"Unknown FeatureAlgorithm: {algo!r}")


@dataclass(frozen=True)
class FeatureSpec:
    """Shape + dtype metadata for one named feature in a :class:`SampleRef`.

    Mirrors the ``{dtype_code, shape}`` pair the EAGLE-3 remote protocol
    encodes in :func:`nemo_automodel.components.speculative.eagle.remote.protocol.encode_nccl_metadata`
    so the client can preallocate the NCCL receive buffer. Consumers here use
    it to preallocate the receive tensor before calling
    :meth:`FeatureStore.get`.

    Attributes:
        shape: Logical shape with the same axis order the producer used.
            ``shape[0]`` is conventionally batch and ``shape[-1]`` the
            feature dimension; algorithms with packed layouts (THD, BHSD)
            document their conventions in the algorithm row of the
            RFC's "Feature schema per algorithm" table.
        dtype: The ``torch.dtype`` the producer will store under
            ``SampleRef.feature_keys[name]``. Validation in
            :func:`assert_no_tensors` rejects stray tensors in the ref but
            a ``torch.dtype`` object on this field is allowed.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        # Equality on tuple-of-ints is what we want; normalize a mis-passed
        # list once so two FeatureSpec objects built from equivalent
        # sequences compare equal.
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))
        if not all(isinstance(d, int) and d >= 0 for d in self.shape):
            raise ValueError(f"FeatureSpec.shape must be a tuple of non-negative ints, got {self.shape}")


@dataclass(frozen=True)
class SampleRef:
    """Tensor-free reference to one produced sample in a feature store.

    Attributes:
        sample_id: Stable identifier within ``run_id``. The consumer uses it
            to partition the stream across DP ranks (RFC §"Consumer-side DP
            resharding") and to ACK / FAIL the corresponding lease.
        run_id: Stable identifier across the whole training run. Same value
            on every ref of one run so producers and consumers can verify
            they are talking about the same run before materializing.
        store_uri: ``scheme://location`` identifying the :class:`FeatureStore`
            back-end (e.g. ``mem://local`` for
            :class:`~nemo_automodel.components.speculative.streaming.stores.local.LocalFeatureStore`).
            The store URI plus ``feature_keys`` form the lookup the consumer
            makes; the store object itself is discovered through the URI at
            materialization time, not stored on the ref.
        feature_keys: Named features this sample contributes, mapped to
            per-store keys (for example filenames for
            :class:`~nemo_automodel.components.speculative.streaming.stores.shared_dir.SharedDirFeatureStore`
            or dict keys for :class:`LocalFeatureStore`).
        feature_specs: Per-feature :class:`FeatureSpec` so the consumer can
            allocate the receive buffer before calling :meth:`FeatureStore.get`.
        algorithm: Which draft family produced this sample; the consumer
            matches it against a registered :class:`FeatureSpec` registry
            to validate the ref before materializing.
        schema_version: Bumped whenever the producer's feature set or tensor
            layout for ``algorithm`` changes incompatibly. The consumer uses
            it as a hard gate on the ref.
        num_tokens: Sum of attended tokens; used by the consumer for empty /
            short loss-mask neutralization and by the queue for backpressure.
        estimated_bytes: Sum of every feature's ``numel() * dtype_bytes``.
            The :class:`SampleRefQueue` reads this against
            :attr:`StoreHealth.resident_bytes` for watermark hysteresis.
        target_model_version: Monotonically increasing identifier of the
            target-model weights that produced this sample. Used to reject
            refs from a stale target once train-with-decode weight resync
            is wired through the recipe.
        draft_weight_version: Same idea for the draft model's weights, so a
            consumer can refuse to train against a ref produced before its
            own weight snapshot. Starts at ``"0"`` until resync is implemented.
    """

    sample_id: str
    run_id: str
    store_uri: str
    feature_keys: dict[str, str]
    feature_specs: dict[str, FeatureSpec]
    algorithm: FeatureAlgorithm
    schema_version: int
    num_tokens: int
    estimated_bytes: int
    target_model_version: str
    draft_weight_version: str

    def __post_init__(self) -> None:
        if not self.sample_id or not isinstance(self.sample_id, str):
            raise ValueError(f"SampleRef.sample_id must be a non-empty str, got {self.sample_id!r}")
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError(f"SampleRef.run_id must be a non-empty str, got {self.run_id!r}")
        if not self.store_uri or "://" not in self.store_uri:
            raise ValueError(f"SampleRef.store_uri must be a scheme://location URI, got {self.store_uri!r}")
        if set(self.feature_keys) != set(self.feature_specs):
            raise ValueError(
                f"SampleRef.feature_keys and feature_specs must name the same feature set; "
                f"keys={sorted(self.feature_keys)} specs={sorted(self.feature_specs)}"
            )
        required = _algorithm_required_features(self.algorithm)
        missing = required - set(self.feature_keys)
        if missing:
            raise ValueError(
                f"SampleRef for algorithm {self.algorithm} missing required features {sorted(missing)}; "
                f"present={sorted(self.feature_keys)}"
            )
        if self.num_tokens < 0 or self.estimated_bytes < 0:
            raise ValueError(
                f"SampleRef num_tokens/estimated_bytes must be non-negative, got "
                f"num_tokens={self.num_tokens} estimated_bytes={self.estimated_bytes}"
            )
        # The ref itself must be tensor-free; the contract is enforceable at
        # construction time so a misbehaving producer fails fast.
        assert_no_tensors(self, path="SampleRef")

    def feature_names(self) -> tuple[str, ...]:
        """Return the feature names in insertion order.

        The order is significant because some algorithms size the draft's
        ``fc`` projection by the *order* of the aux hidden states, not just
        their count; the producer chooses the order once, the ref freezes it,
        and the consumer preserves it through :meth:`FeatureStore.get`.
        """
        return tuple(self.feature_keys)


_PRIMITIVE_TYPES = (str, int, float, bool, type(None), bytes)


def _is_torch_tensor(obj: Any) -> bool:
    # torch.Tensor is a hard dependency, but the duck-typed check (``hasattr
    # data_ptr``) keeps this safe if a third-party tensor class ever sneaks
    # in -- the contract is structural, not nominal.
    return isinstance(obj, torch.Tensor) or (
        hasattr(obj, "data_ptr") and hasattr(obj, "is_cuda") and hasattr(obj, "numel")
    )


def _is_numpy_array(obj: Any) -> bool:
    if not _np_ok:
        return False
    try:
        return isinstance(obj, _np_module.ndarray)
    except TypeError:
        # safe_import's placeholder triggers TypeError on isinstance; treat as no-numpy.
        return False


def _reject_tensor(obj: Any, path: str) -> NoReturn:
    raise ValueError(
        f"tensor-like object is not allowed on the streaming control plane at {path}: type={type(obj).__name__}"
    )


def assert_no_tensors(obj: Any, *, path: str = "ref") -> None:
    """Recursively validate that ``obj`` carries no tensors.

    Walks dataclasses (any nested dataclass included), ``dict`` with ``str``
    keys, and ``list`` / ``tuple``. Anything else must be a primitive
    (``str``/``int``/``float``/``bool``/``None``/``bytes``) -- a tensor, numpy
    array, or duck-typed tensor-like (``has data_ptr + is_cuda + numel``) at
    any depth raises :class:`ValueError`.

    The check is structural, not nominal: a third-party tensor type that
    quacks like one is rejected. Per-algorithm schema validators layer on
    top of this primitive guard, not instead of it.
    """
    if obj is None or isinstance(obj, _PRIMITIVE_TYPES):
        return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            assert_no_tensors(getattr(obj, f.name), path=f"{path}.{f.name}")
        return
    if isinstance(obj, Mapping):
        if not all(isinstance(k, str) for k in obj):
            raise ValueError(
                f"non-str key on the streaming control plane at {path}; control-plane dicts must be str-keyed"
            )
        for k, v in obj.items():
            assert_no_tensors(v, path=f"{path}[{k!r}]")
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            assert_no_tensors(item, path=f"{path}[{i}]")
        return
    if _is_torch_tensor(obj):
        _reject_tensor(obj, path)
    if _is_numpy_array(obj):
        _reject_tensor(obj, path)
    logger.debug("assert_no_tensors accepted opaque object of type %s at %s", type(obj).__name__, path)


__all__ = [
    "FeatureAlgorithm",
    "FeatureSpec",
    "SampleRef",
    "assert_no_tensors",
]
