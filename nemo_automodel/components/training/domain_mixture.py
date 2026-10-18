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

"""Objective weighting for named pretraining data mixtures."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field

import torch

#: Metric/validation-dataloader key holding the objective-weighted aggregate.
#: Reserved, so a domain may not claim it (that would make the recipe both
#: require and forbid ``validation_dataset_weighted``).
WEIGHTED_AGGREGATE_NAME = "weighted"


@dataclass(frozen=True)
class DomainWeightConfig:
    """Declarative sampling and objective weights for one data domain.

    Attributes:
        name: Domain name. The position in ``DomainMixtureConfig.domains`` must
            match the ``dataset_id`` assigned by the blended dataset.
        sampling_weight: Relative probability used to sample the domain.
        objective_weight: Relative contribution of the domain to the training
            and validation objectives.
        path: Explicit Megatron corpus prefix for this domain. Required by the
            training recipe to verify blend order; optional for standalone
            objective calculations. Names are never inferred from this path.
    """

    name: str
    sampling_weight: float
    objective_weight: float
    path: str | None = None

    def __post_init__(self) -> None:
        if self.path is not None and (not isinstance(self.path, str) or not self.path.strip()):
            raise ValueError(f"domain_mixture path must be a non-empty string for {self.name!r}, got {self.path!r}")
        if not self.name:
            raise ValueError("domain_mixture domain names must be non-empty")
        if self.name == WEIGHTED_AGGREGATE_NAME:
            raise ValueError(
                f"{WEIGHTED_AGGREGATE_NAME!r} is reserved for the objective-weighted aggregate "
                "and cannot be used as a domain name"
            )
        if not math.isfinite(self.sampling_weight) or self.sampling_weight <= 0:
            raise ValueError(
                f"domain_mixture sampling_weight must be finite and positive for {self.name!r}, "
                f"got {self.sampling_weight}"
            )
        if not math.isfinite(self.objective_weight) or self.objective_weight < 0:
            raise ValueError(
                f"domain_mixture objective_weight must be finite and non-negative for {self.name!r}, "
                f"got {self.objective_weight}"
            )


@dataclass(frozen=True)
class DomainMixtureConfig:
    """Configuration for importance-weighted multi-domain pretraining."""

    domains: tuple[DomainWeightConfig, ...]

    def __post_init__(self) -> None:
        if len(self.domains) < 2:
            raise ValueError("domain_mixture.domains must contain at least two domains")
        names = tuple(domain.name for domain in self.domains)
        if len(names) != len(set(names)):
            raise ValueError(f"domain_mixture domain names must be unique, got {names}")
        if sum(domain.objective_weight for domain in self.domains) <= 0:
            raise ValueError("domain_mixture objective weights must have a positive sum")

    def build(self, *, blend: tuple[Sequence[str], Sequence[float]] | None = None) -> "DomainMixture":
        """Build the immutable runtime domain-mixture objective.

        Args:
            blend: Parsed training corpus prefixes and explicit sampling weights,
                in dataset ID order. When provided, construction validates the
                declared domain bindings against this blend. Omit for standalone
                objective calculations.

        Returns:
            The domain-mixture objective with validated bindings when supplied.
        """
        sampling_total = sum(domain.sampling_weight for domain in self.domains)
        objective_total = sum(domain.objective_weight for domain in self.domains)
        sampling_weights = tuple(domain.sampling_weight / sampling_total for domain in self.domains)
        objective_weights = tuple(domain.objective_weight / objective_total for domain in self.domains)
        return DomainMixture(
            blend=blend,
            names=tuple(domain.name for domain in self.domains),
            paths=tuple(domain.path.strip() if domain.path is not None else None for domain in self.domains),
            sampling_weights=sampling_weights,
            objective_weights=objective_weights,
            loss_multipliers=tuple(
                objective_weight / sampling_weight
                for objective_weight, sampling_weight in zip(objective_weights, sampling_weights)
            ),
        )


@dataclass(frozen=True)
class DomainMixture:
    """Runtime importance weights and metrics for a named data mixture.

    ``loss_multipliers[i] = objective_weights[i] / sampling_weights[i]``.
    Applying that multiplier to every supervised token sampled from domain
    ``i`` makes the expected training objective match ``objective_weights``
    without changing the sampler or creating separate optimizers.
    """

    names: tuple[str, ...]
    sampling_weights: tuple[float, ...]
    objective_weights: tuple[float, ...]
    loss_multipliers: tuple[float, ...]
    #: Corpus prefixes in dataset_id order; None means the binding is undeclared.
    #: Keyword-only to preserve the positional constructor for standalone users.
    paths: tuple[str | None, ...] = field(default=(), kw_only=True)
    #: Construction-only validation input; not retained as runtime state.
    blend: InitVar[tuple[Sequence[str], Sequence[float]] | None] = field(default=None, kw_only=True)
    #: Per-device cache of ``loss_multipliers``. Excluded from eq/hash so the
    #: dataclass stays frozen-comparable; mutated in place, never rebound.
    _multiplier_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self, blend: tuple[Sequence[str], Sequence[float]] | None) -> None:
        if blend is None:
            return
        blend_paths, blend_weights = blend
        if len(self.paths) != len(self.names) or any(path is None for path in self.paths):
            raise ValueError(
                "domain_mixture requires a path for every domain; set each domains[i].path to the "
                "corresponding corpus prefix in dataset.paths (or dataset.paths.train). "
                "Sampling weights alone cannot verify domain order when weights are equal."
            )
        if tuple(blend_paths) != self.paths:
            raise ValueError(
                "domain_mixture paths must match the corpus prefixes in dataset.paths in the same order; "
                f"got dataset paths {tuple(blend_paths)} and domain bindings "
                f"{tuple(zip(self.names, self.paths))}"
            )
        blend_total = sum(blend_weights)
        if blend_total <= 0:
            raise ValueError(f"dataset.paths blend weights must have a positive sum, got {blend_weights}")
        normalized = tuple(weight / blend_total for weight in blend_weights)
        if len(normalized) != len(self.sampling_weights) or any(
            not math.isclose(actual, declared, rel_tol=1.0e-6, abs_tol=1.0e-8)
            for actual, declared in zip(normalized, self.sampling_weights)
        ):
            raise ValueError(
                "domain_mixture sampling weights must match the explicit weights in dataset.paths; "
                f"got dataset weights {normalized} and domain_mixture weights {self.sampling_weights}"
            )

    def _multiplier_tensor(self, device: torch.device) -> torch.Tensor:
        """Return the multiplier vector on ``device``, building it at most once.

        ``loss_multipliers`` is immutable after :meth:`DomainMixtureConfig.build`,
        so rebuilding it per microbatch only adds a pageable host-to-device copy
        on the critical path.
        """
        cached = self._multiplier_cache.get(device)
        if cached is None:
            cached = torch.tensor(self.loss_multipliers, dtype=torch.float32, device=device)
            self._multiplier_cache[device] = cached
        return cached

    def _domain_ids(self, dataset_ids: torch.Tensor, *, validate_range: bool = True) -> torch.Tensor:
        """Validate and flatten per-sample dataset IDs to shape ``[batch]``.

        Args:
            dataset_ids: Integer tensor of per-sample domain IDs.
            validate_range: Bounds-check the IDs. This costs two blocking
                device-to-host syncs, so callers on the per-microbatch critical
                path pass ``False`` once a step-level pass has already validated
                the very same batches.
        """
        if not isinstance(dataset_ids, torch.Tensor):
            raise TypeError(f"dataset_id must be a torch.Tensor, got {type(dataset_ids).__name__}")
        if dataset_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"dataset_id must use an integer dtype, got {dataset_ids.dtype}")
        domain_ids = dataset_ids.reshape(-1).to(dtype=torch.long)
        if domain_ids.numel() == 0:
            raise ValueError("dataset_id must contain at least one value")
        if validate_range:
            minimum = int(domain_ids.min().item())
            maximum = int(domain_ids.max().item())
            if minimum < 0 or maximum >= len(self.names):
                raise ValueError(
                    f"dataset_id values must be in [0, {len(self.names) - 1}], got min={minimum}, max={maximum}"
                )
        return domain_ids

    def loss_weights(
        self,
        dataset_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        validate_range: bool = True,
    ) -> torch.Tensor:
        """Expand per-sample importance weights to the label layout.

        Args:
            dataset_ids: Integer tensor of shape ``[batch]`` (or a scalar for a
                single sample). IDs follow the domain order in the config.
            labels: Target token IDs of shape ``[batch, sequence]`` or
                ``[sequence]`` for a single flattened sample.
            validate_range: Bounds-check ``dataset_ids``. Costs two device syncs;
                see :meth:`_domain_ids`.

        Returns:
            Float32 tensor matching ``labels.shape``. Every row is constant
            because sequence packing is not supported by this component.
        """
        if not isinstance(labels, torch.Tensor) or labels.ndim == 0:
            raise ValueError("labels must be a non-scalar torch.Tensor")
        domain_ids = self._domain_ids(dataset_ids, validate_range=validate_range)
        if labels.ndim == 1:
            if domain_ids.numel() != 1:
                raise ValueError(
                    "Flat labels can only be weighted when dataset_id contains one value; "
                    f"got labels.shape={tuple(labels.shape)} and {domain_ids.numel()} IDs"
                )
        elif labels.shape[0] != domain_ids.numel():
            raise ValueError(
                "dataset_id must provide one value per label row; "
                f"got labels.shape={tuple(labels.shape)} and {domain_ids.numel()} IDs"
            )

        multipliers = self._multiplier_tensor(domain_ids.device)
        per_sample = multipliers.index_select(0, domain_ids).to(labels.device)
        if labels.ndim == 1:
            return per_sample[0].expand_as(labels)
        return per_sample.reshape((-1,) + (1,) * (labels.ndim - 1)).expand_as(labels)

    def label_counts(
        self,
        dataset_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Count supervised tokens per domain.

        Args:
            dataset_ids: Integer tensor with one domain ID per label row.
            labels: Target token IDs of shape ``[batch, ...]`` or a flat
                single-sample tensor.
            ignore_index: Label value excluded from the count.

        Returns:
            Int64 tensor of shape ``[num_domains]``.
        """
        domain_ids = self._domain_ids(dataset_ids)
        if labels.ndim == 1:
            if domain_ids.numel() != 1:
                raise ValueError("Flat labels require exactly one dataset_id")
            per_sample = (labels != ignore_index).sum().reshape(1)
        else:
            if labels.shape[0] != domain_ids.numel():
                raise ValueError("dataset_id must provide one value per label row")
            per_sample = (labels != ignore_index).reshape(labels.shape[0], -1).sum(dim=1)
        counts = torch.zeros(len(self.names), dtype=torch.long, device=labels.device)
        counts.scatter_add_(0, domain_ids.to(labels.device), per_sample.to(dtype=torch.long))
        return counts

    def weighted_validation_loss(self, losses: Mapping[str, float]) -> float:
        """Combine named per-domain validation losses using objective weights."""
        missing = [name for name in self.names if name not in losses]
        if missing:
            raise ValueError(f"Missing validation losses for domain(s): {', '.join(missing)}")
        return sum(weight * float(losses[name]) for name, weight in zip(self.names, self.objective_weights))
