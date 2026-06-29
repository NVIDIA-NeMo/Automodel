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

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from torchdata.stateful_dataloader import StatefulDataLoader

_DP_STATE_KEY = "_automodel_dp_state"
_SAMPLER_ITER_STATE = "_sampler_iter_state"
_SAMPLER_ITER_YIELDED = "_sampler_iter_yielded"
_NUM_YIELDED = "_num_yielded"
_FETCHER_STATE = "fetcher_state"
_DATASET_STATE = "dataset_state"
_PROGRESS_UNIT_BATCH = "batch"
_PROGRESS_UNIT_SAMPLE = "sample"


def instantiate_dataloader(cfg_dl: Any, *, dp_rank: int, dp_world_size: int, **kwargs: Any) -> Any:
    """Instantiate a configured DataLoader, upgrading torchdata loaders to DP-aware state.

    Args:
        cfg_dl: Config node with an ``instantiate`` method and optional ``_target_``.
        dp_rank: Current data-parallel rank.
        dp_world_size: Current data-parallel world size.
        **kwargs: Runtime DataLoader kwargs.

    Returns:
        The instantiated DataLoader.
    """
    target = getattr(cfg_dl, "_target_", None)
    if not _is_torchdata_stateful_dataloader_target(target):
        return cfg_dl.instantiate(**kwargs)

    original_target = cfg_dl._target_
    cfg_dl._target_ = DPAwareStatefulDataLoader
    try:
        return cfg_dl.instantiate(dp_rank=dp_rank, dp_world_size=dp_world_size, **kwargs)
    finally:
        cfg_dl._target_ = original_target


class DPAwareStatefulDataLoader(StatefulDataLoader):
    """StatefulDataLoader with metadata for data-parallel checkpoint resharding.

    The native ``torchdata`` state is preserved and annotated with Automodel
    metadata. When DP size changes at restore time, Automodel can derive each
    new rank's local fast-forward offset from the saved global progress.
    """

    def __init__(self, *args: Any, dp_rank: int = 0, dp_world_size: int = 1, **kwargs: Any) -> None:
        if dp_rank < 0:
            raise ValueError(f"dp_rank must be non-negative, got {dp_rank}")
        if dp_world_size <= 0:
            raise ValueError(f"dp_world_size must be positive, got {dp_world_size}")
        if dp_rank >= dp_world_size:
            raise ValueError(f"dp_rank must be smaller than dp_world_size, got {dp_rank} >= {dp_world_size}")

        self.dp_rank = int(dp_rank)
        self.dp_world_size = int(dp_world_size)
        super().__init__(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict[_DP_STATE_KEY] = self._build_dp_state(state_dict)
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(_strip_dp_state(state_dict))

    def load_state_dict_from_dp_rank_states(self, rank_states: Mapping[int, dict[str, Any]]) -> None:
        """Load dataloader state from all checkpointed DP rank states.

        Args:
            rank_states: Mapping from saved DP-rank id to the state dict stored
                for that rank.
        """
        if not rank_states:
            self.load_state_dict({})
            return

        metadata = self._select_metadata(rank_states)
        saved_world_size = metadata["dp_world_size"]
        if metadata.get("is_legacy", False):
            if saved_world_size != self.dp_world_size:
                raise ValueError(
                    "Cannot reshard a legacy dataloader checkpoint without Automodel DP metadata: "
                    f"saved dp_world_size={saved_world_size}, current dp_world_size={self.dp_world_size}."
                )
            if self.dp_rank not in rank_states:
                raise ValueError(
                    "Cannot load legacy dataloader checkpoint because it does not contain "
                    f"state for current dp_rank={self.dp_rank}."
                )
            self.load_state_dict(rank_states[self.dp_rank])
            return

        if saved_world_size == self.dp_world_size and self.dp_rank in rank_states:
            self.load_state_dict(rank_states[self.dp_rank])
            return

        template_state = _strip_dp_state(next(iter(rank_states.values())))
        if template_state.get(_FETCHER_STATE) is not None or template_state.get(_DATASET_STATE) is not None:
            raise ValueError(
                "Dataloader DP resharding is only supported for map-style dataloaders without dataset/fetcher state."
            )

        resharded_state = self._reshard_native_state(
            template_state,
            progress_unit=metadata["progress_unit"],
            global_yielded=metadata["global_yielded"],
        )
        super().load_state_dict(resharded_state)

    def _build_dp_state(self, native_state: Mapping[str, Any]) -> dict[str, Any]:
        progress_unit, local_yielded = _extract_local_progress(native_state)
        return {
            "version": 1,
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "progress_unit": progress_unit,
            "local_yielded": local_yielded,
            "global_yielded": local_yielded * self.dp_world_size,
        }

    def can_load_single_dp_rank_state(
        self, state_dict: Mapping[str, Any], saved_dp_world_size: int | None = None
    ) -> bool:
        """Return whether one rank state is enough to restore this dataloader."""
        metadata = state_dict.get(_DP_STATE_KEY)
        if isinstance(metadata, Mapping):
            return True
        return saved_dp_world_size == self.dp_world_size

    def _select_metadata(self, rank_states: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
        metadata_by_rank = {
            rank: state[_DP_STATE_KEY]
            for rank, state in rank_states.items()
            if isinstance(state.get(_DP_STATE_KEY), Mapping)
        }
        if not metadata_by_rank:
            return {
                "version": 0,
                "is_legacy": True,
                "dp_world_size": len(rank_states),
            }

        metadata = next(iter(metadata_by_rank.values()))
        expected = {
            "dp_world_size": metadata["dp_world_size"],
            "progress_unit": metadata["progress_unit"],
            "global_yielded": metadata["global_yielded"],
        }
        for rank, rank_metadata in metadata_by_rank.items():
            for key, value in expected.items():
                if rank_metadata.get(key) != value:
                    raise ValueError(
                        "Dataloader DP checkpoint states disagree: "
                        f"rank {rank} has {key}={rank_metadata.get(key)!r}, expected {value!r}."
                    )
        return dict(metadata)

    def _reshard_native_state(
        self,
        native_state: dict[str, Any],
        *,
        progress_unit: str,
        global_yielded: int,
    ) -> dict[str, Any]:
        state = deepcopy(native_state)
        local_yielded = _local_yielded_for_rank(global_yielded, self.dp_world_size, self.dp_rank)

        if progress_unit == _PROGRESS_UNIT_SAMPLE:
            batch_size = _infer_batch_size(self)
            state[_SAMPLER_ITER_YIELDED] = local_yielded // batch_size
            state[_NUM_YIELDED] = local_yielded // batch_size
            sampler_iter_state = state.get(_SAMPLER_ITER_STATE)
            if isinstance(sampler_iter_state, dict):
                sampler_iter_state["samples_yielded"] = local_yielded
                sampler_state = sampler_iter_state.get("sampler_state")
                if isinstance(sampler_state, dict) and "yielded" in sampler_state:
                    sampler_state["yielded"] = local_yielded
            return state

        if progress_unit == _PROGRESS_UNIT_BATCH:
            state[_SAMPLER_ITER_YIELDED] = local_yielded
            state[_NUM_YIELDED] = local_yielded
            return state

        raise ValueError(f"Unsupported dataloader progress unit: {progress_unit!r}")


def _is_torchdata_stateful_dataloader_target(target: Any) -> bool:
    if target is StatefulDataLoader:
        return True
    if not isinstance(target, str):
        return False
    return target in {
        "torchdata.stateful_dataloader.StatefulDataLoader",
        "torchdata.stateful_dataloader.stateful_dataloader.StatefulDataLoader",
    }


def _strip_dp_state(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(state_dict)
    state.pop(_DP_STATE_KEY, None)
    return state


def _extract_local_progress(native_state: Mapping[str, Any]) -> tuple[str, int]:
    sampler_iter_state = native_state.get(_SAMPLER_ITER_STATE)
    if isinstance(sampler_iter_state, Mapping) and "samples_yielded" in sampler_iter_state:
        return _PROGRESS_UNIT_SAMPLE, int(sampler_iter_state["samples_yielded"])

    sampler_iter_yielded = native_state.get(_SAMPLER_ITER_YIELDED)
    if sampler_iter_yielded is not None:
        return _PROGRESS_UNIT_BATCH, int(sampler_iter_yielded)

    num_yielded = native_state.get(_NUM_YIELDED, 0)
    return _PROGRESS_UNIT_BATCH, int(num_yielded)


def _local_yielded_for_rank(global_yielded: int, dp_world_size: int, dp_rank: int) -> int:
    if global_yielded <= dp_rank:
        return 0
    return (global_yielded + dp_world_size - 1 - dp_rank) // dp_world_size


def _infer_batch_size(dataloader: StatefulDataLoader) -> int:
    batch_size = getattr(dataloader, "batch_size", None)
    if batch_size is None:
        raise ValueError("Cannot reshard sample-based dataloader state when DataLoader.batch_size is unavailable.")
    return int(batch_size)
