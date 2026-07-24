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

import random
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import numpy as np
import torch


def _ranked_seed(seed: int, ranked: bool) -> int:
    """Return the process-local seed used by ranked RNG streams."""
    if not ranked:
        return seed
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return seed + dist.get_rank()
    except ImportError:
        pass
    return seed


def init_all_rng(seed: int, ranked: bool = False) -> None:
    """Initialize RNGs for Python, NumPy, and PyTorch (incl. CUDA) with a seed.

    Args:
        seed (int): Base seed value.
        ranked (bool): Adjust seed by process rank if True.
    """
    assert isinstance(seed, int) and seed >= 0, ("Seed must be a non-negative integer", seed)
    assert isinstance(ranked, bool), "Ranked must be a boolean"

    seed = _ranked_seed(seed, ranked)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RNGState:
    """Snapshot of Python, NumPy, Torch, and CUDA RNG states."""

    random_rng_state: tuple
    np_rng_state: tuple | dict[str, Any]
    torch_rng_state: torch.Tensor
    cuda_rng_state: list[torch.Tensor]
    generator_states: dict[str, torch.Tensor] = field(default_factory=dict)


def _get_rng_state() -> RNGState:
    """Get current RNG states.

    Returns:
        dict: RNG states for random, NumPy, and PyTorch.
    """
    return RNGState(
        random_rng_state=random.getstate(),
        np_rng_state=np.random.get_state(),
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all(),
    )


def _restore_rng_state(state: RNGState) -> None:
    """Restore RNG states from a saved state.

    Args:
        state (dict): RNG states as returned by state_dict().
    """
    random.setstate(state.random_rng_state)
    np.random.set_state(state.np_rng_state)
    torch.set_rng_state(state.torch_rng_state)
    torch.cuda.set_rng_state_all(state.cuda_rng_state)


class StatefulRNG:
    """
    RNG manager for reproducible RNG states across random, NumPy, and PyTorch."""

    def __init__(self, seed: int, ranked: bool = False) -> None:
        """Initialize and optionally rank-adjust RNGs with a given seed.

        Args:
            seed (int): Base seed for RNGs.
            ranked (bool): Adjust seed based on process rank.
        """
        self.seed = seed
        self.ranked = ranked
        self._effective_seed = _ranked_seed(seed, ranked)
        self._generators: dict[str, torch.Generator] = {}
        self._pending_generator_states: dict[str, torch.Tensor] = {}
        init_all_rng(self.seed, self.ranked)

    def generator(self, device: torch.device | str = "cpu") -> torch.Generator:
        """Return a checkpointable process-local generator for ``device``.

        Args:
            device: Device on which random tensors will be sampled.

        Returns:
            Rank-local generator whose state is included in ``state_dict()``.
        """
        resolved_device = torch.device(device)
        key = str(resolved_device)
        if key not in self._generators:
            generator = torch.Generator(device=resolved_device)
            generator.manual_seed(self._effective_seed)
            pending_state = self._pending_generator_states.pop(key, None)
            if pending_state is not None:
                generator.set_state(pending_state)
            self._generators[key] = generator
        return self._generators[key]

    def state_dict(self) -> RNGState:
        """Get current RNG states.

        Returns:
            RNG snapshot containing the Python and NumPy states, the CPU Torch
            uint8 state tensor of shape ``[cpu_state_bytes]``, one CUDA uint8
            state tensor of shape ``[cuda_state_bytes]`` per device, and any
            explicit generator uint8 state tensors of shape
            ``[generator_state_bytes]`` keyed by device.
        """
        state = _get_rng_state()
        state.generator_states = {
            key: generator.get_state() for key, generator in self._generators.items()
        } | self._pending_generator_states
        return state

    def load_state_dict(self, state: RNGState) -> None:
        """Restore RNG states from a saved state.

        Args:
            state: RNG snapshot returned by ``state_dict()``. Torch CPU, CUDA,
                and explicit generator state tensors are uint8 tensors of shape
                ``[cpu_state_bytes]``, ``[cuda_state_bytes]``, and
                ``[generator_state_bytes]``, respectively.
        """
        _restore_rng_state(state)
        generator_states = dict(getattr(state, "generator_states", {}))
        for key, generator in self._generators.items():
            generator_state = generator_states.pop(key, None)
            if generator_state is not None:
                generator.set_state(generator_state)
        self._pending_generator_states = generator_states


class ScopedRNG:
    """Context manager for reproducible RNG states across random, NumPy, and PyTorch."""

    def __init__(self, seed: int = 95050, ranked: bool = False) -> None:
        """Initialize and optionally rank-adjust RNGs with a given seed.

        Args:
            seed (int): Base seed for RNGs.
            ranked (bool): Adjust seed based on process rank.
        """
        self._saved_state: RNGState | None = None
        self.seed = seed
        self.ranked = ranked

    def __enter__(self) -> "ScopedRNG":
        """Save current RNG states."""
        assert self._saved_state is None
        self._saved_state = _get_rng_state()
        init_all_rng(self.seed, self.ranked)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore RNG states on context exit."""
        assert self._saved_state is not None
        _restore_rng_state(self._saved_state)
        self._saved_state = None
