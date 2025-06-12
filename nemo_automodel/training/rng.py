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
import numpy as np
import torch

from nemo_automodel.utils.dist_utils import get_rank_safe



def init_all_rng(seed: int, ranked: bool = False):
    """
    Initialize all RNGs (Python, NumPy, PyTorch) with a seed.
    Optionally adjust for ranked environments like distributed training.
    """
    assert isinstance(seed, int) and seed > 0, "Seed must be a positive integer"
    assert isinstance(ranked, bool), "Ranked must be a boolean"

    if ranked:
        # Example: use PyTorch's distributed rank if available
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                seed += rank
        except ImportError:
            pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StatefulRNG:
    """
    A context manager for managing reproducible RNG states across random,
    NumPy, and PyTorch (including CUDA).
    """

    def __init__(self, seed: int, ranked: bool = False):
        init_all_rng(seed, ranked)
        self.seed = seed
        self.ranked = ranked
        self._saved_state = None

    def state_dict(self):
        return {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }

    def load_state_dict(self, state):  # pragma: no cover
        random.setstate(state["random_rng_state"])
        np.random.set_state(state["np_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    def __enter__(self):
        self._saved_state = self.state_dict()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.load_state_dict(self._saved_state)