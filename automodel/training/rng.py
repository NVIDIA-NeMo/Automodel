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

from automodel.utils.dist_utils import get_rank_safe

class StatefulRNG:
    def __init__(self, seed: int, ranked: bool = False):
        """Set random seed for reproducability."""
        assert isinstance(seed, int), "Expected seed to be of type int"
        assert seed > 0, "Expected seed ({}) to be a positive integer.".format(seed)
        assert isinstance(ranked, bool), "Expected ranked to a bool"

        if ranked:
            seed += get_rank_safe()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def state_dict(self):
        """Capture Python / NumPy / Torch RNG states for reproducibility."""

        return {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }

    def load_state_dict(self, state):  # pragma: no cover
        """Restore RNG state collected by :func:`_collect_rng_state`."""

        random.setstate(state["random_rng_state"])
        np.random.set_state(state["np_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
