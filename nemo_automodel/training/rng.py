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

class StatefulRNG:
    """
    A class to handle random number generator (RNG) states for reproducibility
    across Python's random module, NumPy, and PyTorch, including CUDA.

    The RNG state can be captured and restored, making it useful in settings
    where reproducible experiments are essential.
    """

    def __init__(self, seed: int, ranked: bool = False):
        """
        Initialize the RNG states using a provided seed and optionally
        modify the seed based on the rank of the process.

        Parameters:
            seed (int): A positive integer used as the base seed for all RNGs.
            ranked (bool): Flag indicating whether to adjust the seed based on
                           the process rank. Default is False.

        Raises:
            AssertionError: If `seed` is not an integer or is not positive,
                            or if `ranked` is not a boolean.
        """
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
        """
        Capture the current random number generator states for Python,
        NumPy, and PyTorch (including CUDA).

        Returns:
            dict: A dictionary containing the RNG states with the keys:
                - "random_rng_state": The state of Python's random module.
                - "np_rng_state": The state of NumPy's random module.
                - "torch_rng_state": The state of PyTorch's CPU random generator.
                - "cuda_rng_state": The state of PyTorch's CUDA random generators.
        """
        return {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }

    def load_state_dict(self, state):  # pragma: no cover
        """
        Restore the random number generator states for Python, NumPy, and PyTorch
        (including CUDA) from a previously captured state dictionary.

        Parameters:
            state (dict): A dictionary containing the RNG states as returned
                          by the `state_dict()` method.
        """
        random.setstate(state["random_rng_state"])
        np.random.set_state(state["np_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
