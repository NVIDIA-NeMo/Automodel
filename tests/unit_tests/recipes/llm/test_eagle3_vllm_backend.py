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

"""CPU unit tests for the co-located vLLM target backend in the EAGLE-3 recipe.

The real vLLM construction needs a GPU and is validated on the server; here
``VLLMEagle3TargetModel.from_pretrained`` is mocked and the tests pin the
recipe-side wiring: backend dispatch, environment guards (CUDA-only,
single-process-only), and how ``recipe_args`` flow into the vLLM kwargs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe


class _RecipeCfg(SimpleNamespace):
    """recipe_args stand-in: attribute access plus the ConfigNode-style ``get``."""

    def get(self, key, default=None):
        return getattr(self, key, default)


class _ToDictNode:
    """ConfigNode stand-in: a mapping exposed only through ``to_dict()``."""

    def __init__(self, values):
        self._values = dict(values)

    def to_dict(self):
        return dict(self._values)


def _make_recipe(world_size: int = 1, device: str = "cuda") -> TrainEagle3Recipe:
    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.device = torch.device(device)
    recipe.dist_env = SimpleNamespace(world_size=world_size)
    recipe.compute_dtype = torch.bfloat16
    # ``_setup_online_target`` reads ``cfg.get("distributed.cp_size", ...)`` for the
    # CP gate; an empty config stand-in defaults it to cp_size=1 (no CP).
    recipe.cfg = _RecipeCfg()
    return recipe


_FROM_PRETRAINED = "nemo_automodel.components.speculative.eagle.vllm_target.VLLMEagle3TargetModel.from_pretrained"


def test_setup_vllm_target_requires_cuda():
    recipe = _make_recipe(device="cpu")
    with pytest.raises(ValueError, match="requires CUDA"):
        recipe._setup_vllm_target(_RecipeCfg(), "target/path")


def test_setup_vllm_target_rejects_multi_process():
    recipe = _make_recipe(world_size=2)
    with pytest.raises(ValueError, match="single-process training only"):
        recipe._setup_vllm_target(_RecipeCfg(), "target/path")


def test_setup_vllm_target_builds_wrapper_with_defaults():
    recipe = _make_recipe()
    wrapper = object()
    with patch(_FROM_PRETRAINED, MagicMock(return_value=wrapper)) as from_pretrained:
        recipe._setup_vllm_target(_RecipeCfg(), "target/path")

    assert recipe.target_wrapper is wrapper
    assert recipe.target_model is None
    from_pretrained.assert_called_once_with(
        "target/path",
        aux_layer_ids=None,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        gpu_memory_utilization=0.5,
    )


@pytest.mark.parametrize("as_node", [False, True], ids=["plain-dict", "to_dict-node"])
def test_setup_vllm_target_forwards_vllm_args(as_node):
    """``recipe_args.vllm_args`` overrides the defaults and passes extras through."""
    recipe = _make_recipe()
    args = {"gpu_memory_utilization": 0.35, "max_model_len": 2048}
    cfg = _RecipeCfg(
        vllm_args=_ToDictNode(args) if as_node else args,
        aux_layer_ids=[1, 2, 3],
        trust_remote_code=True,
    )
    with patch(_FROM_PRETRAINED, MagicMock(return_value=object())) as from_pretrained:
        recipe._setup_vllm_target(cfg, "target/path")

    from_pretrained.assert_called_once_with(
        "target/path",
        aux_layer_ids=[1, 2, 3],
        dtype=torch.bfloat16,
        trust_remote_code=True,
        gpu_memory_utilization=0.35,
        max_model_len=2048,
    )


class _DispatchReached(Exception):
    """Sentinel: the dispatch reached the expected backend setup method."""


def _sentinel(self, *args, **kwargs):
    raise _DispatchReached()


def test_online_target_dispatches_vllm(monkeypatch):
    recipe = _make_recipe()
    monkeypatch.setattr(TrainEagle3Recipe, "_setup_vllm_target", _sentinel)
    with pytest.raises(_DispatchReached):
        recipe._setup_online_target(_RecipeCfg(target_model_backend="vllm"), "target/path", None)


@pytest.mark.parametrize("backend", ["sglang", "vllm", "remote"])
def test_online_target_rejects_packing_on_non_colocated(backend):
    """packed_sequence_size > 0 is colocated-only; vLLM/SGLang/remote leak across docs."""
    recipe = _make_recipe()
    cfg = _RecipeCfg(target_model_backend=backend, packed_sequence_size=4)
    with pytest.raises(NotImplementedError, match="only supported with the colocated"):
        recipe._setup_online_target(cfg, "target/path", None)
