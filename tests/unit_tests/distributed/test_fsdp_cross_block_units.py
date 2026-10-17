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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.parallelizer_utils import (
    fully_shard_across_blocks,
    fully_shard_by_dtype,
)


class _Holder(nn.Module):
    """fp32 parameter holder whose forward ignores its input (the attention-sinks pattern)."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n, dtype=torch.float32))

    def forward(self, reference=None):
        del reference
        return self.weight


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
        self.sinks_param = _Holder(4)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.norm = nn.LayerNorm(8).to(torch.bfloat16)


def _policy() -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)


def test_one_unit_for_all_blocks_with_an_fp32_policy():
    blocks = [_Block() for _ in range(3)]
    calls = []

    def shard(module, **kwargs):
        calls.append((module, kwargs))

    owned = fully_shard_across_blocks(
        blocks, ["attn.sinks_param"], mesh="mesh", mp_policy=_policy(), offload_policy=None, shard_module=shard
    )

    assert len(calls) == 1
    modules, kwargs = calls[0]
    assert modules == [block.attn.sinks_param for block in blocks]
    assert kwargs["mesh"] == "mesh"
    assert kwargs["reshard_after_forward"] is False
    assert kwargs["mp_policy"].param_dtype == torch.float32
    assert kwargs["mp_policy"].reduce_dtype == torch.float32
    assert kwargs["mp_policy"].cast_forward_inputs is False
    assert owned == {block.attn.sinks_param.weight for block in blocks}


def test_an_output_dtype_policy_does_not_cast_the_sink_unit():
    """An explicit bf16 ``output_dtype`` on the blocks' policy must not reach the fp32 holders: the unit is internal to
    the block's forward, so its outputs keep their natural dtype (otherwise FSDP rounds every sink to bf16 before
    attention reads it)."""
    blocks = [_Block() for _ in range(2)]
    calls = []
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16, cast_forward_inputs=False
    )

    fully_shard_across_blocks(
        blocks,
        ["attn.sinks_param"],
        mesh="mesh",
        mp_policy=policy,
        offload_policy=None,
        shard_module=lambda module, **kwargs: calls.append(kwargs),
    )

    (kwargs,) = calls
    assert kwargs["mp_policy"].output_dtype is None
    assert kwargs["mp_policy"].param_dtype == torch.float32
    assert kwargs["mp_policy"].reduce_dtype == torch.float32
    assert kwargs["mp_policy"].cast_forward_inputs is False
    assert policy.output_dtype == torch.bfloat16  # the caller's policy is untouched


def _swap_parameters(modules, **kwargs):
    """Stand-in for FSDP2: ``fully_shard`` installs new (sharded DTensor) parameter objects on the modules."""
    for module in modules:
        for name, param in list(module.named_parameters(recurse=False)):
            setattr(module, name, nn.Parameter(param.detach().clone()))


def test_owned_parameters_are_the_objects_installed_by_the_sharding():
    blocks = [_Block() for _ in range(2)]
    before = {block.attn.sinks_param.weight for block in blocks}
    owned = fully_shard_across_blocks(
        blocks,
        ["attn.sinks_param"],
        mesh="mesh",
        mp_policy=_policy(),
        offload_policy=None,
        shard_module=_swap_parameters,
    )
    assert owned == {block.attn.sinks_param.weight for block in blocks}
    assert not (owned & before)


def test_owned_parameters_are_left_out_of_the_per_block_units():
    blocks = [_Block() for _ in range(2)]
    owned = fully_shard_across_blocks(
        blocks,
        ["attn.sinks_param"],
        mesh="mesh",
        mp_policy=_policy(),
        offload_policy=None,
        shard_module=_swap_parameters,
    )
    block_calls = []

    def shard(module, **kwargs):
        block_calls.append((module, kwargs))

    fully_shard_by_dtype(
        blocks[0],
        mesh="mesh",
        mp_policy=_policy(),
        offload_policy=None,
        fp32_compute_module_names=("attn.sinks_param",),
        ignored_params=owned & set(blocks[0].parameters()),
        model_parallelizer=SimpleNamespace(_fully_shard_module=shard),
    )
    # the block keeps a single bf16 unit: no block-level fp32 unit claims the sinks again
    sharded_modules = [module for module, _ in block_calls]
    assert blocks[0].attn.sinks_param not in sharded_modules
    assert blocks[0] in sharded_modules
    block_kwargs = dict(block_calls)[blocks[0]]
    assert block_kwargs.get("ignored_params") == {blocks[0].attn.sinks_param.weight}


def test_no_names_or_no_match_is_a_no_op():
    blocks = [_Block()]
    calls = []
    assert (
        fully_shard_across_blocks(
            blocks, [], mesh="m", mp_policy=_policy(), offload_policy=None, shard_module=calls.append
        )
        == set()
    )
    assert (
        fully_shard_across_blocks(
            blocks, ["attn.missing"], mesh="m", mp_policy=_policy(), offload_policy=None, shard_module=calls.append
        )
        == set()
    )
    assert calls == []


def test_rejects_non_fp32_holders():
    blocks = [_Block()]
    blocks[0].attn.sinks_param.weight.data = blocks[0].attn.sinks_param.weight.data.to(torch.bfloat16)
    with pytest.raises(ValueError, match="fp32"):
        fully_shard_across_blocks(
            blocks,
            ["attn.sinks_param"],
            mesh="m",
            mp_policy=_policy(),
            offload_policy=None,
            shard_module=lambda *a, **k: None,
        )
