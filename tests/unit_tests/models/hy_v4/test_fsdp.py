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

"""CPU tests for HY4's mixed-dtype FSDP policy and explicit dispatch."""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed import parallelizer
from nemo_automodel.components.models.hy_v4 import fsdp as hy_fsdp
from nemo_automodel.components.models.hy_v4.hc import HyV4HCLayer
from nemo_automodel.components.models.hy_v4.layers import HyV4FP32Parameter


def test_mixed_block_isolates_typed_fp32_islands_before_parent(monkeypatch, tiny_hy_v4_model):
    """iHC and sink FP32 parameters never share an FSDP unit with BF16 weights."""
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(hy_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(hy_fsdp, "_has_fsdp_state", lambda module: False)
    block = tiny_hy_v4_model.model.layers["0"]
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )

    hy_fsdp.fully_shard_hy_v4(
        block,
        mesh=object(),
        mp_policy=policy,
        reshard_after_forward=True,
    )

    assert calls[-1][0] is block
    assert calls[-1][1]["mp_policy"] is policy
    fp32_calls = [
        (module, kwargs) for module, kwargs in calls[:-1] if isinstance(module, (HyV4HCLayer, HyV4FP32Parameter))
    ]
    assert len(fp32_calls) == 3
    assert all(kwargs["mp_policy"].param_dtype is torch.float32 for _, kwargs in fp32_calls)
    assert all(kwargs["mp_policy"].reduce_dtype is torch.float32 for _, kwargs in fp32_calls)
    sink_call = next((module, kwargs) for module, kwargs in fp32_calls if isinstance(module, HyV4FP32Parameter))
    assert sink_call[1]["reshard_after_forward"] is False


def test_dense_parallelizer_explicitly_selects_hy_v4_typed_policy(monkeypatch, tiny_hy_v4_model):
    captured = {}

    def fake_default_parallelize(self, model, device_mesh, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(parallelizer.DefaultParallelizationStrategy, "parallelize", fake_default_parallelize)
    strategy = parallelizer.get_parallelization_strategy(tiny_hy_v4_model)

    assert isinstance(strategy, parallelizer.HyV4ParallelizationStrategy)
    assert strategy.parallelize(tiny_hy_v4_model, object()) is tiny_hy_v4_model
    assert captured["fully_shard_fn"] is hy_fsdp.fully_shard_hy_v4
