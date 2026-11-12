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

"""Materialization-device selection for FSDP2 CPU offload vs. offload_experts_only.

Review finding: with ``moe.offload_experts_only=true`` the dense units keep
``offload_policy=None`` (GPU residency), but infrastructure materialized *all*
parameters on CPU whenever a global CPUOffloadPolicy existed. The dense FSDP
units then held CPU master storage and only worked through FSDP's temporary
CUDA gather buffers — not the documented dense-GPU residency, and it kept the
host-to-device copies the option is supposed to avoid.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from nemo_automodel._transformers.infrastructure import apply_model_infrastructure
from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
from nemo_automodel.components.distributed.mesh import MeshContext


def _mesh_ctx(moe: MoEParallelizerConfig | None) -> MeshContext:
    return MeshContext(moe_parallel_config=moe)


def _run_materialize_device(strategy_config, moe_config) -> torch.device:
    """Invoke apply_model_infrastructure's meta-materialization path and
    capture the init_device passed to checkpointer.initialize_model_weights.

    ``torch.nn.Module.to`` is stubbed so the check runs on CPU-only test hosts;
    only the device *selection* is asserted here (residency itself is verified
    by the GPU functional suites).
    """
    device = torch.device("cuda", 0)
    captured = {}
    with (
        patch("nemo_automodel._transformers.infrastructure.Checkpointer") as ckpt_cls,
        patch("nemo_automodel._transformers.infrastructure.print_trainable_parameters"),
        patch("nemo_automodel._transformers.infrastructure._verify_safe_moe_tp_weights_loaded"),
        patch("nemo_automodel._transformers.infrastructure._apply_trainability_policy"),
        patch.object(torch.nn.Module, "to", lambda self, *a, **k: self),
    ):
        # No checkpoint path (empty name) + parallelize_fn -> need_materialize
        # reaches the init_device selection at the post-shard call site.
        ckpt_cls.return_value.initialize_model_weights.side_effect = lambda mp, init_device, peft_init_method=None: (
            captured.setdefault("init_device", init_device)
        )
        apply_model_infrastructure(
            torch.nn.Linear(2, 2),
            is_meta_device=True,
            device=device,
            model_wrapper=SimpleNamespace(
                config=strategy_config,  # legacy-wrapper attribute read by _get_strategy_config
            ),
            mesh=_mesh_ctx(moe_config),
            parallelize_fn=lambda *a, **k: None,
        )
    return captured["init_device"]


def test_full_cpu_offload_materializes_on_cpu():
    cfg = FSDP2Config(offload_policy=torch.distributed.fsdp.CPUOffloadPolicy())
    assert _run_materialize_device(cfg, MoEParallelizerConfig()) == torch.device("cpu")


def test_offload_experts_only_materializes_on_device():
    """Dense units stay GPU-resident: materialization must target the device —
    the expert FSDP units move their own shards to CPU during wrap."""
    cfg = FSDP2Config(offload_policy=torch.distributed.fsdp.CPUOffloadPolicy())
    moe = MoEParallelizerConfig(offload_experts_only=True)
    assert _run_materialize_device(cfg, moe) == torch.device("cuda", 0)


def test_no_offload_materializes_on_device():
    cfg = FSDP2Config(offload_policy=None)
    assert _run_materialize_device(cfg, MoEParallelizerConfig()) == torch.device("cuda", 0)
