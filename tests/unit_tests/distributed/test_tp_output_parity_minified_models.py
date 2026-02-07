# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
import tempfile
from datetime import timedelta
from typing import Literal, cast

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Replicate

from nemo_automodel.components.distributed.parallelizer import _get_parallel_plan
from nemo_automodel.components.models.mistral3.model import Ministral3Config, Ministral3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

ModelKind = Literal["ministral3", "qwen3"]


def _maybe_to_replicated_local(logits, *, tp_mesh: DeviceMesh) -> torch.Tensor:
    if isinstance(logits, DTensor):
        logits = logits.redistribute(device_mesh=tp_mesh, placements=[Replicate()]).to_local()
    assert isinstance(logits, torch.Tensor)
    return logits


def _build_minified_model(kind: ModelKind):
    if kind == "ministral3":
        cfg = Ministral3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            use_cache=False,
            tie_word_embeddings=True,
        )
        return cfg, Ministral3ForCausalLM(cfg)

    if kind == "qwen3":
        cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            use_cache=False,
            tie_word_embeddings=True,
            use_sliding_window=False,
        )
        return cfg, Qwen3ForCausalLM(cfg)

    raise ValueError(f"Unknown model kind: {kind}")


def _tp_parity_worker(
    rank: int,
    world_size: int,
    init_method: str,
    kind: ModelKind,
) -> None:
    # Reduce flakiness on hosts/containers with limited hostname resolution.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(1234)

        cfg, baseline_model = _build_minified_model(kind)
        baseline_model.eval()

        # Keep the input deterministic across ranks.
        torch.manual_seed(999)
        input_ids = torch.randint(0, int(cfg.vocab_size), (2, 8), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            baseline_out = baseline_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        baseline_logits = cast(torch.Tensor, baseline_out.logits).detach().cpu()

        # Build a second model with identical weights to baseline, then TP-parallelize it.
        _, tp_model = _build_minified_model(kind)
        tp_model.load_state_dict(baseline_model.state_dict(), strict=True)
        tp_model.eval()

        tp_mesh = DeviceMesh("cpu", torch.arange(world_size), mesh_dim_names=("tp",))
        plan = _get_parallel_plan(tp_model, sequence_parallel=False)
        parallelize_module(tp_model, tp_mesh, plan)

        with torch.no_grad():
            tp_out = tp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        tp_logits_local = _maybe_to_replicated_local(tp_out.logits, tp_mesh=tp_mesh).detach().cpu()

        torch.testing.assert_close(tp_logits_local, baseline_logits, rtol=1e-4, atol=1e-4)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_tp_parity(kind: ModelKind) -> None:
    world_size = 2
    with tempfile.NamedTemporaryFile(prefix="nemo_automodel_tp_parity_", delete=False) as f:
        init_file = f.name
    init_method = f"file://{init_file}"
    try:
        torch.multiprocessing.spawn(
            _tp_parity_worker,
            args=(world_size, init_method, kind),
            nprocs=world_size,
            join=True,
        )
    finally:
        try:
            os.remove(init_file)
        except OSError:
            pass


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend not available")
def test_ministral3_tp2_matches_tp1_outputs_minified():
    _run_tp_parity("ministral3")


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend not available")
def test_qwen3_tp2_matches_tp1_outputs_minified():
    _run_tp_parity("qwen3")

