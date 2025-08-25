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

import types
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoConfig, AutoModelForCausalLM

from nemo_automodel.components.distributed.autopipeline.functional import (
    split_model_into_stages,
    generate_hf_model_fqn_per_model_part,
)
# from nemo_automodel.components.training import pp_utils as pp_utils_mod  # Not available


class DummyRotaryEmb(nn.Module):
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        return torch.zeros_like(hidden_states)


class DummyDecoderLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return (hidden_states,)


class DummyInnerModel(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 64, num_layers: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device="meta")
        self.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size, device="meta")
        self.rotary_emb = DummyRotaryEmb()


class DummyQwenForCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 64, num_layers: int = 8):
        super().__init__()
        self.model = DummyInnerModel(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, device="meta")
        # minimal config stub for any accesses
        self.config = types.SimpleNamespace(output_attentions=False, output_hidden_states=False, use_cache=False)


def setup_fake_distributed(pp_size: int, local_rank: int = 0):
    """Setup fake distributed environment and return mesh."""
    if not dist.is_initialized():
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=local_rank, world_size=pp_size, store=store)

    # Create device mesh with fake devices
    devices = [torch.device("fake") for _ in range(pp_size)]
    mesh = DeviceMesh("fake", list(range(pp_size)), mesh_dim_names=("pp",))
    return mesh


class FakePPMesh:
    def __init__(self, size: int, local_rank: int):
        self._size = size
        self._local_rank = local_rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._local_rank

    def get_group(self, *_, **__):
        return None


class FakeWorldMesh(dict):
    # simple mapping wrapper: world_mesh[pp_axis_name] -> FakePPMesh
    pass


class DummyPipelineStage:
    def __init__(self, submod: nn.Module, stage_idx: int, num_stages: int, device: torch.device, group=None):
        self.submod = submod
        self.stage_index = stage_idx
        self.num_stages = num_stages
        self.device = device
        self.group = group
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == (num_stages - 1)


def _patch_pipeline_stage(monkeypatch):
    # Replace real PipelineStage with our dummy to avoid distributed setup
    import nemo_automodel.components.distributed.autopipeline.functional as pipe

    monkeypatch.setattr(pipe, "PipelineStage", DummyPipelineStage)


class TestModelSplittingBasics:
    """Test basic model splitting functionality with dummy models."""

    @pytest.mark.parametrize("pp_size", [2, 4])
    @pytest.mark.parametrize("local_rank", [0, 1, 2, 3])
    def test_split_qwen_like_model_into_stages_and_verify_keys(self, monkeypatch, pp_size, local_rank):
        _patch_pipeline_stage(monkeypatch)
        if local_rank >= pp_size:
            pytest.skip("local_rank not part of this pp_size")

        # Build a Qwen-like model on meta
        num_layers = 8
        model = DummyQwenForCausalLM(num_layers=num_layers)

        # Build explicit module FQNs for pp_size*2 stages (2 stages per rank)
        num_stages = pp_size * 2
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=num_stages,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        # Fake PP mesh and split
        pp_mesh = FakePPMesh(size=pp_size, local_rank=local_rank)
        device = torch.device("cpu")

        stages, model_parts = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            device=device,
            module_names_per_stage=module_fqns,
            layers_per_stage=None,
            round_to_pp_multiple=None,
        )

        # Expect 2 local stages per rank
        assert len(stages) == 2
        assert len(model_parts) == 2

        # Determine expected stage indices for this rank
        local_stage_indices = [local_rank + s * pp_size for s in range(2)]

        for part, global_stage_idx in zip(model_parts, local_stage_indices):
            # layers should be ModuleDict with only indices assigned to this global stage
            assert isinstance(part.model.layers, nn.ModuleDict)
            expected_layer_indices = sorted(
                int(name.split(".")[-1])
                for name in module_fqns[global_stage_idx]
                if name.startswith("model.layers.")
            )
            assert sorted(map(int, part.model.layers.keys())) == expected_layer_indices

            # rotary_emb should be present for every stage
            assert part.model.rotary_emb is not None

            # embed_tokens only on first global stage
            if global_stage_idx == 0:
                assert part.model.embed_tokens is not None
            else:
                assert part.model.embed_tokens is None

            # norm only on last global stage
            if global_stage_idx == num_stages - 1:
                assert part.model.norm is not None
            else:
                assert part.model.norm is None

            # lm_head only on last global stage (top-level attribute)
            if global_stage_idx == num_stages - 1:
                assert part.lm_head is not None
            else:
                assert part.lm_head is None

    def test_meta_model_splitting_basic(self, monkeypatch):
        """Test basic meta model splitting without external dependencies."""
        _patch_pipeline_stage(monkeypatch)

        # Build model on meta device
        with torch.device("meta"):
            model = DummyQwenForCausalLM(num_layers=8)

        # Verify meta placement for key parameters
        assert model.model.embed_tokens.weight.device.type == "meta"
        assert model.lm_head.weight.device.type == "meta"

        # Now split explicitly
        num_layers = len(model.model.layers)
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=4,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        pp_mesh = FakePPMesh(size=2, local_rank=0)
        device = torch.device("cpu")
        _, parts_rank0 = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            device=device,
            module_names_per_stage=module_fqns,
        )

        # rank0 holds stages 0 and 2
        assert len(parts_rank0) == 2
        s0, s2 = parts_rank0

        # Quick key sanity checks
        assert "0" in s0.model.layers.keys()
        assert s0.model.embed_tokens is not None
        assert s0.lm_head is None

        assert s2.model.embed_tokens is None
        assert s2.lm_head is None

    @pytest.mark.parametrize("num_layers,num_stages", [(8, 2), (16, 4), (12, 3)])
    def test_different_layer_stage_combinations(self, monkeypatch, num_layers, num_stages):
        """Test model splitting with different layer/stage combinations."""
        _patch_pipeline_stage(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=num_layers)

        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=num_stages,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        pp_mesh = FakePPMesh(size=num_stages, local_rank=0)
        device = torch.device("cpu")

        stages, model_parts = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            device=device,
            module_names_per_stage=module_fqns,
        )

        # Verify we get the expected number of stages for rank 0
        expected_stages_for_rank0 = 1  # Since pp_size == num_stages, rank 0 gets 1 stage
        assert len(stages) == expected_stages_for_rank0
        assert len(model_parts) == expected_stages_for_rank0

        # Verify layer distribution
        total_layers_assigned = 0
        for part in model_parts:
            if isinstance(part.model.layers, nn.ModuleDict):
                total_layers_assigned += len(part.model.layers)

        # The layers should be a subset of total layers (since we're only looking at rank 0)
        assert total_layers_assigned <= num_layers

class TestHFModelSplitting:
    """Test model splitting with HF models (end-to-end tests from original file)."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen/Qwen3-0.6B-Base",
            "Qwen/Qwen3-1.7B-Base",
            "Qwen/Qwen3-4B-Base",
        ],
    )
    @pytest.mark.parametrize("pp_size", [2, 4, 8])
    @pytest.mark.parametrize("local_rank", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_split_qwen3_models_on_meta(self, monkeypatch, model_id, pp_size, local_rank):
        _patch_pipeline_stage(monkeypatch)
        if local_rank >= pp_size:
            pytest.skip("local_rank not part of this pp_size")

        # Load actual Qwen3 config; skip if not available or offline
        try:
            cfg = AutoConfig.from_pretrained(model_id)
        except Exception as e:  # pragma: no cover - network dependent
            pytest.skip(f"Skipping {model_id}: {e}")

        # Instantiate the model on meta to avoid memory use
        try:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(cfg)
        except Exception as e:  # pragma: no cover - arch unsupported in local transformers
            pytest.skip(f"Model init on meta failed for {model_id}: {e}")

        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            pytest.skip(f"Model {model_id} does not expose model.layers; skipping structural test")

        num_layers = len(model.model.layers)
        if num_layers < pp_size * 2:
            pytest.skip(f"Model {model_id} has too few layers ({num_layers}) for pp_size={pp_size}")
        num_stages = pp_size * 2

        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=num_stages,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=hasattr(model, "lm_head"),
            include_rotary_emb=hasattr(model.model, "rotary_emb"),
            fqn_prefix="model.",
        )

        pp_mesh = FakePPMesh(size=pp_size, local_rank=local_rank)
        device = torch.device("cpu")

        stages, model_parts = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            device=device,
            module_names_per_stage=module_fqns,
            layers_per_stage=None,
        )

        assert len(stages) == (num_stages // pp_size)
        assert len(model_parts) == (num_stages // pp_size)

        local_stage_indices = [local_rank + i * pp_size for i in range(num_stages // pp_size)]

        for part, global_stage_idx in zip(model_parts, local_stage_indices):
            assert isinstance(part.model.layers, nn.ModuleDict)
            expected_layer_indices = sorted(
                int(name.split(".")[-1])
                for name in module_fqns[global_stage_idx]
                if name.startswith("model.layers.")
            )
            assert sorted(map(int, part.model.layers.keys())) == expected_layer_indices

            if hasattr(model.model, "rotary_emb"):
                assert part.model.rotary_emb is not None
            else:
                assert not hasattr(part.model, "rotary_emb") or part.model.rotary_emb is None

            if global_stage_idx == 0:
                assert part.model.embed_tokens is not None
            else:
                assert part.model.embed_tokens is None

            if global_stage_idx == num_stages - 1:
                assert part.model.norm is not None
            else:
                assert part.model.norm is None

            if hasattr(model, "lm_head"):
                if global_stage_idx == num_stages - 1:
                    assert part.lm_head is not None
                else:
                    assert part.lm_head is None


class TestSplittingEdgeCases:
    """Test edge cases and error conditions in model splitting."""

    def test_single_stage_splitting(self, monkeypatch):
        """Test splitting into a single stage."""
        _patch_pipeline_stage(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=4)
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=1,
            num_layers=4,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        pp_mesh = FakePPMesh(size=1, local_rank=0)
        device = torch.device("cpu")

        stages, model_parts = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="PipelineScheduleSingle",
            device=device,
            module_names_per_stage=module_fqns,
        )

        assert len(stages) == 1
        assert len(model_parts) == 1

        # Single stage should contain all components
        part = model_parts[0]
        assert part.model.embed_tokens is not None
        assert part.model.norm is not None
        assert part.lm_head is not None
        assert len(part.model.layers) == 4

    def test_1f1b_single_stage_schedule(self, monkeypatch):
        """Test 1f1b schedule (single-stage) with layers_per_stage parameter."""
        _patch_pipeline_stage(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=8)

        # Provide module_fqns but also layers_per_stage
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=8,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        pp_mesh = FakePPMesh(size=2, local_rank=0)
        device = torch.device("cpu")

        # 1f1b is treated as a single-stage schedule - requires 1 stage per rank
        stages, model_parts = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            device=device,
            module_names_per_stage=module_fqns,
            layers_per_stage=8,  # Use layers_per_stage=8 to get 1+1=2 stages total (1 per rank)
        )

        # Single-stage schedule: 1 stage per rank
        assert len(stages) == 1  # rank 0 gets 1 stage
        assert len(model_parts) == 1

    def test_interleaved1f1b_multi_stage_schedule(self, monkeypatch):
        """Test interleaved1f1b schedule (multi-stage) with layers_per_stage parameter."""
        _patch_pipeline_stage(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=8)

        # Provide module_fqns but also layers_per_stage
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=4,
            num_layers=8,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        pp_mesh = FakePPMesh(size=2, local_rank=0)
        device = torch.device("cpu")

        # interleaved1f1b is treated as a multi-stage schedule - requires at least 2 stages per rank
        stages, model_parts = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="interleaved1f1b",
            device=device,
            module_names_per_stage=module_fqns,
            layers_per_stage=2,  # Use layers_per_stage=2 to get 4+1=5 stages, round up to 6 (3 per rank)
            round_to_pp_multiple="up",
        )

        # Multi-stage schedule: at least 2 stages per rank
        # With layers_per_stage=2, we get ceil(8/2)+1=5 stages, rounded up to 6 for pp_size=2
        # But the actual allocation gives us 2 stages per rank
        assert len(stages) == 2  # rank 0 gets 2 stages
        assert len(model_parts) == 2

    def test_single_stage_schedule_layers_per_stage(self, monkeypatch):
        """Test layers_per_stage with single stage schedule."""
        _patch_pipeline_stage(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=8)

        # For single stage schedule, we need exactly 1 stage per rank
        pp_mesh = FakePPMesh(size=2, local_rank=0)
        device = torch.device("cpu")

        stages, model_parts = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="PipelineScheduleSingle",
            device=device,
            layers_per_stage=8,  # 8 layers / 8 = 1 stage + 1 = 2 stages total, perfect for pp_size=2
        )

        # Single stage schedule: 1 stage per rank
        assert len(stages) == 1  # rank 0 gets 1 stage
        assert len(model_parts) == 1

    def test_multi_stage_schedule_layers_per_stage(self, monkeypatch):
        """Test layers_per_stage with multi-stage schedules."""
        _patch_pipeline_stage(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=8)

        pp_mesh = FakePPMesh(size=2, local_rank=0)
        device = torch.device("cpu")

        # Use a known multi-stage schedule
        stages, model_parts = split_model_into_stages(
            model=model,
            pp_mesh=pp_mesh,
            pp_axis_name="pp",
            pp_schedule="PipelineScheduleMulti",  # Explicit multi-stage schedule
            device=device,
            layers_per_stage=2,  # 8 layers / 2 = 4 stages + 1 = 5 stages, round up to 6
            round_to_pp_multiple="up",
        )

        # Multi-stage schedule: at least 2 stages per rank
        # With 6 total stages and pp_size=2, each rank gets 3 stages
        assert len(stages) == 3  # rank 0 gets 3 stages
        assert len(model_parts) == 3
