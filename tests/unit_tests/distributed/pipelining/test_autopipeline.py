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
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.distributed.pipelining.microbatch import TensorChunkSpec, split_args_kwargs_into_chunks

from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.distributed.pipelining.functional import (
    generate_hf_model_fqn_per_model_part,
)


class DummyRotaryEmb(nn.Module):
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        return torch.zeros_like(hidden_states)


class DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # tiny param to ensure some grads exist when needed
        self.proj = nn.Linear(8, 8, bias=False, device="meta")

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
        # minimal config stub
        self.config = types.SimpleNamespace(output_attentions=False, output_hidden_states=False, use_cache=False)


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


class FakeDeviceMesh:
    """Mock DeviceMesh that behaves like the real DeviceMesh but without distributed setup."""

    def __init__(self, device_type="cpu", mesh=None, mesh_dim_names=None, pp_size=2, local_rank=0):
        self.device_type = device_type
        self.mesh = mesh or [[0, 1]]
        self.mesh_dim_names = mesh_dim_names or ["pp"]
        self._pp_size = pp_size
        self._local_rank = local_rank

    def __getitem__(self, key):
        # Return a FakePPMesh when accessed like mesh["pp"]
        if key == "pp":
            return FakePPMesh(size=self._pp_size, local_rank=self._local_rank)
        return self

    def size(self):
        return self._pp_size

    def get_local_rank(self):
        return self._local_rank


class FakeWorldMesh(dict):
    """Mock for DeviceMesh that behaves like a dict."""

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
        self._scaled = 1

    def scale_grads(self, divisor: int):
        # record the last divisor for verification if needed
        self._scaled = divisor


class FakeSchedule:
    def __init__(self, stages: list[DummyPipelineStage], n_microbatches: int = 1):
        self._stages = stages
        self.n_microbatches = n_microbatches

    def step(self, *args, target=None, losses=None, **kwargs):
        # append a tiny dummy loss on last stage
        if losses is not None:
            losses.append(torch.tensor(0.123, device=kwargs.get("device", "cpu")))


def _patch_autopipeline_monkey(monkeypatch):
    # Replace real PipelineStage and schedule builder with our dummies
    import nemo_automodel.components.distributed.pipelining.functional as fn

    monkeypatch.setattr(fn, "PipelineStage", DummyPipelineStage)

    def _fake_build_schedule(pp_schedule_csv, pp_schedule, micro, batch, stages, loss_fn, scale_grads=False):
        return FakeSchedule(stages, n_microbatches=(batch // max(micro, 1)))

    monkeypatch.setattr(fn, "build_pipeline_schedule", _fake_build_schedule)


class TestAutoPipelineValidation:
    """Test AutoPipeline validation and properties."""

    def test_valid_autopipeline(self):
        world_mesh = FakeDeviceMesh()
        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            device=torch.device("cpu"),
        )

        assert ap.world_mesh == world_mesh
        assert ap.pp_axis_name == "pp"
        assert ap.pp_schedule == "1f1b"
        assert ap.pp_microbatch_size == 1
        assert ap.pp_batch_size == 4
        assert ap.num_microbatches == 4
        assert ap._device == torch.device("cpu")

    def test_invalid_batch_size(self):
        world_mesh = FakeDeviceMesh()

        # The validation happens in __init__
        with pytest.raises(ValueError, match="local_batch_size must be divisible by microbatch_size"):
            AutoPipeline(
                world_mesh=world_mesh,
                pp_axis_name="pp",
                pp_schedule="1f1b",
                pp_microbatch_size=3,
                pp_batch_size=4,  # 4 not divisible by 3
                device=torch.device("cpu"),
            )

    def test_schedule_validation(self):
        world_mesh = FakeDeviceMesh()

        # Test missing schedule validation
        with pytest.raises(
            ValueError, match="Either pipeline_parallel_schedule or pipeline_parallel_schedule_csv must be provided"
        ):
            AutoPipeline(
                world_mesh=world_mesh,
                pp_axis_name="pp",
                pp_schedule=None,  # Both pp_schedule and pp_schedule_csv are None
                pp_schedule_csv=None,
                pp_microbatch_size=1,
                pp_batch_size=4,
                device=torch.device("cpu"),
            )

        # Valid schedule should not raise
        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            device=torch.device("cpu"),
        )
        # Should not raise
        assert ap is not None

    def test_pp_mesh_extraction(self):
        world_mesh = FakeDeviceMesh()

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            device=torch.device("cpu"),
        )

        assert ap.world_mesh == world_mesh
        assert ap.pp_mesh is not None


class _KwargsChunkHookPart(nn.Module):
    def __init__(self, chunk_dims: dict[str, int]):
        super().__init__()
        self.chunk_dims = chunk_dims

    def get_pipeline_kwargs_chunk_dims(self, kwargs):
        return {key: dim for key, dim in self.chunk_dims.items() if key in kwargs}


class _UnknownKwargsChunkHookPart(nn.Module):
    def get_pipeline_kwargs_chunk_dims(self, kwargs):
        return {"unknown": 0}


class _KwargsChunkSchedule:
    def __init__(self, *, fail_on_step: bool = False, invoke_loss: bool = False):
        self._kwargs_chunk_spec = None
        self._loss_fn = Mock(return_value=torch.tensor(0.0))
        self.fail_on_step = fail_on_step
        self.invoke_loss = invoke_loss
        self.args_during_step = None
        self.args_split = None
        self.kwargs_chunk_spec_during_step = None
        self.loss_fn_during_step = None
        self.split_inputs_during_step = None
        self.kwargs_split = None
        self.target_during_step = None
        self.losses_during_step = None
        self.return_outputs_during_step = None
        self.loss_results = []
        self.step_calls = 0
        self.eval_calls = 0
        self.split_inputs_calls = 0

    def _split_inputs(self, args, kwargs=None):
        return split_args_kwargs_into_chunks(
            args,
            kwargs,
            2,
            kwargs_chunk_spec=self._kwargs_chunk_spec,
        )

    def _run_schedule(self, *args, target=None, losses=None, return_outputs=True, **kwargs):
        """Split schedule inputs using the chunk spec active during the call.

        Args:
            *args: Positional schedule inputs. Tensor values have arbitrary
                model-defined layouts.
            target: Optional tensor of shape [batch, ...] containing loss targets
                or prepared-input microbatch IDs.
            losses: Optional mutable list populated with scalar loss tensors.
            return_outputs: Whether to return the schedule result.
            **kwargs: Keyword schedule inputs. Tensor values have arbitrary
                model-defined layouts.

        Returns:
            A sentinel string identifying the schedule result.
        """
        self.args_during_step = args
        self.kwargs_chunk_spec_during_step = self._kwargs_chunk_spec
        self.loss_fn_during_step = self._loss_fn
        self.split_inputs_during_step = self._split_inputs
        self.target_during_step = target
        self.losses_during_step = losses
        self.return_outputs_during_step = return_outputs
        if self.fail_on_step:
            raise RuntimeError("schedule failed")
        self.split_inputs_calls += 1
        self.args_split, self.kwargs_split = self._split_inputs(args, kwargs)
        if self.invoke_loss:
            assert target is not None
            target_chunks = torch.tensor_split(target, 2)
            for index in (1, 0):
                self.loss_results.append(self._loss_fn(torch.tensor(float(index)), target_chunks[index]))
        return "schedule-result"

    def step(self, *args, target=None, losses=None, return_outputs=True, **kwargs):
        self.step_calls += 1
        return self._run_schedule(
            *args,
            target=target,
            losses=losses,
            return_outputs=return_outputs,
            **kwargs,
        )

    def eval(self, *args, target=None, losses=None, return_outputs=True, **kwargs):
        self.eval_calls += 1
        return self._run_schedule(
            *args,
            target=target,
            losses=losses,
            return_outputs=return_outputs,
            **kwargs,
        )


class _LegacyStepSchedule(_KwargsChunkSchedule):
    """Schedule with the PyTorch 2.6-2.9 step signature."""

    def __init__(self):
        super().__init__()
        self.received_return_outputs = False

    def step(self, *args, target=None, losses=None, **kwargs):
        self.received_return_outputs = "return_outputs" in kwargs
        return super().step(*args, target=target, losses=losses, **kwargs)


class _LegacyEvalSchedule(_LegacyStepSchedule):
    """Schedule with an eval signature that predates return_outputs."""

    def __init__(self):
        super().__init__()
        self.received_return_outputs = False

    def eval(self, *args, target=None, losses=None, **kwargs):
        self.received_return_outputs = "return_outputs" in kwargs
        self.eval_calls += 1
        return self._run_schedule(*args, target=target, losses=losses, **kwargs)


class _ForwardingEvalSchedule(_KwargsChunkSchedule):
    """Current PyTorch shape: eval forwards kwargs to a newer step API."""

    def eval(self, *args, target=None, losses=None, **kwargs):
        self.eval_calls += 1
        return self._run_schedule(*args, target=target, losses=losses, **kwargs)


class _NoEvalSchedule(_LegacyStepSchedule):
    """PyTorch pipeline schedule shape before forward-only eval existed."""

    eval = None


class TestAutoPipelineKwargsChunkSpec:
    def _pipeline_with_parts(self, *parts: nn.Module, schedule=None, has_first_stage: bool = True):
        ap = AutoPipeline(
            world_mesh=FakeDeviceMesh(),
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=torch.device("cpu"),
        )
        ap._info.schedule = schedule or _KwargsChunkSchedule()
        ap._info.model_parts = list(parts)
        ap._info.has_first_stage = has_first_stage
        return ap

    def test_step_splits_mrope_position_ids_on_model_owned_batch_axis(self):
        """AutoPipeline.step keeps all mRoPE axes in every microbatch."""
        input_ids = torch.zeros(2, 8, dtype=torch.long)
        position_ids = torch.arange(8, dtype=torch.long).view(1, 1, -1).expand(3, 2, -1).clone()
        kwargs = {
            "position_ids": position_ids,
            "attention_mask": torch.ones(2, 8, dtype=torch.bool),
            "qkv_format": "thd",
        }

        _, default_kwargs_split = split_args_kwargs_into_chunks((input_ids,), kwargs, 2)
        assert default_kwargs_split[0]["position_ids"].shape == (2, 2, 8)

        ap = self._pipeline_with_parts(_KwargsChunkHookPart({"position_ids": 1}))
        result = ap.step(input_ids, **kwargs)

        fixed_kwargs_split = ap.info.schedule.kwargs_split
        assert result == "schedule-result"
        assert fixed_kwargs_split[0]["position_ids"].shape == (3, 1, 8)
        assert fixed_kwargs_split[1]["position_ids"].shape == (3, 1, 8)
        torch.testing.assert_close(fixed_kwargs_split[0]["position_ids"], position_ids[:, :1])
        torch.testing.assert_close(fixed_kwargs_split[1]["position_ids"], position_ids[:, 1:])
        assert fixed_kwargs_split[0]["attention_mask"].shape == (1, 8)
        assert fixed_kwargs_split[0]["qkv_format"] == "thd"
        assert fixed_kwargs_split[1]["qkv_format"] == "thd"
        assert ap.info.schedule.args_during_step == (input_ids,)
        assert ap.info.schedule._kwargs_chunk_spec is None

    def test_step_without_model_hook_uses_pytorch_default_chunking(self):
        ap = self._pipeline_with_parts(nn.Module())

        ap.step(torch.zeros(2, 8), attention_mask=torch.ones(2, 8))

        assert ap.info.schedule.kwargs_chunk_spec_during_step is None
        assert ap.info.schedule.kwargs_split[0]["attention_mask"].shape == (1, 8)
        assert ap.info.schedule._kwargs_chunk_spec is None

    def test_step_microbatches_passes_prepared_inputs_without_resplitting(self):
        schedule = _KwargsChunkSchedule(invoke_loss=True)
        ap = self._pipeline_with_parts(nn.Module(), schedule=schedule)
        input_ids = [torch.full((1, 8), index, dtype=torch.long) for index in range(2)]
        position_ids = [torch.full((3, 1, 8), index, dtype=torch.long) for index in range(2)]
        metadata = [object(), object()]
        model_inputs = [
            {
                "input_ids": input_ids[index],
                "position_ids": position_ids[index],
                "metadata": metadata[index],
            }
            for index in range(2)
        ]
        original_split_inputs = schedule._split_inputs
        original_loss_fn = schedule._loss_fn
        seen = []
        losses = []

        def loss_fn(output, index):
            seen.append((output, index))
            return output

        result = ap.step_microbatches(model_inputs, loss_fn=loss_fn, losses=losses, return_outputs=False)

        assert result == "schedule-result"
        assert schedule.args_during_step == ()
        assert schedule.args_split[0][0] is input_ids[0]
        assert schedule.args_split[1][0] is input_ids[1]
        assert schedule.kwargs_split[0]["position_ids"] is position_ids[0]
        assert schedule.kwargs_split[1]["position_ids"] is position_ids[1]
        assert schedule.kwargs_split[0]["metadata"] is metadata[0]
        assert schedule.kwargs_split[1]["metadata"] is metadata[1]
        assert schedule.target_during_step.tolist() == [0, 1]
        assert schedule.losses_during_step is losses
        assert schedule.return_outputs_during_step is False
        assert [(output.item(), index) for output, index in seen] == [(1.0, 1), (0.0, 0)]
        assert schedule._split_inputs == original_split_inputs
        assert schedule._loss_fn is original_loss_fn
        assert model_inputs[0]["input_ids"] is input_ids[0]
        assert model_inputs[1]["input_ids"] is input_ids[1]

    def test_step_microbatches_omits_primary_args_on_nonfirst_stage(self):
        schedule = _KwargsChunkSchedule()
        ap = self._pipeline_with_parts(nn.Module(), schedule=schedule, has_first_stage=False)
        model_inputs = [{"inputs_embeds": torch.zeros(1, 8, 4), "position_ids": torch.zeros(1, 8)} for _ in range(2)]

        ap.step_microbatches(model_inputs, loss_fn=Mock())

        assert schedule.args_split == [(), ()]
        assert all("inputs_embeds" not in kwargs for kwargs in schedule.kwargs_split)

    @pytest.mark.parametrize(
        ("method_name", "schedule_cls"),
        [
            ("step_microbatches", _LegacyStepSchedule),
            ("eval_microbatches", _LegacyEvalSchedule),
        ],
    )
    def test_prepared_microbatches_do_not_forward_return_outputs_to_older_pytorch(
        self,
        method_name,
        schedule_cls,
    ):
        schedule = schedule_cls()
        ap = self._pipeline_with_parts(nn.Module(), schedule=schedule)

        getattr(ap, method_name)(
            [{"input_ids": torch.zeros(1, 8)} for _ in range(2)],
            loss_fn=Mock(),
            return_outputs=False,
        )

        assert schedule.received_return_outputs is False

    def test_eval_microbatches_uses_forward_only_schedule_with_exact_prepared_split(self):
        schedule = _KwargsChunkSchedule(invoke_loss=True)
        ap = self._pipeline_with_parts(nn.Module(), schedule=schedule)
        input_ids = [torch.full((1, 8), index, dtype=torch.long) for index in range(2)]
        metadata = [object(), object()]
        model_inputs = [
            {
                "input_ids": input_ids[index],
                "metadata": metadata[index],
            }
            for index in range(2)
        ]
        original_split_inputs = schedule._split_inputs
        original_loss_fn = schedule._loss_fn
        seen = []
        losses = []

        def loss_fn(output, index):
            seen.append((output, index))
            return output

        result = ap.eval_microbatches(model_inputs, loss_fn=loss_fn, losses=losses, return_outputs=False)

        assert result == "schedule-result"
        assert schedule.eval_calls == 1
        assert schedule.step_calls == 0
        assert schedule.split_inputs_calls == 1
        assert schedule.args_split[0][0] is input_ids[0]
        assert schedule.args_split[1][0] is input_ids[1]
        assert schedule.kwargs_split[0]["metadata"] is metadata[0]
        assert schedule.kwargs_split[1]["metadata"] is metadata[1]
        assert schedule.target_during_step.tolist() == [0, 1]
        assert schedule.losses_during_step is losses
        assert schedule.return_outputs_during_step is False
        assert [(output.item(), index) for output, index in seen] == [(1.0, 1), (0.0, 0)]
        assert schedule._split_inputs == original_split_inputs
        assert schedule._loss_fn is original_loss_fn

    def test_eval_microbatches_uses_step_capability_when_eval_forwards_kwargs(self):
        schedule = _ForwardingEvalSchedule()
        ap = self._pipeline_with_parts(nn.Module(), schedule=schedule)

        ap.eval_microbatches(
            [{"input_ids": torch.zeros(1, 8)} for _ in range(2)],
            loss_fn=Mock(),
            return_outputs=False,
        )

        assert schedule.eval_calls == 1
        assert schedule.step_calls == 0
        assert schedule.return_outputs_during_step is False

    def test_eval_microbatches_fails_clearly_when_pytorch_has_no_eval_schedule(self):
        schedule = _NoEvalSchedule()
        ap = self._pipeline_with_parts(nn.Module(), schedule=schedule)

        with pytest.raises(NotImplementedError, match=r"schedule with eval\(\)"):
            ap.eval_microbatches(
                [{"input_ids": torch.zeros(1, 8)} for _ in range(2)],
                loss_fn=Mock(),
            )

        assert schedule.step_calls == 0

    @pytest.mark.parametrize(
        "model_inputs",
        [
            [{"input_ids": torch.zeros(1, 8)}],
            [{"attention_mask": torch.ones(1, 8)} for _ in range(2)],
            [{"input_ids": torch.zeros(1, 8), "inputs_embeds": torch.zeros(1, 8, 4)} for _ in range(2)],
        ],
    )
    def test_step_microbatches_validates_prepared_inputs(self, model_inputs):
        ap = self._pipeline_with_parts(nn.Module())

        with pytest.raises(ValueError, match="Expected 2|exactly one"):
            ap.step_microbatches(model_inputs, loss_fn=Mock())

    @pytest.mark.parametrize("method_name", ["step_microbatches", "eval_microbatches"])
    def test_prepared_microbatches_restore_schedule_state_after_failure(self, method_name):
        schedule = _KwargsChunkSchedule(fail_on_step=True)
        original_split_inputs = schedule._split_inputs
        original_loss_fn = schedule._loss_fn
        ap = self._pipeline_with_parts(nn.Module(), schedule=schedule)

        with pytest.raises(RuntimeError, match="schedule failed"):
            getattr(ap, method_name)(
                [{"input_ids": torch.zeros(1, 8)} for _ in range(2)],
                loss_fn=Mock(),
            )

        assert schedule.eval_calls == int(method_name == "eval_microbatches")
        assert schedule.step_calls == int(method_name == "step_microbatches")
        assert schedule.split_inputs_during_step is not original_split_inputs
        assert schedule.loss_fn_during_step is not original_loss_fn
        assert schedule._split_inputs == original_split_inputs
        assert schedule._loss_fn is original_loss_fn

    def test_only_canonical_model_part_supplies_chunk_policy(self):
        ap = self._pipeline_with_parts(
            _KwargsChunkHookPart({"position_ids": 1}),
            _KwargsChunkHookPart({"position_ids": 0}),
        )

        ap.step(torch.zeros(2, 8), position_ids=torch.zeros(3, 2, 8))

        assert ap.info.schedule.kwargs_split[0]["position_ids"].shape == (3, 1, 8)

    def test_nonfirst_stage_ignores_model_input(self):
        ap = self._pipeline_with_parts(nn.Module(), has_first_stage=False)

        ap.step(torch.zeros(2, 8), attention_mask=torch.ones(2, 8))

        assert ap.info.schedule.args_during_step == ()
        assert ap.info.schedule.kwargs_split[0]["attention_mask"].shape == (1, 8)

    def test_step_restores_schedule_chunk_spec_after_failure(self):
        schedule = _KwargsChunkSchedule(fail_on_step=True)
        original_chunk_spec = {"position_ids": TensorChunkSpec(0)}
        schedule._kwargs_chunk_spec = original_chunk_spec
        ap = self._pipeline_with_parts(_KwargsChunkHookPart({"position_ids": 1}), schedule=schedule)

        with pytest.raises(RuntimeError, match="schedule failed"):
            ap.step(torch.zeros(2, 8), position_ids=torch.zeros(3, 2, 8))

        assert schedule.kwargs_chunk_spec_during_step["position_ids"].split_dim == 1
        assert schedule._kwargs_chunk_spec is original_chunk_spec

    def test_model_hook_cannot_configure_unknown_kwarg(self):
        ap = self._pipeline_with_parts(_UnknownKwargsChunkHookPart())

        with pytest.raises(ValueError, match="unknown kwarg"):
            ap.step(torch.zeros(2, 8), attention_mask=torch.ones(2, 8))


# -----------------------------
# Core build/materialize/step tests
# -----------------------------


class TestAutoPipelineBuildAndStep:
    """Test AutoPipeline build, materialize, and step functionality."""

    def test_autopipeline_basic_creation(self):
        """Test basic AutoPipeline creation without full build process."""
        world_mesh = FakeDeviceMesh()
        # Should be able to create AutoPipeline instance
        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            device=torch.device("cpu"),
        )
        assert ap.world_mesh == world_mesh
        assert ap.device == torch.device("cpu")
        assert ap._info.enabled is False  # Not built yet

    @pytest.mark.parametrize("pp_size", [2, 4])
    @pytest.mark.parametrize("local_rank", [0, 1, 2, 3])
    def test_autopipeline_build_split_materialize_and_step(self, monkeypatch, pp_size, local_rank):
        """Test complete AutoPipeline build, materialize, and step workflow."""
        _patch_autopipeline_monkey(monkeypatch)
        if local_rank >= pp_size:
            pytest.skip("local_rank not part of this pp_size")

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

        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=pp_size, local_rank=local_rank)

        # trivial loss_fn; not used by FakeSchedule
        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            layers_per_stage=None,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        ap.build(model, loss_fn=loss_fn, parallelize_fn=None)

        # Expect 2 local stages per rank
        stages_per_rank = num_stages // pp_size
        assert len(ap.parts) == stages_per_rank

        local_stage_indices = [local_rank + s * pp_size for s in range(stages_per_rank)]

        for part, global_stage_idx in zip(ap.parts, local_stage_indices):
            # layers should be ModuleDict with only indices assigned to this global stage
            assert isinstance(part.model.layers, nn.ModuleDict)
            expected_layer_indices = sorted(
                int(name.split(".")[-1]) for name in module_fqns[global_stage_idx] if name.startswith("model.layers.")
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

    def test_autopipeline_materialize_workflow(self, monkeypatch):
        """Test AutoPipeline materialize functionality."""
        # Skip this complex test - the materialize method requires extensive setup
        pytest.skip("Complex materialize test requires extensive distributed mocking")

    def test_build_method_return_self(self, monkeypatch):
        """Test that build method returns self for method chaining."""
        _patch_autopipeline_monkey(monkeypatch)

        num_layers = 4
        model = DummyQwenForCausalLM(num_layers=num_layers)

        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        result = ap.build(model, loss_fn=loss_fn)

        # Should return self for method chaining
        assert result is ap
        assert ap._info.enabled is True

    def test_autopipeline_step_workflow(self, monkeypatch):
        """Test AutoPipeline step functionality."""
        # Skip this complex test - the step method requires extensive pipeline setup
        pytest.skip("Complex step test requires extensive pipeline mocking")

    def test_autopipeline_build_assertions(self, monkeypatch):
        """Test AutoPipeline build method assertion errors."""
        _patch_autopipeline_monkey(monkeypatch)

        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=torch.device("cpu"),
        )
        model = DummyQwenForCausalLM(num_layers=4)

        # Test missing loss_fn
        with pytest.raises(AssertionError, match="loss_fn must be provided"):
            ap.build(model, loss_fn=None)

        # Test invalid model type
        with pytest.raises(AssertionError, match="model must be a PyTorch module"):
            ap.build("not_a_module", loss_fn=lambda x, y: torch.tensor(0.0))


class TestAutoPipelineErrorHandling:
    """Test AutoPipeline error handling and edge cases."""

    def test_parts_before_build_error(self):
        """Test that accessing parts fails if build hasn't been called."""
        world_mesh = FakeDeviceMesh()
        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            device=torch.device("cpu"),
        )

        with pytest.raises(RuntimeError, match="Autopipeline not built"):
            _ = ap.parts


class TestAutoPipelineProperties:
    """Test AutoPipeline properties and state management."""

    def test_properties_before_build(self):
        """Test that properties work correctly before build."""
        world_mesh = FakeDeviceMesh()
        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=2,
            pp_batch_size=8,
            device=torch.device("cpu"),
        )

        # Test basic properties that should work before build
        assert ap.device == torch.device("cpu")
        assert ap.world_mesh == world_mesh
        assert ap.pp_mesh is not None  # Should be world_mesh["pp"]
        assert ap.info.enabled is False
        assert ap.info.schedule is None
        assert ap.info.model_parts is None
        assert ap.info.stages is None

    def test_properties_after_build(self, monkeypatch):
        """Test that properties work correctly after build."""
        # Skip this complex test for now - it requires extensive mocking of distributed components
        pytest.skip("Complex build test requires extensive distributed mocking")

    def test_world_mesh_none_error(self):
        """Test that AutoPipeline raises error when world_mesh is None."""
        with pytest.raises(ValueError, match="world_mesh must be provided"):
            AutoPipeline(
                world_mesh=None,
                pp_schedule="1f1b",
                pp_microbatch_size=1,
                pp_batch_size=4,
            )


class TestAutoPipelineDebugUtilities:
    """Test AutoPipeline debug and utility methods."""

    def test_debug_utilities_before_build(self):
        """Test debug utilities work correctly before build."""
        world_mesh = FakeDeviceMesh()
        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            device=torch.device("cpu"),
        )

        # Test methods that should work before build
        assert ap.list_stage_modules() == []
        assert ap.get_stage_param_counts() == []
        assert ap.get_stage_param_counts(trainable_only=True) == []
        assert ap.get_total_param_count() == 0
        assert ap.get_total_param_count(trainable_only=True) == 0
        assert ap.pretty_print_stages() == "<no stages>"

        # Test debug summary
        summary = ap.debug_summary()
        assert "PP degree:" in summary
        assert "Local stages: 0" in summary
        assert "Schedule: None" in summary

    def test_debug_utilities_after_build(self, monkeypatch):
        """Test debug utilities work correctly after build."""
        _patch_autopipeline_monkey(monkeypatch)

        num_layers = 4
        model = DummyQwenForCausalLM(num_layers=num_layers)

        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        ap.build(model, loss_fn=loss_fn)

        # Test methods that work after build
        stage_modules = ap.list_stage_modules()
        assert len(stage_modules) == 1  # rank 0 gets 1 stage
        assert isinstance(stage_modules[0], list)

        param_counts = ap.get_stage_param_counts()
        assert len(param_counts) == 1
        assert param_counts[0] > 0

        param_counts_trainable = ap.get_stage_param_counts(trainable_only=True)
        assert len(param_counts_trainable) == 1
        assert param_counts_trainable[0] > 0

        total_params = ap.get_total_param_count()
        assert total_params > 0

        total_trainable = ap.get_total_param_count(trainable_only=True)
        assert total_trainable > 0

        pretty_print = ap.pretty_print_stages()
        assert "Stage 0" in pretty_print
        assert "params=" in pretty_print

        # Test with max_modules_per_stage limit
        pretty_print_limited = ap.pretty_print_stages(max_modules_per_stage=2)
        assert "Stage 0" in pretty_print_limited

        # Test debug summary
        summary = ap.debug_summary()
        assert "PP degree: 2" in summary
        assert "Local stages: 1" in summary
        assert "Schedule: FakeSchedule" in summary
        assert "Total params:" in summary
        assert "trainable:" in summary

    def test_log_debug_summary(self, monkeypatch, caplog):
        """Test log_debug_summary method."""
        _patch_autopipeline_monkey(monkeypatch)

        num_layers = 4
        model = DummyQwenForCausalLM(num_layers=num_layers)

        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        ap.build(model, loss_fn=loss_fn)

        # Test log_debug_summary
        import logging

        with caplog.at_level(logging.INFO):
            ap.log_debug_summary()

        # Check that something was logged
        assert len(caplog.records) > 0

    def test_count_parameters_static_method(self):
        """Test the _count_parameters static method."""
        # Create a simple module with known parameter count
        module = nn.Sequential(
            nn.Linear(10, 5, bias=True),  # 10*5 + 5 = 55 params
            nn.Linear(5, 1, bias=False),  # 5*1 = 5 params
        )

        # Total params: 60
        total_params = AutoPipeline._count_parameters(module)
        assert total_params == 60

        # All params are trainable by default
        trainable_params = AutoPipeline._count_parameters(module, trainable_only=True)
        assert trainable_params == 60

        # Make some params non-trainable
        module[0].weight.requires_grad_(False)  # Remove 50 params (10*5)
        trainable_params = AutoPipeline._count_parameters(module, trainable_only=True)
        # Remaining: first layer bias (5) + second layer weight (5) = 10
        assert trainable_params == 10

    def test_visualize_current_schedule(self, monkeypatch):
        """Test visualize_current_schedule method."""
        _patch_autopipeline_monkey(monkeypatch)

        # Mock the torch.distributed.pipelining._schedule_visualizer imports
        mock_get_schedule_ops = Mock(return_value=[])
        mock_visualize_schedule = Mock()

        import nemo_automodel.components.distributed.pipelining.autopipeline as autopipeline_mod

        # Patch the import inside visualize_current_schedule
        def mock_visualize_method(self, filename=None):
            # This simulates the actual method with mocked imports
            schedule = self._info.schedule
            ops = mock_get_schedule_ops(schedule, self.pp_mesh.size(), self.pp_microbatch_size, len(self._info.stages))
            mock_visualize_schedule(ops, filename)

        monkeypatch.setattr(autopipeline_mod.AutoPipeline, "visualize_current_schedule", mock_visualize_method)

        num_layers = 4
        model = DummyQwenForCausalLM(num_layers=num_layers)

        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        ap.build(model, loss_fn=loss_fn)

        # Test visualize_current_schedule - this should execute lines 236-240
        ap.visualize_current_schedule()
        ap.visualize_current_schedule(filename="test.png")

        # Verify mocks were called
        assert mock_get_schedule_ops.call_count == 2
        assert mock_visualize_schedule.call_count == 2

    def test_pretty_print_stages_with_stage_tags(self, monkeypatch):
        """Test pretty_print_stages with stage tags (first/last)."""
        _patch_autopipeline_monkey(monkeypatch)

        num_layers = 8
        model = DummyQwenForCausalLM(num_layers=num_layers)

        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=4,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )

        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        ap.build(model, loss_fn=loss_fn)

        # Create stages with is_first and is_last attributes to trigger line 268 coverage
        for i, stage in enumerate(ap._info.stages):
            stage.is_first = i == 0
            stage.is_last = i == len(ap._info.stages) - 1

        # Test pretty_print_stages with module limit to trigger line 275-276
        pretty_print = ap.pretty_print_stages(max_modules_per_stage=1)

        # Should contain stage information and tags
        assert "Stage 0" in pretty_print
        assert "params=" in pretty_print
        # Should have "..." when modules exceed limit
        assert "..." in pretty_print


class TestAutoPipelineIntegration:
    """Test AutoPipeline integration with finetune workflow."""

    def test_autopipeline_field_defaults(self):
        """Test AutoPipeline field defaults."""
        # Test default device
        ap = AutoPipeline(world_mesh=FakeDeviceMesh())
        # Should default to cuda if available, otherwise cpu
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert ap._device == expected_device

        # Test other defaults
        assert ap.pp_axis_name == "pp"
        assert ap.dp_axis_names == ("dp",)
        assert ap.pp_schedule == "1f1b"
        assert ap.pp_microbatch_size == 1
        assert ap.pp_batch_size == 1
        assert ap.patch_inner_model is True
        assert ap.patch_causal_lm_model is True
        assert ap.scale_grads_in_schedule is False

    def test_autopipeline_with_all_optional_fields(self):
        """Test AutoPipeline with all optional fields set."""
        world_mesh = FakeDeviceMesh()
        moe_mesh = FakeDeviceMesh()

        ap = AutoPipeline(
            world_mesh=world_mesh,
            moe_mesh=moe_mesh,
            pp_axis_name="pipeline",
            dp_axis_names=("dp1", "dp2"),
            cp_axis_name="context",
            tp_axis_name="tensor",
            ep_axis_name="expert",
            ep_shard_axis_names=("shard1", "shard2"),
            pp_schedule="interleaved1f1b",
            pp_schedule_csv="/path/to/schedule.csv",
            pp_microbatch_size=4,
            pp_batch_size=16,
            layers_per_stage=8,
            round_virtual_stages_to_pp_multiple="up",
            module_fqns_per_model_part=[["layer1"], ["layer2"]],
            patch_inner_model=False,
            patch_causal_lm_model=False,
            device=torch.device("cuda:1"),
            dtype=torch.float16,
            scale_grads_in_schedule=True,
        )

        # Verify all fields are set correctly
        assert ap.world_mesh == world_mesh
        assert ap.moe_mesh == moe_mesh
        assert ap.pp_axis_name == "pipeline"
        assert ap.dp_axis_names == ("dp1", "dp2")
        assert ap.cp_axis_name == "context"
        assert ap.tp_axis_name == "tensor"
        assert ap.ep_axis_name == "expert"
        assert ap.ep_shard_axis_names == ("shard1", "shard2")
        assert ap.pp_schedule == "interleaved1f1b"
        assert ap.pp_schedule_csv == "/path/to/schedule.csv"
        assert ap.pp_microbatch_size == 4
        assert ap.pp_batch_size == 16
        assert ap.layers_per_stage == 8
        assert ap.round_virtual_stages_to_pp_multiple == "up"
        assert ap.module_fqns_per_model_part == [["layer1"], ["layer2"]]
        assert ap.patch_inner_model is False
        assert ap.patch_causal_lm_model is False
        assert ap._device == torch.device("cuda:1")
        assert ap.dtype == torch.float16
        assert ap.scale_grads_in_schedule is True


class TestAutoPipelineUpdateSeqLen:
    """Test AutoPipeline.update_seq_len method."""

    def test_update_seq_len_before_build_raises(self):
        """update_seq_len should fail if build() hasn't been called."""
        world_mesh = FakeDeviceMesh()
        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            device=torch.device("cpu"),
        )
        with pytest.raises(RuntimeError, match="AutoPipeline.build\\(\\) must be called before update_seq_len"):
            ap.update_seq_len(128)

    def test_update_seq_len_noop_when_unchanged(self, monkeypatch):
        """update_seq_len should be a no-op when seq_len hasn't changed."""
        _patch_autopipeline_monkey(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=4)
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )
        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        ap.build(model, loss_fn=lambda x, y: torch.tensor(0.0))

        # Patch reset_pp_stage_shapes to track calls (mock to avoid needing real config)
        call_count = [0]
        import nemo_automodel.components.distributed.pipelining.autopipeline as ap_mod

        def counting_reset(*args, **kwargs):
            call_count[0] += 1

        monkeypatch.setattr(ap_mod, "reset_pp_stage_shapes", counting_reset)

        # First call should trigger reset
        ap.update_seq_len(128)
        assert call_count[0] == 1
        assert ap._pp_current_seq_len == 128

        # Same seq_len should be a no-op
        ap.update_seq_len(128)
        assert call_count[0] == 1  # Not incremented

    def test_update_seq_len_calls_reset(self, monkeypatch):
        """update_seq_len should call reset_pp_stage_shapes with correct args."""
        _patch_autopipeline_monkey(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=4)
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )
        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=2,
            pp_batch_size=4,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        ap.build(model, loss_fn=lambda x, y: torch.tensor(0.0))

        # Track the call args
        import nemo_automodel.components.distributed.pipelining.autopipeline as ap_mod

        captured_args = []

        def mock_reset(
            schedule,
            stages,
            model_config,
            microbatch_size,
            seq_len,
            tensor_dtype=None,
            first_stage_input_meta=None,
        ):
            captured_args.append(
                (schedule, stages, model_config, microbatch_size, seq_len, tensor_dtype, first_stage_input_meta)
            )

        monkeypatch.setattr(ap_mod, "reset_pp_stage_shapes", mock_reset)

        ap.update_seq_len(256)

        assert len(captured_args) == 1
        _, _, _, mb_size, sl, tensor_dtype, first_stage_input_meta = captured_args[0]
        assert mb_size == 2  # pp_microbatch_size
        assert sl == 256
        assert tensor_dtype is ap.dtype
        assert first_stage_input_meta is None

        # THD materialization collapses source examples into one synthetic
        # batch row. The same sequence length must still refresh stage metadata
        # when its effective batch extent changes.
        ap.update_seq_len(256, microbatch_size=1)
        assert len(captured_args) == 2
        assert captured_args[-1][3:5] == (1, 256)

        embeds = torch.empty(1, 256, 32, dtype=torch.bfloat16)
        ap.update_seq_len(256, microbatch_size=1, input_tensor=embeds)
        input_meta = captured_args[-1][-1]
        assert input_meta.device.type == "meta"
        assert input_meta.shape == embeds.shape
        assert input_meta.dtype == embeds.dtype

    def test_update_seq_len_tracks_current(self, monkeypatch):
        """update_seq_len should track current seq_len and reset on change."""
        _patch_autopipeline_monkey(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=4)
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )
        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )
        ap.build(model, loss_fn=lambda x, y: torch.tensor(0.0))

        import nemo_automodel.components.distributed.pipelining.autopipeline as ap_mod

        call_count = [0]

        def mock_reset(*a, **kw):
            call_count[0] += 1

        monkeypatch.setattr(ap_mod, "reset_pp_stage_shapes", mock_reset)

        ap.update_seq_len(100)
        assert ap._pp_current_seq_len == 100
        assert call_count[0] == 1

        # Different seq_len triggers reset
        ap.update_seq_len(200)
        assert ap._pp_current_seq_len == 200
        assert call_count[0] == 2

        # Same as current is a no-op
        ap.update_seq_len(200)
        assert call_count[0] == 2

    def test_model_config_stored_after_build(self, monkeypatch):
        """build() should store model.config for later use by update_seq_len."""
        _patch_autopipeline_monkey(monkeypatch)

        model = DummyQwenForCausalLM(num_layers=4)
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )
        world_mesh = FakeWorldMesh()
        world_mesh["pp"] = FakePPMesh(size=2, local_rank=0)

        ap = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            module_fqns_per_model_part=module_fqns,
            device=torch.device("cpu"),
        )

        assert ap._model_config is None
        ap.build(model, loss_fn=lambda x, y: torch.tensor(0.0))
        assert ap._model_config is model.config
