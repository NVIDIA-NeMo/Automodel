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
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import PipelineScheduleMulti, PipelineScheduleSingle
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.pipelining.model_parts import (
    PipelineModelPart,
    _wrap_stage_forward_to_emit_tensor,
    split_model_into_parts,
)
from nemo_automodel.components.distributed.pipelining.module_plan import (
    calculate_virtual_stages,
    generate_hf_model_fqn_per_model_part,
    stage_ids_this_rank,
)
from nemo_automodel.components.distributed.pipelining.schedules import build_pipeline_schedule
from nemo_automodel.components.distributed.pipelining.stage_runtime import (
    _get_hidden_and_vocab_size,
    _get_stage_metas,
    configure_pipeline_stage_backward,
    create_pipeline_stages,
    warmup_pipeline_stage_neighbors,
)
from nemo_automodel.shared.pipeline import PipelineModelMixin


@pytest.mark.parametrize(
    ("rank", "size", "num_stages", "style", "expected"),
    [
        (0, 4, 4, "loop", (0,)),
        (2, 4, 4, "loop", (2,)),
        (1, 4, 8, "loop", (1, 5)),
        (0, 4, 8, "v", (0, 7)),
        (3, 4, 8, "v", (3, 4)),
    ],
)
def test_stage_ids_this_rank(rank, size, num_stages, style, expected):
    assert stage_ids_this_rank(rank, size, num_stages, style) == expected


def test_stage_ids_reject_invalid_distributions():
    with pytest.raises(ValueError, match="evenly divisible"):
        stage_ids_this_rank(0, 4, 5)
    with pytest.raises(ValueError, match="2 stages per rank"):
        stage_ids_this_rank(0, 4, 12, "v")


def test_generate_standard_hf_stage_modules():
    stages = generate_hf_model_fqn_per_model_part(num_stages=3, num_layers=8)

    assert stages[0][0] == "model.embed_tokens"
    assert {"model.layers.0", "model.layers.1", "model.layers.2"} <= set(stages[0])
    assert "model.layers.3" in stages[1]
    assert stages[-1][-3:] == ["model.norm", "lm_head", "model.rotary_emb"]
    assert sum(name.startswith("model.layers.") for stage in stages for name in stage) == 8


def test_generate_hf_stage_modules_respects_layout_options():
    stages = generate_hf_model_fqn_per_model_part(
        num_stages=2,
        num_layers=4,
        include_embeddings=False,
        include_lm_head=False,
        include_rotary_emb=False,
        include_multimodal_encoders=False,
        extra_module_fqns=["vision_tower"],
        fqn_prefix="language_model.",
        lm_head_fqn="language_model.lm_head",
    )

    assert stages[0] == ["vision_tower", "language_model.layers.0", "language_model.layers.1"]
    assert stages[1] == ["language_model.layers.2", "language_model.layers.3", "language_model.norm"]


@pytest.mark.parametrize("num_stages", [0, 5])
def test_generate_hf_stage_modules_rejects_invalid_stage_count(num_stages):
    with pytest.raises(ValueError):
        generate_hf_model_fqn_per_model_part(num_stages=num_stages, num_layers=4)


@pytest.mark.parametrize(
    ("layers", "layers_per_stage", "pp_size", "single", "rounding", "expected"),
    [
        (32, 8, 4, True, None, (4, 1)),
        (32, 4, 4, False, None, (8, 2)),
        (30, 4, 4, False, "up", (8, 2)),
        (34, 4, 4, False, "down", (8, 2)),
        (32, None, 4, True, None, (4, 1)),
        (32, None, 4, False, None, (8, 2)),
    ],
)
def test_calculate_virtual_stages(layers, layers_per_stage, pp_size, single, rounding, expected):
    assert calculate_virtual_stages(layers, layers_per_stage, pp_size, single, rounding) == expected


def test_calculate_virtual_stages_rejects_invalid_topology():
    with pytest.raises(ValueError, match="divisible"):
        calculate_virtual_stages(30, 4, 3, False)
    with pytest.raises(ValueError, match="Single stage schedule"):
        calculate_virtual_stages(32, 4, 4, True)
    with pytest.raises(ValueError, match="Multi-stage schedule"):
        calculate_virtual_stages(32, 8, 4, False)
    with pytest.raises(ValueError, match="Invalid value"):
        calculate_virtual_stages(33, 4, 4, False, "nearest")


class _SingleSchedule(PipelineScheduleSingle):
    def __init__(self, stage, n_microbatches, loss_fn, scale_grads):
        self.stage = stage
        self.n_microbatches = n_microbatches
        self.loss_fn = loss_fn
        self.scale_grads = scale_grads

    def _step_microbatches(self, arg_mbs=None, kwarg_mbs=None, target_mbs=None, losses=None):
        del arg_mbs, kwarg_mbs, target_mbs, losses


class _MultiSchedule(PipelineScheduleMulti):
    def __init__(self, stages, n_microbatches, loss_fn, scale_grads):
        self.stages = stages
        self.n_microbatches = n_microbatches
        self.loss_fn = loss_fn
        self.scale_grads = scale_grads

    def _step_microbatches(self, arg_mbs=None, kwarg_mbs=None, target_mbs=None, losses=None):
        del arg_mbs, kwarg_mbs, target_mbs, losses


@pytest.mark.parametrize(("schedule_cls", "stage_count"), [(_SingleSchedule, 1), (_MultiSchedule, 2)])
def test_build_pipeline_schedule(schedule_cls, stage_count):
    stages = [Mock() for _ in range(stage_count)]
    loss_fn = Mock()
    with patch(
        "nemo_automodel.components.distributed.pipelining.schedules.get_schedule_class",
        return_value=schedule_cls,
    ):
        schedule = build_pipeline_schedule(None, "test", 2, 8, stages, loss_fn, scale_grads=True)

    assert schedule.n_microbatches == 4
    assert schedule.loss_fn is loss_fn
    assert schedule.scale_grads is True
    if stage_count == 1:
        assert schedule.stage is stages[0]
    else:
        assert schedule.stages is stages


def test_build_pipeline_schedule_validates_batch_and_csv(tmp_path):
    with pytest.raises(ValueError, match="must be divisible"):
        build_pipeline_schedule(None, "1f1b", 3, 8, [Mock()], Mock())
    with pytest.raises(FileNotFoundError):
        build_pipeline_schedule(str(tmp_path / "missing.csv"), None, 2, 8, [Mock()], Mock())


class _MetaModule(nn.Module):
    def __init__(self, *, lm_head: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, dtype=dtype))
        self.lm_head = nn.Identity() if lm_head else None


@pytest.mark.parametrize(
    ("index", "lm_head", "input_shape", "output_shape"),
    [
        (0, False, (2, 16), (2, 16, 64)),
        (1, False, (2, 16, 64), (2, 16, 64)),
        (2, True, (2, 16, 64), (2, 16, 128)),
    ],
)
def test_default_stage_metadata(index, lm_head, input_shape, output_shape):
    part = PipelineModelPart(_MetaModule(lm_head=lm_head), index, 3)
    config = types.SimpleNamespace(hidden_size=64, vocab_size=128)

    inputs, outputs = _get_stage_metas(part, config, microbatch_size=2, seq_len=16)

    assert inputs[0].shape == input_shape
    assert outputs[0].shape == output_shape
    assert inputs[0].device.type == outputs[0].device.type == "meta"


def test_default_stage_metadata_uses_nested_config_dtype_and_hidden_output_flag():
    module = _MetaModule(lm_head=True, dtype=torch.float16)
    module._pp_return_hidden_states = True
    part = PipelineModelPart(module, 1, 2)
    config = types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=32, vocab_size=96))

    inputs, outputs = _get_stage_metas(part, config, microbatch_size=3, seq_len=7)

    assert inputs[0].shape == outputs[0].shape == (3, 7, 32)
    assert inputs[0].dtype == outputs[0].dtype == torch.float16


def test_model_owned_stage_metadata_is_used():
    class CustomModule(PipelineModelMixin, _MetaModule):
        def __init__(self):
            super().__init__(dtype=torch.float16)
            self.calls = []

        def pipeline_stage_metas(self, *, is_first, microbatch_size, seq_len, dtype):
            self.calls.append((is_first, microbatch_size, seq_len, dtype))
            hidden = torch.empty(microbatch_size, seq_len, 3, device="meta", dtype=dtype)
            carry = torch.empty(microbatch_size, seq_len, 5, device="meta", dtype=torch.float32)
            return (hidden, carry), (hidden, carry)

    module = CustomModule()
    part = PipelineModelPart(module, 1, 2)
    inputs, outputs = _get_stage_metas(
        part,
        types.SimpleNamespace(),
        microbatch_size=2,
        seq_len=11,
    )

    assert module.calls == [(False, 2, 11, torch.float16)]
    assert [tensor.shape for tensor in inputs] == [(2, 11, 3), (2, 11, 5)]
    assert [tensor.dtype for tensor in outputs] == [torch.float16, torch.float32]


class _FakeMesh:
    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank
        self.group = object()

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    def get_group(self, _axis=None):
        return self.group


class _RecordingStage:
    def __init__(
        self,
        submod,
        stage_index,
        num_stages,
        device,
        *,
        group,
        input_args=None,
        output_args=None,
    ):
        self.submod = submod
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group
        self.inputs_meta = input_args
        self.outputs_meta = output_args
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1


def test_create_pipeline_stages_supplies_constructor_metadata():
    parts = [PipelineModelPart(_MetaModule(), 0, 2), PipelineModelPart(_MetaModule(lm_head=True), 1, 2)]
    mesh = _FakeMesh()
    config = types.SimpleNamespace(hidden_size=64, vocab_size=128)
    with patch("nemo_automodel.components.distributed.pipelining.stage_runtime.PipelineStage", _RecordingStage):
        stages = create_pipeline_stages(
            parts,
            mesh,
            "pp",
            torch.device("cpu"),
            model_config=config,
            microbatch_size=2,
            seq_len=16,
        )

    assert stages[0].submod is parts[0].module
    assert stages[0].inputs_meta[0].shape == (2, 16)
    assert stages[1].outputs_meta[0].shape == (2, 16, 128)
    assert all(stage.group is mesh.group for stage in stages)


def test_create_pipeline_stages_allows_dynamic_inference():
    part = PipelineModelPart(_MetaModule(), 0, 1)
    with patch("nemo_automodel.components.distributed.pipelining.stage_runtime.PipelineStage", _RecordingStage):
        stage = create_pipeline_stages(
            [part],
            _FakeMesh(size=1),
            "pp",
            torch.device("cpu"),
            model_config=types.SimpleNamespace(hidden_size=64, vocab_size=128),
            microbatch_size=1,
            seq_len=None,
        )[0]

    assert stage.inputs_meta is None
    assert stage.outputs_meta is None


def test_create_pipeline_stages_reports_unsupported_public_metadata_api():
    class IncompatibleStage:
        def __init__(self, submod, stage_index, num_stages, device, *, group):
            del submod, stage_index, num_stages, device, group

    with (
        patch("nemo_automodel.components.distributed.pipelining.stage_runtime.PipelineStage", IncompatibleStage),
        pytest.raises(RuntimeError, match="accepts input_args and output_args"),
    ):
        create_pipeline_stages(
            [PipelineModelPart(_MetaModule(), 0, 1)],
            _FakeMesh(size=1),
            "pp",
            torch.device("cpu"),
            model_config=types.SimpleNamespace(hidden_size=64, vocab_size=128),
            microbatch_size=1,
            seq_len=8,
        )


def test_warmup_pipeline_stage_neighbors_uses_symmetric_edges():
    stage = Mock(device=torch.device("cpu"), group_size=3)
    stage.group.rank.return_value = 1
    with (
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1, 2]),
        patch("torch.distributed.isend") as isend,
        patch("torch.distributed.irecv") as irecv,
        patch("torch.cuda.synchronize") as synchronize,
        patch("torch.cuda.device_count", return_value=8),
        patch("nemo_automodel.components.distributed.pipelining.stage_runtime.time.sleep"),
    ):
        warmup_pipeline_stage_neighbors(stage)

    assert [call.kwargs["group_dst"] for call in isend.call_args_list] == [0, 2]
    assert [call.kwargs["group_src"] for call in irecv.call_args_list] == [0, 2]
    synchronize.assert_called_once_with(stage.device)


def test_configure_pipeline_stage_backward_is_noop_by_default():
    stage = Mock()
    configure_pipeline_stage_backward(
        [stage],
        patch_stage_backward_maybe_with_nosync=False,
        reduce_grad_per_microbatch=False,
    )
    assert "backward_maybe_with_nosync" not in stage.__dict__


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (types.SimpleNamespace(hidden_size=64, vocab_size=128), (64, 128)),
        (types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=32, vocab_size=96)), (32, 96)),
        (
            types.SimpleNamespace(
                hidden_size=64,
                vocab_size=None,
                text_config=types.SimpleNamespace(hidden_size=32, vocab_size=96),
            ),
            (64, 96),
        ),
    ],
)
def test_get_hidden_and_vocab_size(config, expected):
    assert _get_hidden_and_vocab_size(config) == expected


def test_get_hidden_and_vocab_size_reports_missing_fields():
    with pytest.raises(ValueError, match="hidden_size"):
        _get_hidden_and_vocab_size(types.SimpleNamespace())
    with pytest.raises(ValueError, match="vocab_size"):
        _get_hidden_and_vocab_size(types.SimpleNamespace(hidden_size=64))


def test_split_model_parts_preserves_model_owned_stage_customization():
    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 8, device="meta")
            self.layers = nn.ModuleList([nn.Linear(8, 8, device="meta") for _ in range(4)])
            self.norm = nn.LayerNorm(8, device="meta")
            self.rotary_emb = nn.Identity()
            self.custom_shared = nn.Identity()

    class Wrapper(PipelineModelMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TextModel()
            self.lm_head = nn.Linear(8, 16, device="meta")
            self.config = types.SimpleNamespace(hidden_size=8, vocab_size=16)

        def pipeline_stage_modules(self, module_names_per_stage, *, layers_prefix, text_model):
            assert text_model is self.model
            return [list(names) + [f"{layers_prefix}custom_shared"] for names in module_names_per_stage]

    with patch(
        "nemo_automodel.components.distributed.pipelining.model_parts.patch_hf_model_for_pp",
        lambda *args, **kwargs: None,
    ):
        parts = split_model_into_parts(
            Wrapper(),
            _FakeMesh(size=2, rank=0),
            "PipelineScheduleSingle",
            layers_per_stage=2,
        )

    assert len(parts) == 1
    assert parts[0].stage_index == 0
    assert parts[0].module.model.custom_shared is not None
    assert list(parts[0].module.model.layers) == ["0", "1"]


class _OutputModule(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, hidden_states: torch.Tensor):
        """Return the configured output.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            The configured output, whose tensor leaves preserve the input layout.
        """
        if callable(self.output):
            return self.output(hidden_states)
        return self.output


def test_stage_forward_unwraps_model_output_and_preserves_gradients():
    module = _OutputModule(lambda hidden: CausalLMOutputWithPast(logits=hidden.square()))
    _wrap_stage_forward_to_emit_tensor(module)
    hidden = torch.randn(2, 3, 4, requires_grad=True)

    output = module(hidden)
    output.sum().backward()

    assert torch.equal(output, hidden.detach().square())
    assert torch.equal(hidden.grad, 2 * hidden.detach())


@pytest.mark.parametrize("output", [torch.ones(2, 3), (torch.ones(2, 3), torch.zeros(2, 3))])
def test_stage_forward_preserves_tensor_outputs(output):
    module = _OutputModule(output)
    original_signature = str(__import__("inspect").signature(module.forward))
    _wrap_stage_forward_to_emit_tensor(module)
    _wrap_stage_forward_to_emit_tensor(module)

    result = module(torch.zeros(2, 3, 4))

    if isinstance(output, tuple):
        assert all(torch.equal(actual, expected) for actual, expected in zip(result, output))
    else:
        assert torch.equal(result, output)
    assert str(__import__("inspect").signature(module.forward)) == original_signature
