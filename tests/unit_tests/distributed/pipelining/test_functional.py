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
from nemo_automodel.components.distributed.pipelining.schedules import (
    build_pipeline_schedule,
    resolve_pipeline_schedule,
)
from nemo_automodel.components.distributed.pipelining.stage_runtime import (
    build_pipeline_runtime,
    configure_pipeline_stage_backward,
    create_pipeline_stages,
)
from nemo_automodel.shared.pipeline import PP_MEDIA_INDEX_KEY, PipelineModelMixin, pp_media_chunk


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


@pytest.mark.parametrize(
    ("schedule_name", "schedule_type", "stage_count"),
    [("1f1b", "Schedule1F1B", 1), ("interleaved1f1b", "ScheduleInterleaved1F1B", 2)],
)
def test_build_pipeline_schedule(schedule_name, schedule_type, stage_count):
    """A schedule name resolves to the matching PyTorch schedule around the local stages."""
    stages = [
        Mock(stage_index=index, num_stages=stage_count, group_size=1, submod=Mock()) for index in range(stage_count)
    ]
    loss_fn = Mock()

    schedule = build_pipeline_schedule(None, schedule_name, 2, 8, stages, loss_fn, scale_grads=True)

    assert type(schedule).__name__ == schedule_type
    assert schedule._n_microbatches == 4
    assert schedule._loss_fn is loss_fn
    assert schedule.scale_grads is True
    if stage_count == 1:
        assert schedule._stage is stages[0]
    else:
        assert schedule._stages is stages


def test_resolve_pipeline_schedule_maps_names_to_stage_styles():
    """Supported schedules resolve to the stage-assignment style they are split with."""
    assert resolve_pipeline_schedule("1f1b")[1] == "loop"
    assert resolve_pipeline_schedule("ZBVZeroBubble")[1] == "v"


def test_resolve_pipeline_schedule_rejects_unmapped_and_unknown_names():
    """Names PyTorch does not know, and ones this package cannot split, are rejected."""
    with pytest.raises(ValueError, match="Unknown pipeline schedule"):
        resolve_pipeline_schedule("not_a_schedule")
    with pytest.raises(ValueError, match="must be a schedule name"):
        resolve_pipeline_schedule(None)


def test_build_pipeline_schedule_validates_batch_and_csv(tmp_path):
    with pytest.raises(ValueError, match="must be divisible"):
        build_pipeline_schedule(None, "1f1b", 3, 8, [Mock()], Mock())
    with pytest.raises(FileNotFoundError):
        build_pipeline_schedule(str(tmp_path / "missing.csv"), None, 2, 8, [Mock()], Mock())


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


def test_configure_pipeline_stage_backward_is_noop_by_default():
    stage = Mock()
    configure_pipeline_stage_backward(
        [stage],
        patch_stage_backward_maybe_with_nosync=False,
        reduce_grad_per_microbatch=False,
    )
    assert "backward_maybe_with_nosync" not in stage.__dict__


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


# ---------------------------------------------------------------------------
# Real single-rank pipeline runtime
#
# These tests construct a real ``torch.distributed.pipelining.PipelineStage`` and
# drive a real schedule over a gloo group of size one, so they fail when stage
# construction, runtime shape inference, or kwarg microbatching regresses.
# ---------------------------------------------------------------------------

requires_gloo = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="requires torch.distributed with the gloo backend",
)


@pytest.fixture
def single_rank_pp_mesh(tmp_path):
    """Yield a one-rank pipeline mesh backed by a real gloo process group.

    Yields:
        The ``pp`` submesh of a one-dimensional CPU device mesh.
    """
    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    from torch.distributed.device_mesh import init_device_mesh

    torch.distributed.init_process_group(
        "gloo",
        rank=0,
        world_size=1,
        init_method=f"file://{tmp_path / 'pp_store'}",
    )
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("pp",))["pp"]
    finally:
        torch.distributed.destroy_process_group()


class _TinyStageModule(nn.Module):
    """Embedding plus projection standing in for one pipeline stage."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Embed and project a microbatch of token ids.

        Args:
            input_ids: Token ids of shape [microbatch, sequence].
            **kwargs: Ignored schedule keyword arguments.

        Returns:
            Logits of shape [microbatch, sequence, vocab].
        """
        del kwargs
        return self.head(self.embed(input_ids))


def _token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean token cross-entropy.

    Args:
        logits: Tensor of shape [microbatch, sequence, vocab].
        labels: Token ids of shape [microbatch, sequence].

    Returns:
        Scalar loss tensor.
    """
    return nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1))


@requires_gloo
def test_create_pipeline_stages_defers_boundary_metadata_to_runtime(single_rank_pp_mesh):
    """Stages are constructed without hand-written boundary metadata."""
    parts = [PipelineModelPart(_TinyStageModule(), 0, 1)]

    stages = create_pipeline_stages(parts, single_rank_pp_mesh, "pp", torch.device("cpu"))

    assert len(stages) == 1
    assert stages[0].submod is parts[0].module
    assert stages[0].stage_index == 0
    assert stages[0].num_stages == 1
    # Metadata is measured by the first schedule step, not declared up front.
    assert stages[0].inputs_meta is None


@requires_gloo
def test_pipeline_runtime_infers_stage_metadata_from_the_real_input(single_rank_pp_mesh):
    """A real schedule step measures stage metadata and produces gradients."""
    part = PipelineModelPart(_TinyStageModule(), 0, 1)
    runtime = build_pipeline_runtime(
        [part],
        single_rank_pp_mesh,
        "pp",
        torch.device("cpu"),
        microbatch_size=1,
        local_batch_size=2,
        schedule_name="1f1b",
        schedule_csv=None,
        loss_fn=_token_loss,
        scale_grads=False,
        patch_stage_backward_maybe_with_nosync=False,
        reduce_grad_per_microbatch=False,
    )
    input_ids = torch.randint(0, 16, (2, 5))
    labels = torch.randint(0, 16, (2, 5))
    losses: list[torch.Tensor] = []

    output = runtime.schedule.step(input_ids, target=labels, losses=losses)

    assert output.shape == (2, 5, 16)
    assert len(losses) == 2
    # The measured metadata describes one microbatch of the real input.
    assert runtime.stages[0].inputs_meta[0].shape == (1, 5)
    assert runtime.stages[0].inputs_meta[0].dtype == torch.int64
    assert part.module.head.weight.grad is not None


class _MediaStageModule(nn.Module):
    """Stage-0 module selecting its media chunk by microbatch index."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)
        self.observed: list[tuple[int, float]] = []

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Consume the staged media chunk for this microbatch.

        Args:
            input_ids: Token ids of shape [microbatch, sequence].
            **kwargs: Schedule keyword arguments carrying ``pp_media_index``,
                an int64 tensor of shape [microbatch].

        Returns:
            Logits of shape [microbatch, sequence, vocab].
        """
        media_index = kwargs.get(PP_MEDIA_INDEX_KEY)
        pixel_values = pp_media_chunk(self, "pixel_values", media_index)
        if pixel_values is not None:
            self.observed.append((int(media_index.reshape(-1)[0]), float(pixel_values[0])))
        return self.head(self.embed(input_ids))


@requires_gloo
def test_media_chunks_are_selected_by_index_not_by_a_cursor(single_rank_pp_mesh):
    """Every forward reads the chunk its own microbatch index names.

    A cursor advanced inside ``forward`` mis-assigns chunks as soon as the
    schedule reorders microbatches or runs the shape-inference probe forward,
    which executes microbatch 0 an extra time.
    """
    module = _MediaStageModule()
    module._pp_media_chunks = {"pixel_values": [torch.tensor([10.0]), torch.tensor([20.0])]}
    part = PipelineModelPart(module, 0, 1)
    runtime = build_pipeline_runtime(
        [part],
        single_rank_pp_mesh,
        "pp",
        torch.device("cpu"),
        microbatch_size=1,
        local_batch_size=2,
        schedule_name="1f1b",
        schedule_csv=None,
        loss_fn=_token_loss,
        scale_grads=False,
        patch_stage_backward_maybe_with_nosync=False,
        reduce_grad_per_microbatch=False,
    )
    input_ids = torch.randint(0, 16, (2, 5))
    labels = torch.randint(0, 16, (2, 5))

    runtime.schedule.step(
        input_ids,
        target=labels,
        losses=[],
        **{PP_MEDIA_INDEX_KEY: torch.tensor([0, 1], dtype=torch.int64)},
    )

    assert module.observed, "the stage never received a media index"
    expected = {0: 10.0, 1: 20.0}
    for index, value in module.observed:
        assert value == expected[index]
    assert {index for index, _ in module.observed} == {0, 1}


def test_pp_media_chunk_returns_none_without_staged_media():
    """A module with no staged media yields None instead of raising."""
    module = nn.Linear(2, 2)

    assert pp_media_chunk(module, "pixel_values", torch.tensor([0])) is None

    module._pp_media_chunks = {"pixel_values": [torch.zeros(1)]}
    assert pp_media_chunk(module, "image_grid_hws", torch.tensor([0])) is None
    assert pp_media_chunk(module, "pixel_values", None) is None


def test_pp_media_chunk_rejects_an_index_the_staging_cannot_satisfy():
    """A microbatch index beyond the staged chunks is a staging bug, not a silent reuse."""
    module = nn.Linear(2, 2)
    module._pp_media_chunks = {"pixel_values": [torch.zeros(1)]}

    with pytest.raises(IndexError, match="out of range"):
        pp_media_chunk(module, "pixel_values", torch.tensor([1]))
