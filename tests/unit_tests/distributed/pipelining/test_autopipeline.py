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

"""Tests for :class:`AutoPipeline`.

The first half covers construction and model splitting, which need no process
group. The second half spawns two real gloo ranks and drives a real
``torch.distributed.pipelining`` schedule, so the runtime-identity contract is
exercised against PyTorch's own stage machinery instead of a stand-in schedule.
"""

import os
import socket
import sys
import traceback
import types

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.distributed.pipelining.module_plan import generate_hf_model_fqn_per_model_part
from nemo_automodel.shared.pipeline import PipelineForwardStyle, PipelineModelMixin


class DummyRotaryEmb(nn.Module):
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Return zero rotary embeddings.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            position_ids: Tensor of shape [batch, sequence].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        del position_ids
        return torch.zeros_like(hidden_states)


class DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # tiny param to ensure some grads exist when needed
        self.proj = nn.Linear(8, 8, bias=False, device="meta")

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        """Return the input unchanged.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            **kwargs: Ignored decoder-layer keyword arguments.

        Returns:
            A one-tuple holding a tensor of shape [batch, sequence, hidden].
        """
        del kwargs
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
        self.config = types.SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )


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
    """Device mesh stand-in that yields a pipeline submesh without distributed setup."""

    def __init__(self, device_type="cpu", mesh=None, mesh_dim_names=None, pp_size=2, local_rank=0):
        self.device_type = device_type
        self.mesh = mesh or [[0, 1]]
        self.mesh_dim_names = mesh_dim_names or ["pp"]
        self._pp_size = pp_size
        self._local_rank = local_rank

    def __getitem__(self, key):
        if key == "pp":
            return FakePPMesh(size=self._pp_size, local_rank=self._local_rank)
        return self

    def size(self):
        return self._pp_size

    def get_local_rank(self):
        return self._local_rank


class FakeWorldMesh(dict):
    """Mapping-based stand-in for a DeviceMesh keyed by axis name."""


def _make_pipeline(pp_size: int = 2, local_rank: int = 0, **kwargs) -> AutoPipeline:
    """Build an AutoPipeline over a fake mesh with the given pipeline topology."""
    world_mesh = FakeWorldMesh()
    world_mesh["pp"] = FakePPMesh(size=pp_size, local_rank=local_rank)
    options = {
        "pp_axis_name": "pp",
        "pp_schedule": "1f1b",
        "pp_microbatch_size": 1,
        "pp_batch_size": 2,
        "device": torch.device("cpu"),
    }
    options.update(kwargs)
    return AutoPipeline(world_mesh=world_mesh, **options)


def _loss_stub(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return a constant loss.

    Args:
        logits: Tensor of shape [batch, sequence, vocab].
        labels: Tensor of shape [batch, sequence].

    Returns:
        Scalar loss tensor.
    """
    del logits, labels
    return torch.tensor(0.0)


class TestAutoPipelineValidation:
    """Constructor validation and stored configuration."""

    def test_valid_autopipeline(self):
        world_mesh = FakeDeviceMesh()
        pipeline = AutoPipeline(
            world_mesh=world_mesh,
            pp_axis_name="pp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            device=torch.device("cpu"),
        )

        assert pipeline.world_mesh is world_mesh
        assert pipeline.pp_axis_name == "pp"
        assert pipeline.pp_schedule == "1f1b"
        assert pipeline.pp_microbatch_size == 1
        assert pipeline.pp_batch_size == 4
        assert pipeline.device == torch.device("cpu")

    def test_batch_size_must_be_divisible_by_microbatch_size(self):
        with pytest.raises(ValueError, match="must be divisible by pp_microbatch_size"):
            AutoPipeline(
                world_mesh=FakeDeviceMesh(),
                pp_microbatch_size=3,
                pp_batch_size=4,
                device=torch.device("cpu"),
            )

    def test_microbatch_size_must_be_positive(self):
        with pytest.raises(ValueError, match="pp_microbatch_size must be positive"):
            AutoPipeline(
                world_mesh=FakeDeviceMesh(),
                pp_microbatch_size=0,
                pp_batch_size=4,
                device=torch.device("cpu"),
            )

    def test_a_schedule_must_be_named(self):
        with pytest.raises(ValueError, match="Either pp_schedule or pp_schedule_csv must be provided"):
            AutoPipeline(
                world_mesh=FakeDeviceMesh(),
                pp_schedule=None,
                pp_schedule_csv=None,
                pp_microbatch_size=1,
                pp_batch_size=4,
                device=torch.device("cpu"),
            )

    def test_world_mesh_is_required(self):
        with pytest.raises(ValueError, match="world_mesh must be provided"):
            AutoPipeline(world_mesh=None, pp_schedule="1f1b", pp_microbatch_size=1, pp_batch_size=4)

    def test_field_defaults(self):
        pipeline = AutoPipeline(world_mesh=FakeDeviceMesh())

        assert pipeline.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert pipeline.pp_axis_name == "pp"
        assert pipeline.dp_axis_names == ("dp",)
        assert pipeline.pp_schedule == "1f1b"
        assert pipeline.pp_microbatch_size == 1
        assert pipeline.pp_batch_size == 1
        assert pipeline.patch_inner_model is True
        assert pipeline.patch_causal_lm_model is True
        assert pipeline.scale_grads_in_schedule is False
        assert pipeline.emits_hidden_states is False

    def test_optional_fields_are_stored(self):
        world_mesh = FakeDeviceMesh()
        moe_mesh = FakeDeviceMesh()

        pipeline = AutoPipeline(
            world_mesh=world_mesh,
            moe_mesh=moe_mesh,
            pp_axis_name="pp",
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

        assert pipeline.moe_mesh is moe_mesh
        assert pipeline.dp_axis_names == ("dp1", "dp2")
        assert pipeline.cp_axis_name == "context"
        assert pipeline.tp_axis_name == "tensor"
        assert pipeline.ep_axis_name == "expert"
        assert pipeline.ep_shard_axis_names == ("shard1", "shard2")
        assert pipeline.pp_schedule == "interleaved1f1b"
        assert pipeline.pp_schedule_csv == "/path/to/schedule.csv"
        assert pipeline.layers_per_stage == 8
        assert pipeline.round_virtual_stages_to_pp_multiple == "up"
        assert pipeline.module_fqns_per_model_part == [["layer1"], ["layer2"]]
        assert pipeline.device == torch.device("cuda:1")
        assert pipeline.dtype is torch.float16
        assert pipeline.scale_grads_in_schedule is True


class TestAutoPipelineBuild:
    """Model splitting performed by ``build``."""

    @pytest.mark.parametrize("pp_size", [2, 4])
    @pytest.mark.parametrize("local_rank", [0, 1, 2, 3])
    def test_build_assigns_the_declared_modules_to_each_local_stage(self, pp_size, local_rank):
        if local_rank >= pp_size:
            pytest.skip("local_rank not part of this pp_size")

        num_layers = 8
        num_stages = pp_size * 2
        model = DummyQwenForCausalLM(num_layers=num_layers)
        module_fqns = generate_hf_model_fqn_per_model_part(
            num_stages=num_stages,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
            fqn_prefix="model.",
        )
        pipeline = _make_pipeline(
            pp_size=pp_size,
            local_rank=local_rank,
            module_fqns_per_model_part=module_fqns,
        )

        pipeline.build(model, loss_fn=_loss_stub)

        stages_per_rank = num_stages // pp_size
        assert len(pipeline.parts) == stages_per_rank
        local_stage_indices = [local_rank + s * pp_size for s in range(stages_per_rank)]
        for part, global_stage_idx in zip(pipeline.parts, local_stage_indices):
            assert isinstance(part.model.layers, nn.ModuleDict)
            expected_layer_indices = sorted(
                int(name.split(".")[-1]) for name in module_fqns[global_stage_idx] if name.startswith("model.layers.")
            )
            assert sorted(map(int, part.model.layers.keys())) == expected_layer_indices
            assert part.model.rotary_emb is not None
            assert (part.model.embed_tokens is not None) == (global_stage_idx == 0)
            assert (part.model.norm is not None) == (global_stage_idx == num_stages - 1)
            assert (part.lm_head is not None) == (global_stage_idx == num_stages - 1)

    def test_build_returns_self_and_defers_the_runtime_to_the_first_step(self):
        module_fqns = generate_hf_model_fqn_per_model_part(num_stages=2, num_layers=4, fqn_prefix="model.")
        pipeline = _make_pipeline(module_fqns_per_model_part=module_fqns)

        result = pipeline.build(DummyQwenForCausalLM(num_layers=4), loss_fn=_loss_stub)

        assert result is pipeline
        assert pipeline.loss_fn is _loss_stub
        assert pipeline.info.model_parts is not None
        # Stage boundary metadata is measured on the first step, so no schedule
        # or stages exist yet.
        assert pipeline.info.schedule is None
        assert pipeline.info.stages is None
        # Stage ownership is known from the split, before any stage exists.
        assert pipeline.info.enabled is True
        assert pipeline.info.stage_indices == (0,)
        assert pipeline.info.num_stages == 2
        assert pipeline.info.has_first_stage is True
        assert pipeline.info.has_last_stage is False

    def test_build_rejects_missing_loss_and_non_modules(self):
        pipeline = _make_pipeline()
        model = DummyQwenForCausalLM(num_layers=4)

        with pytest.raises(ValueError, match="loss_fn must be provided"):
            pipeline.build(model, loss_fn=None)
        with pytest.raises(TypeError, match="model must be a torch.nn.Module"):
            pipeline.build("not_a_module", loss_fn=_loss_stub)

    def test_build_requires_more_than_one_pipeline_rank(self):
        pipeline = _make_pipeline(pp_size=1)

        with pytest.raises(ValueError, match="at least two ranks"):
            pipeline.build(DummyQwenForCausalLM(num_layers=4), loss_fn=_loss_stub)

    def test_configure_loss_fn_requires_a_built_pipeline(self):
        pipeline = _make_pipeline()

        with pytest.raises(RuntimeError, match="build\\(\\) must be called before configure_loss_fn"):
            pipeline.configure_loss_fn(_loss_stub)


class TestAutoPipelineErrorHandling:
    """State-dependent errors raised before the pipeline is usable."""

    def test_parts_before_build_error(self):
        pipeline = _make_pipeline()

        with pytest.raises(RuntimeError, match="Autopipeline not built"):
            _ = pipeline.parts

    def test_step_before_build_raises(self):
        pipeline = _make_pipeline()

        with pytest.raises(RuntimeError, match="AutoPipeline.build\\(\\) must be called before step"):
            pipeline.step(torch.ones(2, 8, dtype=torch.long))

    def test_step_rejects_a_rank_one_model_input(self):
        module_fqns = generate_hf_model_fqn_per_model_part(num_stages=2, num_layers=4, fqn_prefix="model.")
        pipeline = _make_pipeline(module_fqns_per_model_part=module_fqns)
        pipeline.build(DummyQwenForCausalLM(num_layers=4), loss_fn=_loss_stub)

        with pytest.raises(ValueError, match="at least two dimensions"):
            pipeline.step(torch.ones(8, dtype=torch.long))


class TestAutoPipelineProperties:
    """Properties and debug helpers."""

    def test_properties_before_build(self):
        pipeline = _make_pipeline(pp_microbatch_size=2, pp_batch_size=8)

        assert pipeline.device == torch.device("cpu")
        assert pipeline.pp_mesh is not None
        assert pipeline.info.enabled is False
        assert pipeline.info.schedule is None
        assert pipeline.info.model_parts is None
        assert pipeline.info.stages is None
        assert pipeline.loss_fn is None
        assert pipeline.last_stage_part is None
        assert pipeline.get_stage_param_counts() == []
        assert pipeline.get_stage_param_counts(trainable_only=True) == []

    def test_debug_summary_before_build(self):
        summary = _make_pipeline().debug_summary()

        assert "PP degree: 2" in summary
        assert "Local stages: 0" in summary
        assert "Schedule: not built" in summary
        assert "Runtime key: None" in summary

    def test_debug_summary_after_build(self, caplog):
        import logging

        module_fqns = generate_hf_model_fqn_per_model_part(num_stages=2, num_layers=4, fqn_prefix="model.")
        pipeline = _make_pipeline(module_fqns_per_model_part=module_fqns)
        # Real parameters so the counts are meaningful.
        model = DummyQwenForCausalLM(num_layers=4).to_empty(device="cpu")
        pipeline.build(model, loss_fn=_loss_stub)

        counts = pipeline.get_stage_param_counts()
        assert len(counts) == 1
        assert counts[0] > 0
        assert pipeline.get_stage_param_counts(trainable_only=True) == counts

        summary = pipeline.debug_summary()
        assert "PP degree: 2" in summary
        assert "Schedule: not built" in summary
        assert "Stage part 0: params=" in summary

        with caplog.at_level(logging.INFO):
            pipeline.log_debug_summary()
        assert any("PP degree: 2" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# Two-rank pipeline runtime
#
# These tests drive a real gloo process group and a real
# ``torch.distributed.pipelining`` schedule, so they fail on a hang, a rank
# disagreement, or a stale runtime -- none of which a stand-in schedule detects.
# ---------------------------------------------------------------------------

HIDDEN = 16
VOCAB = 32
NUM_LAYERS = 4
BATCH = 2
SEQ = 8

requires_gloo_ranks = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available() or sys.platform != "linux",
    reason="requires torch.distributed with the gloo backend on linux",
)


class _RuntimeConfig:
    """Minimal HuggingFace-style config for the two-rank toy model."""

    def __init__(self) -> None:
        self.hidden_size = HIDDEN
        self.vocab_size = VOCAB
        self.num_hidden_layers = NUM_LAYERS
        self.tie_word_embeddings = False
        self.is_encoder_decoder = False
        self.model_type = "runtime_toy"
        self.text_config = None
        self.output_attentions = False
        self.output_hidden_states = False


class _RuntimeInner(nn.Module):
    """Embedding, decoder layers, and final norm."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(HIDDEN)


class _RuntimeCausalLM(PipelineModelMixin, nn.Module):
    """Toy causal LM owning a pipeline forward that honors the fused-loss contract."""

    pipeline_forward_style = PipelineForwardStyle.MODEL
    pipeline_supports_hidden_state_output = True

    def __init__(self) -> None:
        super().__init__()
        self.config = _RuntimeConfig()
        self.model = _RuntimeInner()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Run this stage's slice of the model.

        Args:
            input_ids: Token ids of shape [microbatch, sequence] on the first
                stage, or hidden states of shape [microbatch, sequence, hidden]
                on any later stage.
            **kwargs: Ignored schedule keyword arguments.

        Returns:
            Hidden states of shape [microbatch, sequence, hidden] when this
            stage does not own ``lm_head`` or when ``_pp_return_hidden_states``
            is set, otherwise logits of shape [microbatch, sequence, vocab].
        """
        del kwargs
        inner = self.model
        hidden_states = inner.embed_tokens(input_ids) if inner.embed_tokens is not None else input_ids
        layers = getattr(inner, "layers", None)
        if layers is not None:
            iterator = layers.values() if hasattr(layers, "values") else layers
            for layer in iterator:
                hidden_states = hidden_states + torch.tanh(layer(hidden_states))
        if inner.norm is not None:
            hidden_states = inner.norm(hidden_states)
        if getattr(self, "_pp_return_hidden_states", False):
            return hidden_states
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return hidden_states


def _logits_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean token cross-entropy.

    Args:
        logits: Tensor of shape [microbatch, sequence, vocab].
        labels: Token ids of shape [microbatch, sequence].

    Returns:
        Scalar loss tensor.
    """
    return nn.functional.cross_entropy(logits.reshape(-1, VOCAB).float(), labels.reshape(-1))


def _hidden_state_loss(hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Stand-in fused loss consuming unprojected hidden states.

    Args:
        hidden_states: Tensor of shape [microbatch, sequence, hidden].
        labels: Token ids of shape [microbatch, sequence].

    Returns:
        Scalar loss tensor.
    """
    del labels
    return hidden_states.float().pow(2).mean()


def _runtime_batch(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic batch identical on every rank.

    Args:
        seq_len: Sequence length of the batch.

    Returns:
        Token ids of shape [batch, sequence] and labels of the same shape.
    """
    generator = torch.Generator().manual_seed(seq_len)
    input_ids = torch.randint(0, VOCAB, (BATCH, seq_len), generator=generator)
    labels = torch.randint(0, VOCAB, (BATCH, seq_len), generator=generator)
    return input_ids, labels


def _local_grads(pipeline: AutoPipeline) -> list[torch.Tensor]:
    """Return every non-None parameter gradient owned by this rank."""
    return [param.grad for part in pipeline.parts for param in part.parameters() if param.grad is not None]


def _clear_grads(pipeline: AutoPipeline) -> None:
    """Drop every parameter gradient owned by this rank."""
    for part in pipeline.parts:
        for param in part.parameters():
            param.grad = None


def _scenario_runtime_key_reuse(pipeline: AutoPipeline, rank: int) -> None:
    """A repeated shape reuses the runtime; a new shape rebuilds it."""
    assert pipeline._runtime_key is None, "build() must not freeze a runtime"

    input_ids, labels = _runtime_batch(SEQ)
    pipeline.step(input_ids, target=labels, losses=[])
    first_key = pipeline._runtime_key
    first_stages = pipeline.info.stages
    first_parts = tuple(pipeline.parts)
    assert first_key.input_shape == (SEQ,)
    assert first_key.input_dtype == torch.int64
    assert first_key.forward_only is False
    assert first_key.emits_hidden_states is False

    pipeline.step(input_ids, target=labels, losses=[])
    assert pipeline._runtime_key == first_key, f"rank {rank}: identical input must not change the runtime key"
    assert pipeline.info.stages is first_stages, f"rank {rank}: identical input must reuse the stages"

    long_ids, long_labels = _runtime_batch(SEQ * 2)
    pipeline.step(long_ids, target=long_labels, losses=[])
    assert pipeline._runtime_key != first_key, f"rank {rank}: a new sequence length must change the runtime key"
    assert pipeline._runtime_key.input_shape == (SEQ * 2,)
    assert pipeline.info.stages is not first_stages, f"rank {rank}: a new sequence length must rebuild the stages"
    # Rebuilding the runtime must not replace the model parts or the loss.
    assert tuple(pipeline.parts) == first_parts
    assert pipeline.loss_fn is _logits_loss
    assert _local_grads(pipeline), f"rank {rank}: no gradients after the rebuild"


def _scenario_eval_then_train(pipeline: AutoPipeline, rank: int) -> None:
    """A forward-only step must not poison the next training step at the same shape.

    ``PipelineStage`` only marks its receive buffers as requiring grad when
    ``has_backward`` is true while ``_prepare_forward_infra`` runs, and
    ``schedule.eval()`` clears that flag for the duration of the call. Reusing an
    eval-initialized runtime therefore left the non-first stages unable to
    backpropagate, which is why ``forward_only`` is part of the runtime key.
    """
    input_ids, labels = _runtime_batch(SEQ)

    with torch.no_grad():
        pipeline.step(input_ids, target=labels, losses=[], forward_only=True)
    eval_key = pipeline._runtime_key
    eval_stages = pipeline.info.stages
    assert eval_key.forward_only is True
    assert not _local_grads(pipeline), f"rank {rank}: forward_only must not produce gradients"

    losses: list[torch.Tensor] = []
    pipeline.step(input_ids, target=labels, losses=losses)

    assert pipeline._runtime_key.forward_only is False
    assert pipeline._runtime_key != eval_key, f"rank {rank}: training must not reuse the eval runtime"
    assert pipeline.info.stages is not eval_stages
    grads = _local_grads(pipeline)
    assert grads, f"rank {rank}: training after a forward-only step produced no gradients"
    for grad in grads:
        assert torch.isfinite(grad).all()
    if pipeline.info.has_last_stage:
        assert losses, f"rank {rank}: the last stage produced no loss"


def _scenario_fused_loss_contract(pipeline: AutoPipeline, rank: int) -> None:
    """``configure_loss_fn`` re-keys the runtime on every rank, not just the last."""
    input_ids, labels = _runtime_batch(SEQ)
    logits_output = pipeline.step(input_ids, target=labels, losses=[])
    logits_key = pipeline._runtime_key
    logits_stages = pipeline.info.stages
    has_last = pipeline.info.has_last_stage
    if has_last:
        assert logits_output.shape == (BATCH, SEQ, VOCAB)
    else:
        assert logits_output is None
    _clear_grads(pipeline)

    pipeline.configure_loss_fn(_hidden_state_loss, emits_hidden_states=True)

    assert pipeline.loss_fn is _hidden_state_loss
    assert pipeline.emits_hidden_states is True
    # The rebuild is declarative: it is deferred to the next step so that every
    # rank performs it, including the ranks that own no last stage.
    assert pipeline._runtime_key == logits_key, f"rank {rank}: configure_loss_fn must not rebuild eagerly"
    assert (pipeline.last_stage_part is not None) == has_last

    losses: list[torch.Tensor] = []
    hidden_output = pipeline.step(input_ids, target=labels, losses=losses)

    assert pipeline._runtime_key.emits_hidden_states is True, (
        f"rank {rank}: the fused-loss contract must re-key the runtime on every rank"
    )
    assert pipeline._runtime_key != logits_key
    assert pipeline.info.stages is not logits_stages
    if has_last:
        assert hidden_output.shape == (BATCH, SEQ, HIDDEN), "the last stage must skip its lm_head"
        assert losses, f"rank {rank}: the fused loss produced no loss values"
    else:
        assert hidden_output is None
    assert _local_grads(pipeline), f"rank {rank}: the fused-loss step produced no gradients"


_SCENARIOS = {
    "runtime_key_reuse": _scenario_runtime_key_reuse,
    "eval_then_train": _scenario_eval_then_train,
    "fused_loss_contract": _scenario_fused_loss_contract,
}


def _free_port() -> int:
    """Return a currently free localhost TCP port for the rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pipeline_worker(rank: int, world_size: int, scenario: str, port: int, error_queue) -> None:
    """Run one pipeline rank through a scenario, forwarding failures to the parent.

    Args:
        rank: Global rank of this process.
        world_size: Number of pipeline ranks.
        scenario: Key into ``_SCENARIOS``.
        port: Localhost rendezvous port shared by every rank.
        error_queue: Queue receiving a traceback string on failure.
    """
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("pp",))
        torch.manual_seed(1234)
        pipeline = AutoPipeline(
            world_mesh=mesh,
            pp_axis_name="pp",
            dp_axis_names=(),
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=BATCH,
            device=torch.device("cpu"),
        ).build(_RuntimeCausalLM(), loss_fn=_logits_loss)

        _SCENARIOS[scenario](pipeline, rank)
        torch.distributed.barrier()
    except Exception:
        error_queue.put(f"rank {rank}:\n{traceback.format_exc()}")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_scenario(scenario: str, world_size: int = 2) -> None:
    """Spawn pipeline ranks for a scenario and re-raise any child failure."""
    context = mp.get_context("spawn")
    error_queue = context.SimpleQueue()
    port = _free_port()
    processes = [
        context.Process(target=_pipeline_worker, args=(rank, world_size, scenario, port, error_queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)

    failures = []
    while not error_queue.empty():
        failures.append(error_queue.get())
    for process in processes:
        if process.is_alive():
            process.terminate()
            failures.append(f"a pipeline rank hung running scenario {scenario!r} and was terminated")
    if failures:
        pytest.fail("\n\n".join(failures))
    for process in processes:
        assert process.exitcode == 0, f"pipeline rank exited with {process.exitcode}"


@requires_gloo_ranks
def test_runtime_is_rebuilt_only_when_the_input_shape_changes():
    """Repeating a shape reuses the stages; changing it re-keys and rebuilds them."""
    _run_scenario("runtime_key_reuse")


@requires_gloo_ranks
def test_training_step_after_a_forward_only_step_at_the_same_shape():
    """A cached eval runtime must not break the following training step."""
    _run_scenario("eval_then_train")


@requires_gloo_ranks
def test_fused_loss_contract_is_applied_on_every_rank():
    """``emits_hidden_states`` re-keys the runtime uniformly across pipeline ranks."""
    _run_scenario("fused_loss_contract")
