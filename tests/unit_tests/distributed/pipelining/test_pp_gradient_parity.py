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

"""Two-rank CPU gradient parity for pipeline-parallel execution.

These tests spawn a real gloo process group and drive a real
``torch.distributed.pipelining`` schedule, so they fail on a hang, a rank
mismatch, or wrong gradients -- unlike mock-based pipeline tests. Every
parameter gradient produced under ``pp_size=2`` is compared against a
single-process microbatch loop over the same weights.
"""

import contextlib
import os
import socket
import sys
import traceback

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from nemo_automodel.shared.pipeline import PipelineForwardStyle, PipelineModelMixin, pp_media_chunk

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="requires torch.distributed with the gloo backend",
)

HIDDEN = 16
VOCAB = 32
NUM_LAYERS = 4
BATCH = 4
SEQ = 8
MICROBATCH = 2


class _TinyConfig:
    """Minimal stand-in for a HuggingFace config."""

    def __init__(self) -> None:
        self.hidden_size = HIDDEN
        self.vocab_size = VOCAB
        self.num_hidden_layers = NUM_LAYERS
        self.tie_word_embeddings = False
        self.is_encoder_decoder = False
        self.model_type = "tiny"
        self.text_config = None
        self.output_attentions = False
        self.output_hidden_states = False


class _TinyLayer(nn.Module):
    """One residual MLP block."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply a residual projection.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return hidden_states + torch.tanh(self.proj(hidden_states))


class _TinyModel(nn.Module):
    """Embedding, decoder layers, and final norm."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([_TinyLayer() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(HIDDEN)


class _TinyForCausalLM(nn.Module, PipelineModelMixin):
    """Causal-LM wrapper that owns its pipeline-aware forward.

    Uses the ``PipelineForwardStyle.MODEL`` contract shared by the migrated
    models: a stage without ``embed_tokens`` receives the previous stage's
    hidden states in the ``input_ids`` slot as a floating-point tensor.
    """

    pipeline_forward_style = PipelineForwardStyle.MODEL
    pipeline_supports_hidden_state_output = True

    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig()
        self.model = _TinyModel()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self._seen_media: list[int | None] = []

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Run the stage's slice of the model.

        Args:
            input_ids: Token ids of shape [batch, sequence] on the first stage,
                or hidden states of shape [batch, sequence, hidden] on any
                later stage.
            **kwargs: Ignored; present for schedule compatibility.

        Returns:
            Logits of shape [batch, sequence, vocab] when this stage owns
            ``lm_head``, otherwise hidden states of shape
            [batch, sequence, hidden].
        """
        inner = self.model
        if getattr(inner, "embed_tokens", None) is not None:
            hidden_states = inner.embed_tokens(input_ids)
            # Record which staged media chunk this forward resolved, so a test can
            # assert the pairing survives the schedule's execution order and the
            # runtime shape-inference probe forward.
            if getattr(self, "_pp_media_chunks", None) is not None:
                chunk = pp_media_chunk(self, "pixel_values", kwargs.get("pp_media_index"))
                self._seen_media.append(None if chunk is None else int(chunk.sum().item()))
        else:
            hidden_states = input_ids

        layers = getattr(inner, "layers", None)
        if layers is not None:
            iterator = layers.values() if hasattr(layers, "values") else layers
            for layer in iterator:
                hidden_states = layer(hidden_states)

        if getattr(inner, "norm", None) is not None:
            hidden_states = inner.norm(hidden_states)

        if getattr(self, "_pp_return_hidden_states", False):
            # Fused-loss contract: the final stage skips its vocabulary
            # projection so the loss can fuse it with the cross entropy.
            return hidden_states
        if getattr(self, "lm_head", None) is not None:
            return self.lm_head(hidden_states)
        return hidden_states


def _loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean token cross-entropy.

    Args:
        logits: Tensor of shape [batch, sequence, vocab].
        labels: Tensor of shape [batch, sequence] of int64 token ids.

    Returns:
        Scalar loss tensor.
    """
    return nn.functional.cross_entropy(logits.reshape(-1, VOCAB).float(), labels.reshape(-1))


def _build_model(seed: int = 1234) -> _TinyForCausalLM:
    """Build a model with deterministic weights, identical on every rank."""
    torch.manual_seed(seed)
    return _TinyForCausalLM()


def _make_batch(seed: int = 7, seq_len: int = SEQ, batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic batch.

    Args:
        seed: Seed controlling the generated token ids.
        seq_len: Sequence length of the generated batch.
        batch: Number of samples in the batch.

    Returns:
        Token ids of shape [batch, sequence] and labels of the same shape.
    """
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (batch, seq_len), generator=generator)
    labels = torch.randint(0, VOCAB, (batch, seq_len), generator=generator)
    return input_ids, labels


def _reference_grads(seq_len: int = SEQ, batch: int = BATCH) -> dict[str, torch.Tensor]:
    """Accumulate gradients with a single-process microbatch loop.

    The pipeline schedule runs with ``scale_grads=False``, so it sums the
    per-microbatch gradients; the reference loop does the same.

    Args:
        seq_len: Sequence length of the batch to run.
        batch: Number of samples in the batch.

    Returns:
        Mapping of parameter name to gradient tensor of the parameter's shape.
    """
    model = _build_model()
    input_ids, labels = _make_batch(seq_len=seq_len, batch=batch)
    for input_chunk, label_chunk in zip(input_ids.split(MICROBATCH), labels.split(MICROBATCH)):
        _loss_fn(model(input_chunk), label_chunk).backward()
    return {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}


#: scenario -> (schedule name, batch size). Interleaved schedules place two
#: virtual stages on each rank, so they need at least one microbatch per global
#: stage.
_SCENARIOS = {
    "grad_parity": ("1f1b", BATCH),
    "media_index": ("1f1b", BATCH),
    "fused_loss_contract": ("1f1b", BATCH),
    "eval_then_train": ("1f1b", BATCH),
    "shape_change": ("1f1b", BATCH),
    "interleaved_grad_parity": ("interleaved1f1b", 8),
}


def _pp_worker(rank: int, world_size: int, scenario: str, error_queue: mp.SimpleQueue) -> None:
    """Run one pipeline rank and assert gradient parity against the reference.

    Args:
        rank: Global rank of this process.
        world_size: Number of pipeline ranks.
        scenario: Which behavior to exercise; see the tests below.
        error_queue: Queue receiving a traceback string on failure.
    """
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
        from torch.distributed.device_mesh import init_device_mesh

        from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline

        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("pp",))
        model = _build_model()
        schedule_name, batch_size = _SCENARIOS[scenario]

        pipeline = AutoPipeline(
            world_mesh=mesh,
            pp_axis_name="pp",
            dp_axis_names=(),
            pp_schedule=schedule_name,
            pp_microbatch_size=MICROBATCH,
            pp_batch_size=batch_size,
            device=torch.device("cpu"),
        ).build(model, loss_fn=_loss_fn)

        def run_step(seq_len: int, forward_only: bool = False) -> list[torch.Tensor]:
            input_ids, labels = _make_batch(seq_len=seq_len, batch=batch_size)
            losses: list[torch.Tensor] = []
            pipeline.step(input_ids, target=labels, losses=losses, forward_only=forward_only)
            return losses

        if scenario in ("grad_parity", "interleaved_grad_parity"):
            run_step(SEQ)
            expected = _reference_grads(SEQ, batch=batch_size)
            local = {
                name: param.grad
                for part in pipeline.parts
                for name, param in part.named_parameters()
                if param.grad is not None
            }
            assert local, f"rank {rank} produced no gradients"
            for name, grad in local.items():
                assert name in expected, f"rank {rank}: unexpected parameter {name}"
                torch.testing.assert_close(grad, expected[name], rtol=2e-4, atol=2e-5, msg=f"grad mismatch: {name}")

        elif scenario == "eval_then_train":
            # Regression test: torch's PipelineStage only marks its receive
            # buffers as requiring grad when has_backward is true at
            # _prepare_forward_infra time, and schedule.eval() clears that flag.
            # A runtime cached from an eval step therefore broke the next
            # training step at the same shape.
            with torch.no_grad():
                run_step(SEQ, forward_only=True)
            run_step(SEQ)
            local = [p.grad for part in pipeline.parts for p in part.parameters() if p.grad is not None]
            assert local, f"rank {rank}: training step after eval produced no gradients"
            for grad in local:
                assert torch.isfinite(grad).all()

        elif scenario == "shape_change":
            run_step(SEQ)
            first_stages = pipeline.info.stages
            run_step(SEQ)
            assert pipeline.info.stages is first_stages, "repeated shape must reuse the runtime"
            run_step(SEQ * 2)
            assert pipeline.info.stages is not first_stages, "new shape must rebuild the runtime"
            for part in pipeline.parts:
                for param in part.parameters():
                    param.grad = None
            run_step(SEQ * 2)
            local = [p.grad for part in pipeline.parts for p in part.parameters() if p.grad is not None]
            assert local, f"rank {rank}: no gradients after rebuild"
            expected = _reference_grads(SEQ * 2)
            for part in pipeline.parts:
                for name, param in part.named_parameters():
                    if param.grad is not None:
                        torch.testing.assert_close(
                            param.grad, expected[name], rtol=2e-4, atol=2e-5, msg=f"post-rebuild grad: {name}"
                        )
        elif scenario == "fused_loss_contract":
            run_step(SEQ)
            assert pipeline.emits_hidden_states is False

            observed: list[int] = []

            def fused_loss(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                """Consume last-stage hidden states directly.

                Args:
                    output: Tensor of shape [microbatch, sequence, hidden].
                    labels: Token ids of shape [microbatch, sequence].

                Returns:
                    Scalar loss tensor.
                """
                del labels
                observed.append(output.shape[-1])
                return output.float().pow(2).mean()

            # Called with identical arguments on every rank; the rebuild must
            # therefore happen uniformly on the next step rather than on the
            # last-stage rank alone, which would desynchronize the group.
            pipeline.configure_loss_fn(fused_loss, emits_hidden_states=True)
            assert pipeline.emits_hidden_states is True
            run_step(SEQ)

            if pipeline.info.has_last_stage:
                assert observed, f"rank {rank}: fused loss never ran"
                assert set(observed) == {HIDDEN}, (
                    f"rank {rank}: fused loss saw width {set(observed)}, expected hidden size {HIDDEN}"
                )

        elif scenario == "media_index":
            # Microbatch 0 is staged as an EMPTY chunk (an all-text microbatch);
            # microbatch 1 carries real media. The old cursor protocol advanced
            # only when media tokens were present, so microbatch 1 read
            # microbatch 0's chunk. Addressing by index cannot desynchronize.
            if pipeline.info.has_first_stage:
                stage0 = pipeline.parts[0]
                stage0._pp_media_chunks = {"pixel_values": [torch.zeros(0), torch.full((3,), 7.0)]}
                stage0._seen_media = []

            input_ids, labels = _make_batch(seq_len=SEQ, batch=batch_size)
            media_index = torch.repeat_interleave(torch.arange(batch_size // MICROBATCH, dtype=torch.long), MICROBATCH)
            pipeline.step(input_ids, target=labels, losses=[], pp_media_index=media_index)

            if pipeline.info.has_first_stage:
                seen = pipeline.parts[0]._seen_media
                # One shape-inference probe forward on microbatch 0, then the two
                # real microbatches. The probe must not consume anything.
                assert seen == [None, None, 21], f"rank {rank}: media pairing broke, saw {seen}"
        else:
            raise ValueError(f"unknown scenario {scenario!r}")

        torch.distributed.barrier()
    except Exception:  # noqa: BLE001 - forwarded to the parent process
        error_queue.put(f"rank {rank}:\n{traceback.format_exc()}")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _free_port() -> int:
    """Return a port the OS reports as free, for the gloo rendezvous.

    A fixed or hash-derived port collides when two test sessions run at once,
    and a collision does not fail fast -- the ranks block until the join
    timeout. Asking the OS for an ephemeral port keeps concurrent sessions and
    xdist workers independent.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _run_scenario(scenario: str, world_size: int = 2) -> None:
    """Spawn pipeline ranks and re-raise any child failure in the parent."""
    context = mp.get_context("spawn")
    error_queue = context.SimpleQueue()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    processes = [
        context.Process(target=_pp_worker, args=(rank, world_size, scenario, error_queue)) for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)

    failures = []
    while not error_queue.empty():
        failures.append(error_queue.get())
    for process in processes:
        if process.is_alive():
            process.terminate()
            failures.append("a pipeline rank hung and was terminated")
    if failures:
        pytest.fail("\n\n".join(failures))
    for process in processes:
        assert process.exitcode == 0, f"pipeline rank exited with {process.exitcode}"


@pytest.mark.skipif(sys.platform != "linux", reason="spawned gloo process groups are exercised on linux CI")
def test_pp2_gradient_parity_with_single_process_reference():
    """pp_size=2 must produce the same gradients as a single-process microbatch loop."""
    _run_scenario("grad_parity")


@pytest.mark.skipif(sys.platform != "linux", reason="spawned gloo process groups are exercised on linux CI")
def test_interleaved_pp2_gradient_parity_with_single_process_reference():
    """An interleaved schedule with two virtual stages per rank must also match the reference."""
    _run_scenario("interleaved_grad_parity")


@pytest.mark.skipif(sys.platform != "linux", reason="spawned gloo process groups are exercised on linux CI")
def test_training_step_after_eval_step_at_same_shape():
    """A forward-only step must not poison the runtime reused by the next training step."""
    _run_scenario("eval_then_train")


@pytest.mark.skipif(sys.platform != "linux", reason="spawned gloo process groups are exercised on linux CI")
def test_fused_loss_output_contract_rebuilds_on_every_rank():
    """Switching the last stage to hidden-state output must not desynchronize the group."""
    _run_scenario("fused_loss_contract")


@pytest.mark.skipif(sys.platform != "linux", reason="spawned gloo process groups are exercised on linux CI")
def test_media_chunks_are_addressed_by_index_not_a_cursor():
    """An all-text microbatch must not shift the media of the microbatches after it."""
    _run_scenario("media_index")


@pytest.mark.skipif(sys.platform != "linux", reason="spawned gloo process groups are exercised on linux CI")
def test_runtime_rebuilds_on_shape_change_and_stays_correct():
    """Changing the sequence length rebuilds the runtime and keeps gradients correct."""
    _run_scenario("shape_change")
