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

"""Whole-step CUDA graph: one graph for every microbatch's forward, loss, and backward."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

StepFunction = Callable[[list[dict[str, torch.Tensor]]], list[torch.Tensor]]
"""Runs forward, loss, and backward for every microbatch of one optimizer step.

The argument is the list of device-resident microbatches. The result holds one
detached scalar loss tensor per microbatch. The function must not touch the
optimizer, read device values on the host, or allocate tensors that outlive the
call other than parameter gradients.
"""


@dataclass(frozen=True)
class WholeStepGraphValidation:
    """Comparison between one graph replay and an eager rerun on identical weights and inputs.

    Attributes:
        num_gradients: Number of parameter gradients compared.
        max_relative_rmse: Largest per-gradient RMSE divided by the eager gradient RMS.
        max_eager_repeat_relative_rmse: Same statistic between two eager reruns; the
            run-to-run noise floor of the eager path itself.
        max_abs_loss_difference: Largest absolute difference between graph and eager
            per-microbatch losses.
        all_finite: Whether every graph and eager gradient was finite.
    """

    num_gradients: int
    max_relative_rmse: float
    max_eager_repeat_relative_rmse: float
    max_abs_loss_difference: float
    all_finite: bool


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return a parameter or gradient's local shard without a collective.

    Args:
        tensor: Tensor of arbitrary shape, or a DTensor whose placements already
            describe the training shard.

    Returns:
        Tensor aliasing the local storage of ``tensor``.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return RMSE of ``actual - expected`` relative to the RMS of ``expected``.

    Args:
        actual: Tensor of arbitrary shape.
        expected: Tensor with the same shape as ``actual``.

    Returns:
        Zero for empty tensors, otherwise the relative RMSE as a host float.
    """
    if actual.numel() == 0:
        return 0.0
    difference = actual.float() - expected.float()
    return (difference.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-20)).item()


class WholeStepGraph:
    """Capture and replay one optimizer step's forward, loss, and backward as a single CUDA graph.

    All gradient-accumulation microbatches of a step are captured into one graph.
    Each microbatch owns a set of static device buffers; ``replay`` copies the next
    step's host batches into them and launches the graph. The optimizer step, gradient
    clipping, and logging stay eager and run after ``replay`` returns.

    The backward pass is captured with gradients set to ``None``, so the captured
    kernels write fresh gradients into graph-owned storage on every replay. Because
    the optimizer's ``zero_grad`` detaches those tensors from the parameters,
    ``replay`` re-binds every captured gradient to its parameter before returning.
    """

    def __init__(
        self,
        step_fn: StepFunction,
        parameters: list[tuple[str, torch.Tensor]],
        *,
        device: torch.device,
        warmup_iterations: int = 2,
    ) -> None:
        """Create an un-captured whole-step graph.

        Args:
            step_fn: Forward/loss/backward over all microbatches of one optimizer step.
            parameters: Named parameters whose gradients the graph produces. Each
                tensor keeps its training layout (local tensor or DTensor shard) and
                must keep the same storage for the graph's lifetime.
            device: CUDA device holding the static input buffers.
            warmup_iterations: Eager iterations on the capture stream before capture,
                so lazily initialized state (allocator pools, NCCL communicators,
                autograd nodes) exists before recording.
        """
        if warmup_iterations < 1:
            raise ValueError("warmup_iterations must be at least 1")
        self._step_fn = step_fn
        self._parameters = parameters
        self._device = device
        self._warmup_iterations = warmup_iterations
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_batches: list[dict[str, torch.Tensor]] = []
        self._losses: list[torch.Tensor] = []
        self._grads: dict[str, torch.Tensor] = {}
        self._parameter_ptrs: dict[str, int] = {}

    @property
    def captured(self) -> bool:
        """Whether ``capture`` has completed and ``replay`` may be called."""
        return self._graph is not None

    def _check_batches(self, batches: list[dict[str, torch.Tensor]]) -> None:
        """Reject batches the graph cannot replay.

        Args:
            batches: Host or device microbatches. Every value must be a tensor; the
                layout of each tensor is the model's own input contract and is
                only required to be fixed across steps.
        """
        if not batches:
            raise ValueError("Whole-step graph requires at least one microbatch")
        for batch in batches:
            for key, value in batch.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"Whole-step graph requires a flat tensor batch; entry {key!r} is {type(value).__name__}"
                    )

    def _clear_gradients(self) -> None:
        """Detach every tracked gradient so the next backward allocates fresh storage."""
        for _, parameter in self._parameters:
            parameter.grad = None

    def capture(self, batches: list[dict[str, torch.Tensor]]) -> None:
        """Record the step on ``batches`` after warming up on a side stream.

        Args:
            batches: One optimizer step's microbatches with the shapes and dtypes
                every later step must reproduce. Each tensor is cloned into a static
                device buffer; the arguments themselves are not retained.
        """
        if self._graph is not None:
            raise RuntimeError("Whole-step graph is already captured")
        self._check_batches(batches)
        self._static_batches = [
            {key: value.to(self._device, copy=True) for key, value in batch.items()} for batch in batches
        ]
        self._parameter_ptrs = {name: _local(parameter).data_ptr() for name, parameter in self._parameters}
        torch.cuda.synchronize(self._device)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        capture_stream = torch.cuda.Stream(device=self._device)
        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        # Eager warmup created the AccumulateGrad nodes FSDP keeps across steps;
        # autograd must accept them on the capture stream.
        torch.autograd.graph.set_override_stale_capture_stream(True)
        with torch.cuda.stream(capture_stream):
            for _ in range(self._warmup_iterations):
                self._clear_gradients()
                self._step_fn(self._static_batches)
            self._clear_gradients()
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        torch.cuda.synchronize(self._device)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        graph = torch.cuda.CUDAGraph()
        # Own the graph before recording so a failed capture is still released by reset().
        self._graph = graph
        with torch.cuda.graph(graph, stream=capture_stream):
            self._losses = self._step_fn(self._static_batches)
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        torch.cuda.synchronize(self._device)
        self._grads = {name: parameter.grad for name, parameter in self._parameters if parameter.grad is not None}
        logger.info(
            "Whole-step CUDA graph captured: microbatches=%d gradients=%d allocated=%.3f GiB reserved=%.3f GiB",
            len(self._static_batches),
            len(self._grads),
            torch.cuda.memory_allocated(self._device) / 1024**3,
            torch.cuda.memory_reserved(self._device) / 1024**3,
        )

    def _copy_inputs(self, batches: list[dict[str, torch.Tensor]]) -> None:
        """Copy ``batches`` into the static buffers, rejecting any metadata change.

        Args:
            batches: Microbatches with exactly the keys, shapes, and dtypes captured.
        """
        if len(batches) != len(self._static_batches):
            raise ValueError(
                f"Whole-step graph captured {len(self._static_batches)} microbatches per step, got {len(batches)}"
            )
        for index, (batch, static_batch) in enumerate(zip(batches, self._static_batches)):
            if batch.keys() != static_batch.keys():
                raise ValueError(f"Whole-step graph microbatch {index} keys changed since capture")
            for key, value in batch.items():
                target = static_batch[key]
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError(
                        f"Whole-step graph microbatch {index} entry {key!r} changed from "
                        f"{tuple(target.shape)}/{target.dtype} to {tuple(value.shape)}/{value.dtype}"
                    )
                target.copy_(value, non_blocking=True)

    def _bind_gradients(self) -> None:
        """Point every parameter at the gradient storage the graph writes."""
        for name, parameter in self._parameters:
            if _local(parameter).data_ptr() != self._parameter_ptrs[name]:
                raise RuntimeError(f"Parameter storage changed after whole-step graph capture: {name}")
            parameter.grad = self._grads.get(name)

    def replay(self, batches: list[dict[str, torch.Tensor]]) -> list[torch.Tensor]:
        """Run the captured step on new inputs.

        Args:
            batches: Microbatches with the captured keys, shapes, and dtypes.

        Returns:
            The captured per-microbatch loss tensors. They alias graph-owned storage
            and hold this replay's values until the next ``replay``.
        """
        if self._graph is None:
            raise RuntimeError("Whole-step graph has not been captured")
        self._check_batches(batches)
        self._copy_inputs(batches)
        self._graph.replay()
        self._bind_gradients()
        return self._losses

    def validate(self, batches: list[dict[str, torch.Tensor]]) -> WholeStepGraphValidation:
        """Compare the latest replay's gradients and losses against two eager reruns.

        Call right after ``replay(batches)`` and before the optimizer step. The
        graph's gradients are copied to the host, the eager step runs twice on the
        same weights and inputs, and the graph's gradients are re-bound afterwards
        so the optimizer step that follows uses them, not the eager ones.

        Args:
            batches: The microbatches passed to the preceding ``replay``.

        Returns:
            Summary statistics; the caller decides the acceptance threshold.
        """
        if self._graph is None:
            raise RuntimeError("Whole-step graph has not been captured")
        graph_grads = {name: _local(grad).detach().to("cpu", copy=True) for name, grad in self._grads.items()}
        graph_losses = torch.stack(self._losses).float().cpu()
        device_batches = [
            {key: value.to(self._device, non_blocking=True) for key, value in batch.items()} for batch in batches
        ]
        self._clear_gradients()
        eager_losses = torch.stack(self._step_fn(device_batches)).float().cpu()
        eager_grads = {
            name: _local(parameter.grad).detach().to("cpu", copy=True)
            for name, parameter in self._parameters
            if parameter.grad is not None
        }
        self._clear_gradients()
        self._step_fn(device_batches)
        repeat_grads = {
            name: _local(parameter.grad).detach().to("cpu", copy=True)
            for name, parameter in self._parameters
            if parameter.grad is not None
        }
        self._bind_gradients()
        if graph_grads.keys() != eager_grads.keys():
            missing = sorted(set(graph_grads) ^ set(eager_grads))
            raise AssertionError(f"Whole-step graph and eager produced gradients for different parameters: {missing}")
        max_rmse = 0.0
        max_repeat_rmse = 0.0
        all_finite = True
        for name, expected in eager_grads.items():
            actual = graph_grads[name]
            if actual.shape != expected.shape:
                raise AssertionError(f"Gradient shard shape differs between graph and eager for {name}")
            all_finite = all_finite and bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
            max_rmse = max(max_rmse, _relative_rmse(actual, expected))
            max_repeat_rmse = max(max_repeat_rmse, _relative_rmse(repeat_grads[name], expected))
        return WholeStepGraphValidation(
            num_gradients=len(eager_grads),
            max_relative_rmse=max_rmse,
            max_eager_repeat_relative_rmse=max_repeat_rmse,
            max_abs_loss_difference=(graph_losses - eager_losses).abs().max().item(),
            all_finite=all_finite,
        )

    def reset(self) -> None:
        """Release the graph, its static inputs, and the captured gradients."""
        if self._graph is not None:
            torch.cuda.synchronize(self._device)
            self._graph.reset()
        self._graph = None
        self._static_batches = []
        self._losses = []
        self._grads = {}
        self._parameter_ptrs = {}
