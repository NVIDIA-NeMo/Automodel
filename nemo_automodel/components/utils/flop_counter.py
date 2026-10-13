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

"""FLOPs measurement for eager and Inductor-compiled forward/backward execution."""

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from types import TracebackType

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._inductor.compile_fx import compile_fx
from torch._inductor.decomposition import select_decomp_table
from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket
from torch.fx.node import map_arg
from torch.utils.flop_counter import FlopCounterMode as TorchFlopCounterMode
from torch.utils.flop_counter import _FlopCounterMode, shape_wrapper

logger = logging.getLogger(__name__)


def _grouped_mm_flops(a_shape: Sequence[int], b_shape: Sequence[int], *args: object, **kwargs: object) -> int:
    """Count grouped GEMMs, excluding bias just like PyTorch's MM formula.

    Shapes are [groups, rows, reduction] / [groups, reduction, columns]
    for 3D operands. A 2D operand packs the variable group dimension:
    rows for 2D/3D, columns for 3D/2D, or reduction for 2D/2D.
    Summing group sizes therefore requires no offset tensor values.
    """
    groups = a_shape[0] if len(a_shape) == len(b_shape) == 3 else 1
    return 2 * groups * a_shape[-2] * a_shape[-1] * b_shape[-1]


class _CompileAwareDispatchMode(_FlopCounterMode):
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # Compilation must not contribute FLOPs or force fullgraph code eager.
        # The AOT graph wrappers below account for compiled execution instead.
        return True

    def __torch_dispatch__(
        self,
        func: OpOverload | HigherOrderOperator,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        """Follow PyTorch's dispatch contract for arbitrary operator tensor layouts."""
        if self.counter._compiled_depth:
            # Inductor can call ATen GEMMs as well as fused kernels. Both are
            # already included in the graph count; do not count GEMMs twice.
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)


class FlopCounterMode(TorchFlopCounterMode):
    """Count eager operators and executed compiled graphs with PyTorch's formulas.

    During measurement, existing ``torch.compile`` functions use an instrumented
    Inductor backend. AOTAutograd exposes separate forward and backward graphs;
    their registered operator FLOPs are added each time the compiled graph runs,
    including cache hits. Compilation itself contributes no FLOPs. Eager regions
    and graph breaks use the ordinary PyTorch dispatch counter.

    Put both forward and backward inside this context. Entering it may compile a
    separate measurement variant even for warmed functions. The original compiler
    stance/backend is restored on exit. These instrumented steps must not be used
    for throughput timing. This is an Inductor measurement, not a measurement of
    another backend's optimizations or exact fused-kernel instructions.
    The measurement backend replaces each function's ``mode`` and ``options``
    with Inductor defaults, so kernel selection and rounding may differ.

    Counts describe registered ATen graph operations before Inductor fusion, with
    PyTorch's usual omissions (for example, elementwise operations). Custom opaque
    kernels still need a FLOPs formula. Only global totals are collected;
    measurement does not install module tracking hooks.
    Activation-checkpoint and Inductor rematerialization FLOPs are included
    whenever their registered operations execute in the measured graphs.
    Symbolic FLOP totals specialize the measurement graph to their integer value.
    This adapter uses the dispatch/compiler interfaces in the CI PyTorch 2.14.
    Like PyTorch's counter, instances are not reentrant or thread safe.

    Args:
        display: Whether to log the table when leaving the context.
        custom_mapping: Additional operator-to-FLOPs formulas, following
            ``torch.utils.flop_counter.FlopCounterMode`` conventions.
    """

    def __init__(
        self,
        *,
        display: bool = True,
        custom_mapping: dict[OpOverloadPacket | HigherOrderOperator, Callable[..., int]] | None = None,
    ) -> None:
        super().__init__(display=display, custom_mapping=custom_mapping)
        # Fill the gap in PyTorch's registry without replacing an upstream or
        # caller-supplied formula. Both eager dispatch and AOT graphs use this.
        self.flop_registry.setdefault(torch.ops.aten._grouped_mm, shape_wrapper(_grouped_mm_flops))
        self._active = False
        self._compiled_depth = 0
        self._stack: ExitStack | None = None
        # Keep the backend identity stable when reusing this counter so that
        # measurement graphs can be reused without recompiling each iteration.
        self._backend = aot_autograd(
            fw_compiler=self._compile_graph,
            bw_compiler=self._compile_graph,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=select_decomp_table(),
        )

    def _compile_graph(
        self, graph: torch.fx.GraphModule, inputs: Sequence[torch.Tensor | int | torch.SymInt]
    ) -> Callable[..., Sequence[torch.Tensor | int | torch.SymInt | None]]:
        """Compile an AOT graph and attach its FLOPs to actual invocations.

        Args:
            graph: Flat ATen forward or backward graph with fake tensor metadata.
            inputs: Flat example arguments; tensor shapes/layouts follow the
                graph placeholders, with arbitrary rank and dtype.

        Returns:
            Boxed callable returning the graph's flat outputs in their original
            shapes, layouts and dtypes.
        """
        counts: Counter[OpOverloadPacket | HigherOrderOperator] = Counter()
        for node in graph.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if isinstance(target, OpOverload):
                packet = target._overloadpacket
            elif isinstance(target, HigherOrderOperator):
                packet = target
            else:
                continue
            formula = self.flop_registry.get(packet)
            if formula is None and isinstance(target, torch._ops.HigherOrderOperator):
                raise NotImplementedError(
                    f"Compiled FLOPs counting requires a formula for higher-order operator {target}"
                )
            if formula is not None:
                args = map_arg(node.args, lambda arg: arg.meta["val"])
                kwargs = map_arg(node.kwargs, lambda arg: arg.meta["val"])
                counts[packet] += int(formula(*args, **kwargs, out_val=node.meta["val"]))

        # Lazy backward compilation can run outside Dynamo's mode suspension.
        # Inductor autotuning may execute real candidate GEMMs there.
        self._compiled_depth += 1
        try:
            compiled = compile_fx(graph, inputs)
        finally:
            self._compiled_depth -= 1

        def run(*args: torch.Tensor | int | torch.SymInt) -> Sequence[torch.Tensor | int | torch.SymInt | None]:
            """Execute the compiled graph and count it on success.

            Args:
                args: Flat runtime arguments with tensor shapes/layouts matching
                    the AOT graph placeholders, of arbitrary rank and dtype.

            Returns:
                Flat graph outputs with unchanged tensor shapes/layouts/dtypes.
            """
            self._compiled_depth += 1
            try:
                result = compiled(*args)
            finally:
                self._compiled_depth -= 1
            if self._active:
                for packet, count in counts.items():
                    self.flop_counts["Global"][packet] += count
            return result

        return make_boxed_func(run)

    def __enter__(self) -> "FlopCounterMode":
        if self._active:
            raise RuntimeError("FlopCounterMode is not reentrant")
        self.flop_counts.clear()
        with ExitStack() as stack:
            stack.enter_context(torch.compiler.set_stance("default", force_backend=self._backend))
            stack.enter_context(_CompileAwareDispatchMode(self))
            self._stack = stack.pop_all()
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._active = False
        stack, self._stack = self._stack, None
        if stack is None:
            raise RuntimeError("FlopCounterMode must be entered before exiting")
        result = stack.__exit__(exc_type, exc_value, traceback)
        if self.display:
            logger.info("\n%s", self.get_table(self.depth))
        return result
