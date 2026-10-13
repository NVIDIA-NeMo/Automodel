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
from typing import Any

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func
from torch._inductor.compile_fx import compile_fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_arg
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import FlopCounterMode as TorchFlopCounterMode
from torch.utils.flop_counter import _FlopCounterMode

logger = logging.getLogger(__name__)


class _CompileAwareDispatchMode(_FlopCounterMode):
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # Compilation must not contribute FLOPs or force fullgraph code eager.
        # The AOT graph wrappers below account for compiled execution instead.
        return True

    def __torch_dispatch__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Follow PyTorch's dispatch contract for arbitrary operator tensor layouts."""
        if self.counter._compiled_depth or any(isinstance(arg, FakeTensor) for arg in tree_leaves((args, kwargs))):
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

    Counts describe registered ATen graph operations before Inductor fusion, with
    PyTorch's usual omissions (for example, elementwise operations). Custom opaque
    kernels still need a FLOPs formula. Only global totals are collected;
    measurement does not install module tracking hooks.
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
        custom_mapping: dict[Any, Callable[..., int]] | None = None,
    ) -> None:
        super().__init__(display=display, custom_mapping=custom_mapping)
        self._active = False
        self._compiled_depth = 0
        self._stack: ExitStack | None = None
        # Keep the backend identity stable when reusing this counter so that
        # measurement graphs can be reused without recompiling each iteration.
        self._backend = aot_autograd(fw_compiler=self._compile_graph, bw_compiler=self._compile_graph)

    def _compile_graph(self, graph: torch.fx.GraphModule, inputs: Sequence[Any]) -> Callable[..., Any]:
        """Compile an AOT graph and attach its FLOPs to actual invocations.

        Args:
            graph: Flat ATen forward or backward graph with fake tensor metadata.
            inputs: Flat example arguments; tensor shapes/layouts follow the
                graph placeholders, with arbitrary rank and dtype.

        Returns:
            Boxed callable returning the graph's flat outputs in their original
            shapes, layouts and dtypes.
        """
        counts: Counter[Any] = Counter()
        for node in graph.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            packet = target._overloadpacket if isinstance(target, torch._ops.OpOverload) else target
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

        def run(*args: Any) -> Any:
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
