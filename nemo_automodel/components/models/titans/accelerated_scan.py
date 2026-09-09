# Copyright (c) 2024 Volodymyr Kyrylov
# SPDX-License-Identifier: MIT
#
# Adapted from accelerated-scan 0.3.1:
# https://github.com/proger/accelerated-scan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Triton first-order scan with a bounds fix over accelerated-scan 0.3.1."""

import torch
import triton
import triton.language as tl


@triton.jit
def _combine(left_x, left_gate, right_x, right_gate):
    output = left_x * right_gate + right_x
    gate = left_gate * right_gate
    return output, gate


@triton.jit
def _forward_scan(gates, inputs, outputs, sequence_length, BLOCK: tl.constexpr = 2048):
    block_count = tl.cdiv(sequence_length, BLOCK)
    batch, channel = tl.program_id(axis=0), tl.program_id(axis=1)
    sequence = tl.num_programs(axis=1) * batch + channel
    indices = tl.arange(0, BLOCK)
    carry = tl.zeros((), dtype=tl.float32)

    for block_index in tl.range(0, block_count):
        time = block_index * BLOCK + indices
        offset = sequence * sequence_length + time
        valid = time < sequence_length
        value = tl.load(inputs + offset, mask=valid, other=0.0)
        gate = tl.load(gates + offset, mask=valid, other=1.0)
        value, gate = tl.associative_scan((value, gate), axis=0, combine_fn=_combine)
        block_value = tl.sum(tl.where(indices == BLOCK - 1, value, 0.0), axis=0)
        block_gate = tl.sum(tl.where(indices == BLOCK - 1, gate, 0.0), axis=0)
        value = value + carry * gate
        tl.store(outputs + offset, value, mask=valid)
        carry = block_value + carry * block_gate


@triton.jit
def _backward_scan(
    gates,
    outputs,
    output_grads,
    input_grads,
    gate_grads,
    sequence_length,
    BLOCK: tl.constexpr = 2048,
):
    block_count = tl.cdiv(sequence_length, BLOCK)
    batch, channel = tl.program_id(axis=0), tl.program_id(axis=1)
    sequence = tl.num_programs(axis=1) * batch + channel
    indices = tl.arange(0, BLOCK)
    carry = tl.zeros((), dtype=tl.float32)

    for block_index in tl.range(0, block_count):
        reverse_block = block_count - 1 - block_index
        time = reverse_block * BLOCK + indices
        offset = sequence * sequence_length + time
        valid = time < sequence_length
        output_grad = tl.load(output_grads + offset, mask=valid, other=0.0)
        shifted_gate = tl.load(
            gates + offset + 1,
            mask=time < sequence_length - 1,
            other=1.0,
        )
        input_grad, gate = tl.associative_scan(
            (output_grad, shifted_gate),
            axis=0,
            combine_fn=_combine,
            reverse=True,
        )
        block_grad = tl.sum(tl.where(indices == 0, input_grad, 0.0), axis=0)
        block_gate = tl.sum(tl.where(indices == 0, gate, 0.0), axis=0)
        input_grad = input_grad + carry * gate
        tl.store(input_grads + offset, input_grad, mask=valid)

        # accelerated-scan 0.3.1 only checked ``time > 0`` here. Threads with
        # ``time >= sequence_length`` consequently loaded beyond ``outputs``.
        previous_output = tl.load(
            outputs + offset - 1,
            mask=(time > 0) & valid,
            other=0.0,
        )
        tl.store(gate_grads + offset, previous_output * input_grad, mask=valid)
        carry = block_grad + carry * block_gate


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, sequence_length = gates.shape
        if inputs.shape != (batch, channels, sequence_length):
            raise ValueError(f"scan inputs must match gates; got {inputs.shape} and {gates.shape}")
        if not gates.is_contiguous() or not inputs.is_contiguous():
            raise ValueError("scan gates and inputs must be contiguous")
        outputs = torch.empty_like(inputs)
        _forward_scan[(batch, channels)](
            gates,
            inputs,
            outputs,
            sequence_length=sequence_length,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(gates, outputs)
        return outputs

    @staticmethod
    def backward(ctx, output_grads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates, outputs = ctx.saved_tensors
        batch, channels, sequence_length = gates.shape
        input_grads = torch.empty_like(outputs)
        gate_grads = torch.empty_like(gates)
        _backward_scan[(batch, channels)](
            gates,
            outputs,
            output_grads.contiguous(),
            input_grads,
            gate_grads,
            sequence_length=sequence_length,
            enable_fp_fusion=False,
        )
        return gate_grads, input_grads


def scan(gates: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Solve ``state_t = gate_t * state_(t-1) + input_t`` along the last axis."""
    return _Scan.apply(gates, inputs)
