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

"""CPU coverage for the selective-AC save-set build (FFPA fold-in wiring) and vision checkpointing."""

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, checkpoint_wrapper
from torch.nn.attention import SDPBackend

from nemo_automodel.components.distributed import activation_checkpointing as ac


def test_unwrap_checkpoint_wrapper_returns_input_module_when_unwrapped():
    module = nn.Linear(2, 2)

    assert ac.unwrap_checkpoint_wrapper(module) is module


def test_unwrap_checkpoint_wrapper_returns_inner_module_when_wrapped():
    module = nn.Linear(2, 2)
    wrapped = checkpoint_wrapper(module)

    assert ac.unwrap_checkpoint_wrapper(wrapped) is module


def test_ffpa_forward_ops_folded_into_save_set(monkeypatch):
    """Ops returned by _ffpa_forward_ops() land in the save set (kernel-free wiring)."""
    dense, varlen = object(), object()
    monkeypatch.setattr(ac, "_ffpa_forward_ops", lambda: (dense, varlen))

    save_ops = ac._build_selective_ac_save_ops()
    assert dense in save_ops
    assert varlen in save_ops


def test_build_save_set_ok_when_ffpa_absent(monkeypatch):
    """CPU degrade path: _ffpa_forward_ops() -> () must not break the build."""
    monkeypatch.setattr(ac, "_ffpa_forward_ops", lambda: ())

    save_ops = ac._build_selective_ac_save_ops()
    assert isinstance(save_ops, frozenset) and len(save_ops) > 0


_D = 8

_DEFAULT_PINNED_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


def _sdp_backend_state() -> tuple[bool, bool, bool]:
    """Return the global (math, flash, mem_efficient) SDPA backend enablement flags."""
    return (
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )


class _RecordingVisionBlock(nn.Module):
    """Minimal HF-style vision block (``attn``, no ``self_attn``) that runs real SDPA.

    Records the enabled SDPA backends (``math``, ``flash``, ``mem_efficient``) on
    every forward call, so tests can assert both the checkpoint forward and its
    recompute ran with the same pinned backend set.
    """

    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(_D, 3 * _D)
        self.mlp = nn.Linear(_D, _D)
        self.backend_states: list[tuple[bool, bool, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SDPA + MLP to patch embeddings.

        Args:
            x: Patch embeddings of shape ``[S, H]`` (``S`` = patches, ``H`` = hidden).

        Returns:
            Block output of shape ``[S, H]``.
        """
        self.backend_states.append(_sdp_backend_state())
        q, k, v = (t.unsqueeze(0).unsqueeze(0) for t in self.attn(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v).squeeze(0).squeeze(0)
        return x + self.mlp(out)


class _DropoutVisionBlock(nn.Module):
    """Minimal vision block with dropout, exercising RNG-state handling across recompute."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(_D, _D)
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Linear(_D, _D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention proxy + dropout + MLP to patch embeddings of shape ``[S, H]``."""
        return x + self.mlp(self.dropout(self.attn(x)))


class _VisionModel(nn.Module):
    def __init__(self, num_blocks: int = 1, block_cls: type[nn.Module] = _RecordingVisionBlock):
        super().__init__()
        self.blocks = nn.ModuleList(block_cls() for _ in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all vision blocks to patch embeddings of shape ``[S, H]``, returning ``[S, H]``."""
        for block in self.blocks:
            x = block(x)
        return x


def test_pinned_sdpa_context_fn_yields_identical_backend_sets_for_forward_and_recompute():
    """Forward and recompute contexts must enable the exact same SDPA backend set."""
    for backends, expected in (
        (_DEFAULT_PINNED_BACKENDS, (False, True, True)),
        ([SDPBackend.MATH], (True, False, False)),
    ):
        forward_ctx, recompute_ctx = ac._make_pinned_sdpa_context_fn(backends)()

        states = []
        for ctx in (forward_ctx, recompute_ctx):
            with ctx:
                states.append(_sdp_backend_state())
        assert states[0] == states[1] == expected


def test_apply_vision_block_checkpointing_recomputes_under_pinned_backends_and_matches_reference():
    model = _VisionModel()
    reference = copy.deepcopy(model)
    block = model.blocks[0]

    ac.apply_vision_block_checkpointing(model, [block])
    assert isinstance(model.blocks[0], CheckpointWrapper)
    assert ac.unwrap_checkpoint_wrapper(model.blocks[0]) is block

    x = torch.randn(4, _D)
    out = model(x)
    out.sum().backward()

    # Non-reentrant checkpointing runs the block forward twice (forward + backward
    # recompute); both runs must see the same default pinned backend set
    # (flash + mem-efficient enabled, math disabled).
    assert len(block.backend_states) == 2
    assert all(state == (False, True, True) for state in block.backend_states)

    # Run the reference under the same pin so the comparison is kernel-identical
    # (different SDPA backends differ from each other by float noise).
    ref_ctx, _ = ac._make_pinned_sdpa_context_fn(_DEFAULT_PINNED_BACKENDS)()
    with ref_ctx:
        ref_out = reference(x)
        ref_out.sum().backward()
    assert torch.allclose(out, ref_out)
    for got, want in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(got.grad, want.grad)


def test_apply_vision_block_checkpointing_math_backend_escape_hatch():
    """``backends=[MATH]`` must pin MATH on both the checkpoint forward and its recompute."""
    model = _VisionModel()
    block = model.blocks[0]

    ac.apply_vision_block_checkpointing(model, [block], backends=[SDPBackend.MATH])
    assert isinstance(model.blocks[0], CheckpointWrapper)

    out = model(torch.randn(4, _D))
    out.sum().backward()

    assert len(block.backend_states) == 2
    assert all(state == (True, False, False) for state in block.backend_states)
    assert block.attn.weight.grad is not None


def test_apply_vision_block_checkpointing_preserves_dropout_rng_on_recompute():
    """Backward recompute must redraw the forward's dropout mask, or gradients silently corrupt."""
    model = _VisionModel(block_cls=_DropoutVisionBlock)
    reference = copy.deepcopy(model)
    model.train()
    reference.train()

    ac.apply_vision_block_checkpointing(model, list(model.blocks))
    assert isinstance(model.blocks[0], CheckpointWrapper)

    x = torch.randn(64, _D)
    torch.manual_seed(1234)
    out = model(x)
    out.sum().backward()

    torch.manual_seed(1234)
    ref_out = reference(x)
    ref_out.sum().backward()

    # Same seed -> same forward mask; grads only match if the recompute (which runs
    # after the global RNG has advanced) restores the checkpoint-time RNG state.
    assert torch.allclose(out, ref_out)
    for got, want in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(got.grad, want.grad)
