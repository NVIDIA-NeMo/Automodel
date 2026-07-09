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
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, checkpoint_wrapper

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


class _RecordingVisionBlock(nn.Module):
    """Minimal HF-style vision block (``attn``, no ``self_attn``) that runs real SDPA.

    Records the enabled SDPA backends (``math``, ``flash``, ``mem_efficient``) on
    every forward call, so tests can assert both the checkpoint forward and its
    recompute ran with the MATH backend pinned.
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
        self.backend_states.append(
            (
                torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
            )
        )
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


def test_vision_math_sdpa_context_fn_pins_math_on_forward_and_recompute():
    forward_ctx, recompute_ctx = ac._vision_math_sdpa_context_fn()

    for ctx in (forward_ctx, recompute_ctx):
        with ctx:
            assert torch.backends.cuda.math_sdp_enabled()
            assert not torch.backends.cuda.flash_sdp_enabled()
            assert not torch.backends.cuda.mem_efficient_sdp_enabled()


def test_apply_vision_block_checkpointing_recomputes_under_math_and_matches_reference():
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
    # recompute); both runs must see only the MATH SDPA backend enabled.
    assert len(block.backend_states) == 2
    assert all(state == (True, False, False) for state in block.backend_states)

    # Run the reference under the same MATH pin so the comparison is
    # kernel-identical (the fused SDPA kernels differ from MATH by float noise).
    ref_ctx, _ = ac._vision_math_sdpa_context_fn()
    with ref_ctx:
        ref_out = reference(x)
        ref_out.sum().backward()
    assert torch.allclose(out, ref_out)
    for got, want in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(got.grad, want.grad)


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


def test_apply_vision_block_checkpointing_flash_attention_config_wraps_mlp_only():
    model = _VisionModel()
    model.config = SimpleNamespace(vision_config=SimpleNamespace(_attn_implementation="flash_attention_2"))

    ac.apply_vision_block_checkpointing(model, list(model.blocks))

    assert not isinstance(model.blocks[0], CheckpointWrapper)
    assert isinstance(model.blocks[0].mlp, CheckpointWrapper)
    assert not isinstance(model.blocks[0].attn, CheckpointWrapper)


def test_apply_vision_block_checkpointing_detects_flash_attention_on_blocks():
    model = _VisionModel()
    model.blocks[0].attn.config = SimpleNamespace(_attn_implementation="flash_attention_3")

    ac.apply_vision_block_checkpointing(model, list(model.blocks))

    assert not isinstance(model.blocks[0], CheckpointWrapper)
    assert isinstance(model.blocks[0].mlp, CheckpointWrapper)


def test_apply_vision_block_checkpointing_detects_kernel_hub_flash_attention():
    """transformers-5 kernel-hub identifiers must also route to the MLP-only path."""
    model = _VisionModel()
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(_attn_implementation="kernels-community/vllm-flash-attn3")
    )

    ac.apply_vision_block_checkpointing(model, list(model.blocks))

    assert not isinstance(model.blocks[0], CheckpointWrapper)
    assert isinstance(model.blocks[0].mlp, CheckpointWrapper)
    assert not isinstance(model.blocks[0].attn, CheckpointWrapper)
