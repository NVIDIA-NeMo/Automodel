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

import contextlib
import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, checkpoint_wrapper
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import CheckpointError

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

# (flash, mem_efficient, cudnn, math) -- the flag order of _sdp_backend_state.
_MATH_ONLY = (False, False, False, True)


def _sdp_backend_state() -> tuple[bool, bool, bool, bool]:
    """Return the global (flash, mem_efficient, cudnn, math) SDPA backend enablement flags."""
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )


class _RecordingSdpaAttention(nn.Module):
    """Minimal HF-style vision attention running real SDPA.

    Records the enabled SDPA backend flags on every forward call, so tests can
    assert which backend set the checkpoint forward and its backward-time
    recompute ran under.
    """

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(_D, 3 * _D)
        self.proj = nn.Linear(_D, _D)
        self.backend_states: list[tuple[bool, bool, bool, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SDPA over a flattened patch sequence.

        Args:
            x: Patch embeddings of shape ``[S, H]`` (``S`` = patches, ``H`` = hidden).

        Returns:
            Attention output of shape ``[S, H]``.
        """
        self.backend_states.append(_sdp_backend_state())
        q, k, v = (t.unsqueeze(0).unsqueeze(0) for t in self.qkv(x).chunk(3, dim=-1))
        return self.proj(F.scaled_dot_product_attention(q, k, v).squeeze(0).squeeze(0))


class _SdpaVisionBlock(nn.Module):
    """Minimal vision block with ``attn``/``mlp`` submodules for ``apply_submodule_checkpointing``."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.attn = _RecordingSdpaAttention()
        self.mlp = nn.Sequential(nn.Linear(_D, _D), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention + MLP to patch embeddings of shape ``[S, H]``, returning ``[S, H]``."""
        return x + self.mlp(self.attn(x))


def test_snapshot_context_fn_recompute_ctx_restores_captured_backend_set():
    """The recompute context must re-pin exactly the backend set captured at snapshot time."""
    with sdpa_kernel([SDPBackend.MATH]):
        forward_ctx, recompute_ctx = ac.sdpa_backend_snapshot_context_fn()
        with forward_ctx:
            # The forward context is a no-op: the ambient forcing stays in effect.
            assert _sdp_backend_state() == _MATH_ONLY

    # Simulate a divergent backward-time ambient state (the forward pin has exited).
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        with recompute_ctx:
            assert _sdp_backend_state() == _MATH_ONLY
        # The backward-time ambient state is restored once the recompute exits.
        assert _sdp_backend_state() == (True, True, False, False)


def test_submodule_checkpointing_with_snapshot_context_reruns_recompute_under_forward_backends():
    """Recompute must rerun under the forward-time SDPA backend set even after the pin exits."""
    block = _SdpaVisionBlock()
    ac.apply_submodule_checkpointing([block], has_kv_sharing=False, context_fn=ac.sdpa_backend_snapshot_context_fn)
    assert isinstance(block.attn, CheckpointWrapper)
    assert isinstance(block.mlp, CheckpointWrapper)

    attn = ac.unwrap_checkpoint_wrapper(block.attn)
    with sdpa_kernel([SDPBackend.MATH]):  # ambient forcing active only during the forward
        out = block(torch.randn(4, _D))
    out.sum().backward()  # recompute runs here, after the pin has exited

    # Non-reentrant checkpointing runs the attention forward twice (forward +
    # backward-time recompute). The snapshot is taken at checkpoint-region entry
    # (while the ambient pin is still active) and re-pinned for the recompute.
    assert attn.backend_states == [_MATH_ONLY, _MATH_ONLY]
    assert block.attn._checkpoint_wrapped_module.qkv.weight.grad is not None


def test_submodule_checkpointing_without_snapshot_context_recompute_sees_divergent_backends():
    """Without the snapshot context_fn, the recompute runs under whatever is ambient at backward time."""
    block = _SdpaVisionBlock()
    ac.apply_submodule_checkpointing([block], has_kv_sharing=False)

    attn = ac.unwrap_checkpoint_wrapper(block.attn)
    with sdpa_kernel([SDPBackend.MATH]):
        out = block(torch.randn(4, _D))
    # On fused-CUDA stacks the divergence trips the checkpoint determinism check;
    # on CPU both passes may dispatch the same kernel and only the recorded flags
    # diverge. Either way, the forward-time backend set is not preserved.
    with contextlib.suppress(CheckpointError):
        out.sum().backward()
    assert attn.backend_states[0] == _MATH_ONLY
    assert attn.backend_states[1] != _MATH_ONLY


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Backend-set divergence across checkpoint recompute requires fused CUDA SDPA backends",
)
def test_checkpointed_sdpa_replay_faults_without_snapshot_context_and_passes_with_it():
    """One-sided ambient forcing faults plain checkpoint replay; the snapshot context_fn restores parity."""
    device = torch.device("cuda")
    x = torch.randn(4, _D, device=device)

    # WITHOUT the snapshot context_fn: forward under an ambient MATH pin that has
    # exited by backward time -> the recompute dispatches a fused backend that
    # saves different tensors -> deterministic CheckpointError.
    plain = _SdpaVisionBlock().to(device)
    ac.apply_submodule_checkpointing([plain], has_kv_sharing=False)
    with sdpa_kernel([SDPBackend.MATH]):
        out = plain(x)
    with pytest.raises(CheckpointError):
        out.sum().backward()

    # WITH the snapshot context_fn: the same one-sided pin replays cleanly, with
    # gradient parity against an unwrapped reference run entirely under the pin.
    block = _SdpaVisionBlock().to(device)
    reference = copy.deepcopy(block)
    ac.apply_submodule_checkpointing([block], has_kv_sharing=False, context_fn=ac.sdpa_backend_snapshot_context_fn)
    with sdpa_kernel([SDPBackend.MATH]):
        out = block(x)
    out.sum().backward()

    with sdpa_kernel([SDPBackend.MATH]):
        ref_out = reference(x)
        ref_out.sum().backward()

    assert torch.allclose(out, ref_out)
    for got, want in zip(block.parameters(), reference.parameters()):
        assert torch.allclose(got.grad, want.grad)


def test_submodule_checkpointing_with_snapshot_context_preserves_dropout_rng_on_recompute():
    """Backward recompute must redraw the forward's dropout mask, or gradients silently corrupt."""
    model = _SdpaVisionBlock(dropout=0.5)
    reference = copy.deepcopy(model)
    model.train()
    reference.train()

    ac.apply_submodule_checkpointing([model], has_kv_sharing=False, context_fn=ac.sdpa_backend_snapshot_context_fn)

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
