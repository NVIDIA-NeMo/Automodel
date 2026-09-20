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

"""Opt-in fused kernels for the Kimi-K3 KDA chunked delta rule (``KimiK3TextConfig.kda_chunk_impl = "fused"``).

Forward: a fused CUDA kernel (``csrc/chunk_kda_fwd.cu``, JIT-built once per build directory through
``torch.utils.cpp_extension.load`` and imported straight from the built ``.so`` afterwards; build dir
``$NEMO_KDA_FUSED_BUILD_DIR`` or ``~/.cache/nemo_automodel/kda_fused``, keyed by sources, torch version and device arch).
Backward: a Triton kernel (``chunk_kda_bwd_triton.py``) that recomputes every intermediate from the raw forward inputs,
so the autograd function saves only q, k, v, g, beta and cu_seqlens — no chunk states or WY factors.

Both kernels are specialised to the K3 production call: head dims K = V = 128, packed layout (batch 1 with int32
``cu_seqlens``; a dense batch of equal-length sequences is viewed as B packed documents), q/k l2-normalised inside the kernel, a bounded log gate (``safe_gate``), no initial or final state,
no context parallelism, ``kda_mode = "chunk"``. ``fused_kda_unsupported_reason`` names the first violated condition;
the model raises instead of silently falling back, so a benchmark that asks for the fused kernels either runs them or
stops. FLA's ``chunk_kda`` remains the default (``kda_chunk_impl = "fla"``); ``kda_transpose_state_layout`` and
``kda_disable_recompute`` are FLA-only knobs and have no effect under ``"fused"``.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")
_FWD_SOURCES = [os.path.join(_CSRC, "chunk_kda_fwd_binding.cpp"), os.path.join(_CSRC, "chunk_kda_fwd.cu")]
_FWD_HEADERS = [os.path.join(_CSRC, "chunk_kda_fwd.h")]
_HEAD_DIM = 128
_EXT_NAME = "nemo_fused_chunk_kda_fwd"
# A single-arch build of the forward took ~15 s on GB200 (first-call 17.6 s including the Triton compiles); a lock
# older than 40x that cannot belong to a live build. The build is pinned to the current device's arch (below), so the
# multi-arch TORCH_CUDA_ARCH_LIST of NGC containers cannot stretch a live build past this threshold.
_STALE_LOCK_SECONDS = 600.0

_ext_lock = threading.Lock()
_fwd_ext: Any = None


def _build_dir() -> str:
    """One build directory per (sources, torch, device arch); shared file systems are fine (torch takes a file lock)."""
    root = os.environ.get("NEMO_KDA_FUSED_BUILD_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "nemo_automodel", "kda_fused"
    )
    h = hashlib.sha1()
    for p in _FWD_SOURCES + _FWD_HEADERS:
        with open(p, "rb") as f:
            h.update(f.read())
    h.update(torch.__version__.encode())
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    h.update(f"sm_{cap[0]}{cap[1]}".encode())
    d = os.path.join(root, h.hexdigest()[:12])
    os.makedirs(d, exist_ok=True)
    return d


def _lock_age(build_dir: str) -> float | None:
    try:
        import time

        return time.time() - os.path.getmtime(os.path.join(build_dir, "lock"))
    except OSError:
        return None


def _import_prebuilt(build_dir: str) -> Any | None:
    """Import the extension straight from its built ``.so`` when it exists and no build is in progress.

    ``torch.utils.cpp_extension.load`` takes a file lock in the build directory on EVERY call, even when nothing needs
    rebuilding, and a process killed while holding it (a cancelled job) leaves a lock that hangs every later caller
    for good. Importing the finished artifact directly needs no lock; only a genuine build goes through ``load``. A
    fresh ``lock`` means another process may still be linking the ``.so``: return None and let the caller wait in
    ``load`` instead of importing a half-written file.
    """
    so = os.path.join(build_dir, f"{_EXT_NAME}.so")
    if not os.path.isfile(so):
        return None
    age = _lock_age(build_dir)
    if age is not None and age < _STALE_LOCK_SECONDS:
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(_EXT_NAME, so)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_stale_lock(build_dir: str) -> None:
    """Drop a build lock nobody can be holding (older than the longest build by far), with a warning."""
    import warnings

    lock = os.path.join(build_dir, "lock")
    age = _lock_age(build_dir)
    if age is not None and age > _STALE_LOCK_SECONDS:
        warnings.warn(
            f"kda_fused: removing a {age:.0f}s-old cpp_extension build lock at {lock} (a previous build was interrupted)",
            stacklevel=2,
        )
        try:
            os.remove(lock)
        except OSError:
            pass


def _forward_ext() -> Any:
    """Return the CUDA forward extension: prebuilt .so if present, else JIT-build it once (file-locked by torch)."""
    global _fwd_ext
    if _fwd_ext is None:
        with _ext_lock:
            if _fwd_ext is None:
                build_dir = _build_dir()
                _fwd_ext = _import_prebuilt(build_dir)
                if _fwd_ext is None:
                    from torch.utils.cpp_extension import load

                    _clear_stale_lock(build_dir)
                    cap = torch.cuda.get_device_capability()
                    cc = f"{cap[0]}{cap[1]}"
                    _fwd_ext = load(
                        name=_EXT_NAME,
                        sources=_FWD_SOURCES,
                        extra_include_paths=[_CSRC],
                        # one arch (the current device) — torch drops its own TORCH_CUDA_ARCH_LIST when a flag names one
                        extra_cuda_cflags=["-O3", "-lineinfo", "-gencode", f"arch=compute_{cc},code=sm_{cc}"],
                        build_directory=build_dir,
                        verbose=False,
                    )
    return _fwd_ext


def fused_kda_unsupported_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    *,
    mode: str = "chunk",
    cp_context: Any = None,
    use_qk_l2norm_in_kernel: bool = True,
    safe_gate: bool = True,
) -> str | None:
    """Return why this call cannot run on the fused kernels, or None when it can."""
    if mode != "chunk":
        return f"kda_mode must be 'chunk', got {mode!r}"
    if cp_context is not None:
        return "context parallelism (cp_context) is not supported"
    if not use_qk_l2norm_in_kernel:
        return "kda_use_qk_l2norm_in_kernel must be True (the kernels l2-normalise q and k themselves)"
    if not safe_gate:
        return "a bounded log gate (gate_lower_bound / safe_gate) is required"
    if q.dim() != 4:
        return f"expected q of shape [B, T, H, K], got {tuple(q.shape)}"
    if cu_seqlens is not None and q.shape[0] != 1:
        return f"packed layout (cu_seqlens given) needs a batch of 1, got batch {q.shape[0]}"
    if q.shape[-1] != _HEAD_DIM or k.shape[-1] != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        return f"head dims must be K = V = {_HEAD_DIM}, got K_q={q.shape[-1]} K_k={k.shape[-1]} V={v.shape[-1]}"
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return f"q, k, v must be bfloat16, got {q.dtype}, {k.dtype}, {v.dtype}"
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        return f"g and beta must be float32, got {g.dtype}, {beta.dtype}"
    if g.shape != q.shape:
        return f"g must have q's shape [B, T, H, K] = {tuple(q.shape)}, got {tuple(g.shape)}"
    if beta.shape != q.shape[:3]:
        return f"beta must have shape [B, T, H] = {tuple(q.shape[:3])}, got {tuple(beta.shape)}"
    if cu_seqlens is not None and (cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2):
        return f"cu_seqlens must be a 1-D tensor of at least two offsets, got {tuple(cu_seqlens.shape)}"
    return None


class _FusedChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu=None):  # noqa: D102
        q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
        cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        o = torch.empty_like(v)
        _forward_ext().run(q, k, v, g, beta, cu, o)
        ctx.save_for_backward(q, k, v, g, beta, cu)
        ctx.cu_seqlens_cpu = None if cu_seqlens_cpu is None else tuple(int(x) for x in cu_seqlens_cpu)
        return o

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        q, k, v, g, beta, cu = ctx.saved_tensors
        from nemo_automodel.components.models.kimi_k3.kda_fused.chunk_kda_bwd_triton import run as bwd_run

        do = do.to(v.dtype).contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dg, dbeta = torch.empty_like(g), torch.empty_like(beta)
        bwd_run(q, k, v, g, beta, cu, do, dq, dk, dv, dg, dbeta, cu_seqlens_cpu=ctx.cu_seqlens_cpu)
        return dq, dk, dv, dg, dbeta, None, None


def fused_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, None]:
    """Drop-in for FLA's ``chunk_kda(..., output_final_state=False)`` return convention: ``(o, None)``.

    Packed input (``cu_seqlens`` given, batch 1) goes straight to the kernels. Dense input (``cu_seqlens`` None,
    batch B of equal-length sequences — the fixed-length benchmark data) is viewed as one packed batch of B documents
    with ``cu_seqlens = [0, T, 2T, ..., BT]``: the recurrent state resets at every document boundary, so this is the
    same computation FLA performs on the batched layout, the views cost no copies, and the offsets are handed to the
    backward as host ints (no device->host sync, matching FLA's sync-free dense path). Packed input syncs once in the
    backward, as FLA's varlen path does.
    """
    reason = fused_kda_unsupported_reason(q, k, v, g, beta, cu_seqlens)
    if reason is not None:
        raise ValueError(f"fused_chunk_kda cannot run this call: {reason}")
    B, T = q.shape[0], q.shape[1]
    cu_seqlens_cpu = None
    if cu_seqlens is None:
        # dense batch: the document offsets follow from the shape, so the backward gets them without a host sync
        cu_seqlens_cpu = tuple(range(0, B * T + 1, T))
        cu_seqlens = torch.arange(0, B * T + 1, T, device=q.device, dtype=torch.int32)
        if B > 1:
            q, k, v, g = (t.reshape(1, B * T, *t.shape[2:]) for t in (q, k, v, g))
            beta = beta.reshape(1, B * T, beta.shape[2])
    o = _FusedChunkKDAFunction.apply(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu)
    if cu_seqlens_cpu is not None and B > 1:
        o = o.reshape(B, T, *o.shape[2:])
    return o, None


__all__ = ["fused_chunk_kda", "fused_kda_unsupported_reason"]
