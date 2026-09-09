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

"""Setup-time prewarm for Mamba SSD Triton autotuners."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import torch

from nemo_automodel.components.training._prewarm_utils import _resolve_cuda_device
from nemo_automodel.shared.import_utils import safe_import_from

logger = logging.getLogger(__name__)


@runtime_checkable
class _MambaSSDContextParallel(Protocol):
    """Structural type of the local Mamba geometry under context parallelism."""

    num_heads_local: int
    n_groups_local: int


@runtime_checkable
class _MambaSSDMixer(Protocol):
    """Structural type of a mixer backed by the Mamba SSD Triton operators."""

    num_heads: int
    head_dim: int
    ssm_state_size: int
    n_groups: int
    chunk_size: int
    cp: _MambaSSDContextParallel | None


_MambaSSDShape = tuple[int, int, int, int, int, torch.dtype]


def _collect_mamba_ssd_autotune_shapes(model_parts: list[torch.nn.Module]) -> dict[_MambaSSDShape, str]:
    """Discover the unique Mamba SSD kernel geometries in ``model_parts``.

    The pinned Mamba SSD Triton autotuners key their caches on subsets of the
    head count, head dimension, state size, group count, and chunk size. Batch
    size and sequence length affect launch-grid size and transient memory, but
    are not autotune keys.

    Args:
        model_parts: Model parts to scan for Mamba SSD mixer modules.

    Returns:
        Mapping of ``(heads, head_dim, state_size, groups, chunk_size, dtype)``
        to the qualified name of one module with that geometry.
    """
    shapes: dict[_MambaSSDShape, str] = {}
    for part in model_parts:
        for name, module in part.named_modules():
            if not isinstance(module, _MambaSSDMixer):
                continue

            num_heads = int(module.num_heads)
            n_groups = int(module.n_groups)
            if module.cp is not None:
                if not isinstance(module.cp, _MambaSSDContextParallel):
                    logger.warning("Skipping Mamba SSD prewarm for %s: unrecognized context-parallel geometry.", name)
                    continue
                num_heads = int(module.cp.num_heads_local)
                n_groups = int(module.cp.n_groups_local)

            dtype = torch.bfloat16
            for param in module.parameters(recurse=True):
                if param.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    dtype = param.dtype
                    break
            shape = (
                num_heads,
                int(module.head_dim),
                int(module.ssm_state_size),
                n_groups,
                int(module.chunk_size),
                dtype,
            )
            shapes.setdefault(shape, name)
    return shapes


def _prewarm_mamba_ssd_autotune(
    model_parts: list[torch.nn.Module],
    device: torch.device | int | str | None,
) -> bool:
    """Pre-populate Mamba SSD Triton autotune caches before real activations exist.

    Each unique mixer geometry runs through two chunks at batch size one. This
    reaches the public SSD forward/backward call graph, including inter-chunk
    state passing, while matching every current autotune key and avoiding the
    activation footprint of the real batch and sequence length.

    Args:
        model_parts: Model parts to scan for Mamba SSD mixer modules.
        device: Target CUDA device (skipped when None or not CUDA).

    Returns:
        True if the warmup ran for at least one shape.
    """
    device = _resolve_cuda_device(device, "Mamba SSD autotune")
    if device is None:
        return False

    shapes = _collect_mamba_ssd_autotune_shapes(model_parts)
    if not shapes:
        logger.warning("Mamba SSD autotune prewarm enabled but no Mamba SSD mixer modules were found.")
        return False

    logger.info(
        "Prewarming Mamba SSD autotune caches for %d unique shape(s): %s",
        len(shapes),
        sorted((*shape[:-1], str(shape[-1]), name) for shape, name in shapes.items()),
    )
    return _prewarm_mamba_ssd_end_to_end(shapes, device)


def _prewarm_mamba_ssd_end_to_end(shapes: dict[_MambaSSDShape, str], device: torch.device) -> bool:
    """Run a tiny public Mamba SSD forward/backward per unique kernel geometry.

    Synthetic tensors match the training operator contract: ``x`` has layout
    ``[batch, sequence, heads, head_dim]``; ``dt`` is
    ``[batch, sequence, heads]``; ``B`` and ``C`` are
    ``[batch, sequence, groups, state_size]``; ``A`` and ``dt_bias`` are
    ``[heads]``; and ``D`` is ``[heads]``. The floating-point activation
    tensors use the mixer's compute dtype, while ``A``, ``D``, and ``dt_bias``
    use float32 as in the Nemotron Mamba training path.

    Args:
        shapes: Unique Mamba SSD kernel geometries and representative module
            names, as returned by :func:`_collect_mamba_ssd_autotune_shapes`.
        device: Target device. Production callers pass CUDA; CPU is supported
            so the operator contract can be tested with a stub.

    Returns:
        True if the warmup ran for at least one shape.
    """
    has_mamba_ssd, mamba_chunk_scan_combined = safe_import_from(
        "mamba_ssm.ops.triton.ssd_combined", "mamba_chunk_scan_combined"
    )
    if not has_mamba_ssd:
        logger.info("Skipping Mamba SSD end-to-end prewarm: mamba_ssm SSD operators are not importable.")
        return False

    ran = False
    rng_devices = []
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        rng_devices = [device_index]

    with torch.random.fork_rng(devices=rng_devices), torch.enable_grad():
        for (num_heads, head_dim, state_size, n_groups, chunk_size, dtype), module_name in shapes.items():
            seq_len = 2 * chunk_size
            x = dt = a = b = c = d = dt_bias = seq_idx = out = loss = None
            try:
                x = torch.randn((1, seq_len, num_heads, head_dim), device=device, dtype=dtype, requires_grad=True)
                dt = torch.zeros((1, seq_len, num_heads), device=device, dtype=dtype, requires_grad=True)
                a = torch.full((num_heads,), -1.0, device=device, dtype=torch.float32, requires_grad=True)
                b = torch.randn((1, seq_len, n_groups, state_size), device=device, dtype=dtype, requires_grad=True)
                c = torch.randn((1, seq_len, n_groups, state_size), device=device, dtype=dtype, requires_grad=True)
                d = torch.ones((num_heads,), device=device, dtype=torch.float32, requires_grad=True)
                dt_bias = torch.zeros((num_heads,), device=device, dtype=torch.float32, requires_grad=True)
                seq_idx = torch.zeros((1, seq_len), device=device, dtype=torch.int32)
                out = mamba_chunk_scan_combined(
                    x,
                    dt,
                    a,
                    b,
                    c,
                    chunk_size,
                    D=d,
                    z=None,
                    dt_bias=dt_bias,
                    seq_idx=seq_idx,
                    dt_softplus=True,
                )
                loss = out.float().sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                ran = True
            except Exception:
                logger.exception("Mamba SSD end-to-end prewarm failed for %s; continuing.", module_name)
            finally:
                del x, dt, a, b, c, d, dt_bias, seq_idx, out, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    logger.info("Finished Mamba SSD autotune prewarm.")
    return ran
