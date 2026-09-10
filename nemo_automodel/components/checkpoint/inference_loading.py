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

"""Single-process loaders for sharded training checkpoints, for inference-only use.

Training under FSDP2 with ``save_consolidated: false`` (the default for every
non-final checkpoint save) writes per-rank shards instead of a single
HF-loadable directory. These helpers reconstruct the full, unsharded weights
directly from those shards into an already-instantiated ``nn.Module`` --
no offline consolidation step, and no live distributed job required (a
throwaway single-rank ``gloo`` process group is created on demand for the
``torch.distributed.checkpoint`` APIs, which require one to be initialized).

Used by both ``examples/diffusion/generate/generate.py`` and
``tools/diffusion/inference_dmd2_qwen_image.py`` to load a training
checkpoint straight into an inference pipeline's transformer.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def load_sharded_fsdp_checkpoint(
    transformer: torch.nn.Module, sharded_dir: str, torch_dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    """Load a sharded FSDP1 ``.distcp`` checkpoint into a transformer module.

    Creates a temporary ``gloo`` process group for single-GPU loading if
    ``torch.distributed`` is not already initialized.

    Args:
        transformer: The transformer module to load weights into. Its
            ``state_dict()`` keys/shapes must already match the checkpoint
            (e.g. constructed from the same base architecture/config).
        sharded_dir: Path to the directory containing ``.distcp`` shard files.
        torch_dtype: The dtype to cast the transformer to before loading.

    Returns:
        The unwrapped transformer module (``nn.Module``, unsharded, on
        ``cuda``) with the checkpoint's weights loaded.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load as dist_load
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.api import ShardedStateDictConfig

    init_dist = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        init_dist = True

    try:
        transformer.to(device="cuda", dtype=torch_dtype)
        fsdp_transformer = FSDP(transformer, use_orig_params=True)
        FSDP.set_state_dict_type(
            fsdp_transformer,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )
        model_state = fsdp_transformer.state_dict()
        dist_load(state_dict=model_state, storage_reader=FileSystemReader(sharded_dir))
        fsdp_transformer.load_state_dict(model_state)
        return fsdp_transformer.module
    finally:
        if init_dist:
            dist.destroy_process_group()


def load_sharded_hf_safetensors_checkpoint(
    transformer: torch.nn.Module, sharded_dir: str, torch_dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    """Load a NeMo-AutoModel sharded HF safetensors checkpoint into a transformer.

    Handles directories containing ``shard-XXXXX-model-XXXXX-of-XXXXX.safetensors``
    files -- one per FSDP rank, produced by a training run with
    ``save_consolidated: false``. Uses DCP's ``HuggingFaceStorageReader`` to
    gather all shards into the target state dict.

    Args:
        transformer: The transformer module to load weights into. Its
            ``state_dict()`` keys/shapes must already match the checkpoint
            (e.g. constructed from the same base architecture/config).
        sharded_dir: Path to the directory containing ``shard-*.safetensors``
            files.
        torch_dtype: The dtype to cast the transformer to before loading.

    Returns:
        The transformer module (on ``cuda``) with the merged state dict loaded.
    """
    from torch.distributed.checkpoint import load as dist_load

    # Prefer the upstream HF storage reader; fall back to NeMo's backport if
    # the torch version is too old to ship it.
    try:
        from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
    except ImportError:
        from nemo_automodel.components.checkpoint._backports.hf_storage import (
            _HuggingFaceStorageReader as HuggingFaceStorageReader,
        )

    init_dist = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        init_dist = True

    try:
        transformer.to(device="cuda", dtype=torch_dtype)
        state_dict = transformer.state_dict()
        dist_load(state_dict=state_dict, storage_reader=HuggingFaceStorageReader(path=sharded_dir))
        transformer.load_state_dict(state_dict, strict=True)
        return transformer
    finally:
        if init_dist:
            dist.destroy_process_group()
