# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Block-diagonal (per-document) varlen context parallelism for packed sequences.

Packed-sequence training concatenates many documents into one long sequence; a
correct attention mask is block-causal per document. The stock DTensor
``context_parallel`` path assumes a single causal document, so packed VLM/LLM
batches need a CP implementation that reshards by contiguous chunks and rebuilds
per-document masking on every rank.

This package provides that implementation, split by responsibility:

- :mod:`.state`   -- runtime knob normalization + activation-checkpoint-safe step state.
- :mod:`.kernels` -- dense-mask and varlen (FlashAttention / TransformerEngine) kernels.
- :mod:`.exchange` -- differentiable K/V collectives (all-gather, left-halo, all-to-all-v).
- :mod:`.runtime` -- the SDPA entry point and collective-safe path selection.
- :mod:`.batch`   -- batch padding/sharding and the per-step train context.
- :mod:`.packed`  -- the cp_size==1 packed-sequence varlen SDPA hook.

Integration follows the model-owned CP convention of
:func:`nemo_automodel.components.distributed.cp_utils.make_cp_batch_and_ctx`: a model
attaches :func:`make_cp_blockdiag_batch_and_ctx` to the batch as ``_cp_make_batch_fn``
and routes its softmax attention through :func:`cp_blockdiag_sdpa` while the returned
context is active.
"""

from nemo_automodel.components.distributed.blockdiag_cp.batch import make_cp_blockdiag_batch_and_ctx
from nemo_automodel.components.distributed.blockdiag_cp.kernels import precompute_blockdiag_varlen_meta
from nemo_automodel.components.distributed.blockdiag_cp.packed import (
    cp1_packed_varlen_backend,
    disable_cp1_packed_varlen,
    enable_cp1_packed_varlen,
)
from nemo_automodel.components.distributed.blockdiag_cp.runtime import cp_blockdiag_sdpa
from nemo_automodel.components.distributed.blockdiag_cp.state import (
    configure_cp_varlen,
    cp_attn_fire_count,
    cp_varlen_runtime_config,
    normalize_attn_backend,
    normalize_kv_exchange,
    reset_cp_attn_fire_count,
)

__all__ = [
    "configure_cp_varlen",
    "cp_attn_fire_count",
    "cp_varlen_runtime_config",
    "cp_blockdiag_sdpa",
    "cp1_packed_varlen_backend",
    "disable_cp1_packed_varlen",
    "enable_cp1_packed_varlen",
    "make_cp_blockdiag_batch_and_ctx",
    "normalize_attn_backend",
    "normalize_kv_exchange",
    "precompute_blockdiag_varlen_meta",
    "reset_cp_attn_fire_count",
]
