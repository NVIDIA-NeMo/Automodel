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

"""Native THD hooks override generic Magi only with explicit backend ownership."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel import utils
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder, ShardLayout


class _Mesh(dict):
    mesh_dim_names = ("cp",)


@pytest.mark.parametrize("owns_magi", [False, True])
def test_explicit_magi_ownership_preserves_generic_thd_fallback(
    owns_magi: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_calls = []
    generic_calls = []
    cp_mesh = SimpleNamespace(size=lambda: 2)
    mesh = _Mesh(cp=cp_mesh)

    def generic(cp, batch, **kwargs):
        """Use first-half [1,4] IDs as a visibly distinct generic ownership map.

        Args:
            cp: Runtime context-parallel mesh.
            batch: Global IDs [1,8] and qkv_format metadata.
            **kwargs: Existing generic Magi sharder controls.

        Returns:
            Batch with local IDs [1,4] and global indices [4].
        """
        generic_calls.append(True)
        return {**batch, "input_ids": batch["input_ids"][:, :4]}, torch.arange(4)

    def native(cp, tp, batch, **kwargs):
        """Use alternating global IDs [1,4] to expose model-owned token dispatch.

        Args:
            cp: Runtime context-parallel mesh.
            tp: Unused tensor-parallel mesh.
            batch: Global IDs [1,8] and qkv_format metadata.
            **kwargs: Existing shared sharder controls.

        Returns:
            Null context, local IDs [1,4] and explicit global coordinates [4].
        """
        return (
            contextlib.nullcontext,
            {**batch, "input_ids": batch["input_ids"][:, ::2]},
            ShardLayout(local_token_global_indices=torch.arange(0, 8, 2), padded_seq_len=8),
        )

    def hook(batch, *, num_chunks):
        """Return the native sharder for the global [1,8] token batch."""
        native_calls.append(True)
        return {"cp_sharder": ContextParallelSharder(shard_batch=native)}

    state = SimpleNamespace(enabled=True, domain="llm", cp_size=2, make_cp_batch=generic)
    monkeypatch.setattr(utils, "_magi_state_from_model", lambda model, device_mesh: state)
    model = SimpleNamespace(
        supports_thd=True,
        _packed_cp_attn_backends=("te", "magi") if owns_magi else ("te",),
        prepare_model_inputs_for_cp=hook,
    )
    batch = {"input_ids": torch.arange(8)[None], "qkv_format": "thd"}
    sharder = ContextParallelSharder(model, mesh, batch)
    _, result = sharder.shard(batch)
    assert native_calls == ([True] if owns_magi else [])
    assert generic_calls == ([] if owns_magi else [True])
    torch.testing.assert_close(
        result["input_ids"], torch.tensor([[0, 2, 4, 6]]) if owns_magi else torch.tensor([[0, 1, 2, 3]])
    )
