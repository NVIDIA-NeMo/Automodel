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

"""CPU contract coverage through the production TE factory/callable."""

import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Callable, Unpack
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.attention.utils import _TEAttentionKwargs, initialize_attn_module_and_func

pytestmark = pytest.mark.cpu_only


class _RecordingAttention(torch.nn.Module):
    def __init__(self, **kwargs: int | float | str | None) -> None:
        super().__init__()
        self.qkv_format = kwargs["qkv_format"]
        self.cp_group = None
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Unpack[_TEAttentionKwargs]
    ) -> torch.Tensor:
        """Record metadata and preserve autograd for a CPU stand-in.

        Args:
            q: Query [tokens, heads, head_dim] or [batch, sequence, heads, head_dim].
            k: Key with q's layout and KV heads.
            v: Value with k's layout.
            **kwargs: Logical/physical offsets [documents + 1], or a padding mask
                [batch, 1, 1, sequence], plus scalar TE options.

        Returns:
            Weighted q with the same shape; numerical attention is GPU-tested.
        """
        self.calls.append(kwargs)
        return q * self.weight


_Factory = tuple[_RecordingAttention, Callable[..., torch.Tensor], SimpleNamespace]


@pytest.fixture
def factory() -> _Factory:
    state = SimpleNamespace(enabled=False)
    modules = {
        "transformer_engine.pytorch.attention": SimpleNamespace(DotProductAttention=_RecordingAttention),
        "transformer_engine.pytorch.quantization": SimpleNamespace(
            FP8GlobalStateManager=SimpleNamespace(is_fp8_enabled=lambda: state.enabled)
        ),
    }
    with patch.dict(sys.modules, modules):
        module, call = initialize_attn_module_and_func("te", 2, 8, 8, 8**-0.5, qkv_format="thd")
    return module, call, state


@pytest.mark.parametrize("documents,capacity", [(1, 128), (127, 128), (128, 128), (129, 256), (255, 256), (256, 256)])
def test_metadata_preserves_boundaries_and_physical_offsets(factory: _Factory, documents: int, capacity: int) -> None:
    module, call, _ = factory
    q = torch.ones(documents * 4, 2, 8, dtype=torch.bfloat16)
    original = {
        "cu_seqlens_q": torch.arange(documents + 1, dtype=torch.int32) * 2,
        "cu_seqlens_kv": torch.arange(documents + 1, dtype=torch.int32) * 3,
        "cu_seqlens_q_padded": torch.arange(documents + 1, dtype=torch.int32) * 4,
        "cu_seqlens_kv_padded": torch.arange(documents + 1, dtype=torch.int32) * 5,
    }
    snapshots = {key: value.clone() for key, value in original.items()}
    call(q, q, q, **original, pad_between_seqs=True, max_seqlen_q=2, max_seqlen_kv=3)
    recorded = module.calls[-1]
    for key, value in original.items():
        torch.testing.assert_close(value, snapshots[key])
        assert recorded[key].shape == (capacity + 1,)
        torch.testing.assert_close(recorded[key][: documents + 1], value)
        assert torch.all(recorded[key][documents:] == value[-1])
    assert recorded["pad_between_seqs"] is True
    assert recorded["max_seqlen_kv"] == 3


@pytest.mark.parametrize("guard", ["cp", "dropout", "quantization", "float32", "bshd", "bias", "mask", "empty"])
def test_unsupported_modes_preserve_original_metadata(factory: _Factory, guard: str) -> None:
    module, call, state = factory
    q = torch.ones(8, 2, 8, dtype=torch.bfloat16)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    extra = {}
    if guard == "cp":
        module.cp_group = [None]
    elif guard == "dropout":
        module.attention_dropout = 0.1
    elif guard == "quantization":
        state.enabled = True
    elif guard == "float32":
        q = q.float()
    elif guard == "bshd":
        extra["qkv_format"] = "bshd"
    elif guard == "bias":
        extra["core_attention_bias"] = torch.zeros(2, 2, 5, 5)
    elif guard == "mask":
        extra["attention_mask"] = torch.zeros(1, 1, 1, 8, dtype=torch.bool)
    elif guard == "empty":
        q = q[:0]
        cu = cu[:1]
    call(q, q, q, cu_seqlens_q=cu, cu_seqlens_kv=cu, **extra)
    assert module.calls[-1]["cu_seqlens_q"] is cu
    assert module.calls[-1]["cu_seqlens_kv"] is cu


def test_logical_physical_aliases_are_preserved(factory: _Factory) -> None:
    module, call, _ = factory
    q = torch.ones(8, 2, 8, dtype=torch.bfloat16)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    call(q, q, q, cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu)
    recorded = module.calls[-1]
    assert recorded["cu_seqlens_q"] is recorded["cu_seqlens_q_padded"]
    assert recorded["cu_seqlens_kv"] is recorded["cu_seqlens_kv_padded"]
    assert "pad_between_seqs" not in recorded
    torch.testing.assert_close(cu, torch.tensor([0, 3, 8], dtype=torch.int32))


def test_capacity_is_repeatable_after_larger_forward_and_deepcopy(factory: _Factory) -> None:
    module, call, _ = factory
    q = torch.ones(8, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    for count in (2, 129, 2):
        cu = torch.arange(count + 1, dtype=torch.int32)
        call(q, q, q, cu_seqlens_q=cu, cu_seqlens_kv=cu).sum().backward()
    assert [row["cu_seqlens_q"].numel() for row in module.calls] == [129, 257, 129]
    clone_module, clone_call = deepcopy((module, call))
    clone_call(q, q, q)
    assert len(module.calls) == 3
    assert len(clone_module.calls) == 4
    assert list(module.state_dict()) == ["weight"]


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[0, 2, 8]], dtype=torch.int32),
        torch.tensor([0, 2, 8], dtype=torch.int64),
    ],
)
def test_rejects_inconsistent_logical_or_physical_arrays(factory: _Factory, bad: torch.Tensor) -> None:
    """Reject invalid physical offsets before invoking TE.

    Args:
        factory: CPU stand-in and actual production callable.
        bad: Invalid cumulative offsets [2], [1, 3], or int64 [3].
    """
    module, call, _ = factory
    q = torch.ones(8, 2, 8, dtype=torch.bfloat16)
    cu = torch.tensor([0, 2, 8], dtype=torch.int32)
    with pytest.raises(ValueError, match="int32 vectors with matching document counts"):
        call(q, q, q, cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_kv_padded=bad)
    assert not module.calls
