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

"""Tests for the interleaved Wan-Animate-2 traversal.

The traversal itself needs the upstream transformer, but everything that decides
*whether* and *how* it is installed is plain Python and is what actually broke in
practice: a cache handed across a call boundary, a method installed on a wrapper
rather than on the block underneath it, and an unguarded optional import.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.wan_animate2.interleaved import (
    _block_forward_origin,
    _precompile_attention_without_cudagraphs,
    _unwrap_module,
    install_forward_origin,
    supports_interleaved_forward,
)


class _Block(nn.Module):
    """A block exposing the two per-block passes the traversal calls."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []

    def forward_ref(self, x_ref: torch.Tensor, index: int, k_cache: dict, v_cache: dict, **kwargs: Any):
        """Record the index and write this block's keys and values."""
        self.seen.append(index)
        k_cache[index] = x_ref * 2
        v_cache[index] = x_ref * 3
        return x_ref

    def forward_gen(self, x: torch.Tensor, index: int, k_cache: dict, v_cache: dict, **kwargs: Any):
        """Read back this block's keys and values, failing loudly if absent."""
        return x + k_cache[index] + v_cache[index]


class _Wrapper(nn.Module):
    """Stand-in for an activation-checkpoint wrapper around a block."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._checkpoint_wrapped_module = inner


class _Transformer(nn.Module):
    """A transformer exposing everything the traversal depends on."""

    def __init__(self, *, num_blocks: int = 3, wrap: bool = False) -> None:
        super().__init__()
        blocks = [_Block() for _ in range(num_blocks)]
        self.blocks = nn.ModuleList(_Wrapper(b) if wrap else b for b in blocks)
        self.patch_embedding = nn.Conv3d(4, 4, kernel_size=1)
        self.time_embedding = nn.Identity()
        self.time_projection = nn.Identity()
        self.text_embedding = nn.Identity()
        self.head = nn.Identity()
        self.block_masks: dict[Any, Any] = {}

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Present the upstream method name."""
        return x

    def create_mask(self, origin_len: int, origin_area: list[int], device: torch.device) -> None:
        """Present the upstream method name."""
        return None


def test_block_traversal_round_trips_its_own_cache() -> None:
    """The generation pass reads exactly what the reference pass just wrote."""
    block = _Block()
    x = torch.zeros(2, 2)
    x_ref = torch.ones(2, 2)

    out, out_ref = _block_forward_origin(block, x, x_ref, {}, {})

    # forward_gen reads index 0, so a cache that did not survive the call would
    # raise KeyError rather than return.
    torch.testing.assert_close(out, x + x_ref * 2 + x_ref * 3)
    torch.testing.assert_close(out_ref, x_ref)
    assert block.seen == [0]


def test_block_traversal_uses_a_fresh_cache_each_call() -> None:
    """Nothing carries over between calls, so no state can leak across steps."""
    block = _Block()
    x, x_ref = torch.zeros(2, 2), torch.ones(2, 2)

    _block_forward_origin(block, x, x_ref, {}, {})
    _block_forward_origin(block, x, x_ref, {}, {})

    # Both calls addressed index 0 of their own cache, never index 1.
    assert block.seen == [0, 0]


def test_unwrap_module_reaches_through_checkpoint_wrappers() -> None:
    """Wrapped blocks resolve to the block underneath."""
    inner = _Block()
    assert _unwrap_module(inner) is inner
    assert _unwrap_module(_Wrapper(inner)) is inner
    assert _unwrap_module(_Wrapper(_Wrapper(inner))) is inner


def test_unwrap_module_stops_on_a_wrapper_cycle() -> None:
    """A self-referential wrapper terminates instead of looping forever."""

    class _Cyclic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._checkpoint_wrapped_module = self

    cyclic = _Cyclic()
    assert _unwrap_module(cyclic) is cyclic


def test_supports_interleaved_forward_accepts_a_complete_transformer() -> None:
    """A transformer carrying every dependency is accepted."""
    assert supports_interleaved_forward(_Transformer()) is True


def test_supports_interleaved_forward_rejects_a_missing_attribute() -> None:
    """Dropping any dependency is refused rather than failing mid-traversal."""
    model = _Transformer()
    del model.patch_embedding
    assert supports_interleaved_forward(model) is False


def test_supports_interleaved_forward_rejects_blocks_without_the_two_passes() -> None:
    """Blocks lacking the per-block passes are refused."""
    model = _Transformer()
    model.blocks = nn.ModuleList([nn.Identity()])
    assert supports_interleaved_forward(model) is False


def test_supports_interleaved_forward_rejects_an_empty_block_list() -> None:
    """A transformer with no blocks is refused."""
    model = _Transformer()
    model.blocks = nn.ModuleList()
    assert supports_interleaved_forward(model) is False


def test_install_forward_origin_adds_the_method_to_model_and_block() -> None:
    """Both levels gain the traversal, so both are entered through __call__."""

    class _FreshTransformer(_Transformer):
        pass

    class _FreshBlock(_Block):
        pass

    model = _FreshTransformer()
    model.blocks = nn.ModuleList([_FreshBlock() for _ in range(2)])

    assert install_forward_origin(model) is True
    assert hasattr(type(model), "forward_origin")
    assert hasattr(type(model.blocks[0]), "forward_origin")


def test_install_forward_origin_targets_the_block_not_its_wrapper() -> None:
    """The method lands on the block underneath an activation-checkpoint wrapper.

    Installing on the wrapper's class would leave the real block without it, and
    the traversal would fail with AttributeError on the first block.
    """

    class _FreshBlock(_Block):
        pass

    class _FreshTransformer(_Transformer):
        pass

    model = _FreshTransformer()
    model.blocks = nn.ModuleList([_Wrapper(_FreshBlock()) for _ in range(2)])

    assert install_forward_origin(model) is True
    assert hasattr(_FreshBlock, "forward_origin")


def test_install_forward_origin_refuses_an_unsupported_transformer() -> None:
    """An incomplete transformer is reported rather than patched."""
    model = _Transformer()
    del model.head
    assert install_forward_origin(model) is False


def test_precompiling_attention_is_a_no_op_without_the_integration() -> None:
    """A diffusers build lacking Wan-Animate-2 does not raise here.

    The model load reports missing support with a useful message; this helper
    must not pre-empt it with an ImportError of its own.
    """
    _precompile_attention_without_cudagraphs()
