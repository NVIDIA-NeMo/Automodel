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

"""Unit tests for CPAwareGatedDeltaNet (cp_linear_attn.py).

Tests cover:
  - _extract_local_positions: various tensor shapes and fallback behavior
  - _undo_attention_load_balancing / _redo_attention_load_balancing: correctness
  - _AllGatherConcatFn: forward/backward in a single-rank mock scenario
  - CPAwareGatedDeltaNet.forward: fast path delegation when CP is disabled
  - _conv1d_with_cp: boundary token exchange logic
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
    CPAwareGatedDeltaNet,
    _AllGatherConcatFn,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def text_config():
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        layer_types=["full_attention", "linear_attention"],
    )


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@pytest.fixture
def module(text_config, device):
    """Create a CPAwareGatedDeltaNet on device with no CP mesh."""
    m = CPAwareGatedDeltaNet(text_config, layer_idx=1)
    m = m.to(device)
    return m


# -- helpers for mocking dist.all_gather in a CP world_size=2 scenario -------

def _make_fake_all_gather(rank0_pos, rank1_pos, rank0_hidden, rank1_hidden, device):
    """Return a fake all_gather that fills gathered lists for a 2-rank CP setup."""

    def fake_all_gather(gathered, tensor, group=None):
        if tensor.ndim == 1:
            # position tensor (1-D)
            gathered[0].copy_(rank0_pos.to(device))
            gathered[1].copy_(rank1_pos.to(device))
        else:
            # hidden states (B, S, D)
            gathered[0].copy_(rank0_hidden.to(device) if tensor.shape == rank0_hidden.shape else tensor)
            gathered[1].copy_(rank1_hidden.to(device) if tensor.shape == rank1_hidden.shape else torch.randn_like(tensor))

    return fake_all_gather


def _patch_dist_for_cp(rank=0, world_size=2):
    """Context manager that patches dist calls for CP testing."""
    return (
        patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=world_size),
        patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=rank),
    )


# ============================================================================
# _extract_local_positions
# ============================================================================


class TestExtractLocalPositions:
    def test_1d_seq_index(self, module, device):
        seq_index = torch.tensor([3, 1, 0, 2], device=device)
        result = module._extract_local_positions(position_ids=None, seq_index=seq_index, seq_len=4)
        assert result is not None
        assert torch.equal(result, seq_index.long())

    def test_2d_position_ids(self, module, device):
        position_ids = torch.tensor([[0, 1, 2, 3]], device=device)
        result = module._extract_local_positions(position_ids=position_ids, seq_index=None, seq_len=4)
        assert result is not None
        assert torch.equal(result, position_ids[0].long())

    def test_3d_mrope_position_ids(self, module, device):
        # 3D mRoPE: [1, num_axes, seq_len]
        position_ids = torch.arange(4, device=device).unsqueeze(0).unsqueeze(0).expand(1, 3, 4)
        result = module._extract_local_positions(position_ids=position_ids, seq_index=None, seq_len=4)
        assert result is not None
        assert result.shape == (4,)

    def test_seq_index_preferred_over_position_ids(self, module, device):
        seq_index = torch.tensor([3, 1, 0, 2], device=device)
        position_ids = torch.tensor([[0, 1, 2, 3]], device=device)
        result = module._extract_local_positions(position_ids=position_ids, seq_index=seq_index, seq_len=4)
        # seq_index is checked first
        assert torch.equal(result, seq_index.long())

    def test_returns_none_when_both_none(self, module, device):
        result = module._extract_local_positions(position_ids=None, seq_index=None, seq_len=4)
        assert result is None

    def test_returns_none_when_length_mismatch(self, module, device):
        seq_index = torch.tensor([0, 1, 2], device=device)  # length 3 != seq_len 4
        result = module._extract_local_positions(position_ids=None, seq_index=seq_index, seq_len=4)
        assert result is None

    def test_4d_tensor_skipped(self, module, device):
        """Tensors with ndim > 3 should be ignored."""
        position_ids = torch.arange(4, device=device).reshape(1, 1, 1, 4)
        result = module._extract_local_positions(position_ids=position_ids, seq_index=None, seq_len=4)
        assert result is None


# ============================================================================
# _undo_attention_load_balancing
# ============================================================================


class TestUndoAttentionLoadBalancing:
    """Test load-balancing undo using mocked dist calls (simulating CP world_size=2)."""

    def test_reorders_to_dense(self, module, device):
        """Tokens in load-balanced order should be sorted to dense 0..S-1 order."""
        B, S_local, D = 1, 4, module.config.hidden_size
        hidden = torch.randn(B, S_local, D, device=device)
        positions = torch.tensor([0, 3, 4, 7], device=device, dtype=torch.long)

        rank1_positions = torch.tensor([1, 2, 5, 6], dtype=torch.long)
        rank1_hidden = torch.randn(B, S_local, D)

        fake_ag = _make_fake_all_gather(positions, rank1_positions, hidden, rank1_hidden, device)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_ag),
            *_patch_dist_for_cp(rank=0, world_size=2),
        ):
            result_hidden, sorted_pos = module._undo_attention_load_balancing(
                hidden, positions, MagicMock()
            )

        # sorted_pos should be 0..7
        assert torch.equal(sorted_pos, torch.arange(8, device=device, dtype=torch.long))
        # result_hidden is rank 0's chunk of the dense order (positions 0..3)
        assert result_hidden.shape == (B, S_local, D)

    def test_raises_on_non_dense_positions(self, module, device):
        """Should raise if gathered positions don't form a dense 0..S-1 sequence."""
        B, S_local, D = 1, 4, module.config.hidden_size
        hidden = torch.randn(B, S_local, D, device=device)
        positions = torch.tensor([0, 2, 4, 8], device=device, dtype=torch.long)

        rank1_positions = torch.tensor([1, 3, 5, 9], dtype=torch.long)  # gap at 6,7
        rank1_hidden = torch.randn(B, S_local, D)

        fake_ag = _make_fake_all_gather(positions, rank1_positions, hidden, rank1_hidden, device)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_ag),
            *_patch_dist_for_cp(rank=0, world_size=2),
        ):
            with pytest.raises(RuntimeError, match="dense global token positions"):
                module._undo_attention_load_balancing(hidden, positions, MagicMock())


# ============================================================================
# _redo_attention_load_balancing
# ============================================================================


class TestRedoAttentionLoadBalancing:
    """Test that _redo restores the original load-balanced CP layout."""

    def test_restores_original_layout(self, module, device):
        """Output gathered in dense order should be scattered back to load-balanced order."""
        B, S_local, D = 1, 4, module.config.hidden_size

        # Dense-order output from the attention computation
        output = torch.arange(S_local, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(-1).expand(B, S_local, D)

        # Rank 0 originally held positions [0, 3, 4, 7]
        original_positions = torch.tensor([0, 3, 4, 7], device=device, dtype=torch.long)
        sorted_positions = torch.arange(8, device=device, dtype=torch.long)

        rank1_output = torch.arange(S_local, S_local * 2, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(-1).expand(B, S_local, D)

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)
            gathered[1].copy_(rank1_output)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
            *_patch_dist_for_cp(rank=0, world_size=2),
        ):
            result = module._redo_attention_load_balancing(
                output, original_positions, sorted_positions, MagicMock()
            )

        # Result should have the same shape as input
        assert result.shape == (B, S_local, D)
        # The tokens at positions [0,3,4,7] should be selected from the full dense output
        expected_indices = original_positions
        for i, pos in enumerate(expected_indices):
            assert result[0, i, 0].item() == pos.item()

    def test_raises_on_position_mismatch(self, module, device):
        """Should raise if sorted_positions don't cover the original_positions."""
        B, S_local, D = 1, 4, module.config.hidden_size
        output = torch.randn(B, S_local, D, device=device)

        original_positions = torch.tensor([0, 3, 4, 10], device=device, dtype=torch.long)  # 10 out of range
        sorted_positions = torch.arange(8, device=device, dtype=torch.long)

        rank1_output = torch.randn(B, S_local, D, device=device)

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)
            gathered[1].copy_(rank1_output)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
            *_patch_dist_for_cp(rank=0, world_size=2),
        ):
            with pytest.raises(RuntimeError, match="Failed to restore"):
                module._redo_attention_load_balancing(
                    output, original_positions, sorted_positions, MagicMock()
                )


# ============================================================================
# forward fast path (no CP)
# ============================================================================


class TestForwardFastPath:
    def test_no_cp_mesh_delegates_to_super(self, module, device):
        """When _cp_mesh is None, forward should delegate to the HF parent class."""
        assert module._cp_mesh is None
        B, S, D = 1, 8, module.config.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        with patch.object(
            type(module).__bases__[0], "forward", return_value=torch.randn(B, S, D, device=device)
        ) as mock_super_fwd:
            module.forward(hidden)
            mock_super_fwd.assert_called_once()

    def test_cp_mesh_size_1_delegates_to_super(self, module, device):
        """When _cp_mesh.size() == 1, forward should delegate to the HF parent class."""
        mesh = MagicMock()
        mesh.size.return_value = 1
        module._cp_mesh = mesh

        B, S, D = 1, 8, module.config.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        with patch.object(
            type(module).__bases__[0], "forward", return_value=torch.randn(B, S, D, device=device)
        ) as mock_super_fwd:
            module.forward(hidden)
            mock_super_fwd.assert_called_once()

    def test_cp_mesh_gt_1_calls_forward_with_cp(self, module, device):
        """When _cp_mesh.size() > 1, forward should call _forward_with_cp."""
        mesh = MagicMock()
        mesh.size.return_value = 2
        module._cp_mesh = mesh

        B, S, D = 1, 8, module.config.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        with patch.object(
            module, "_forward_with_cp", return_value=torch.randn(B, S, D, device=device)
        ) as mock_cp_fwd:
            module.forward(hidden, position_ids=torch.arange(S, device=device).unsqueeze(0))
            mock_cp_fwd.assert_called_once()


# ============================================================================
# _conv1d_with_cp
# ============================================================================


class TestConv1dWithCP:
    def test_output_shape_matches_input(self, module, device):
        """Conv1d output should preserve [B, D, S_local] shape."""
        B = 1
        conv_dim = module.conv1d.weight.shape[0]
        S_local = 8
        mixed_qkv = torch.randn(B, conv_dim, S_local, device=device)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=0),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=1),
            patch(
                "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_process_group_ranks",
                return_value=[0],
            ),
        ):
            result = module._conv1d_with_cp(mixed_qkv, MagicMock())

        assert result.shape == (B, conv_dim, S_local)

    def test_rank0_sends_but_does_not_recv(self, module, device):
        """Rank 0 has no predecessor — should only send, not receive."""
        B = 1
        conv_dim = module.conv1d.weight.shape[0]
        S_local = 8
        mixed_qkv = torch.randn(B, conv_dim, S_local, device=device)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=0),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=2),
            patch(
                "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_process_group_ranks",
                return_value=[0, 1],
            ),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.batch_isend_irecv") as mock_batch,
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.P2POp") as mock_p2p,
        ):
            mock_batch.return_value = [MagicMock()]
            result = module._conv1d_with_cp(mixed_qkv, MagicMock())

            # Rank 0 should only issue 1 P2P op (isend to rank 1), no irecv
            assert mock_p2p.call_count == 1
            # First arg to P2POp should be dist.isend
            call_args = mock_p2p.call_args_list[0]
            assert "isend" in str(call_args)

        assert result.shape == (B, conv_dim, S_local)

    def test_middle_rank_sends_and_recvs(self, module, device):
        """A middle rank should both send and receive."""
        B = 1
        conv_dim = module.conv1d.weight.shape[0]
        S_local = 8
        mixed_qkv = torch.randn(B, conv_dim, S_local, device=device)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=1),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=3),
            patch(
                "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_process_group_ranks",
                return_value=[0, 1, 2],
            ),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.batch_isend_irecv") as mock_batch,
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.P2POp") as mock_p2p,
        ):
            mock_batch.return_value = [MagicMock(), MagicMock()]
            result = module._conv1d_with_cp(mixed_qkv, MagicMock())

            # Middle rank: 1 irecv from rank 0 + 1 isend to rank 2
            assert mock_p2p.call_count == 2

        assert result.shape == (B, conv_dim, S_local)


# ============================================================================
# _AllGatherConcatFn
# ============================================================================


class TestAllGatherConcatFn:
    def test_forward_concatenates_gathered_shards(self, device):
        """Forward should gather and concatenate along the specified dim."""
        local = torch.tensor([[1.0, 2.0]], device=device)
        group = MagicMock()

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)
            gathered[1].copy_(tensor * 2)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=2),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=0),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
        ):
            result = _AllGatherConcatFn.apply(local, group, 1)

        expected = torch.tensor([[1.0, 2.0, 2.0, 4.0]], device=device)
        assert torch.equal(result, expected)

    def test_forward_dim0(self, device):
        """Forward should work along dim=0."""
        local = torch.tensor([[1.0], [2.0]], device=device)
        group = MagicMock()

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)
            gathered[1].copy_(tensor + 10)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=2),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=0),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
        ):
            result = _AllGatherConcatFn.apply(local, group, 0)

        expected = torch.tensor([[1.0], [2.0], [11.0], [12.0]], device=device)
        assert torch.equal(result, expected)


# ============================================================================
# _all_gather_concat: differentiable vs non-differentiable
# ============================================================================


class TestAllGatherConcat:
    def test_non_differentiable_path(self, module, device):
        """Non-differentiable path should use plain dist.all_gather."""
        local = torch.randn(1, 4, device=device)

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=1),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
        ):
            result = module._all_gather_concat(local, MagicMock(), dim=1, differentiable=False)

        assert result.shape == (1, 4)

    def test_differentiable_path_requires_grad(self, module, device):
        """Differentiable path should produce output that requires grad when input does."""
        local = torch.randn(1, 4, device=device, requires_grad=True)

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=1),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=0),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
        ):
            result = module._all_gather_concat(local, MagicMock(), dim=1, differentiable=True)

        assert result.requires_grad


# ============================================================================
# CPAwareGatedDeltaNet init
# ============================================================================


class TestInit:
    def test_cp_mesh_defaults_to_none(self, module):
        """Freshly created module should have _cp_mesh == None."""
        assert module._cp_mesh is None

    def test_inherits_hf_weights(self, module):
        """Module should have the same projection layers as the HF parent."""
        assert hasattr(module, "in_proj_qkv")
        assert hasattr(module, "in_proj_z")
        assert hasattr(module, "in_proj_b")
        assert hasattr(module, "in_proj_a")
        assert hasattr(module, "out_proj")
        assert hasattr(module, "conv1d")
