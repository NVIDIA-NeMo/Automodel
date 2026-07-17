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
"""Contract and GPU parity tests for the optional GLM-5.2 cuDNN DSA adapter."""

from __future__ import annotations

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa import layers as layer_mod
from nemo_automodel.components.models.glm_moe_dsa.kernels import cudnn_dsa
from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaIndexer, GlmMoeDsaMLA
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM

TOKENS = 5
INDEX_HEADS = 64
INDEX_DIM = 128
SPARSE_TOKENS = 2
SPARSE_HEADS = 8
QK_DIM = 576
VALUE_DIM = 512


def _has_cudnn_dsa_gpu() -> bool:
    """Return whether this worker can execute the SM90+ cuDNN DSA kernels."""
    return (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9 and cudnn_dsa.is_cudnn_dsa_available()
    )


requires_cudnn_dsa_gpu = pytest.mark.skipif(
    not _has_cudnn_dsa_gpu(), reason="cuDNN DSA parity requires SM90+ and the CUDA optional dependencies"
)


def _accept_cpu_tensors(operation: str, *tensors: torch.Tensor) -> tuple[int, int]:
    """Replace the CUDA device check while retaining dtype/shape validation.

    Args:
        operation: Name of the kernel operation being validated.
        *tensors: Tensors of arbitrary shape whose CPU devices are accepted by the fake kernels.

    Returns:
        Hopper compute capability represented as ``(major, minor)``.
    """
    assert operation in {"cuDNN DSA indexer", "cuDNN DSA sparse attention"}
    assert tensors
    return 9, 0


class _FakeIndexerDsa:
    """Assert the cuDNN indexer API contract and return deterministic local indices."""

    def __init__(self) -> None:
        self.forward_called = False
        self.topk_called = False
        self.scores: torch.Tensor | None = None

    def indexer_forward_wrapper(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        head_weights: torch.Tensor,
        *,
        ratio: int,
        sm_scale: float,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> dict[str, torch.Tensor]:
        """Return fake packed indexer scores after checking kernel layouts.

        Args:
            index_q: Tensor of shape [tokens, index_heads, index_dim].
            index_k: Tensor of shape [tokens, 1, index_dim].
            head_weights: Tensor of shape [tokens, index_heads].
            ratio: Query-to-key sequence ratio.
            sm_scale: Scale applied inside the score kernel.
            cu_seqlens_q: Tensor of shape [sequences + 1] for packed queries.
            cu_seqlens_k: Tensor of shape [sequences + 1] for packed keys.
            max_seqlen_q: Maximum packed query length.
            max_seqlen_k: Maximum packed key length.

        Returns:
            Mapping whose ``scores`` tensor has shape [tokens, max_sequence].
        """
        assert index_q.shape == (TOKENS, INDEX_HEADS, INDEX_DIM)
        assert index_q.dtype == torch.bfloat16
        assert index_q.is_contiguous()
        assert index_k.shape == (TOKENS, 1, INDEX_DIM)
        assert index_k.dtype == torch.bfloat16
        assert index_k.is_contiguous()
        assert head_weights.shape == (TOKENS, INDEX_HEADS)
        assert head_weights.dtype == torch.bfloat16
        assert head_weights.is_contiguous()
        torch.testing.assert_close(head_weights.float(), torch.full_like(head_weights.float(), 0.25))
        assert ratio == 1
        assert sm_scale == 1.0
        torch.testing.assert_close(cu_seqlens_q, torch.tensor([0, 2, 5], dtype=torch.int32))
        torch.testing.assert_close(cu_seqlens_k, cu_seqlens_q)
        assert max_seqlen_q == max_seqlen_k == 3

        self.forward_called = True
        self.scores = torch.arange(TOKENS * 3, dtype=torch.float32).reshape(TOKENS, 3)
        return {"scores": self.scores}

    def indexer_top_k_wrapper(
        self,
        scores: torch.Tensor,
        causal_lengths: torch.Tensor,
        *,
        top_k: int,
        next_n: int,
        return_val: bool,
    ) -> dict[str, torch.Tensor]:
        """Return unsorted local indices with invalid entries for adapter cleanup.

        Args:
            scores: Tensor of shape [tokens, max_sequence] containing indexer scores.
            causal_lengths: Tensor of shape [tokens] containing per-query causal key counts.
            top_k: Number of local indices produced by the fake kernel.
            next_n: Number of next-token top-k sets requested.
            return_val: Whether selected score values are requested.

        Returns:
            Mapping whose ``indices`` tensor has shape [tokens, top_k].
        """
        assert self.scores is not None
        torch.testing.assert_close(scores, self.scores)
        torch.testing.assert_close(causal_lengths, torch.tensor([1, 2, 1, 2, 3], dtype=torch.int32))
        assert top_k == 2
        assert next_n == 1
        assert return_val is False

        self.topk_called = True
        return {
            "indices": torch.tensor(
                [
                    [0, 2],
                    [1, 0],
                    [0, -1],
                    [1, 0],
                    [2, 0],
                ],
                dtype=torch.int32,
            )
        }


class _FakeCpIndexerDsa:
    """Check segmented local-Q/global-K dispatch and return deterministic indices."""

    def indexer_forward_wrapper(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        head_weights: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        assert index_q.shape == (2, INDEX_HEADS, INDEX_DIM)
        assert index_k.shape == (4, 1, INDEX_DIM)
        torch.testing.assert_close(index_k[:, 0, 0].float(), torch.tensor([4.0, 5.0, 6.0, 7.0]))
        assert head_weights.shape == (2, INDEX_HEADS)
        torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, 2], dtype=torch.int32))
        torch.testing.assert_close(kwargs["cu_seqlens_k"], torch.tensor([0, 4], dtype=torch.int32))
        assert kwargs["max_seqlen_q"] == 2
        assert kwargs["max_seqlen_k"] == 4
        assert kwargs["ratio"] == 1
        assert kwargs["sm_scale"] == 1.0
        return {"scores": torch.arange(8, dtype=torch.float32).reshape(2, 4)}

    def indexer_top_k_wrapper(
        self,
        scores: torch.Tensor,
        causal_lengths: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        assert scores.shape == (2, 4)
        torch.testing.assert_close(causal_lengths, torch.tensor([3, 4], dtype=torch.int32))
        assert kwargs == {"top_k": 2, "next_n": 1, "return_val": False}
        return {"indices": torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)}


class _FakeFlashMla:
    """Assert FlashMLA forward arguments and return a head-distinct value tensor."""

    def __init__(self) -> None:
        self.called = False
        self.output: torch.Tensor | None = None
        self.lse: torch.Tensor | None = None

    def __call__(
        self,
        q: torch.Tensor,
        kv_latent: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        *,
        d_v: int,
        attn_sink: torch.Tensor,
        topk_length: torch.Tensor,
        indexer_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fake sparse-attention output after checking padded kernel layouts.

        Args:
            q: Tensor of shape [tokens, padded_heads, qk_dim].
            kv_latent: Tensor of shape [tokens, 1, qk_dim].
            indices: Tensor of shape [tokens, 1, aligned_topk].
            softmax_scale: Scale applied to sparse attention logits.
            d_v: Number of latent value channels.
            attn_sink: Tensor of shape [padded_heads] containing attention sink logits.
            topk_length: Tensor of shape [tokens] containing valid-prefix lengths.
            indexer_topk: Number of indexer positions consumed by FlashMLA itself.

        Returns:
            Tuple containing output of shape [tokens, padded_heads, value_dim], maximum logits
            of shape [tokens, padded_heads], and LSE of shape [tokens, padded_heads].
        """
        assert q.shape == (SPARSE_TOKENS, 64, QK_DIM)
        assert q.dtype == torch.bfloat16
        assert q.is_contiguous()
        torch.testing.assert_close(q[:, :SPARSE_HEADS], torch.ones_like(q[:, :SPARSE_HEADS]))
        torch.testing.assert_close(q[:, SPARSE_HEADS:], torch.zeros_like(q[:, SPARSE_HEADS:]))
        assert kv_latent.shape == (SPARSE_TOKENS, 1, QK_DIM)
        assert kv_latent.dtype == torch.bfloat16
        assert kv_latent.is_contiguous()
        assert indices.shape == (SPARSE_TOKENS, 1, 512)
        assert indices.dtype == torch.int32
        assert indices.is_contiguous()
        assert softmax_scale == 0.125
        assert d_v == VALUE_DIM
        assert attn_sink.shape == (64,)
        assert attn_sink.dtype == torch.float32
        assert torch.isneginf(attn_sink).all()
        torch.testing.assert_close(topk_length, torch.tensor([1, 2], dtype=torch.int32))
        assert indexer_topk == 0

        head_values = torch.arange(64, dtype=torch.bfloat16).view(1, 64, 1)
        self.output = head_values.expand(SPARSE_TOKENS, 64, VALUE_DIM).contiguous()
        max_logits = torch.zeros(SPARSE_TOKENS, 64, dtype=torch.float32)
        self.lse = torch.ones(SPARSE_TOKENS, 64, dtype=torch.float32)
        self.called = True
        return self.output, max_logits, self.lse


class _FakeSparseDsa:
    """Assert cuDNN backward arguments and return known gradients."""

    def __init__(self, flash_mla: _FakeFlashMla) -> None:
        self.flash_mla = flash_mla
        self.called = False

    def sparse_attention_backward_wrapper(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        d_out: torch.Tensor,
        lse: torch.Tensor,
        attn_sink: torch.Tensor,
        indices: torch.Tensor,
        *,
        softmax_scale: float,
        topk_length: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic query and latent-KV gradients.

        Args:
            q: Tensor of shape [tokens, padded_heads, qk_dim].
            kv: Tensor of shape [tokens, qk_dim].
            out: Tensor of shape [tokens, padded_heads, value_dim].
            d_out: Tensor of shape [tokens, padded_heads, value_dim].
            lse: Tensor of shape [tokens, padded_heads].
            attn_sink: Tensor of shape [padded_heads] containing attention sink logits.
            indices: Tensor of shape [tokens, aligned_topk] with ignored suffix entries sanitized to zero.
            softmax_scale: Scale applied to sparse attention logits.
            topk_length: Tensor of shape [tokens] containing valid-prefix lengths.

        Returns:
            Mapping containing ``dq`` of shape [tokens, padded_heads, qk_dim] and ``dkv``
            of shape [tokens, qk_dim].
        """
        assert q.shape == (SPARSE_TOKENS, 64, QK_DIM)
        assert kv.shape == (SPARSE_TOKENS, QK_DIM)
        assert self.flash_mla.output is not None
        assert out.shape == (SPARSE_TOKENS, 64, VALUE_DIM)
        torch.testing.assert_close(out[:, :SPARSE_HEADS], self.flash_mla.output[:, :SPARSE_HEADS])
        torch.testing.assert_close(out[:, SPARSE_HEADS:], torch.zeros_like(out[:, SPARSE_HEADS:]))
        assert d_out.shape == (SPARSE_TOKENS, 64, VALUE_DIM)
        torch.testing.assert_close(d_out[:, :SPARSE_HEADS], torch.ones_like(d_out[:, :SPARSE_HEADS]))
        torch.testing.assert_close(d_out[:, SPARSE_HEADS:], torch.zeros_like(d_out[:, SPARSE_HEADS:]))
        assert self.flash_mla.lse is not None
        assert lse.shape == (SPARSE_TOKENS, 64)
        torch.testing.assert_close(lse[:, :SPARSE_HEADS], self.flash_mla.lse[:, :SPARSE_HEADS])
        torch.testing.assert_close(lse[:, SPARSE_HEADS:], torch.zeros_like(lse[:, SPARSE_HEADS:]))
        assert attn_sink.shape == (64,)
        assert indices.shape == (SPARSE_TOKENS, 512)
        assert indices.dtype == torch.int32
        assert (indices >= 0).all()
        torch.testing.assert_close(indices[0, :3], torch.tensor([0, 0, 0], dtype=torch.int32))
        torch.testing.assert_close(indices[1, :3], torch.tensor([0, 1, 0], dtype=torch.int32))
        assert softmax_scale == 0.125
        torch.testing.assert_close(topk_length, torch.tensor([1, 2], dtype=torch.int32))

        self.called = True
        return {"dq": torch.full_like(q, 3), "dkv": torch.full_like(kv, 5)}


@pytest.mark.parametrize(
    ("has_cudnn", "has_flash_mla", "expected"),
    [(False, False, False), (True, False, False), (False, True, False), (True, True, True)],
)
def test_cudnn_dsa_availability_requires_both_runtimes(
    monkeypatch: pytest.MonkeyPatch,
    has_cudnn: bool,
    has_flash_mla: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(cudnn_dsa, "_HAS_CUDNN_DSA", has_cudnn)
    monkeypatch.setattr(cudnn_dsa, "_HAS_FLASH_MLA", has_flash_mla)

    assert cudnn_dsa.is_cudnn_dsa_available() is expected

    if not expected:
        with pytest.raises(RuntimeError, match="requires both nvidia-cudnn-frontend"):
            cudnn_dsa.cudnn_indexer_topk(
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                index_topk=1,
            )


def test_cudnn_indexer_uses_packed_offsets_fixed_width_and_reference_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dsa = _FakeIndexerDsa()
    monkeypatch.setattr(cudnn_dsa, "_HAS_CUDNN_DSA", True)
    monkeypatch.setattr(cudnn_dsa, "_HAS_FLASH_MLA", True)
    monkeypatch.setattr(cudnn_dsa, "_CUDNN_DSA", fake_dsa)
    monkeypatch.setattr(cudnn_dsa, "_require_cuda_tensors", _accept_cpu_tensors)

    index_q = torch.ones(TOKENS, INDEX_HEADS, INDEX_DIM, dtype=torch.bfloat16)
    index_k = torch.ones(TOKENS, INDEX_DIM, dtype=torch.bfloat16)
    head_weights = torch.full((TOKENS, INDEX_HEADS), 0.25, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    actual = cudnn_dsa.cudnn_indexer_topk(
        index_q,
        index_k,
        head_weights,
        cu_seqlens,
        index_topk=2,
        query_indices=torch.arange(TOKENS),
        cu_seqlens_padded=cu_seqlens.clone(),
    )

    expected = torch.tensor(
        [
            [[0, -1]],
            [[0, 1]],
            [[2, -1]],
            [[2, 3]],
            [[2, 4]],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()
    assert fake_dsa.forward_called
    assert fake_dsa.topk_called


def test_cudnn_metadata_segments_cp_queries_and_padded_documents() -> None:
    """Local CP queries use positive Q segments and matching global K prefixes."""
    metadata = cudnn_dsa.prepare_cudnn_dsa_packed_metadata(
        torch.tensor([0, 3, 7, 9], dtype=torch.int32),
        total_key_tokens=12,
        query_indices=torch.arange(2, 10, dtype=torch.int32),
        cu_seqlens_padded=torch.tensor([0, 4, 9, 12], dtype=torch.int32),
    )

    torch.testing.assert_close(metadata.segment_cu_q, torch.tensor([0, 2, 7, 8], dtype=torch.int32))
    torch.testing.assert_close(metadata.segment_cu_k, torch.tensor([0, 4, 9, 10], dtype=torch.int32))
    torch.testing.assert_close(metadata.key_source_indices, torch.arange(10))
    torch.testing.assert_close(metadata.starts, torch.tensor([0, 0, 4, 4, 4, 4, 4, 9], dtype=torch.int32))
    torch.testing.assert_close(metadata.causal_lengths, torch.tensor([3, 3, 1, 2, 3, 4, 4, 1], dtype=torch.int32))
    assert metadata.max_seqlen_q == 5
    assert metadata.max_seqlen_k == 5
    assert metadata.total_key_tokens == 12


def test_cudnn_indexer_maps_cp_segment_indices_to_global_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Segment-local cuDNN columns are canonicalized in gathered global-K coordinates."""
    monkeypatch.setattr(cudnn_dsa, "_HAS_CUDNN_DSA", True)
    monkeypatch.setattr(cudnn_dsa, "_HAS_FLASH_MLA", True)
    monkeypatch.setattr(cudnn_dsa, "_CUDNN_DSA", _FakeCpIndexerDsa())
    monkeypatch.setattr(cudnn_dsa, "_require_cuda_tensors", _accept_cpu_tensors)

    index_q = torch.ones(2, INDEX_HEADS, INDEX_DIM, dtype=torch.bfloat16)
    index_k = torch.arange(8, dtype=torch.bfloat16).view(8, 1).expand(-1, INDEX_DIM).contiguous()
    actual = cudnn_dsa.cudnn_indexer_topk(
        index_q,
        index_k,
        torch.ones(2, INDEX_HEADS),
        torch.tensor([0, 4, 8], dtype=torch.int32),
        index_topk=2,
        query_indices=torch.tensor([6, 7], dtype=torch.int32),
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[[4, 6]], [[5, 7]]], dtype=torch.int32),
    )


def test_cudnn_indexer_rejects_unsupported_sm90_head_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """cuDNN frontend 1.25 supports 32 or 64 index heads on Hopper."""
    monkeypatch.setattr(cudnn_dsa, "_HAS_CUDNN_DSA", True)
    monkeypatch.setattr(cudnn_dsa, "_HAS_FLASH_MLA", True)
    monkeypatch.setattr(cudnn_dsa, "_require_cuda_tensors", _accept_cpu_tensors)

    with pytest.raises(ValueError, match="supports H_index in \\(32, 64\\)"):
        cudnn_dsa.cudnn_indexer_topk(
            torch.ones(2, 16, INDEX_DIM, dtype=torch.bfloat16),
            torch.ones(2, INDEX_DIM, dtype=torch.bfloat16),
            torch.ones(2, 16, dtype=torch.float32),
            torch.tensor([0, 2], dtype=torch.int32),
            index_topk=1,
        )


def test_cudnn_sparse_attention_dispatches_padded_forward_and_exact_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_flash_mla = _FakeFlashMla()
    fake_dsa = _FakeSparseDsa(fake_flash_mla)
    monkeypatch.setattr(cudnn_dsa, "_HAS_CUDNN_DSA", True)
    monkeypatch.setattr(cudnn_dsa, "_HAS_FLASH_MLA", True)
    monkeypatch.setattr(cudnn_dsa, "_CUDNN_DSA", fake_dsa)
    monkeypatch.setattr(cudnn_dsa, "_FLASH_MLA_SPARSE_FWD", fake_flash_mla)
    monkeypatch.setattr(cudnn_dsa, "_require_cuda_tensors", _accept_cpu_tensors)

    q = torch.ones(SPARSE_TOKENS, SPARSE_HEADS, QK_DIM, dtype=torch.bfloat16, requires_grad=True)
    kv_latent = torch.ones(SPARSE_TOKENS, 1, QK_DIM, dtype=torch.bfloat16, requires_grad=True)
    topk_indices = torch.tensor([[[0, -1, -1]], [[0, 1, -1]]], dtype=torch.int32)

    actual = cudnn_dsa.cudnn_sparse_attention(q, kv_latent, topk_indices, softmax_scale=0.125)

    assert actual.shape == (SPARSE_TOKENS, SPARSE_HEADS, VALUE_DIM)
    assert actual.dtype == torch.bfloat16
    for head in range(SPARSE_HEADS):
        torch.testing.assert_close(actual[:, head], torch.full_like(actual[:, head], head))

    actual.sum().backward()

    assert q.grad is not None
    assert kv_latent.grad is not None
    torch.testing.assert_close(q.grad, torch.full_like(q, 3))
    torch.testing.assert_close(kv_latent.grad, torch.full_like(kv_latent, 5))
    assert fake_flash_mla.called
    assert fake_dsa.called


def _small_dsa_config() -> GlmMoeDsaConfig:
    """Return a compact model config that preserves DSA tensor dimensions."""
    return GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=512,
        rms_norm_eps=1.0e-5,
        attention_bias=False,
        kv_lora_rank=512,
        q_lora_rank=256,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=16,
        index_head_dim=128,
        index_topk=64,
        mlp_layer_types=["dense"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        torch_dtype="bfloat16",
    )


def _freqs(tokens: int, rope_dim: int) -> torch.Tensor:
    """Return identity complex RoPE frequencies for packed THD tests."""
    angles = torch.zeros(tokens, rope_dim // 2, dtype=torch.float32)
    return torch.polar(torch.ones_like(angles), angles)


def test_cudnn_indexer_layer_preserves_fp32_reference_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    """The layer must fold both index scales exactly once before adapter dispatch."""
    config = _small_dsa_config()
    backend = BackendConfig(attn="cudnn", linear="torch", rms_norm="torch", rope_fusion=False)
    indexer = GlmMoeDsaIndexer(config, backend)
    captured: dict[str, object] = {}
    expected_topk = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)

    def fake_indexer(
        q: torch.Tensor,
        k: torch.Tensor,
        head_weights: torch.Tensor,
        cu_seqlens: torch.Tensor,
        index_topk: int,
        **kwargs: object,
    ) -> torch.Tensor:
        captured.update(q=q, k=k, head_weights=head_weights, cu_seqlens=cu_seqlens, kwargs=kwargs)
        assert index_topk == config.index_topk
        return expected_topk

    monkeypatch.setattr(layer_mod, "is_cudnn_dsa_available", lambda: True)
    monkeypatch.setattr(layer_mod, "cudnn_indexer_topk", fake_indexer)
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16)
    q_resid = torch.randn(4, config.q_lora_rank, dtype=torch.bfloat16)
    expected_weights = indexer.weights_proj(x).float() * config.index_n_heads**-0.5 * config.index_head_dim**-0.5
    packed_metadata = (
        torch.zeros(4, dtype=torch.int32),
        torch.arange(1, 5, dtype=torch.int32),
        4,
    )

    actual = indexer(
        x,
        q_resid,
        _freqs(4, config.qk_rope_head_dim),
        cu_seqlens=torch.tensor([[0, 4]]),
        _cudnn_dsa_packed_metadata=packed_metadata,
    )

    assert actual is expected_topk
    assert captured["q"].shape == (4, config.index_n_heads, config.index_head_dim)
    assert captured["k"].shape == (4, config.index_head_dim)
    assert captured["head_weights"].dtype == torch.float32
    torch.testing.assert_close(captured["head_weights"], expected_weights)
    torch.testing.assert_close(captured["cu_seqlens"], torch.tensor([0, 4], dtype=torch.int32))
    captured_kwargs = captured["kwargs"]
    assert isinstance(captured_kwargs, dict)
    assert captured_kwargs["packed_metadata"] is packed_metadata


def test_cudnn_indexer_layer_gathers_cp_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model gathers indexer K before dispatching local Q to cuDNN."""
    config = _small_dsa_config()
    backend = BackendConfig(attn="cudnn", linear="torch", rms_norm="torch", rope_fusion=False)
    indexer = GlmMoeDsaIndexer(config, backend)
    captured: dict[str, object] = {}

    def fake_gather(tensor: torch.Tensor, *, dim: int, cp_group: object) -> torch.Tensor:
        assert dim == 0 and cp_group == "cp-group"
        return torch.cat((tensor, tensor), dim=0)

    def fake_indexer(
        q: torch.Tensor,
        k: torch.Tensor,
        head_weights: torch.Tensor,
        cu_seqlens: torch.Tensor,
        index_topk: int,
        **kwargs: object,
    ) -> torch.Tensor:
        captured.update(q=q, k=k, head_weights=head_weights, cu_seqlens=cu_seqlens, kwargs=kwargs)
        return torch.zeros(q.shape[0], 1, index_topk, dtype=torch.int32)

    monkeypatch.setattr(layer_mod, "glm_dsa_cp_enabled", lambda group: group == "cp-group")
    monkeypatch.setattr(layer_mod, "glm_dsa_cp_all_gather", fake_gather)
    monkeypatch.setattr(layer_mod, "is_cudnn_dsa_available", lambda: True)
    monkeypatch.setattr(layer_mod, "cudnn_indexer_topk", fake_indexer)
    query_indices = torch.arange(4, dtype=torch.int32)
    result = indexer(
        torch.randn(4, config.hidden_size, dtype=torch.bfloat16),
        torch.randn(4, config.q_lora_rank, dtype=torch.bfloat16),
        _freqs(4, config.qk_rope_head_dim),
        cu_seqlens=torch.tensor([0, 8]),
        cu_seqlens_padded=torch.tensor([0, 8]),
        glm_dsa_cp_query_indices=query_indices,
        _glm_dsa_cp_group="cp-group",
        _cudnn_dsa_packed_metadata=object(),
    )

    assert result.shape == (4, 1, config.index_topk)
    assert captured["k"].shape == (8, config.index_head_dim)
    assert captured["kwargs"]["query_indices"] is query_indices


def test_cudnn_mla_layer_projects_latent_values_with_model_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cuDNN adapter returns latent values and the layer must apply ``w_vc`` once."""
    config = _small_dsa_config()
    backend = BackendConfig(attn="cudnn", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend, skip_topk=True)
    mla.o_proj = torch.nn.Identity()
    captured: dict[str, torch.Tensor | float | None] = {}
    latent = torch.ones(4, config.num_attention_heads, config.kv_lora_rank, dtype=torch.bfloat16)

    def fake_sparse(
        q: torch.Tensor,
        kv_latent: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        captured.update(
            q=q,
            kv_latent=kv_latent,
            topk_indices=topk_indices,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
        )
        return latent

    monkeypatch.setattr(layer_mod, "is_cudnn_dsa_available", lambda: True)
    monkeypatch.setattr(layer_mod, "cudnn_sparse_attention", fake_sparse)
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16)
    topk = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)
    topk_length = torch.arange(1, 5, dtype=torch.int32)
    actual = mla(
        x,
        _freqs(4, config.qk_rope_head_dim),
        prev_topk_indices=topk,
        _cudnn_dsa_topk_length=topk_length,
    )

    weight = mla.kv_b_proj.weight.view(
        config.num_attention_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    )
    w_vc = weight[:, config.qk_nope_head_dim :, :]
    expected = torch.einsum("thc,hdc->thd", latent, w_vc.to(latent.dtype)).flatten(1)
    torch.testing.assert_close(actual, expected)
    assert captured["q"].shape == (4, config.num_attention_heads, config.kv_lora_rank + config.qk_rope_head_dim)
    assert captured["kv_latent"].shape == (4, 1, config.kv_lora_rank + config.qk_rope_head_dim)
    assert captured["topk_indices"] is topk
    assert captured["softmax_scale"] == mla.softmax_scale
    assert captured["topk_length"] is topk_length


def test_cudnn_mla_layer_gathers_cp_kv(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model gathers latent K/V before local-query FlashMLA/cuDNN dispatch."""
    config = _small_dsa_config()
    backend = BackendConfig(attn="cudnn", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend, skip_topk=True)
    mla.o_proj = torch.nn.Identity()
    gathered: list[tuple[int, ...]] = []
    captured: dict[str, torch.Tensor] = {}

    def fake_gather(tensor: torch.Tensor, *, dim: int, cp_group: object) -> torch.Tensor:
        assert dim == 0 and cp_group == "cp-group"
        gathered.append(tuple(tensor.shape))
        return torch.cat((tensor, tensor), dim=0)

    def fake_sparse(
        q: torch.Tensor,
        kv_latent: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del softmax_scale, topk_length
        captured.update(q=q, kv_latent=kv_latent, topk_indices=topk_indices)
        return torch.ones(q.shape[0], config.num_attention_heads, config.kv_lora_rank, dtype=torch.bfloat16)

    monkeypatch.setattr(layer_mod, "glm_dsa_cp_enabled", lambda group: group == "cp-group")
    monkeypatch.setattr(layer_mod, "glm_dsa_cp_all_gather", fake_gather)
    monkeypatch.setattr(layer_mod, "is_cudnn_dsa_available", lambda: True)
    monkeypatch.setattr(layer_mod, "cudnn_sparse_attention", fake_sparse)
    topk = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)
    result = mla(
        torch.randn(4, config.hidden_size, dtype=torch.bfloat16),
        _freqs(4, config.qk_rope_head_dim),
        prev_topk_indices=topk,
        _glm_dsa_cp_group="cp-group",
    )

    assert result.shape == (4, config.num_attention_heads * config.v_head_dim)
    assert gathered == [(4, config.kv_lora_rank), (4, config.qk_rope_head_dim)]
    assert captured["kv_latent"].shape == (8, 1, config.kv_lora_rank + config.qk_rope_head_dim)
    assert captured["topk_indices"] is topk


def test_cudnn_model_requires_packing_and_fixed_pipeline_topk() -> None:
    """cuDNN uses the packed CP hook and fixed-width pipeline top-k carry."""
    config = _small_dsa_config()
    backend = BackendConfig(attn="cudnn", linear="torch", rms_norm="torch", rope_fusion=False)
    model = GlmMoeDsaForCausalLM(config, backend=backend)
    model.lm_head = None

    assert model.should_pack_validation_with_training() is True
    inputs, outputs = model.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=4,
        seq_len=32,
        dtype=torch.bfloat16,
    )
    assert inputs[0].shape == (32, config.hidden_size)
    assert inputs[1].shape == (32, 1, config.index_topk)
    assert outputs[0].shape == inputs[0].shape
    assert outputs[1].shape == inputs[1].shape
    prepared = model.prepare_model_inputs_for_cp(input_ids=torch.arange(32).view(1, 32))
    assert callable(prepared["_cp_make_batch_fn"])


def _dense_packed_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Return a dense packed-causal top-k oracle in global compact coordinates."""
    result = torch.full((q.shape[0], topk), -1, dtype=torch.int32, device=q.device)
    bounds = cu_seqlens.cpu().tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        scores = torch.relu(torch.einsum("thd,sd->ths", q[start:end].float(), k[start:end].float()))
        scores = torch.einsum("th,ths->ts", weights[start:end].to(torch.bfloat16).float(), scores)
        length = end - start
        scores.masked_fill_(torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        for row in range(length):
            count = min(topk, row + 1)
            selected = scores[row].topk(count).indices.to(torch.int32).add(start).sort().values
            result[start + row, :count] = selected
    return result


def _causal_topk(tokens: int, topk: int, device: torch.device) -> torch.Tensor:
    """Return fixed-width sorted causal indices with an invalid suffix."""
    result = torch.full((tokens, topk), -1, dtype=torch.int32, device=device)
    for row in range(tokens):
        count = min(row + 1, topk)
        result[row, :count] = torch.randperm(row + 1, device=device)[:count].sort().values.to(torch.int32)
    return result.unsqueeze(1)


def _dense_sparse_attention(
    q: torch.Tensor,
    kv_latent: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Return dense attention restricted to the supplied sparse global indices."""
    scores = torch.einsum("thd,sd->ths", q.float(), kv_latent[:, 0].float()) * softmax_scale
    keep = torch.zeros((q.shape[0], kv_latent.shape[0]), dtype=torch.bool, device=q.device)
    flat_indices = topk_indices[:, 0]
    valid = flat_indices >= 0
    rows = torch.arange(q.shape[0], device=q.device).unsqueeze(1).expand_as(flat_indices)[valid]
    keep[rows, flat_indices[valid].long()] = True
    probabilities = scores.masked_fill(~keep.unsqueeze(1), float("-inf")).softmax(dim=-1)
    return torch.einsum("ths,sd->thd", probabilities, kv_latent[:, 0, :VALUE_DIM].float())


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return flattened float32 cosine similarity."""
    return torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return RMSE normalized by the reference RMS."""
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return (error / scale).item()


@requires_cudnn_dsa_gpu
def test_cudnn_dsa_gpu_forward_backward_parity() -> None:
    """Compare the real split kernels with dense packed-causal PyTorch oracles."""
    torch.manual_seed(123)
    device = torch.device("cuda")
    tokens, index_heads, topk = 128, 32, 32
    index_q = torch.randn(tokens, index_heads, INDEX_DIM, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(tokens, INDEX_DIM, device=device, dtype=torch.bfloat16)
    weights = torch.randn(tokens, index_heads, device=device, dtype=torch.float32) * index_heads**-0.5 * INDEX_DIM**-0.5
    cu_seqlens = torch.tensor([0, 48, tokens], dtype=torch.int32, device=device)

    actual_topk = cudnn_dsa.cudnn_indexer_topk(index_q, index_k, weights, cu_seqlens, topk).squeeze(1)
    expected_topk = _dense_packed_indexer_topk(index_q, index_k, weights, cu_seqlens, topk)
    overlaps = []
    for row in range(tokens):
        actual_set = set(actual_topk[row].tolist()) - {-1}
        expected_set = set(expected_topk[row].tolist()) - {-1}
        overlaps.append(len(actual_set & expected_set) / len(expected_set))
    assert sum(overlaps) / len(overlaps) > 0.97

    attention_tokens, attention_heads = 64, 64
    softmax_scale = QK_DIM**-0.5
    indices = _causal_topk(attention_tokens, topk, device)
    q_seed = torch.randn(attention_tokens, attention_heads, QK_DIM, device=device, dtype=torch.bfloat16).mul_(0.25)
    kv_seed = torch.randn(attention_tokens, 1, QK_DIM, device=device, dtype=torch.bfloat16).mul_(0.25)
    output_grad = torch.randn(attention_tokens, attention_heads, VALUE_DIM, device=device, dtype=torch.bfloat16)

    q_actual = q_seed.detach().clone().requires_grad_(True)
    kv_actual = kv_seed.detach().clone().requires_grad_(True)
    packed_metadata = cudnn_dsa.prepare_cudnn_dsa_packed_metadata(
        torch.tensor([0, attention_tokens], dtype=torch.int32, device=device), attention_tokens
    )
    topk_length = packed_metadata.causal_lengths.clamp_max(topk).contiguous()
    actual = cudnn_dsa.cudnn_sparse_attention(
        q_actual,
        kv_actual,
        indices,
        softmax_scale,
        topk_length=topk_length,
    )
    (actual.float() * output_grad.float()).sum().backward()

    q_expected = q_seed.detach().clone().requires_grad_(True)
    kv_expected = kv_seed.detach().clone().requires_grad_(True)
    expected = _dense_sparse_attention(q_expected, kv_expected, indices, softmax_scale)
    (expected * output_grad.float()).sum().backward()

    assert _cosine(actual, expected) > 0.999
    assert _cosine(q_actual.grad, q_expected.grad) > 0.99
    assert _cosine(kv_actual.grad, kv_expected.grad) > 0.99
    assert _relative_rmse(q_actual.grad, q_expected.grad) < 0.15
    assert _relative_rmse(kv_actual.grad, kv_expected.grad) < 0.15


@requires_cudnn_dsa_gpu
def test_cudnn_dsa_gpu_cp_forward_backward_parity() -> None:
    """Validate segmented local Q, global K/V, and global top-k coordinates on Hopper."""
    torch.manual_seed(321)
    device = torch.device("cuda")
    total_keys, local_queries, index_heads, topk = 128, 32, 32, 32
    query_indices = torch.arange(96, 128, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)
    index_q = torch.randn(local_queries, index_heads, INDEX_DIM, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(total_keys, INDEX_DIM, device=device, dtype=torch.bfloat16)
    weights = (
        torch.randn(local_queries, index_heads, device=device, dtype=torch.float32)
        * index_heads**-0.5
        * INDEX_DIM**-0.5
    )

    actual_topk = cudnn_dsa.cudnn_indexer_topk(
        index_q,
        index_k,
        weights,
        cu_seqlens,
        topk,
        query_indices=query_indices,
    ).squeeze(1)
    scores = torch.relu(torch.einsum("thd,sd->ths", index_q.float(), index_k[64:].float()))
    scores = torch.einsum("th,ths->ts", weights.to(torch.bfloat16).float(), scores)
    expected_topk = torch.full_like(actual_topk, -1)
    for row in range(local_queries):
        causal_length = row + 33
        scores[row, causal_length:] = float("-inf")
        expected_topk[row] = scores[row].topk(topk).indices.to(torch.int32).add(64).sort().values
    overlaps = []
    for row in range(local_queries):
        overlaps.append(len(set(actual_topk[row].tolist()) & set(expected_topk[row].tolist())) / topk)
    assert sum(overlaps) / len(overlaps) > 0.97

    softmax_scale = QK_DIM**-0.5
    q_seed = torch.randn(local_queries, 64, QK_DIM, device=device, dtype=torch.bfloat16).mul_(0.25)
    kv_seed = torch.randn(total_keys, 1, QK_DIM, device=device, dtype=torch.bfloat16).mul_(0.25)
    output_grad = torch.randn(local_queries, 64, VALUE_DIM, device=device, dtype=torch.bfloat16)
    indices = actual_topk.unsqueeze(1).contiguous()
    topk_length = torch.full((local_queries,), topk, dtype=torch.int32, device=device)

    q_actual = q_seed.detach().clone().requires_grad_(True)
    kv_actual = kv_seed.detach().clone().requires_grad_(True)
    actual = cudnn_dsa.cudnn_sparse_attention(
        q_actual,
        kv_actual,
        indices,
        softmax_scale,
        topk_length=topk_length,
    )
    (actual.float() * output_grad.float()).sum().backward()

    q_expected = q_seed.detach().clone().requires_grad_(True)
    kv_expected = kv_seed.detach().clone().requires_grad_(True)
    expected = _dense_sparse_attention(q_expected, kv_expected, indices, softmax_scale)
    (expected * output_grad.float()).sum().backward()

    assert _cosine(actual, expected) > 0.999
    assert _cosine(q_actual.grad, q_expected.grad) > 0.99
    assert _cosine(kv_actual.grad, kv_expected.grad) > 0.99
    assert _relative_rmse(q_actual.grad, q_expected.grad) < 0.15
    assert _relative_rmse(kv_actual.grad, kv_expected.grad) < 0.15
