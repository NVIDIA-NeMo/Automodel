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

from unittest.mock import patch

import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss.layers import (
    GptOssAttention,
)
from nemo_automodel.shared.import_utils import is_te_min_version

# Over the default 5s budget on purpose: CUDA FlexAttention compilation takes longer on a cold worker.
# Reduce cold compiler startup before lowering this further.
pytestmark = pytest.mark.timeout(70)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def gpt_config():
    return GptOssConfig(
        vocab_size=1000,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
        num_local_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
    )


@pytest.fixture
def gpt_config_with_rope_scaling():
    return GptOssConfig(
        vocab_size=1000,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
        num_local_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        rope_scaling={"factor": 2.0, "type": "yarn"},
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        rope_fusion=False,
    )


@pytest.fixture
def backend_config_with_rope_fusion():
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        rope_fusion=True,
    )


class TestGptOssAttention:
    """Test GptOssAttention module."""

    def test_gpt_oss_attention_init(self, gpt_config, backend_config):
        """Test GptOssAttention initialization."""
        attention = GptOssAttention(gpt_config, backend_config)

        assert attention.head_dim == gpt_config.head_dim
        assert attention.num_attention_heads == gpt_config.num_attention_heads
        assert attention.num_key_value_heads == gpt_config.num_key_value_heads
        assert attention.hidden_size == gpt_config.hidden_size
        assert attention.sliding_window is None
        assert hasattr(attention, "q_proj")
        assert hasattr(attention, "k_proj")
        assert hasattr(attention, "v_proj")
        assert hasattr(attention, "o_proj")
        assert hasattr(attention, "sinks")

    def test_gpt_oss_attention_init_with_sliding_window(self, gpt_config, backend_config):
        """Test GptOssAttention initialization with sliding window."""
        attention = GptOssAttention(gpt_config, backend_config, use_sliding_attention=True)

        assert attention.sliding_window == gpt_config.sliding_window

    def test_gpt_oss_attention_linear_layer_dimensions(self, gpt_config, backend_config):
        """Test that linear layers have correct dimensions."""
        attention = GptOssAttention(gpt_config, backend_config)

        # q_proj: hidden_size -> num_attention_heads * head_dim
        assert attention.q_proj.in_features == gpt_config.hidden_size
        assert attention.q_proj.out_features == gpt_config.num_attention_heads * gpt_config.head_dim

        # k_proj, v_proj: hidden_size -> num_key_value_heads * head_dim
        assert attention.k_proj.in_features == gpt_config.hidden_size
        assert attention.k_proj.out_features == gpt_config.num_key_value_heads * gpt_config.head_dim

        assert attention.v_proj.in_features == gpt_config.hidden_size
        assert attention.v_proj.out_features == gpt_config.num_key_value_heads * gpt_config.head_dim

        # o_proj: num_attention_heads * head_dim -> hidden_size
        assert attention.o_proj.in_features == gpt_config.num_attention_heads * gpt_config.head_dim
        assert attention.o_proj.out_features == gpt_config.hidden_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.runtime_budget(60, hard_timeout=70, reason="compiles CUDA FlexAttention on a cold worker")
    def test_forward_shape_correctness(self, gpt_config, backend_config, device):
        """Test forward pass output shapes."""
        attention = GptOssAttention(gpt_config, backend_config)
        attention = attention.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
        freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

        # Mock the attn_module call method instead of replacing the module
        with patch.object(attention.attn_module, "__call__") as mock_attn:
            # Mock attention module to return expected shape
            mock_attn.return_value = torch.randn(
                batch_size,
                gpt_config.num_attention_heads,
                seq_len,
                gpt_config.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )

            output = attention(x, freqs_cis)

            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            assert output.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_gpu_execution(self, gpt_config, backend_config):
        """Test forward pass executes correctly on GPU."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        attention = GptOssAttention(gpt_config, backend_config)
        attention = attention.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
        freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

        # Test that the forward pass completes successfully on GPU
        try:
            output = attention(x, freqs_cis)
            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            assert output.device == device
            # Test passes if no exception is raised
        except Exception as e:
            pytest.fail(f"GPU forward pass failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_weights(self, gpt_config, backend_config, device):
        """Test weight initialization."""
        attention = GptOssAttention(gpt_config, backend_config)

        # Store original weights to verify they change
        original_q_weight = attention.q_proj.weight.clone()
        original_sinks = attention.sinks.clone()

        attention.init_weights(device, init_std=0.02)

        # Weights should have changed
        assert not torch.equal(attention.q_proj.weight, original_q_weight)
        assert not torch.equal(attention.sinks, original_sinks)

    def test_init_weights_initializes_projection_biases(self, gpt_config, backend_config):
        """Test that meta-materialized projection biases are initialized."""
        with torch.device("meta"):
            attention = GptOssAttention(gpt_config, backend_config)

        attention.to_empty(device="cpu")
        projections = [attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj]

        with torch.no_grad():
            for projection in projections:
                assert projection.bias is not None
                projection.bias.fill_(float("nan"))

        attention.init_weights(torch.device("cpu"), init_std=0.02)

        assert all(torch.isfinite(parameter).all() for parameter in attention.parameters())
        for projection in projections:
            assert projection.bias is not None
            assert torch.count_nonzero(projection.bias).item() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rotary_embedding_application(self, gpt_config, backend_config, device):
        """Test that rotary embedding is correctly applied to q and k."""
        attention = GptOssAttention(gpt_config, backend_config)
        attention = attention.to(device)

        batch_size, seq_len = 1, 4
        x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)

        # Create simple freqs_cis for testing
        cos = torch.ones(batch_size, seq_len, gpt_config.head_dim // 2, dtype=torch.bfloat16, device=device)
        sin = torch.zeros(batch_size, seq_len, gpt_config.head_dim // 2, dtype=torch.bfloat16, device=device)
        freqs_cis = torch.cat([cos, sin], dim=-1)

        # Test that the forward pass completes successfully with valid inputs
        # The main goal is to ensure rotary embedding doesn't crash
        try:
            output = attention(x, freqs_cis)
            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            assert output.device == device
            # Test passes if no exception is raised
        except Exception as e:
            pytest.fail(f"Forward pass failed with rotary embedding: {e}")

    def test_yarn_concentration_not_set_without_rope_fusion(self, gpt_config, backend_config):
        """Test that yarn_concentration is None when rope_fusion is False."""
        attention = GptOssAttention(gpt_config, backend_config)

        assert attention.yarn_concentration is None

    @pytest.mark.skip(reason="TE fused RoPE force-disabled globally in BackendConfig.__post_init__; see #3027")
    def test_yarn_concentration_set_with_rope_fusion(
        self, gpt_config_with_rope_scaling, backend_config_with_rope_fusion
    ):
        """Test that yarn_concentration is correctly computed when rope_fusion is True."""
        import math

        attention = GptOssAttention(gpt_config_with_rope_scaling, backend_config_with_rope_fusion)

        assert hasattr(attention, "yarn_concentration")
        # yarn_get_mscale(2.0) = 0.1 * 1.0 * math.log(2.0) + 1.0
        expected_concentration = 0.1 * math.log(2.0) + 1.0
        assert abs(attention.yarn_concentration - expected_concentration) < 1e-6

    @pytest.mark.skip(reason="TE fused RoPE force-disabled globally in BackendConfig.__post_init__; see #3027")
    def test_yarn_concentration_different_scaling_factors(self, backend_config_with_rope_fusion):
        """Test yarn_concentration with different scaling factors."""
        import math

        for factor in [1.5, 2.0, 4.0, 8.0]:
            config = GptOssConfig(
                vocab_size=1000,
                hidden_size=128,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=32,
                num_hidden_layers=2,
                intermediate_size=256,
                max_position_embeddings=512,
                rms_norm_eps=1e-6,
                sliding_window=None,
                layer_types=["full_attention", "full_attention"],
                num_local_experts=8,
                num_experts_per_tok=2,
                router_aux_loss_coef=0.01,
                rope_scaling={"factor": factor, "type": "yarn"},
            )

            attention = GptOssAttention(config, backend_config_with_rope_fusion)

            expected_concentration = 0.1 * math.log(factor) + 1.0
            assert abs(attention.yarn_concentration - expected_concentration) < 1e-6


@pytest.mark.skipif(not is_te_min_version("2.8.0"), reason="TE version 2.8.0 or higher is required")
class TestGptOssAttentionWithTE:
    """Test GptOssAttention with Transformer Engine backend."""

    @pytest.fixture
    def te_backend_config(self):
        return BackendConfig(
            linear="torch",
            attn="te",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
            rope_fusion=False,
        )

    def test_te_backend_requires_min_version(self, gpt_config):
        """Test that TE backend requires minimum TE version 2.8.0."""
        te_backend = BackendConfig(attn="te")

        # Mock TE version check to simulate old version
        from unittest.mock import patch

        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=False):
            with pytest.raises(
                ValueError,
                match="Transformer Engine DotProductAttention for GPT-OSS is only supported for TE version 2.8.0 or higher",
            ):
                GptOssAttention(gpt_config, te_backend)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpt_oss_attention_te_init(self, gpt_config, te_backend_config):
        """Test GptOssAttention initialization with TE backend."""
        pytest.importorskip("transformer_engine")

        # Mock version check to allow initialization
        from unittest.mock import patch

        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config)

            assert attention.backend.attn == "te"
            assert attention.head_dim == gpt_config.head_dim
            assert attention.num_attention_heads == gpt_config.num_attention_heads
            # TE backend should not have sinks as a separate parameter
            assert attention.sinks is None
            # TE backend should have softmax_offset in attn_module
            assert hasattr(attention.attn_module, "softmax_offset")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_init_weights(self, gpt_config, te_backend_config, device):
        """Test weight initialization with TE backend."""
        pytest.importorskip("transformer_engine")

        from unittest.mock import patch

        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config)

            # Store original softmax_offset to verify it changes
            original_softmax_offset = attention.attn_module.softmax_offset.clone()

            attention.init_weights(device, init_std=0.02)

            # softmax_offset should have changed
            assert not torch.equal(attention.attn_module.softmax_offset, original_softmax_offset)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_forward_with_bshd_format(self, gpt_config, te_backend_config, device):
        """Test TE attention forward pass with BSHD format."""
        pytest.importorskip("transformer_engine")

        from unittest.mock import patch

        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config)
            attention = attention.to(device)

            batch_size, seq_len = 2, 8
            x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
            freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

            try:
                output = attention(x, freqs_cis)
                assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
                assert output.device == device
            except Exception as e:
                pytest.fail(f"TE forward pass with BSHD format failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_forward_with_thd_format(self, gpt_config, te_backend_config, device):
        """Test TE attention forward pass with THD format."""
        pytest.importorskip("transformer_engine")

        from unittest.mock import patch

        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config)
            attention = attention.to(device)

            # THD format: (num_tokens, hidden_size)
            num_tokens = 16
            x = torch.randn(num_tokens, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
            freqs_cis = torch.randn(num_tokens, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

            # Provide cu_seqlens for packed sequence
            cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)

            try:
                output = attention(x, freqs_cis, cu_seqlens=cu_seqlens, max_seqlen=8)
                assert output.shape == (num_tokens, gpt_config.hidden_size)
                assert output.device == device
            except Exception as e:
                pytest.fail(f"TE forward pass with THD format failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("use_sliding_attention", [False, True], ids=["full", "sliding"])
    def test_te_thd_packed_documents_match_separate_execution(
        self, gpt_config, te_backend_config, device, use_sliding_attention
    ):
        """Packed TE THD attention reproduces separate-document execution.

        The production packer, THD collater and THD preparation flatten two
        48-token rows: documents of 17 and 26 tokens, then one of 35 tokens,
        each row ending in padding. Valid-token outputs and the gradients of
        every projection and of the learnable sinks must match separate BSHD
        runs. Changing the first document or the padding must leave the other
        documents' outputs bit-identical. The attention uses GPT-OSS's released
        head dimension of 64.
        """
        pytest.importorskip("transformer_engine")
        from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset
        from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater
        from nemo_automodel.components.distributed.thd_utils import process_input_for_thd
        from nemo_automodel.components.models.common.utils import get_rope_config
        from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
        from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd

        torch.manual_seed(0)
        # Shorter than every document, so the window limits attention.
        gpt_config.sliding_window = 8
        # Released GPT-OSS head dimension. Before cuDNN 9.26, TE's packed sink
        # backward is wrong for generic head dimensions such as the fixture's 32
        # when a call holds several documents (NVIDIA/TransformerEngine#3249).
        gpt_config.head_dim = 64
        attention = GptOssAttention(gpt_config, te_backend_config, use_sliding_attention=use_sliding_attention)
        attention = attention.to(device)
        attention.init_weights(device)
        rope_theta, rope_scaling, _ = get_rope_config(gpt_config)
        rotary_emb = RotaryEmbedding(
            head_dim=gpt_config.head_dim,
            base=rope_theta,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            device=device,
        )

        doc_lens = (17, 26, 35)
        docs = [{"input_ids": list(range(1, n + 1)), "labels": list(range(1, n + 1))} for n in doc_lens]
        thd = process_input_for_thd(packed_sequence_thd_collater(list(pack_dataset(docs, None, 48))))
        attn_keys = ("qkv_format", "cu_seqlens", "cu_seqlens_padded", "max_seqlen", "pad_between_seqs")
        _, position_ids, _, attn_kwargs = squeeze_input_for_thd(
            None, thd["position_ids"], None, {key: thd[key] for key in attn_keys}
        )
        attn_kwargs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in attn_kwargs.items()}
        # The first row's trailing padding sits between documents in the flattened stream.
        assert attn_kwargs["pad_between_seqs"]
        starts = attn_kwargs["cu_seqlens_padded"][:-1].tolist()
        spans = [slice(start, start + n) for start, n in zip(starts, doc_lens, strict=True)]
        num_tokens = position_ids.numel()
        assert num_tokens == 2 * 48
        packed_freqs_cis = position_ids_to_freqs_cis(
            rotary_emb, position_ids.to(device), qkv_format="thd", for_fused_rope=False
        )

        def run_packed(hidden_states: torch.Tensor) -> torch.Tensor:
            """Run packed TE THD attention.

            Args:
                hidden_states: Tensor of shape [tokens, hidden] holding both
                    flattened packed rows, including their padding.

            Returns:
                Tensor of shape [tokens, hidden]; padding positions are unspecified.
            """
            return attention(hidden_states, packed_freqs_cis, **attn_kwargs)

        def run_separate(hidden_states: torch.Tensor) -> torch.Tensor:
            """Run one document alone through TE BSHD attention.

            Args:
                hidden_states: Tensor of shape [sequence, hidden] for one document.

            Returns:
                Tensor of shape [sequence, hidden].
            """
            positions = torch.arange(hidden_states.shape[0], device=device).unsqueeze(0)
            freqs_cis = position_ids_to_freqs_cis(rotary_emb, positions, qkv_format="bshd", for_fused_rope=False)
            return attention(hidden_states.unsqueeze(0), freqs_cis).squeeze(0)

        x = torch.randn(num_tokens, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
        loss_weights = torch.randn_like(x)

        packed = run_packed(x)
        separate = [run_separate(x[span]) for span in spans]
        for span, expected in zip(spans, separate, strict=True):
            torch.testing.assert_close(packed[span], expected)

        # Both losses read only real tokens and receive identical upstream gradients.
        packed_loss = sum((packed[span] * loss_weights[span]).float().sum() for span in spans)
        separate_loss = sum((out * loss_weights[span]).float().sum() for span, out in zip(spans, separate, strict=True))
        params = dict(attention.named_parameters())
        assert any("softmax_offset" in name for name in params)
        packed_grads = torch.autograd.grad(packed_loss, list(params.values()))
        separate_grads = torch.autograd.grad(separate_loss, list(params.values()))
        for name, got, expected in zip(params, packed_grads, separate_grads, strict=True):
            # Parameter gradients accumulate the same BF16 products in a different
            # order. Observed relative L2 differences are below 0.5%.
            relative = (got.float() - expected.float()).norm() / expected.float().norm()
            assert relative < 2e-2, f"{name} gradient differs by relative L2 {relative.item():.3e}"

        padding = torch.ones(num_tokens, dtype=torch.bool, device=device)
        for span in spans:
            padding[span] = False
        changed = x.clone()
        changed[spans[0]] += torch.randn_like(changed[spans[0]])
        changed[padding] = 100 * torch.randn_like(changed[padding])
        with torch.no_grad():
            reference = run_packed(x)
            perturbed = run_packed(changed)
        assert not torch.equal(perturbed[spans[0]], reference[spans[0]])
        for span in spans[1:]:
            assert torch.equal(perturbed[span], reference[span])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_forward_with_sliding_window(self, gpt_config, te_backend_config, device):
        """Test TE attention forward pass with sliding window."""
        pytest.importorskip("transformer_engine")

        # Set sliding window in config
        gpt_config.sliding_window = 256

        from unittest.mock import patch

        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config, use_sliding_attention=True)
            attention = attention.to(device)

            assert attention.sliding_window == 256

            batch_size, seq_len = 2, 8
            x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
            freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

            try:
                output = attention(x, freqs_cis)
                assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            except Exception as e:
                pytest.fail(f"TE forward pass with sliding window failed: {e}")
