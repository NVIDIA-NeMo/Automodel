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

    @pytest.mark.skip(reason="THD format is not supported for GPT-OSS")
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


@pytest.mark.cpu_only
def test_fa4_preserves_te_sink_owner_order_and_window(
    monkeypatch: pytest.MonkeyPatch, gpt_config: GptOssConfig
) -> None:
    """Actual layer construction preserves checkpoint keys and forwards the original window."""
    import sys
    from types import SimpleNamespace

    import nemo_automodel.components.attention.utils as attention_utils
    import nemo_automodel.components.models.gpt_oss.layers as layers_module

    class SinkOwner(torch.nn.Module):
        """Mock only the external TE parameter owner, not GPT construction."""

        def __init__(self, num_attention_heads, **kwargs):
            super().__init__()
            self.softmax_offset = torch.nn.Parameter(torch.zeros(num_attention_heads))

    calls = []

    def external_kernel(q, k, v, **kwargs):
        """Keep gradient connections while recording external call arguments.

        Args:
            q: Query tensor [tokens, heads, channels].
            k: Key tensor [tokens, kv_heads, channels].
            v: Value tensor [tokens, kv_heads, channels].
            **kwargs: Backend options including sink [heads].

        Returns:
            Mock output [tokens, heads, channels] and absent LSE. This is not attention math.
        """
        calls.append(kwargs)
        ratio = q.shape[-2] // k.shape[-2]
        return q + k.repeat_interleave(ratio, -2) + v.repeat_interleave(ratio, -2) + kwargs["learnable_sink"].to(
            q.dtype
        )[None, :, None], None

    external = SimpleNamespace(flash_attn_func=external_kernel, flash_attn_varlen_func=external_kernel)
    modules = {
        "flash_attn.cute": external,
        "transformer_engine.pytorch.attention": SimpleNamespace(DotProductAttention=SinkOwner),
        "transformer_engine.pytorch.quantization": SimpleNamespace(),
    }
    monkeypatch.setitem(
        sys.modules, "transformer_engine.pytorch.attention", modules["transformer_engine.pytorch.attention"]
    )
    monkeypatch.setattr(attention_utils, "safe_import", lambda name: (name in modules, modules.get(name)))
    monkeypatch.setattr(layers_module, "is_te_min_version", lambda version: True)
    monkeypatch.setattr(layers_module, "apply_rotary_emb_qk", lambda q, k, *args, **kwargs: (q, k))
    gpt_config.sliding_window = 128
    layers = {}
    for backend in ("te", "fa4"):
        torch.manual_seed(909)
        layers[backend] = GptOssAttention(
            gpt_config, BackendConfig(attn=backend, linear="torch", rope_fusion=False), True
        )
    assert list(layers["te"].state_dict()) == list(layers["fa4"].state_dict())
    assert list(dict(layers["te"].named_parameters())) == list(dict(layers["fa4"].named_parameters()))
    for expected, actual in zip(layers["te"].parameters(), layers["fa4"].parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum("softmax_offset" in name for name, _ in layers["fa4"].named_parameters()) == 1
    assert layers["fa4"].sinks is None
    x = torch.randn(2, gpt_config.hidden_size, dtype=layers["fa4"].q_proj.weight.dtype, requires_grad=True)
    output = layers["fa4"](
        x,
        torch.ones(2, gpt_config.head_dim),
        qkv_format="thd",
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        _fa4_cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        _fa4_max_seqlen=2,
        packed_token_indices=torch.tensor([0, 1]),
    )
    output.sum().backward()
    assert calls[-1]["window_size"] == (127, None)
    for name, parameter in layers["fa4"].named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(layers["fa4"].attn_module.softmax_offset.grad)


@pytest.mark.parametrize(
    "change",
    [None, "small_valid", "softcap", "window", "sink_dtype", "sink_device", "missing_used", "cpu", "other_arch"],
)
def test_fa4_physical_canonical_scope(change: str | None) -> None:
    from types import SimpleNamespace

    import nemo_automodel.components.models.gpt_oss.layers as layers_module

    device = torch.device("cuda:0")
    q = SimpleNamespace(shape=(32768, 64, 64), dtype=torch.bfloat16, is_cuda=True, device=device)
    k = SimpleNamespace(shape=(32768, 8, 64), dtype=torch.bfloat16)
    v = SimpleNamespace(shape=(32768, 8, 64), dtype=torch.bfloat16)
    sink = SimpleNamespace(shape=(64,), dtype=torch.bfloat16, device=device)
    kwargs = dict(
        _fa4_seqused=torch.ones(512, dtype=torch.int32),
        packed_token_indices=torch.arange(32768),
        _fa4_physical_cu_seqlens=torch.arange(513, dtype=torch.int32),
        learnable_sink=sink,
        window_size=(128, 0),
    )
    if change == "small_valid":
        kwargs["packed_token_indices"] = torch.arange(128)
    if change == "softcap":
        kwargs["softcap"] = 1.0
    if change == "window":
        kwargs["window_size"] = (65, 0)
    if change == "sink_dtype":
        sink.dtype = torch.float32
    if change == "sink_device":
        sink.device = torch.device("cpu")
    if change == "missing_used":
        kwargs.pop("_fa4_seqused")
    if change == "cpu":
        q.is_cuda = False
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0) if change == "other_arch" else (10, 0)):
        assert layers_module._use_physical_fa4_layout(q, k, v, kwargs) is (change is None)


@pytest.mark.cpu_only
@pytest.mark.parametrize("layout", ["bshd", "thd"])
def test_fa4_sliding_window_reference_boundary(monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    """A width of 128 excludes key zero at query 128, including its gradients."""
    import nemo_automodel.components.models.gpt_oss.layers as layers_module

    class SinkOwner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.softmax_offset = torch.nn.Parameter(torch.zeros(1))

    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        window_size: tuple[int, None],
        learnable_sink: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Evaluate the external kernel's inclusive-distance contract on CPU.

        Args:
            q: Query tensor [1, sequence, heads, channels] or [tokens, heads, channels].
            k: Key tensor with the same layout as q in this single-head fixture.
            v: Value tensor with the same layout as q.
            window_size: Inclusive left distance and unbounded right distance.
            learnable_sink: Sink logits [heads].
            **kwargs: Unused backend metadata.

        Returns:
            Attention output with the same layout as q.
        """
        dense = q.ndim == 4
        q, k, v = (t.squeeze(0) if dense else t for t in (q, k, v))
        positions = torch.arange(q.shape[0])
        left, _ = window_size
        allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] >= positions[:, None] - left)
        scores = torch.einsum("qhd,khd->hqk", q, k) * 0.5
        scores = scores.masked_fill(~allowed, float("-inf"))
        sink = learnable_sink[:, None, None].expand(-1, q.shape[0], 1)
        weights = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
        output = torch.einsum("hqk,khd->qhd", weights, v)
        return output.unsqueeze(0) if dense else output

    monkeypatch.setattr(layers_module, "is_te_min_version", lambda version: True)
    monkeypatch.setattr(layers_module, "initialize_attn_module_and_func", lambda **kwargs: (SinkOwner(), kernel))
    config = GptOssConfig(hidden_size=4, num_attention_heads=1, num_key_value_heads=1, head_dim=4, sliding_window=128)
    attention = GptOssAttention(config, BackendConfig(attn="fa4", linear="torch", rope_fusion=False), True).double()
    with torch.no_grad():
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
            projection.weight.zero_()
            projection.bias.zero_()
        attention.v_proj.weight.copy_(torch.eye(4))
        attention.o_proj.weight.copy_(torch.eye(4))
    x = torch.zeros(130, 4, dtype=torch.float64)
    x[0, 0] = 1
    x.requires_grad_()
    # Identity RoPE: cosine ones followed by sine zeros.
    frequencies = torch.cat((torch.ones(130, 2), torch.zeros(130, 2)), dim=-1)
    metadata = {}
    if layout == "thd":
        metadata = dict(
            qkv_format="thd",
            _fa4_cu_seqlens=torch.tensor([0, 130], dtype=torch.int32),
            _fa4_max_seqlen=130,
            packed_token_indices=torch.arange(130),
        )
    else:
        x = x.unsqueeze(0)
        frequencies = frequencies.unsqueeze(0)
    output = attention(x, frequencies, **metadata).reshape(130, 4)
    # Independent model contract: 128 causal keys including self, plus the sink.
    expected = torch.zeros_like(output)
    expected[:128, 0] = 1 / (torch.arange(128, dtype=torch.float64) + 2)
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-15)
    hidden_grad, sink_grad = torch.autograd.grad(output[128, 0], (x, attention.attn_module.softmax_offset))
    assert torch.count_nonzero(hidden_grad.reshape(130, 4)[0]) == 0
    assert torch.count_nonzero(sink_grad) == 0
