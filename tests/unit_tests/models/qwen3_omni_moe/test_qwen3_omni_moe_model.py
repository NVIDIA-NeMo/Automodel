"""Tests for Qwen3 Omni MoE model wrappers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration as HFQwen3OmniMoeThinkerForConditionalGeneration,
)

from nemo_automodel.components.models.qwen3_omni_moe.model import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def text_config():
    cfg = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=5000.0,
        router_aux_loss_coef=0.0,
        use_sliding_window=False,
    )
    cfg.torch_dtype = "float32"
    return cfg


@pytest.fixture
def moe_config(text_config):
    return MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.intermediate_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=0,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=text_config.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
    )


class IdentityLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.calls = 0

    def forward(self, x, **kwargs):
        self.calls += 1
        return x


@patch("nemo_automodel.components.models.qwen3_omni_moe.model.Qwen3OmniMoeThinkerTextRotaryEmbedding")
def test_text_model_forward_expands_position_ids(rotary_cls, text_config, backend_config, moe_config, device):
    calls = {}

    def rotary_side_effect(hidden_states, position_ids):
        calls["position_ids"] = position_ids
        zeros = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[-1], device=hidden_states.device)
        return zeros, zeros

    rotary_cls.return_value = MagicMock(side_effect=rotary_side_effect)
    model = Qwen3OmniMoeThinkerTextModel(text_config, backend=backend_config, moe_config=moe_config).to(device)
    model.layers = torch.nn.ModuleList(
        [IdentityLayer(text_config.hidden_size) for _ in range(text_config.num_hidden_layers)]
    )

    batch, seq_len = 2, 3
    input_ids = torch.randint(0, text_config.vocab_size, (batch, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    out = model(input_ids=input_ids, position_ids=position_ids)

    assert out.shape == (batch, seq_len, text_config.hidden_size)
    rotary_cls.return_value.assert_called_once()
    pos = calls["position_ids"]
    assert pos.shape[0] == 3
    assert pos.shape[-1] == seq_len
    assert pos[0].device == device
    for layer in model.layers:
        assert layer.calls == 1


def test_deepstack_process_adds_visual_embeddings(text_config, backend_config, moe_config, device):
    with patch(
        "nemo_automodel.components.models.qwen3_omni_moe.model.Qwen3OmniMoeThinkerTextRotaryEmbedding",
        return_value=MagicMock(side_effect=lambda x, y: (torch.zeros_like(x), torch.zeros_like(x))),
    ):
        model = Qwen3OmniMoeThinkerTextModel(text_config, backend=backend_config, moe_config=moe_config).to(device)

    hidden_states = torch.zeros(1, 4, text_config.hidden_size, device=device)
    visual_mask = torch.tensor([[[True], [False], [True], [False]]], device=device)
    visual_embeds = torch.ones(2, text_config.hidden_size, device=device)

    updated = model._deepstack_process(hidden_states, visual_mask, visual_embeds)

    assert torch.count_nonzero(updated[0, 0]) == text_config.hidden_size
    assert torch.count_nonzero(updated[0, 2]) == text_config.hidden_size
    assert torch.all(updated[0, 1] == 0)


@patch("nemo_automodel.components.models.qwen3_omni_moe.model.Qwen3OmniMoeThinkerTextRotaryEmbedding")
def test_text_model_init_weights_calls_layers(rotary_cls, text_config, backend_config, moe_config):
    rotary_cls.return_value = MagicMock(side_effect=lambda x, y: (torch.zeros_like(x), torch.zeros_like(x)))
    model = Qwen3OmniMoeThinkerTextModel(text_config, backend=backend_config, moe_config=moe_config)

    original = model.embed_tokens.weight.clone()
    for layer in model.layers:
        layer.init_weights = MagicMock()

    with patch.object(model.norm, "reset_parameters") as mock_norm:
        model.init_weights(buffer_device=torch.device("cpu"))

    mock_norm.assert_called_once()
    for layer in model.layers:
        layer.init_weights.assert_called_once()
    assert not torch.equal(original, model.embed_tokens.weight)


@pytest.fixture
def thinker_config(text_config):
    vision_config = SimpleNamespace(spatial_merge_size=2)
    return SimpleNamespace(text_config=text_config, vision_config=vision_config, pad_token_id=0)


def _stub_hf_init(self, *args, **kwargs):
    nn.Module.__init__(self)
    config = args[0] if args else kwargs.get("config")
    self.config = config


@patch.object(HFQwen3OmniMoeThinkerForConditionalGeneration, "__init__", new=_stub_hf_init)
@patch("nemo_automodel.components.models.qwen3_omni_moe.model.Qwen3OmniMoeThinkerTextRotaryEmbedding")
def test_thinker_forward_returns_logits(rotary_cls, thinker_config, backend_config, moe_config, device):
    rotary_cls.return_value = MagicMock(side_effect=lambda x, y: (torch.zeros_like(x), torch.zeros_like(x)))
    model = Qwen3OmniMoeThinkerForConditionalGeneration(thinker_config, moe_config=moe_config, backend=backend_config).to(device)
    model.config = thinker_config

    hidden_size = thinker_config.text_config.hidden_size
    vocab_size = thinker_config.text_config.vocab_size
    batch, seq_len = 2, 4

    hidden = torch.randn(
        batch,
        seq_len,
        hidden_size,
        device=device,
        dtype=model.lm_head.weight.dtype,
    )
    with patch.object(model.model, "forward", return_value=hidden) as mock_forward:
        input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        logits = model(input_ids=input_ids)

    assert logits.shape == (batch, seq_len, vocab_size)
    mock_forward.assert_called_once()


@patch.object(HFQwen3OmniMoeThinkerForConditionalGeneration, "__init__", new=_stub_hf_init)
@patch("nemo_automodel.components.models.qwen3_omni_moe.model.Qwen3OmniMoeThinkerTextRotaryEmbedding")
def test_thinker_forward_with_labels_returns_loss_dict(rotary_cls, thinker_config, backend_config, moe_config, device):
    rotary_cls.return_value = MagicMock(side_effect=lambda x, y: (torch.zeros_like(x), torch.zeros_like(x)))
    model = Qwen3OmniMoeThinkerForConditionalGeneration(thinker_config, moe_config=moe_config, backend=backend_config).to(device)
    model.config = thinker_config

    hidden_size = thinker_config.text_config.hidden_size
    vocab_size = thinker_config.text_config.vocab_size
    batch, seq_len = 2, 5

    hidden = torch.randn(
        batch,
        seq_len,
        hidden_size,
        device=device,
        dtype=model.lm_head.weight.dtype,
    )
    with patch.object(model.model, "forward", return_value=hidden) as mock_forward:
        input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)

        output = model(input_ids=input_ids, labels=labels, output_router_logits=True)

    assert isinstance(output, dict)
    assert "logits" in output and output["logits"].shape == (batch, seq_len, vocab_size)
    assert "loss" in output and output["loss"].dim() == 0
    assert "aux_loss" in output
    mock_forward.assert_called_once()


def test_modelclass_export_exists():
    from nemo_automodel.components.models.qwen3_omni_moe import model as omni_module

    assert hasattr(omni_module, "ModelClass")
    assert omni_module.ModelClass is Qwen3OmniMoeThinkerForConditionalGeneration

