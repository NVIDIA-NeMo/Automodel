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

"""CPU coverage for the dense Kimi K3 DSpark draft."""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig
from nemo_automodel.components.speculative.dspark.config import build_kimi_k3_draft_config
from nemo_automodel.components.speculative.dspark.draft_kimi_k3 import KimiK3DSparkModel
from nemo_automodel.components.speculative.dspark.loss import compute_dspark_loss
from nemo_automodel.components.speculative.dspark.registry import resolve_dspark_draft_spec
from nemo_automodel.recipes.llm import _dspark_target_build

VOCAB = 64
HIDDEN = 32
TARGET_LAYERS = [1, 3]


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def _target_config(wrapped: bool = False):
    text_config = KimiK3TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        linear_attn_config={"kda_layers": [1, 2, 3], "full_attn_layers": [4]},
    )
    return KimiK3Config(text_config=text_config) if wrapped else text_config


def _draft_config(wrapped: bool = False):
    return build_kimi_k3_draft_config(
        _target_config(wrapped),
        _Args(
            num_draft_layers=2,
            target_layer_ids=TARGET_LAYERS,
            block_size=3,
            mask_token_id=5,
            num_anchors=4,
            markov_rank=8,
            markov_head_type="vanilla",
            confidence_head_alpha=1.0,
            confidence_head_with_markov=True,
        ),
    )


def _model(config=None):
    model = KimiK3DSparkModel(config if config is not None else _draft_config()).eval()
    model.initialize_embeddings_and_head(
        embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        freeze=True,
    )
    return model


@pytest.mark.parametrize("wrapped", [False, True])
def test_config_builder_uses_dense_mla(wrapped):
    config = _draft_config(wrapped)
    assert config.architectures == ["KimiK3DSparkModel"]
    assert config.num_hidden_layers == 2
    assert config.num_target_layers == 4
    assert config.num_experts is None
    assert config.num_shared_experts == 0
    assert config.linear_attn_config == {"kda_layers": [], "full_attn_layers": [1, 2]}
    assert config._attn_implementation == "sdpa"


def test_forward_loss_and_gradients():
    model = _model()
    generator = torch.Generator().manual_seed(7)
    batch = {
        "input_ids": torch.randint(0, VOCAB, (2, 12), generator=generator),
        "loss_mask": torch.ones(2, 12, dtype=torch.uint8),
        "target_hidden_states": torch.randn(2, 12, len(TARGET_LAYERS) * HIDDEN, generator=generator),
        "target_last_hidden_states": torch.randn(2, 12, HIDDEN, generator=generator),
    }
    torch.manual_seed(11)
    output = model(**batch)
    assert output.draft_logits.shape == (2, 4, 3, VOCAB)
    assert output.confidence_pred.shape == (2, 4, 3)
    assert torch.isfinite(output.draft_logits).all()

    loss = compute_dspark_loss(
        outputs=output,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    )
    loss.backward()
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert torch.isfinite(model.layers[0].self_attn.q_a_proj.weight.grad).all()


def test_attention_rejects_rope_only_mla():
    config = _draft_config()
    config.mla_use_nope = False
    with pytest.raises(ValueError, match="mla_use_nope"):
        KimiK3DSparkModel(config)


def test_forward_without_q_lora_rank():
    config = _draft_config()
    config.q_lora_rank = None
    model = _model(config)
    attention = model.layers[0].self_attn
    assert not hasattr(attention, "q_a_proj")
    assert attention.q_proj.out_features == config.num_attention_heads * (
        config.qk_nope_head_dim + config.qk_rope_head_dim
    )

    generator = torch.Generator().manual_seed(13)
    output = model(
        input_ids=torch.randint(0, VOCAB, (2, 12), generator=generator),
        loss_mask=torch.ones(2, 12, dtype=torch.uint8),
        target_hidden_states=torch.randn(2, 12, len(TARGET_LAYERS) * HIDDEN, generator=generator),
        target_last_hidden_states=torch.randn(2, 12, HIDDEN, generator=generator),
    )
    assert output.draft_logits.shape == (2, 4, 3, VOCAB)
    assert torch.isfinite(output.draft_logits).all()


@pytest.mark.parametrize("architecture", ["KimiK3ForCausalLM", "KimiK3ForConditionalGeneration"])
def test_registry_resolves_kimi_k3(architecture):
    assert resolve_dspark_draft_spec([architecture]).draft_cls is KimiK3DSparkModel


def test_target_backend_defaults_and_overrides():
    backend = _dspark_target_build.build_kimi_k3_backend({})
    assert backend.attn == "eager"
    assert backend.dispatcher == "hybridep"
    assert backend.experts == "torch_mm"

    backend = _dspark_target_build.build_kimi_k3_backend(
        {
            "target_attn_backend": "sdpa",
            "target_dispatcher": "deepep",
            "target_experts": "gmm",
            "target_enable_fsdp_optimizations": False,
        }
    )
    assert backend.attn == "sdpa"
    assert backend.dispatcher == "deepep"
    assert backend.experts == "gmm"
    assert backend.enable_fsdp_optimizations is False


def test_target_builder_requires_cuda():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        _dspark_target_build.build_kimi_k3_target(
            cfg={},
            world_size=1,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            target_path="unused",
            recipe_cfg={},
            trust_remote_code=False,
        )


def test_target_builder_uses_text_config_and_distributed_loader(monkeypatch):
    target_config = KimiK3Config(text_config=_target_config())
    distributed_setup = object()
    target_model = object()
    captured = {}

    monkeypatch.setattr(
        _dspark_target_build.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: target_config,
    )
    monkeypatch.setattr(
        _dspark_target_build,
        "create_distributed_setup_from_config",
        lambda cfg, world_size: distributed_setup,
    )

    def _from_config(**kwargs):
        captured.update(kwargs)
        return target_model

    monkeypatch.setattr(_dspark_target_build.NeMoAutoModelForCausalLM, "from_config", _from_config)
    config, model, setup = _dspark_target_build.build_kimi_k3_target(
        cfg={"distributed": {}},
        world_size=64,
        device=SimpleNamespace(type="cuda"),
        compute_dtype=torch.bfloat16,
        target_path="moonshotai/Kimi-K3",
        recipe_cfg={"target_num_hidden_layers": 3},
        trust_remote_code=False,
    )
    assert config is target_config.text_config
    assert config.num_hidden_layers == 3
    assert config.linear_attn_config == {"kda_layers": [1, 2, 3], "full_attn_layers": []}
    assert config.architectures == ["KimiK3ForCausalLM"]
    assert config.name_or_path == "moonshotai/Kimi-K3"
    assert model is target_model
    assert setup is distributed_setup
    assert captured["config"] is config
    assert captured["distributed_setup"] is distributed_setup
    assert captured["load_base_model"] is True
