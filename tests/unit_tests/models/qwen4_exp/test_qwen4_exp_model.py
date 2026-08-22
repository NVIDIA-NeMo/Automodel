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

"""CPU execution tests for the Qwen4-Exp HC decoder."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig
from nemo_automodel.components.models.qwen4_exp.engram import Qwen4ExpEngramTableConfig
from nemo_automodel.components.models.qwen4_exp.model import ModelClass, Qwen4ExpForConditionalGeneration
from nemo_automodel.components.moe.layers import MoEConfig


def _tiny_config() -> Qwen4ExpConfig:
    text = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        full_attention_interval=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        indexer_budget=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        # The main rotary width is 8 here, so the index head must be at least
        # that wide just like the released 64-rope/128-index configuration.
        indexer_head_dim=8,
        max_position_embeddings=16,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
        },
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    return Qwen4ExpConfig(text_config=text, language_model_only=True, tie_word_embeddings=False)


def _tiny_moe_config(config: Qwen4ExpTextConfig) -> MoEConfig:
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=1,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=config.shared_expert_intermediate_size,
        dtype=torch.float32,
    )


def test_model_class_alias_selects_qwen4_exp_conditional_generation() -> None:
    assert ModelClass is Qwen4ExpForConditionalGeneration


def test_multimodal_configuration_fails_closed() -> None:
    config = _tiny_config()
    config.language_model_only = False

    with pytest.raises(NotImplementedError, match="currently language-only"):
        Qwen4ExpForConditionalGeneration.from_config(
            config,
            moe_config=_tiny_moe_config(config.text_config),
            backend=BackendConfig(enable_hf_state_dict_adapter=False),
        )


def test_tiny_qwen4_exp_forward_backward_and_state_layout() -> None:
    config = _tiny_config()
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = Qwen4ExpForConditionalGeneration.from_config(
        config,
        moe_config=_tiny_moe_config(config.text_config),
        backend=backend,
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.train()

    input_ids = torch.randint(2, config.text_config.vocab_size, (2, 6))
    output = model(input_ids=input_ids, output_hidden_states=True)
    loss = output.logits.square().mean()
    loss.backward()

    assert output.logits.shape == (2, 6, config.text_config.vocab_size)
    assert output.hidden_states[0].shape == (2, 6, config.text_config.hidden_size)
    assert output.hidden_states[1].shape == (
        2,
        6,
        config.text_config.hc_count * config.text_config.hidden_size,
    )
    assert output.hidden_states[-1].shape == (2, 6, config.text_config.hidden_size)
    assert torch.isfinite(output.logits).all()
    assert model.get_input_embeddings().weight.grad is not None

    keys = model.state_dict()
    prefix = "model.language_model.layers.0"
    assert f"{prefix}.attn_hyper_connection.hc_norm.weight" in keys
    assert f"{prefix}.attn_hyper_connection.block_inject_weight.weight" in keys
    assert f"{prefix}.self_attn.indexer.index_qk_proj.weight" in keys
    assert f"{prefix}.input_layernorm.weight" not in keys
    assert "model.language_model.hyper_connection_mixer.block_inject_weight.weight" not in keys


def test_output_hidden_states_defaults_to_model_config() -> None:
    config = _tiny_config()
    config.output_hidden_states = True
    model = Qwen4ExpForConditionalGeneration.from_config(
        config,
        moe_config=_tiny_moe_config(config.text_config),
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        ),
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

    output = model(input_ids=torch.randint(2, config.text_config.vocab_size, (1, 6)))

    assert isinstance(output.hidden_states, torch.Tensor)
    assert output.hidden_states.shape == (1, 6, config.text_config.hidden_size)

    explicit_output = model(
        input_ids=torch.randint(2, config.text_config.vocab_size, (1, 6)),
        output_hidden_states=True,
    )
    assert isinstance(explicit_output.hidden_states, tuple)
    assert len(explicit_output.hidden_states) == config.text_config.num_hidden_layers + 2
    assert explicit_output.hidden_states[-1].shape == (1, 6, config.text_config.hidden_size)


def test_qsa_sparse_training_runs_above_budget() -> None:
    config = _tiny_config()
    config.text_config.indexer_budget = 4
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = Qwen4ExpForConditionalGeneration.from_config(
        config,
        moe_config=_tiny_moe_config(config.text_config),
        backend=backend,
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

    input_ids = torch.randint(2, config.text_config.vocab_size, (1, 9))
    output = model(input_ids=input_ids)
    output.logits.square().mean().backward()

    assert output.logits.shape == (1, 9, config.text_config.vocab_size)
    assert torch.isfinite(output.logits).all()
    attention = model.model.language_model.layers["0"].self_attn
    assert attention.q_proj.weight.grad is not None
    assert all(not parameter.requires_grad for parameter in attention.indexer.parameters())
    assert all(parameter.grad is None for parameter in attention.indexer.parameters())


def test_released_ple_table_requires_distributed_owner_group() -> None:
    config = _tiny_config()
    config.text_config.ple_layer_ids = [1]
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )

    with pytest.raises(RuntimeError, match="must be constructed after torch.distributed initialization"):
        Qwen4ExpForConditionalGeneration.from_config(
            config,
            moe_config=_tiny_moe_config(config.text_config),
            backend=backend,
        )


def test_engram_owner_markers_are_restored_after_meta_materialization() -> None:
    from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only

    config = _tiny_config()
    config.text_config.ple_layer_ids = [1]
    config.text_config.ngram_size = 3
    config.text_config.heads_per_ngram = 8
    config.text_config.ple_embed_dim = 32
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    table_config = Qwen4ExpEngramTableConfig(num_embeddings=36, embedding_dim=2, initializer_range=0.05)
    with torch.device("meta"):
        model = Qwen4ExpForConditionalGeneration.from_config(
            config,
            moe_config=_tiny_moe_config(config.text_config),
            backend=backend,
            engram_table_config=table_config,
        )

    table = model.model.language_model.layers["0"].ple.ple_embedding.ngram_embedding
    assert model.model.language_model.layers["0"]._nemo_disable_activation_checkpointing is True
    original_weight = table.weight
    to_empty_parameters_only(model, device=torch.device("cpu"))
    assert table.weight is not original_weight
    assert not hasattr(table.weight, "_nemo_owner_sharded_spec")
    with torch.no_grad():
        table.weight.fill_(torch.nan)

    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    assert torch.isfinite(table.weight).all()
    assert torch.count_nonzero(table.weight) > 0
    owner_spec = table.weight._nemo_owner_sharded_spec
    assert owner_spec.process_group is None
    assert owner_spec.gradient_divisor == 1.0
    assert owner_spec.optimizer_state_namespace == "__nemo_engram_owner_v1"
