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

"""CPU tests for the Qwen3.8-Flash-Next pipeline-parallel stage contract."""

import copy
import inspect
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig,
    Qwen3_8_FlashNextTextConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
from nemo_automodel.components.models.qwen3_8_flash_next.parallelization import PARALLELIZER
from nemo_automodel.components.moe.layers import MoEConfig

LAYERS_PREFIX = "model.language_model."


def _two_layer_config() -> Qwen3_8_FlashNextConfig:
    """One GatedDeltaNet layer followed by one QSA layer, so the stage cut sits between layer types."""
    text = Qwen3_8_FlashNextTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        full_attention_interval=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
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
        indexer_head_dim=8,
        max_position_embeddings=16,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    return Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True, tie_word_embeddings=False)


def _moe_config(config: Qwen3_8_FlashNextTextConfig) -> MoEConfig:
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


def _reference_kernel(hub_function: Callable) -> Callable:
    """Return the pure-PyTorch body behind a Transformers kernel-hub wrapper.

    With ``fla`` installed the wrapper forwards to CUDA-only kernels even on CPU
    tensors. ``functools.wraps`` keeps the reference body as ``__wrapped__``;
    drop keyword arguments it does not accept, as the wrapper does.
    """
    reference = getattr(hub_function, "__wrapped__", hub_function)
    accepted = tuple(inspect.signature(reference).parameters)

    def call(*args, **kwargs):
        return reference(*args, **{key: value for key, value in kwargs.items() if key in accepted})

    return call


def _build_model() -> Qwen3_8_FlashNextForConditionalGeneration:
    config = _two_layer_config()
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
        config,
        moe_config=_moe_config(config.text_config),
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        ),
    )
    torch.manual_seed(0)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()
    # The GDN layer prefers installed CUDA kernels; on CPU use the reference ops.
    # ``causal_conv1d_fn=None`` selects the layer's own torch convolution path.
    for layer in model.model.language_model.layers.values():
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        linear_attn.causal_conv1d_fn = None
        linear_attn.causal_conv1d_update = None
        linear_attn.chunk_gated_delta_rule = _reference_kernel(torch_chunk_gated_delta_rule)
        linear_attn.recurrent_gated_delta_rule = _reference_kernel(torch_recurrent_gated_delta_rule)
    return model


def _split_into_two_stages(
    model: Qwen3_8_FlashNextForConditionalGeneration,
) -> tuple[Qwen3_8_FlashNextForConditionalGeneration, Qwen3_8_FlashNextForConditionalGeneration]:
    """Mimic the pipeline stage builder: deep-copy, then drop the modules a stage does not own."""
    first = copy.deepcopy(model)
    del first.model.language_model.layers["1"]
    first.model.language_model.hyper_connection_mixer = None
    first.lm_head = None

    last = copy.deepcopy(model)
    del last.model.language_model.layers["0"]
    last.model.language_model.embed_tokens = None
    return first, last


def test_declares_pipeline_support() -> None:
    assert Qwen3_8_FlashNextForConditionalGeneration._pp_keep_self_forward is True
    assert Qwen3_8_FlashNextForConditionalGeneration.ModelCapabilities().supports_pp is True


def test_stage_plan_keeps_hc_mixer_on_last_stage() -> None:
    model = _build_model()
    planned = [
        [f"{LAYERS_PREFIX}embed_tokens", f"{LAYERS_PREFIX}layers.0", f"{LAYERS_PREFIX}rotary_emb"],
        [f"{LAYERS_PREFIX}layers.1", f"{LAYERS_PREFIX}norm", "lm_head", f"{LAYERS_PREFIX}rotary_emb"],
    ]

    customized = model.customize_pipeline_stage_modules(planned, layers_prefix=LAYERS_PREFIX)

    assert customized[0] == planned[0]
    assert customized[1] == planned[1] + [f"{LAYERS_PREFIX}hyper_connection_mixer"]
    # The planner input is not mutated.
    assert "hyper_connection_mixer" not in " ".join(planned[1])


def test_stage_metas_exchange_hc_width() -> None:
    model = _build_model()
    text = model.config.text_config
    hc_width = text.hc_count * text.hidden_size
    first, last = _split_into_two_stages(model)

    inputs_meta, outputs_meta = first.get_pipeline_stage_metas(
        is_first=True, microbatch_size=2, seq_len=5, dtype=torch.float32
    )
    assert inputs_meta[0].shape == (2, 5) and inputs_meta[0].dtype == torch.long
    assert outputs_meta[0].shape == (2, 5, hc_width)

    inputs_meta, outputs_meta = last.get_pipeline_stage_metas(
        is_first=False, microbatch_size=2, seq_len=5, dtype=torch.float32
    )
    assert inputs_meta[0].shape == (2, 5, hc_width)
    assert outputs_meta[0].shape == (2, 5, text.vocab_size)


def test_two_stage_forward_matches_single_model() -> None:
    model = _build_model()
    text = model.config.text_config
    first, last = _split_into_two_stages(model)
    input_ids = torch.randint(2, text.vocab_size, (2, 6))

    with torch.no_grad():
        reference = model(input_ids=input_ids).logits
        stage_output = first(input_ids=input_ids).logits
        assert stage_output.shape == (2, 6, text.hc_count * text.hidden_size)
        # The schedule hands the previous stage's tensor to the next stage positionally.
        logits = last(stage_output).logits

    assert logits.shape == (2, 6, text.vocab_size)
    torch.testing.assert_close(logits, reference)


def test_last_stage_backward_reaches_incoming_state() -> None:
    model = _build_model()
    text = model.config.text_config
    _, last = _split_into_two_stages(model)
    incoming = torch.randn(1, 4, text.hc_count * text.hidden_size, requires_grad=True)

    last(incoming).logits.square().mean().backward()

    assert incoming.grad is not None
    assert torch.isfinite(incoming.grad).all()


@pytest.mark.parametrize(("ple_layer_ids", "rejected"), [([2], True), ([], False)])
def test_parallelizer_rejects_ple_on_dense_fsdp2(monkeypatch, ple_layer_ids, rejected) -> None:
    """Every stage reads the config, so a stage without the PLE layer rejects dense FSDP2 too."""
    parallelized = object()
    monkeypatch.setattr(ModelParallelizer, "_apply", lambda self, model, device_mesh, **kwargs: parallelized)
    stage = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(ple_layer_ids=ple_layer_ids)))

    assert Qwen3_8_FlashNextForConditionalGeneration.parallelizer is PARALLELIZER
    # The MoE (ep_size > 1) dispatch only calls sidecars that customize MoE FSDP, so it never reaches _apply.
    assert PARALLELIZER._customizes_moe_fsdp is False
    if rejected:
        with pytest.raises(ValueError, match="requires distributed.ep_size > 1"):
            PARALLELIZER._apply(stage, device_mesh=None)
    else:
        assert PARALLELIZER._apply(stage, device_mesh=None) is parallelized
