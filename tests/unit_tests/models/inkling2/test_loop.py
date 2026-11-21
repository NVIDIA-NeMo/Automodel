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

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config
from nemo_automodel.components.models.inkling2.config import (
    Inkling2Config,
    Inkling2LoopConfig,
    Inkling2LoopConfiguration,
    Inkling2TextConfig,
)
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.models.inkling2.multimodal import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.state_dict_adapter import Inkling2StateDictAdapter

from . import reference

LOOP: Inkling2LoopConfig = {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": False}


def looped_config(loop: Inkling2LoopConfig | None = None) -> Inkling2TextConfig:
    """Eight layers in blocks of two: local layers 2, 4, 5 and full-attention layer 3 form the loop core."""
    return Inkling2TextConfig(
        hidden_size=16,
        num_hidden_layers=8,
        vocab_size=40,
        unpadded_vocab_size=37,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=3,
        rel_extent=4,
        sliding_window_size=7,
        local_layer_ids=[0, 1, 2, 4, 5, 6],
        n_layers_per_attn_res_block=2,
        n_routed_experts=12,
        num_experts_per_tok=8,
        intermediate_size=8,
        log_scaling_n_floor=4,
        dtype="float32",
        architectures=["Inkling2ForCausalLM"],
        loop_config=LOOP if loop is None else loop,
    )


def looped_model(config: Inkling2TextConfig) -> Inkling2ForCausalLM:
    """Build the model with nonzero queries and gates so every message path changes the output."""
    model = Inkling2ForCausalLM(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "quer" in name:
                parameter.normal_(std=0.5)
            elif "controllers" in name:
                parameter.normal_(std=0.3)
    return model


@pytest.mark.parametrize("seed", [17, 41])
def test_looped_outputs_and_all_parameter_gradients(seed: int) -> None:
    torch.manual_seed(seed)
    config = looped_config()
    model = looped_model(config)
    parameters = dict(model.named_parameters())
    oracle_weights = {
        name: value.detach().clone().requires_grad_(name in parameters) for name, value in model.state_dict().items()
    }
    ids = torch.randint(0, config.unpadded_vocab_size, (1, 9))
    visits: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(lambda module, inputs, output: visits.append(output.partial.detach()))
        for layer in model.layers.values()
    ]
    try:
        actual = model(ids)
    finally:
        for handle in handles:
            handle.remove()
    expected = reference.forward(config, oracle_weights, ids)
    # Two prelude layers, four core layers three times, two coda layers.
    assert len(visits) == len(expected.partials) == 16
    for index, (native, wanted) in enumerate(zip(visits, expected.partials, strict=True)):
        torch.testing.assert_close(native, wanted, rtol=2e-4, atol=1e-5, msg=f"layer visit {index}")
    torch.testing.assert_close(actual, expected.logits, rtol=2e-4, atol=1e-5)
    upstream = torch.randn_like(actual[..., :37])
    (actual[..., :37] * upstream).sum().backward()
    (expected.logits[..., :37] * upstream).sum().backward()
    for name, parameter in parameters.items():
        wanted = oracle_weights[name].grad
        assert wanted is not None and parameter.grad is not None, name
        # The loop core adds three passes of contributions; the absolute bound follows the gradient scale.
        scale = max(1.0, wanted.abs().max().item())
        torch.testing.assert_close(parameter.grad, wanted, rtol=3e-4, atol=2e-5 * scale, msg=name)
    for name in ("message_queries.weight", "message_kv_queries.weight", "layers.3.controllers.v.down_weight"):
        assert parameters[name].grad.abs().sum() > 0, name


def test_each_pass_and_each_message_changes_the_output() -> None:
    torch.manual_seed(5)
    config = looped_config()
    model = looped_model(config)
    ids = torch.randint(0, config.unpadded_vocab_size, (1, 9))
    baseline = model(ids)[..., :37]
    fewer = Inkling2ForCausalLM(looped_config({**LOOP, "num_passes": 2}))
    fewer.load_state_dict(model.state_dict())
    assert (fewer(ids)[..., :37] - baseline).abs().max() > 1e-3
    for name in ("message_queries", "message_kv_queries"):
        changed = Inkling2ForCausalLM(config)
        changed.load_state_dict(model.state_dict())
        with torch.no_grad():
            weight = changed.get_parameter(name + ".weight")
            # Swap the two queries of every core layer: attention and MoE, or K and V.
            weight.copy_(weight.unflatten(0, (-1, 2)).flip(1).flatten(0, 1))
        assert (changed(ids)[..., :37] - baseline).abs().max() > 1e-3, name


def test_loop_parameters_keep_their_checkpoint_names() -> None:
    config = looped_config()
    model = looped_model(config)
    adapter = Inkling2StateDictAdapter(config)
    state = {name: value.detach() for name, value in model.state_dict().items()}
    exported = adapter.to_hf(state)
    assert exported["model.llm.middle_cycle_cross_pass_queries"].shape == (8, 16)
    assert exported["model.llm.middle_cycle_cross_pass_kv_queries"].shape == (8, 16)
    controllers = sorted(key for key in exported if ".controllers." in key)
    assert len(controllers) == 4 * 4 * 3
    assert controllers[0] == "model.llm.layers.2.controllers.attn.down_weight"
    assert not any(
        key.startswith(("model.llm.layers.1.controllers", "model.llm.layers.6.controllers")) for key in exported
    )
    restored = adapter.from_hf(exported)
    assert restored.keys() == state.keys()
    for name, value in state.items():
        assert torch.equal(restored[name], value), name


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"core_start": 1}, "whole attention residual blocks"),
        ({"core_end": 10}, "whole attention residual blocks"),
        ({"num_passes": 1}, "at least two passes"),
        ({"gate_rank": 0}, "positive gate rank"),
        ({"is_varloop": True}, "variable number of passes"),
    ],
)
def test_unsupported_loop_settings_are_rejected(change: dict[str, int | bool], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        looped_config({**LOOP, **change})


def test_looped_checkpoint_config_resolution(tmp_path: Path) -> None:
    text = looped_config().to_dict()
    raw = {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "text_config": text,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = AutoConfig.from_pretrained(tmp_path)
    assert type(config) is Inkling2LoopConfiguration
    assert _resolve_custom_model_cls_for_config(config) is Inkling2ForConditionalGeneration
    assert config.text_config.loop_config == LOOP
    assert list(config.text_config.loop_core_layers) == [2, 3, 4, 5]
    config.save_pretrained(tmp_path / "saved")
    assert AutoConfig.from_pretrained(tmp_path / "saved").text_config.loop_config == LOOP
    with pytest.raises(ValueError, match="requires model_type inkling2_loop"):
        Inkling2Config(text_config=text)


def test_looped_model_rejects_pipeline_parallelism() -> None:
    model = Inkling2ForCausalLM(looped_config())
    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        model.customize_pipeline_stage_modules([["layers.0"], ["layers.1"]], layers_prefix="")
    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        model.get_pipeline_stage_metas(is_first=True, microbatch_size=1, seq_len=8, dtype=torch.float32)
