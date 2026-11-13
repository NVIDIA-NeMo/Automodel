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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.text import Inkling2FullDecoderLayer, Inkling2SlidingDecoderLayer

from .parity_check_inkling2 import TINY_TEXT_CONFIG, _randomize

pytestmark = pytest.mark.cpu_only

_VOCAB = TINY_TEXT_CONFIG["unpadded_vocab_size"]


def _config(dtype: str = "float32", **text_overrides: object) -> Inkling2Config:
    return Inkling2Config(text_config={**TINY_TEXT_CONFIG, **text_overrides}, vision_config=None, torch_dtype=dtype)


def _model(attn: str = "eager", dtype: str = "float32", seed: int = 1, **text_overrides: object):
    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    model = Inkling2ForConditionalGeneration(_config(dtype, **text_overrides), backend=backend)
    _randomize(model, seed=seed)
    return model


def _input_ids(batch: int = 2, sequence: int = 20, seed: int = 0) -> torch.Tensor:
    return torch.randint(2, _VOCAB, (batch, sequence), generator=torch.Generator().manual_seed(seed))


def test_layer_layout_follows_layer_types() -> None:
    layers = _model().model.language_model.layers
    kinds = [type(layer) for layer in layers]
    assert kinds == [
        Inkling2SlidingDecoderLayer,
        Inkling2SlidingDecoderLayer,
        Inkling2FullDecoderLayer,
        Inkling2SlidingDecoderLayer,
        Inkling2SlidingDecoderLayer,
        Inkling2SlidingDecoderLayer,
        Inkling2FullDecoderLayer,
    ]
    assert [layer.is_block_boundary for layer in layers] == [True, False, False, False, False, False, True]
    assert layers[0].attention_input is None
    assert layers[2].attention_input.query.shape == (3, TINY_TEXT_CONFIG["hidden_size"])
    assert layers[1].attention_input.query.shape == (TINY_TEXT_CONFIG["hidden_size"],)


def test_forward_returns_fp32_softcapped_unpadded_logits_and_backward_reaches_every_parameter() -> None:
    model = _model()
    logits = model(input_ids=_input_ids()).logits

    assert logits.shape == (2, 20, _VOCAB)
    assert logits.dtype == torch.float32
    assert logits.abs().max() <= TINY_TEXT_CONFIG["final_logit_softcapping"]
    (logits * torch.randn_like(logits)).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_logits_to_keep_returns_trailing_positions() -> None:
    model = _model()
    input_ids = _input_ids()
    with torch.no_grad():
        full = model(input_ids=input_ids).logits
        last = model(input_ids=input_ids, logits_to_keep=3).logits
    assert last.shape == (2, 3, _VOCAB)
    # The head GEMM over fewer rows may round differently, so compare at FP32 tolerance.
    torch.testing.assert_close(last, full[:, -3:], rtol=1e-5, atol=1e-6)


def test_checkpoint_free_initialization_is_finite_and_trains() -> None:
    backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    with torch.device("meta"):
        model = Inkling2ForConditionalGeneration(_config(), backend=backend)
    model.to_empty(device="cpu")
    torch.manual_seed(0)
    model.initialize_weights(torch.device("cpu"))

    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        assert torch.isfinite(tensor).all(), name
        if name.endswith(("query", "static_weight", "e_score_correction_bias")):
            assert torch.count_nonzero(tensor) == 0, name
        if name.endswith(("norm.weight", "layernorm.weight", "l2_clip.scale")):
            assert torch.all(tensor == 1), name
    logits = model(input_ids=_input_ids()).logits
    logits.float().sum().backward()
    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_sdpa_matches_eager_forward_and_backward() -> None:
    eager, sdpa = _model("eager"), _model("sdpa")
    input_ids = _input_ids()
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :4] = 0
    upstream = torch.randn(2, 20, _VOCAB, generator=torch.Generator().manual_seed(3))
    valid = attention_mask.bool()
    outputs = []
    for model in (eager, sdpa):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        (logits * upstream)[valid].sum().backward()
        outputs.append(logits)

    torch.testing.assert_close(outputs[0][valid], outputs[1][valid], rtol=1e-5, atol=1e-5)
    sdpa_params = dict(sdpa.named_parameters())
    for name, parameter in eager.named_parameters():
        torch.testing.assert_close(parameter.grad, sdpa_params[name].grad, rtol=1e-4, atol=1e-4, msg=name)


def test_bf16_model_keeps_reference_fp32_modules() -> None:
    model = _model(dtype="bfloat16")
    dtypes = {name: tensor.dtype for name, tensor in [*model.named_parameters(), *model.named_buffers()]}
    layer = "model.language_model.layers"
    # The modules the reference keeps in FP32.
    for name in (
        f"{layer}.0.self_attn.k_sconv.conv1d.weight",
        f"{layer}.0.self_attn.v_sconv.conv1d.weight",
        f"{layer}.0.post_attention_convolution.conv1d.weight",
        f"{layer}.0.post_mlp_convolution.conv1d.weight",
        f"{layer}.0.mlp_input.query",
        f"{layer}.1.attention_input.query",
        f"{layer}.2.attention_input.query",
        f"{layer}.0.mlp.gate.weight",
        f"{layer}.0.mlp.gate.e_score_correction_bias",
        "model.language_model.final_input.query",
    ):
        assert dtypes[name] == torch.float32, name
    # Everything else, including the dynamic convolutions of full layers, uses the model dtype.
    strict = Inkling2ForConditionalGeneration._keep_in_fp32_modules_strict
    for name, dtype in dtypes.items():
        expected = torch.float32 if any(token in name for token in strict) else torch.bfloat16
        assert dtype == expected, name
    assert dtypes[f"{layer}.2.self_attn.k_sconv.static_weight"] == torch.bfloat16

    logits = model(input_ids=_input_ids()).logits
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_future_tokens_do_not_change_earlier_logits() -> None:
    model = _model()
    input_ids = _input_ids(batch=1)
    changed = input_ids.clone()
    changed[0, -1] = (changed[0, -1] + 7) % _VOCAB
    with torch.no_grad():
        base = model(input_ids=input_ids).logits
        perturbed = model(input_ids=changed).logits
    # Expert grouping changes with the last token, so allow FP32 GEMM rounding.
    torch.testing.assert_close(base[:, :-1], perturbed[:, :-1], rtol=0, atol=1e-5)
    assert (base[:, -1] - perturbed[:, -1]).abs().max() > 1e-3


def test_left_padding_leaves_valid_token_logits_unchanged() -> None:
    # Positions stay below log_scaling_n_floor, so shifting them by the padding is a no-op.
    model = _model(log_scaling_n_floor=1024)
    input_ids = _input_ids(batch=1, sequence=16)
    padded = torch.cat([torch.zeros(1, 5, dtype=torch.long), input_ids], dim=1)
    attention_mask = torch.cat([torch.zeros(1, 5, dtype=torch.long), torch.ones_like(input_ids)], dim=1)
    with torch.no_grad():
        unpadded_logits = model(input_ids=input_ids).logits
        padded_logits = model(input_ids=padded, attention_mask=attention_mask).logits
    torch.testing.assert_close(padded_logits[:, 5:], unpadded_logits, rtol=0, atol=1e-5)


def test_default_backend_uses_sdpa_attention() -> None:
    model = Inkling2ForConditionalGeneration(_config())
    assert model.backend.attn == "sdpa"
    assert {layer.self_attn.attention_backend for layer in model.model.language_model.layers} == {"sdpa"}


def test_packed_rows_are_rejected() -> None:
    model = _model()
    input_ids = _input_ids(batch=1, sequence=8)
    with pytest.raises(NotImplementedError, match="packed sequences"):
        model(input_ids=input_ids, position_ids=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]))


def test_left_padding_positions_are_not_mistaken_for_packing() -> None:
    model = _model(log_scaling_n_floor=1024)
    input_ids = _input_ids(batch=1, sequence=8)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])
    position_ids = torch.tensor([[1, 1, 0, 1, 2, 3, 4, 5]])
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).logits
    assert torch.isfinite(logits).all()


def test_unsupported_attention_backend_is_rejected() -> None:
    backend = BackendConfig(attn="flex", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    with pytest.raises(ValueError, match="backend.attn"):
        Inkling2ForConditionalGeneration(_config(), backend=backend)


def test_non_fp32_gate_precision_is_rejected() -> None:
    backend = BackendConfig(
        attn="eager", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch", gate_precision="bfloat16"
    )
    with pytest.raises(ValueError, match="FP32"):
        Inkling2ForConditionalGeneration(_config(), backend=backend)


def test_four_dimensional_attention_mask_is_rejected() -> None:
    model = _model()
    input_ids = _input_ids()
    with pytest.raises(ValueError, match="padding mask"):
        model(input_ids=input_ids, attention_mask=torch.ones(2, 1, 20, 20))


def test_pixel_values_without_vision_tower_are_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="vision_config"):
        model(input_ids=_input_ids(), pixel_values=torch.zeros(4, 8, 8, 3), image_grid_thw=torch.tensor([[1, 2, 2]]))
