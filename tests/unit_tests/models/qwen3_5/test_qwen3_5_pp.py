# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the pipeline-parallel-safe Qwen3.5 dense text backbone.

These exercise ``Qwen3_5TextModelPP.forward`` — the override that lets the HF
text model survive a pipeline split (``self.layers`` rewritten from
``nn.ModuleList`` to ``nn.ModuleDict``; ``norm`` dropped to ``None`` on non-last
stages). The override is verified in isolation by stubbing HF's
``Qwen3_5TextModel.forward`` (the ``super()`` target) so no model weights or
distributed setup are required.
"""

from unittest.mock import patch

import torch
import torch.nn as nn
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5.model import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextModelPP,
)


def _bare_pp_model() -> Qwen3_5TextModelPP:
    """A ``Qwen3_5TextModelPP`` instance without HF's heavy ``__init__``."""
    model = Qwen3_5TextModelPP.__new__(Qwen3_5TextModelPP)
    nn.Module.__init__(model)
    return model


def test_split_stage_presents_sliceable_layers_and_identity_norm():
    """On a split stage (ModuleDict layers, norm=None) HF's forward must see a
    slice-able ModuleList over the same layer objects and a callable norm."""
    model = _bare_pp_model()
    l0, l1 = nn.Linear(2, 2), nn.Linear(2, 2)
    # Splitter keys layers by their original (non-contiguous) index.
    model.layers = nn.ModuleDict({"3": l0, "7": l1})
    model.norm = None

    seen = {}

    def probe(self, *args, **kwargs):
        seen["layers_type"] = type(self.layers)
        seen["layers_objs"] = list(self.layers)
        seen["norm_type"] = type(self.norm)
        seen["kwargs"] = kwargs
        # HF slices ``self.layers[: num_hidden_layers]`` — must not raise.
        _ = self.layers[:64]
        return "HIDDEN"

    with patch.object(Qwen3_5TextModel, "forward", probe):
        out = model.forward(position_ids=None, foo=1)

    assert out == "HIDDEN"
    assert seen["layers_type"] is nn.ModuleList
    assert seen["layers_objs"] == [l0, l1]  # same objects, preserved order
    assert seen["norm_type"] is nn.Identity
    assert seen["kwargs"] == {"position_ids": None, "foo": 1}


def test_split_stage_restores_containers_after_forward():
    """After the wrapped forward, the original ModuleDict / None norm are back."""
    model = _bare_pp_model()
    layers = nn.ModuleDict({"0": nn.Linear(2, 2)})
    model.layers = layers
    model.norm = None

    with patch.object(Qwen3_5TextModel, "forward", lambda self, *a, **k: "X"):
        model.forward()

    assert model.layers is layers
    assert isinstance(model.layers, nn.ModuleDict)
    assert model.norm is None


def test_non_split_model_is_pure_passthrough():
    """Full (non-PP) model: ModuleList layers + real norm are left untouched."""
    model = _bare_pp_model()
    layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    norm = nn.LayerNorm(2)
    model.layers = layers
    model.norm = norm

    seen = {}

    def probe(self, *args, **kwargs):
        seen["layers"] = self.layers
        seen["norm"] = self.norm
        return "HIDDEN"

    with patch.object(Qwen3_5TextModel, "forward", probe):
        out = model.forward()

    assert out == "HIDDEN"
    # Same objects passed through — no swap occurred.
    assert seen["layers"] is layers
    assert seen["norm"] is norm
    assert model.layers is layers
    assert model.norm is norm


def test_containers_restored_when_forward_raises():
    """The finally-block must restore layers/norm even if HF's forward raises."""
    model = _bare_pp_model()
    layers = nn.ModuleDict({"0": nn.Linear(2, 2)})
    model.layers = layers
    model.norm = None

    def boom(self, *args, **kwargs):
        raise RuntimeError("forward blew up")

    with patch.object(Qwen3_5TextModel, "forward", boom):
        try:
            model.forward()
        except RuntimeError:
            pass

    assert model.layers is layers
    assert model.norm is None


def test_subclass_relationship():
    """Behavioral invariants the __class__ swap in the VLM __init__ relies on."""
    assert issubclass(Qwen3_5TextModelPP, Qwen3_5TextModel)
    # No extra __init__ — the swap reuses the HF-built instance as-is.
    assert "__init__" not in Qwen3_5TextModelPP.__dict__
    assert "forward" in Qwen3_5TextModelPP.__dict__


# ---------------------------------------------------------------------------
# Outer-forward PP-stage dispatch, exercised on a tiny real VLM (CPU).
# A real pipeline split (FSDP2 + PipelineStage) can't run as a unit test, so we
# simulate the per-stage module layout the splitter produces: stage 0 keeps
# ``embed_tokens`` but not ``lm_head``; the last stage keeps ``lm_head`` but not
# ``embed_tokens``; middle stages keep neither.
# ---------------------------------------------------------------------------


def _tiny_vlm_model():
    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["full_attention", "full_attention"],
        attn_implementation="eager",
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
    )
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )
    model = Qwen3_5ForConditionalGeneration(config, backend=backend, num_nextn_predict_layers=0)
    model.eval()
    return model


def test_init_swaps_text_backbone_class():
    """__init__ points the HF text backbone at the PP-safe subclass."""
    model = _tiny_vlm_model()
    assert isinstance(model.model.language_model, Qwen3_5TextModelPP)


def test_full_model_forward_returns_output_dataclass():
    """Non-PP (embed + lm_head both present) falls through to the normal path."""
    model = _tiny_vlm_model()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    assert isinstance(out, Qwen3_5CausalLMOutputWithPast)
    assert out.logits.shape == (1, 4, model.config.text_config.vocab_size)


def test_first_stage_returns_raw_hidden_states():
    """Stage 0 (embed present, lm_head dropped) returns hidden states, not logits."""
    model = _tiny_vlm_model()
    model.lm_head = None  # splitter drops lm_head on non-last stages
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    assert torch.is_tensor(out)
    assert out.shape == (1, 4, model.config.text_config.hidden_size)


def test_last_stage_consumes_hidden_states_and_applies_lm_head():
    """Last stage (embed dropped, lm_head present) takes hidden states, emits logits."""
    model = _tiny_vlm_model()
    model.model.language_model.embed_tokens = None  # dropped on non-first stages
    dtype = next(model.parameters()).dtype
    hidden = torch.randn(1, 4, model.config.text_config.hidden_size, dtype=dtype)
    out = model(inputs_embeds=hidden, attention_mask=torch.ones(1, 4, dtype=torch.long), use_cache=False)
    assert torch.is_tensor(out)
    assert out.shape == (1, 4, model.config.text_config.vocab_size)


def test_middle_stage_passes_hidden_states_through():
    """Middle stage (neither embed nor lm_head) returns hidden states for the next stage."""
    model = _tiny_vlm_model()
    model.model.language_model.embed_tokens = None
    model.lm_head = None
    dtype = next(model.parameters()).dtype
    hidden = torch.randn(1, 4, model.config.text_config.hidden_size, dtype=dtype)
    out = model(inputs_embeds=hidden, attention_mask=torch.ones(1, 4, dtype=torch.long), use_cache=False)
    assert torch.is_tensor(out)
    assert out.shape == (1, 4, model.config.text_config.hidden_size)
