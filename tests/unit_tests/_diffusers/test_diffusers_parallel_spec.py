# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""The diffusion pipeline binds diffusers transformers' declarations with the transformers loader's resolver."""

import pytest
import torch
import torch.nn as nn

from nemo_automodel._transformers.model_init import bind_model_specs, parallel_spec_for
from nemo_automodel.components.distributed.activation_checkpointing import query_activation_checkpointing_spec
from nemo_automodel.components.distributed.parallelizer import (
    _DEFAULT_STRATEGY,
    get_model_layer_groups,
    get_parallelization_strategy,
    query_parallel_spec,
)
from nemo_automodel.components.models import model_family
from nemo_automodel.components.models.ltx2_video.parallelization import LTX2VideoTransformer3DModel
from nemo_automodel.components.models.wan.parallelization import WAN_TP_PLAN


def _diffusers_double(class_name: str, *bases: type) -> type:
    """A stand-in carrying the diffusers ``ModelMixin`` marker (``config_name``) the family rule keys on."""
    return type(class_name, bases or (nn.Module,), {"config_name": "config.json"})


@pytest.mark.parametrize(
    ("class_name", "family"),
    [
        ("WanTransformer3DModel", "wan"),
        ("HunyuanVideo15Transformer3DModel", "hunyuan_video15"),
        ("LTX2VideoTransformer3DModel", "ltx2_video"),
        ("QwenImageTransformer2DModel", "qwen_image"),
        ("Transformer2DModel", ""),
    ],
)
def test_diffusers_family_is_the_snake_case_stem_before_transformer(class_name, family):
    assert model_family(_diffusers_double(class_name)) == family


def test_modules_from_neither_library_have_no_family():
    assert model_family(type("WanTransformer3DModel", (nn.Module,), {})) is None


@pytest.mark.parametrize(
    "class_name",
    [
        "WanTransformer3DModel",
        "HunyuanVideo15Transformer3DModel",
        "LTX2VideoTransformer3DModel",
        "QwenImageTransformer2DModel",
    ],
)
def test_bind_attaches_whole_block_checkpointing_and_the_default_strategy(class_name):
    """Every diffusers transformer declares whole-block checkpointing and runs the shared FSDP2 flow."""
    upstream = _diffusers_double(class_name)
    module = bind_model_specs(upstream())
    assert isinstance(module, upstream)
    assert type(module).__name__ == class_name
    assert query_activation_checkpointing_spec(module).granularity == "layer"
    assert get_parallelization_strategy(module) is _DEFAULT_STRATEGY


def test_wan_declares_its_tensor_parallel_plan():
    upstream = _diffusers_double("WanTransformer3DModel")
    spec = parallel_spec_for(upstream)
    assert spec.tp_plan is WAN_TP_PLAN
    assert type(bind_model_specs(upstream())).parallel_spec is spec


def test_ltx2_declares_its_block_container():
    """diffusers ships no ``_no_split_modules`` for LTX-2, so the layer group cannot be derived."""
    assert (
        parallel_spec_for(_diffusers_double("LTX2VideoTransformer3DModel")) is LTX2VideoTransformer3DModel.parallel_spec
    )
    assert LTX2VideoTransformer3DModel.parallel_spec.layer_groups == {"backbone": ("transformer_blocks",)}


def test_subclass_inherits_its_base_declaration():
    base = _diffusers_double("WanTransformer3DModel")
    module = bind_model_specs(type("WanVariant", (base,), {})())
    assert query_parallel_spec(module).tp_plan is WAN_TP_PLAN
    assert query_activation_checkpointing_spec(module).granularity == "layer"


def test_modules_without_a_declaration_are_returned_unchanged():
    module = nn.Linear(2, 2)
    assert bind_model_specs(module) is module
    assert type(module) is nn.Linear
    assert query_parallel_spec(module).strategy is None


def _tiny_diffusers_transformers():
    diffusers = pytest.importorskip("diffusers")
    with torch.device("meta"):
        return {
            "blocks": diffusers.WanTransformer3DModel(
                num_attention_heads=2,
                attention_head_dim=8,
                in_channels=4,
                out_channels=4,
                text_dim=16,
                freq_dim=32,
                ffn_dim=32,
                num_layers=2,
            ),
            "transformer_blocks": diffusers.QwenImageTransformer2DModel(
                num_layers=2,
                attention_head_dim=8,
                num_attention_heads=2,
                joint_attention_dim=16,
                axes_dims_rope=(4, 2, 2),
            ),
        }


def test_diffusers_blocks_form_the_backbone_layer_group():
    """The block container diffusers declares in ``_no_split_modules`` is the ``backbone`` group; no declaration needed."""
    for container_name, transformer in _tiny_diffusers_transformers().items():
        model = bind_model_specs(transformer)
        assert query_parallel_spec(model).layer_groups is None
        assert get_model_layer_groups(model) == {"backbone": list(getattr(model, container_name))}
