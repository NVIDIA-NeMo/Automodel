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

import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nemo_automodel.components.models.deprecation import warn_deprecated_checkpoint, warn_deprecated_model_class


@pytest.mark.parametrize(
    ("model_id", "recipe"),
    [
        ("EleutherAI/gpt-j-6b", "examples/llm_finetune/eleutherai/gpt_j_6b_squad_peft.yaml"),
        ("meta-llama/Llama-3.2-1B", "examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml"),
        ("black-forest-labs/FLUX.1-dev", "examples/diffusion/finetune/flux_t2i_flow.yaml"),
    ],
)
def test_checkpoint_warning_names_checkpoint_and_recipe(model_id: str, recipe: str):
    with pytest.warns(FutureWarning, match=model_id) as recorded:
        warn_deprecated_checkpoint(model_id)

    assert len(recorded) == 1
    assert recipe in str(recorded[0].message)
    assert "2024-10-01" in str(recorded[0].message)


@pytest.mark.parametrize(
    "model_id",
    [
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen3-8B",
        "black-forest-labs/FLUX.2-dev",
        Path("/checkpoints/llama-3.2-local"),
    ],
)
def test_checkpoint_deprecation_does_not_apply_to_other_models(model_id: str | Path):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warn_deprecated_checkpoint(model_id)

    assert not recorded


def test_existing_class_deprecation_keeps_removal_schedule():
    with pytest.warns(FutureWarning, match="BaichuanForCausalLM") as recorded:
        warn_deprecated_model_class("BaichuanForCausalLM")

    assert "26.10" in str(recorded[0].message)
    assert "v0.7.0" in str(recorded[0].message)


@pytest.mark.parametrize(
    ("loader_name", "model_id"),
    [
        ("NeMoAutoModelForCausalLM", "EleutherAI/gpt-j-6b"),
        ("NeMoAutoModelForSeq2SeqLM", "google-t5/t5-small"),
        ("NeMoAutoModelForImageTextToText", "llava-hf/llava-1.5-7b-hf"),
    ],
)
def test_pretrained_loader_emits_checkpoint_warning_before_construction(loader_name: str, model_id: str):
    from nemo_automodel._transformers import auto_model

    loader = getattr(auto_model, loader_name)
    with (
        patch.object(auto_model, "_resolve_distributed_setup", side_effect=ValueError("construction stopped")),
        pytest.warns(FutureWarning, match=model_id),
        pytest.raises(ValueError, match="construction stopped"),
    ):
        loader.from_pretrained(model_id)


@pytest.mark.parametrize("config", ["meta-llama/Llama-3.2-1B", SimpleNamespace(name_or_path="meta-llama/Llama-3.2-1B")])
def test_from_config_emits_checkpoint_warning(config: str | SimpleNamespace):
    from nemo_automodel._transformers import auto_model

    with (
        patch.object(auto_model, "resolve_trust_remote_code", side_effect=ValueError("construction stopped")),
        pytest.warns(FutureWarning, match="meta-llama/Llama-3.2-1B"),
        pytest.raises(ValueError, match="construction stopped"),
    ):
        auto_model.NeMoAutoModelForCausalLM.from_config(config)


@pytest.mark.parametrize("loader_name", ["NeMoAutoModelBiEncoder", "NeMoAutoModelCrossEncoder"])
def test_retrieval_loader_emits_checkpoint_warning(loader_name: str):
    from nemo_automodel._transformers import auto_model

    loader = getattr(auto_model, loader_name)
    with (
        patch.object(auto_model, "dtype_from_str", side_effect=ValueError("construction stopped")),
        pytest.warns(FutureWarning, match="meta-llama/Llama-3.2-1B"),
        pytest.raises(ValueError, match="construction stopped"),
    ):
        loader.from_pretrained("meta-llama/Llama-3.2-1B", dtype="float32")


@pytest.mark.parametrize("method", ["from_pretrained", "from_config"])
def test_diffusion_loader_emits_checkpoint_warning(method: str):
    from nemo_automodel._diffusers import auto_diffusion_pipeline

    kwargs = {"pipeline_spec": {"transformer_cls": "FluxTransformer2DModel"}} if method == "from_config" else {}
    with (
        patch.object(auto_diffusion_pipeline, "DIFFUSERS_AVAILABLE", True),
        patch.object(
            auto_diffusion_pipeline, "resolve_diffusion_model_dir", side_effect=ValueError("construction stopped")
        ),
        pytest.warns(FutureWarning, match="black-forest-labs/FLUX.1-dev"),
        pytest.raises(ValueError, match="construction stopped"),
    ):
        getattr(auto_diffusion_pipeline.NeMoAutoDiffusionPipeline, method)("black-forest-labs/FLUX.1-dev", **kwargs)
