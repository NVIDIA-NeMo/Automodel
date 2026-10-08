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

"""Real configuration and checkpoint reads remain on the selected Hub snapshot."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoModel, GPT2Config, LlamaConfig

from nemo_automodel._transformers import retrieval
from nemo_automodel._transformers.auto_model import _BaseNeMoAutoModelClass
from nemo_automodel.components.checkpoint.checkpointing import _get_hf_safetensors_reference_path
from tests.unit_tests._transformers.test_auto_model import TestBuildModelRetryDepth as _TestBuildModelRetryDepth
from tests.unit_tests._transformers.test_retrieval import _tiny_mistral3_vlm_config

A, B = "a" * 40, "b" * 40
REPO = "test/config-race"


@pytest.mark.parametrize(
    "builder_name, config_data",
    [
        ("build_deepseek_v4_target", {"model_type": "deepseek_v4", "num_hidden_layers": 4}),
        ("build_glm_5_2_target", {"model_type": "glm_moe_dsa", "num_hidden_layers": 4}),
        ("build_kimi_k3_target", {"model_type": "kimi_k3", "text_config": {"num_hidden_layers": 4}}),
    ],
)
def test_target_builder_keeps_selected_checkpoint(hf_config_hub, monkeypatch, builder_name, config_data):
    """Actual target config loaders must not route B's build to A's cached weights."""
    from huggingface_hub import constants

    from nemo_automodel.recipes.llm import _dspark_target_build as target_build

    root, cache, ref, _ = hf_config_hub
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(root))
    for sha in (A, B):
        (cache / "snapshots" / sha / "config.json").write_text(json.dumps(config_data))
    (cache / "snapshots" / A / "model.safetensors.index.json").write_text("{}")
    selected_sources = []
    model = MagicMock()

    def load_target(config, **kwargs):
        ref.write_text(A)
        build_kwargs, _ = _TestBuildModelRetryDepth()._make_build_kwargs()
        build_kwargs.update(is_hf_model=False, cache_dir=str(root), revision=kwargs["revision"])
        with (
            patch("nemo_automodel._transformers.auto_model._init_model", return_value=(True, model)),
            patch("nemo_automodel._transformers.auto_model.get_world_size_safe", return_value=1),
            patch("nemo_automodel._transformers.capabilities.attach_capabilities_and_validate"),
            patch("nemo_automodel._transformers.auto_model.apply_model_infrastructure", return_value=model) as apply,
            patch("nemo_automodel._transformers.auto_model._maybe_dequantize_fp8_for_peft", return_value=False),
            patch("torch.cuda.current_device", return_value=0),
        ):
            _BaseNeMoAutoModelClass._build_model(config, **build_kwargs)
        source = apply.call_args.kwargs["pretrained_model_name_or_path"]
        selected_sources.append(Path(_get_hf_safetensors_reference_path(str(root), source)))
        return model

    monkeypatch.setattr(target_build, "NeMoAutoModelForCausalLM", SimpleNamespace(from_config=load_target))
    monkeypatch.setattr(target_build, "create_distributed_setup_from_config", lambda *a, **kw: None)
    config, loaded, _ = getattr(target_build, builder_name)(
        cfg={},
        world_size=1,
        device=torch.device("cuda"),
        compute_dtype=torch.float32,
        target_path=REPO,
        recipe_cfg={},
        trust_remote_code=False,
    )
    assert selected_sources == [cache / "snapshots" / B]
    assert loaded is model
    assert ref.read_text() == A


@pytest.mark.parametrize("family", ["gpt2", "llama", "extracted_llama"])
def test_retrieval_keeps_config_weights_and_prompts_on_one_snapshot(hf_config_hub, monkeypatch, family):
    """Moving main after config loading cannot mix metadata or weights across snapshots."""
    root, cache, ref, requests = hf_config_hub
    if family == "gpt2":
        config = GPT2Config(
            vocab_size=16, n_embd=16, n_head=2, n_layer=1, n_positions=16, bos_token_id=0, eos_token_id=1
        )
    elif family == "llama":
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    else:
        config = _tiny_mistral3_vlm_config("llama")
    reference = AutoModel.from_config(config)
    for sha, weight in ((A, 0.25), (B, 0.75)):
        snapshot = cache / "snapshots" / sha
        with torch.no_grad():
            reference.get_input_embeddings().weight.fill_(weight)
        reference.save_pretrained(snapshot)
        (snapshot / "1_Pooling").mkdir()
        for name, payload in {
            "modules.json": [
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ],
            "1_Pooling/config.json": {"pooling_mode_mean_tokens": True},
            "config_sentence_transformers.json": {"prompts": {"query": f"query from {sha}", "document": "doc"}},
            "sentence_bert_config.json": {},
        }.items():
            (snapshot / name).write_text(json.dumps(payload))
    ref.write_text(B)
    read_metadata = retrieval._load_sentence_transformer_wrapper_options

    def move_main_before_metadata(model_name, kwargs):
        ref.write_text(A)
        return read_metadata(model_name, kwargs)

    monkeypatch.setattr(retrieval, "_load_sentence_transformer_wrapper_options", move_main_before_metadata)
    encoder = retrieval.BiEncoderModel.build(
        REPO,
        task="embedding",
        cache_dir=str(root),
        local_files_only=True,
        **({"extract_submodel": "language_model"} if family == "extracted_llama" else {}),
    )
    assert encoder.sentence_transformer_export_config.query_prompt == f"query from {B}"
    weights = encoder.model.get_input_embeddings().weight
    torch.testing.assert_close(weights, torch.full_like(weights, 0.75), rtol=0, atol=0)
    assert Path(encoder.source_model_path) == cache / "snapshots" / B
    assert ref.read_text() == A
    # Upstream may probe for adapter metadata even with local_files_only=True.
    assert all(f"/resolve/{B}/" in request.url.path for request in requests)
