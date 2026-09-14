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

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_model import NeMoAutoModelBiEncoder, NeMoAutoModelCrossEncoder
from nemo_automodel.components.config.loader import ConfigNode


@pytest.fixture
def checkpoint(tmp_path):
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.fixture
def cpu_model_infrastructure():
    # Keep actual model construction and HF checkpoint loading; bypass only the
    # CUDA/distributed infrastructure and meta allocation needed for GPU sharding.
    with (
        patch("torch.cuda.current_device", return_value=0),
        patch("nemo_automodel._transformers.auto_model.instantiate_infrastructure", return_value=(None,) * 4),
        patch("nemo_automodel._transformers.auto_model.init_empty_weights", side_effect=nullcontext),
        patch("nemo_automodel._transformers.auto_model.no_init_weights", side_effect=nullcontext),
        patch("nemo_automodel._transformers.auto_model.apply_model_infrastructure", side_effect=lambda **kw: kw["model"]),
    ):
        yield


@pytest.mark.parametrize("entrypoint", ["from_pretrained", "from_config"])
@pytest.mark.parametrize("force_hf", [False, True], ids=["custom", "hf"])
@pytest.mark.parametrize(
    "dtype_kwargs, expected",
    [
        pytest.param({"dtype": "float32"}, torch.float32, id="dtype-string"),
        pytest.param({"dtype": torch.float32}, torch.float32, id="dtype-object"),
        pytest.param({"dtype": "torch.float32"}, torch.float32, id="dtype-qualified"),
        pytest.param({"torch_dtype": "float32"}, torch.float32, id="legacy-string"),
        pytest.param({"torch_dtype": torch.float32}, torch.float32, id="legacy-object"),
        pytest.param({"torch_dtype": "torch.float32"}, torch.float32, id="legacy-qualified"),
        pytest.param({"dtype": "bfloat16"}, torch.bfloat16, id="explicit-bf16"),
        pytest.param({"dtype": "float32", "torch_dtype": "bfloat16"}, torch.float32, id="dtype-wins"),
        pytest.param({"dtype": "bfloat16", "torch_dtype": "float32"}, torch.bfloat16, id="dtype-wins-bf16"),
        pytest.param({"dtype": None, "torch_dtype": "float32"}, torch.float32, id="none-falls-back"),
        pytest.param({"dtype": "auto", "torch_dtype": "float32"}, torch.bfloat16, id="auto-wins"),
        pytest.param({"dtype": "auto"}, torch.bfloat16, id="auto"),
        pytest.param({"torch_dtype": "auto"}, torch.bfloat16, id="legacy-auto"),
        pytest.param({"dtype": None}, torch.bfloat16, id="none"),
        pytest.param({}, torch.bfloat16, id="omitted"),
    ],
)
def test_model_storage_dtype(checkpoint, cpu_model_infrastructure, entrypoint, force_hf, dtype_kwargs, expected):
    source = checkpoint if entrypoint == "from_pretrained" else LlamaConfig.from_pretrained(checkpoint)
    model = getattr(NeMoAutoModelForCausalLM, entrypoint)(
        source,
        force_hf=force_hf,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        attn_implementation="eager",
        **dtype_kwargs,
    )

    assert model.model.embed_tokens.weight.dtype == expected
    assert model.model.layers[0].mlp.up_proj.weight.dtype == expected
    assert model.lm_head.weight.dtype == expected
    assert model.config.dtype == expected

    if force_hf and entrypoint == "from_pretrained":
        reference = LlamaForCausalLM.from_pretrained(checkpoint, dtype=expected)
        torch.testing.assert_close(model.model.embed_tokens.weight, reference.model.embed_tokens.weight)
        assert model.model.embed_tokens.weight._hf_compute_dtype == torch.bfloat16


@pytest.mark.parametrize("model_cls", [NeMoAutoModelBiEncoder, NeMoAutoModelCrossEncoder])
@pytest.mark.parametrize(
    "dtype_kwargs, expected",
    [
        ({"dtype": "torch.float32"}, torch.float32),
        ({"torch_dtype": "torch.float32"}, torch.float32),
        ({"dtype": "bfloat16", "torch_dtype": "float32"}, torch.bfloat16),
        ({"dtype": "auto", "torch_dtype": "float32"}, torch.bfloat16),
        ({"dtype": None, "torch_dtype": torch.float32}, torch.float32),
    ],
)
def test_retrieval_storage_dtype(checkpoint, cpu_model_infrastructure, model_cls, dtype_kwargs, expected):
    model = model_cls.from_pretrained(
        checkpoint,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        attn_implementation="eager",
        **dtype_kwargs,
    )

    assert next(model.parameters()).dtype == expected
    assert model.config.dtype == expected


@pytest.mark.parametrize("dtype_key", ["dtype", "torch_dtype"])
@pytest.mark.parametrize("entrypoint", ["from_pretrained", "from_config"])
def test_recipe_model_config_preserves_dtype_after_retry(checkpoint, cpu_model_infrastructure, dtype_key, entrypoint):
    model_settings = {
        "_target_": f"nemo_automodel.NeMoAutoModelForCausalLM.{entrypoint}",
        "pretrained_model_name_or_path" if entrypoint == "from_pretrained" else "config": checkpoint,
        "force_hf": True,
        "use_liger_kernel": True,
        "use_sdpa_patching": False,
        "attn_implementation": "eager",
        dtype_key: "float32",
    }
    config = ConfigNode(model_settings)
    original_config = config.to_dict()
    with patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", side_effect=RuntimeError("retry")):
        model = config.instantiate()

    assert model.model.embed_tokens.weight.dtype == torch.float32
    assert model.config.dtype == torch.float32
    assert config.to_dict() == original_config


@pytest.mark.parametrize("dtype_kwargs", [{}, {"dtype": "auto"}, {"torch_dtype": "auto"}])
def test_hf_auto_preserves_fp32_checkpoint(checkpoint, cpu_model_infrastructure, dtype_kwargs):
    reference = LlamaForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    reference.save_pretrained(checkpoint)
    model = NeMoAutoModelForCausalLM.from_pretrained(
        checkpoint,
        force_hf=True,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        attn_implementation="eager",
        **dtype_kwargs,
    )

    torch.testing.assert_close(model.model.embed_tokens.weight, reference.model.embed_tokens.weight)
