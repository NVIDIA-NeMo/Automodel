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

"""Dense Bolt E2B checkpoint interoperability, KV sharing, and numerical parity."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.api import CheckpointException
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as HFGemma4

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def config() -> Gemma4Config:
    """Retain E2B's PLE, shared KV, wide MLP, and differing attention head widths."""
    cfg = Gemma4Config(
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=Gemma4TextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=16,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
            num_kv_shared_layers=3,
            hidden_size_per_layer_input=4,
            vocab_size_per_layer_input=64,
            use_double_wide_mlp=True,
            enable_moe_block=False,
            max_position_embeddings=64,
            sliding_window=8,
            pad_token_id=0,
            dtype="float32",
            use_cache=False,
        ),
        vision_config=Gemma4VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            patch_size=2,
            pooling_kernel_size=2,
            position_embedding_size=16,
            dtype="float32",
        ),
        audio_config=None,
        image_token_id=60,
        boi_token_id=61,
        eoi_token_id=62,
        tie_word_embeddings=True,
        dtype="float32",
    )
    cfg._attn_implementation = "eager"
    return cfg


def _native(config: Gemma4Config) -> Gemma4ForConditionalGeneration:
    return Gemma4ForConditionalGeneration(
        copy.deepcopy(config),
        backend=BackendConfig(linear="torch", attn="eager", rms_norm="torch", experts="torch", dispatcher="torch"),
    ).float()


def _checkpointer(tmp_path: Path) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="source",
            model_save_format="safetensors",
            save_consolidated=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


def _legacy_checkpoint(path: Path, reference: HFGemma4) -> None:
    """Reproduce E2B's index, omitted tied head, and separate obsolete shared-KV shard."""
    path.mkdir()
    reference.config.save_pretrained(path)
    current = {k: v.contiguous() for k, v in reference.state_dict().items() if k != "lm_head.weight"}
    text = reference.config.text_config
    obsolete = {}
    for i in range(text.num_hidden_layers - text.num_kv_shared_layers, text.num_hidden_layers):
        width = text.per_layer_config[i].head_dim
        prefix = f"model.language_model.layers.{i}.self_attn"
        obsolete[f"{prefix}.k_proj.weight"] = torch.full((width, text.hidden_size), 123.0)
        obsolete[f"{prefix}.v_proj.weight"] = torch.full((width, text.hidden_size), 123.0)
        obsolete[f"{prefix}.k_norm.weight"] = torch.full((width,), 123.0)
    main_name = "model-00001-of-00001.safetensors"
    shared_name = "model-frozen-shared-kv.safetensors"
    save_file(current, path / main_name)
    save_file(obsolete, path / shared_name)
    weight_map = {**dict.fromkeys(current, main_name), **dict.fromkeys(obsolete, shared_name)}
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_legacy_e2b_load_export_and_reload(tmp_path: Path, config: Gemma4Config) -> None:
    """The real DCP loader skips obsolete source keys and restores the tied output head."""
    torch.manual_seed(42)
    reference = HFGemma4(config).float().eval()
    source = tmp_path / "source"
    _legacy_checkpoint(source, reference)
    loaded_config = AutoConfig.from_pretrained(source, local_files_only=True)
    assert isinstance(loaded_config, Gemma4Config)
    assert (
        ModelRegistry.resolve_custom_model_cls(loaded_config.architectures[0], loaded_config)
        is Gemma4ForConditionalGeneration
    )
    native = _native(loaded_config).eval()
    checkpointer = _checkpointer(tmp_path)
    with patch(
        "nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype",
        side_effect=AssertionError("Dense E2B must use direct DCP loading"),
    ):
        checkpointer.load_model(native, str(source), is_init_step=True)
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(native.state_dict()[name], tensor, atol=0, rtol=0)
    assert native.lm_head.weight is native.model.language_model.embed_tokens.weight
    checkpointer.save_model(native, str(tmp_path / "saved"))
    exported = AutoModelForImageTextToText.from_pretrained(tmp_path / "saved/model/consolidated").eval()
    restored = _native(loaded_config).eval()
    checkpointer.load_model(restored, str(tmp_path / "saved/model"))
    for model in (exported, restored):
        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
        assert set(model.state_dict()) == set(reference.state_dict())
        for name, tensor in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], tensor, atol=0, rtol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_e2b_forward_backward_parity(config: Gemma4Config, checkpointing: bool) -> None:
    """Compare padded training with HF, including replay of shared-KV attention layers."""
    torch.manual_seed(12)
    reference = HFGemma4(copy.deepcopy(config)).float().train()
    native = _native(config).train()
    native.load_state_dict(reference.state_dict())
    if checkpointing:
        for model in (reference, native):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [2, 3, 4, 0, 0, 0]])
    mask = ids.ne(0)
    actual = native(input_ids=ids, attention_mask=mask, use_cache=False).logits
    expected = reference(input_ids=ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    upstream = torch.randn_like(expected) * mask.unsqueeze(-1)
    actual.backward(upstream)
    expected.backward(upstream)
    reference_params = dict(reference.named_parameters())
    for name, parameter in native.named_parameters():
        expected_grad = reference_params[name].grad
        if expected_grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected_grad, atol=2e-5, rtol=2e-4, msg=name)
    assert native.model.language_model.embed_tokens_per_layer.weight.grad is not None


def test_e2b_image_text_forward_backward(config: Gemma4Config) -> None:
    """Image patches and text produce equivalent vision/projector gradients with HF."""
    torch.manual_seed(13)
    reference = HFGemma4(copy.deepcopy(config)).float().train()
    native = _native(config).train()
    native.load_state_dict(reference.state_dict())
    inputs = {
        "input_ids": torch.tensor([[2, 60, 60, 60, 60, 3, 4]]),
        "pixel_values": torch.randn(1, 16, 12),
        "image_position_ids": torch.tensor([[[x, y] for y in range(4) for x in range(4)]]),
        "use_cache": False,
    }
    actual = native(**inputs).logits
    expected = reference(**inputs).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    upstream = torch.randn_like(expected)
    actual.backward(upstream)
    expected.backward(upstream)
    for actual_parameter, expected_parameter in zip(
        native.model.vision_tower.parameters(),
        reference.model.vision_tower.parameters(),
        strict=True,
    ):
        if expected_parameter.grad is not None:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=2e-5, rtol=2e-4)
    assert native.model.embed_vision.embedding_projection.weight.grad is not None


def test_e2b_missing_live_tensor_is_rejected(tmp_path: Path, config: Gemma4Config) -> None:
    """Legacy extra tensors must not make missing active model weights acceptable."""
    reference = HFGemma4(config).float()
    source = tmp_path / "source"
    _legacy_checkpoint(source, reference)
    shard = source / "model-00001-of-00001.safetensors"
    tensors = load_file(shard)
    missing_key = "model.language_model.layers.0.self_attn.q_proj.weight"
    tensors.pop(missing_key)
    save_file(tensors, shard)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop(missing_key)
    index_path.write_text(json.dumps(index))
    with pytest.raises(CheckpointException, match="Missing key|missing|not found"):
        _checkpointer(tmp_path).load_model(_native(config), str(source), is_init_step=True)
