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

import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import AutoModelForCausalLM, PretrainedConfig

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_biencoder import (
    _extract_custom_args as _extract_biencoder_custom_args,
)
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _PARITY_DOCUMENT_SHA256,
    _assert_peft_adapter_matches_checkpoint,
    _automodel_reload_parity_policy,
    _compare_logits,
    _compare_source_load_parity,
    _cross_tp_parity_policy,
    _dequantize_hf_fp8_weights_in_place,
    _extract_custom_args,
    _finish_hf_reload_sync,
    _get_input_ids,
    _get_logits_pp,
    _get_parity_document,
    _hf_device_map_max_memory,
    _hf_fp32_module_names,
    _hf_model_load_context,
    _hf_reload_parity_policy,
    _hf_source_load_kwargs,
    _keep_hf_modules_in_fp32,
    _lm_head_embedding_aliased,
    _load_hf_fp8_dequantized_config,
    _load_input_ids_once,
    _LogitParityPolicy,
    _model_pretrained_path,
    _normalize_peft_no_split_modules,
    _patch_remote_fla_api_compatibility,
    _patch_remote_masking_api_compatibility,
    _peft_adapter_load_kwargs,
    _post_load_dequant_max_memory,
    _prepare_consolidated_hf_cache_once,
    _raise_distributed_failure,
    _record_deferred_failure,
    _repair_legacy_partial_rotary_config,
    _repeatability_policy,
    _replace_nemo_owned_reference_config,
    _resolve_hf_attn_implementation,
    _resolve_hf_model_class,
    _run_process_isolated_checkpoint_phase,
    _run_vanilla_hf_reload,
    _set_model_pretrained_path,
    _source_load_parity_policy,
    _trainable_parameter_digests,
    _wait_for_hf_reload_rank0,
    _wait_for_source_load_artifacts,
)
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_vlm import _get_vlm_input_ids
from tests.functional_tests.checkpoint_robustness.test_checkpoint_vllm_deploy import _tokenize_for_generation


def test_model_pretrained_path_resolves_direct_recipe_path():
    model_cfg = SimpleNamespace(pretrained_model_name_or_path="org/direct-model")

    assert _model_pretrained_path(model_cfg) == "org/direct-model"


def test_model_pretrained_path_resolves_config_based_recipe_path():
    model_cfg = SimpleNamespace(
        config=SimpleNamespace(
            pretrained_model_name_or_path="org/config-model",
            name_or_path="org/config-model",
        )
    )

    assert _model_pretrained_path(model_cfg) == "org/config-model"


def test_set_model_pretrained_path_retargets_config_based_recipe():
    nested_config = SimpleNamespace(
        pretrained_model_name_or_path="org/source-model",
        name_or_path="org/source-model",
    )
    model_cfg = SimpleNamespace(config=nested_config)

    _set_model_pretrained_path(model_cfg, "/tmp/exported-model")

    assert nested_config.pretrained_model_name_or_path == "/tmp/exported-model"
    assert nested_config.name_or_path == "/tmp/exported-model"
    assert not hasattr(model_cfg, "pretrained_model_name_or_path")


def test_model_pretrained_path_rejects_recipe_without_source_path():
    with pytest.raises(ValueError, match="model.config.pretrained_model_name_or_path"):
        _model_pretrained_path(SimpleNamespace())


def test_resolve_hf_model_class_uses_advertised_causal_lm_for_vlm_checkpoint():
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    config_dict = {
        "auto_map": {
            "AutoConfig": "configuration_step3p7.Step3p7Config",
            "AutoModelForCausalLM": "modeling_step3p7.Step3p7ForConditionalGeneration",
        }
    }
    with patch("transformers.PretrainedConfig.get_config_dict", return_value=(config_dict, {})):
        resolved_cls = _resolve_hf_model_class("model-path", AutoModelForImageTextToText)

    assert resolved_cls is AutoModelForCausalLM


def test_resolve_hf_model_class_uses_native_image_text_mapping_for_mistral3():
    from transformers import AutoModelForImageTextToText

    with patch(
        "transformers.PretrainedConfig.get_config_dict",
        return_value=({"model_type": "mistral3", "architectures": ["Mistral3ForConditionalGeneration"]}, {}),
    ):
        resolved_cls = _resolve_hf_model_class("model-path", AutoModelForCausalLM)

    assert resolved_cls is AutoModelForImageTextToText


def test_hf_device_map_max_memory_caps_each_visible_gpu():
    with patch("torch.cuda.device_count", return_value=8):
        max_memory = _hf_device_map_max_memory("55")

    assert max_memory == {index: "55GiB" for index in range(8)}


def test_hf_device_map_max_memory_includes_optional_cpu_overflow():
    with patch("torch.cuda.device_count", return_value=8):
        max_memory = _hf_device_map_max_memory("65", "64")

    assert max_memory == {**{index: "65GiB" for index in range(8)}, "cpu": "64GiB"}


def test_peft_adapter_load_reuses_base_model_placement_constraints_without_key_conversion():
    max_memory = {0: "55GiB", "cpu": "128GiB"}

    load_kwargs = _peft_adapter_load_kwargs(
        {
            "device_map": "auto",
            "max_memory": max_memory,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
    )

    assert load_kwargs == {"key_mapping": {}, "device_map": "auto", "max_memory": max_memory}


def test_peft_adapter_load_disables_key_conversion_without_a_base_device_map():
    assert _peft_adapter_load_kwargs({"torch_dtype": torch.bfloat16}) == {"key_mapping": {}}


def test_peft_adapter_fingerprints_match_saved_safetensors(tmp_path):
    from safetensors.torch import save_file

    adapter_path = tmp_path / "adapter_model.safetensors"
    saved_state = {
        "base_model.model.layer.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "base_model.model.layer.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.bfloat16),
    }
    save_file(saved_state, adapter_path)

    with patch(
        "peft.get_peft_model_state_dict", return_value={key: value.clone() for key, value in saved_state.items()}
    ):
        matched, ignored = _assert_peft_adapter_matches_checkpoint(Mock(), adapter_path)

    assert matched == 2
    assert ignored == 0


def test_peft_adapter_fingerprints_disable_peft_auto_embedding_export(tmp_path):
    from safetensors.torch import save_file

    adapter_path = tmp_path / "adapter_model.safetensors"
    adapter_key = "base_model.model.lm_head.lora_A.weight"
    saved_state = {adapter_key: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}
    save_file(saved_state, adapter_path)

    def get_peft_state_dict(_model, *, save_embedding_layers="auto"):
        loaded_state = {adapter_key: saved_state[adapter_key].clone()}
        if save_embedding_layers == "auto":
            loaded_state["base_model.model.lm_head.base_layer.weight"] = torch.tensor(
                [[3.0, 4.0]], dtype=torch.bfloat16
            )
        return loaded_state

    peft_model = Mock()
    with patch("peft.get_peft_model_state_dict", side_effect=get_peft_state_dict) as get_state_dict:
        matched, ignored = _assert_peft_adapter_matches_checkpoint(peft_model, adapter_path)

    assert matched == 1
    assert ignored == 0
    get_state_dict.assert_called_once_with(peft_model, save_embedding_layers=False)


def test_peft_adapter_fingerprints_allow_configured_hf_unsupported_prefix(tmp_path):
    from safetensors.torch import save_file

    adapter_path = tmp_path / "adapter_model.safetensors"
    loaded_key = "base_model.model.layer.lora_A.weight"
    ignored_key = "base_model.model.mtp.layers.0.eh_proj.lora_A.weight"
    saved_state = {
        loaded_key: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        ignored_key: torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
    }
    save_file(saved_state, adapter_path)

    with patch("peft.get_peft_model_state_dict", return_value={loaded_key: saved_state[loaded_key].clone()}):
        matched, ignored = _assert_peft_adapter_matches_checkpoint(
            Mock(),
            adapter_path,
            ignored_key_prefix="base_model.model.mtp.",
        )

    assert matched == 1
    assert ignored == 1


def test_peft_adapter_fingerprints_read_accelerate_offload_backing_tensor(tmp_path):
    from safetensors.torch import save_file

    class OffloadedPeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False, device="meta")})
            self.layer._hf_hook = SimpleNamespace(
                hooks=(
                    SimpleNamespace(
                        weights_map={"lora_A.default.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}
                    ),
                )
            )

    adapter_path = tmp_path / "adapter_model.safetensors"
    key = "layer.lora_A.weight"
    save_file({key: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}, adapter_path)
    model = OffloadedPeftModel()

    with patch("peft.get_peft_model_state_dict", return_value={key: torch.empty(1, 2, device="meta")}):
        matched, ignored = _assert_peft_adapter_matches_checkpoint(model, adapter_path)

    assert matched == 1
    assert ignored == 0


def test_peft_adapter_fingerprints_reject_missing_key_outside_configured_prefix(tmp_path):
    from safetensors.torch import save_file

    adapter_path = tmp_path / "adapter_model.safetensors"
    key = "base_model.model.layer.lora_A.weight"
    save_file({key: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}, adapter_path)

    with (
        patch("peft.get_peft_model_state_dict", return_value={}),
        pytest.raises(AssertionError, match="adapter key mismatch"),
    ):
        _assert_peft_adapter_matches_checkpoint(
            Mock(),
            adapter_path,
            ignored_key_prefix="base_model.model.mtp.",
        )


def test_peft_adapter_fingerprints_report_tensor_mismatch(tmp_path):
    from safetensors.torch import save_file

    adapter_path = tmp_path / "adapter_model.safetensors"
    key = "base_model.model.layer.lora_A.weight"
    save_file({key: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}, adapter_path)

    with (
        patch("peft.get_peft_model_state_dict", return_value={key: torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)}),
        pytest.raises(AssertionError, match="adapter tensor mismatch"),
    ):
        _assert_peft_adapter_matches_checkpoint(Mock(), adapter_path)


def test_remote_masking_api_compatibility_drops_removed_cache_position(monkeypatch):
    import transformers.masking_utils as masking_utils

    calls = []

    def create_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
        calls.append((config, inputs_embeds, attention_mask, past_key_values, position_ids))
        return "mask"

    monkeypatch.setattr(masking_utils, "create_causal_mask", create_mask)
    monkeypatch.setattr(masking_utils, "create_sliding_window_causal_mask", create_mask)

    _patch_remote_masking_api_compatibility()
    _patch_remote_masking_api_compatibility()

    for function_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        # Pre-v5 remote code passes the removed ``cache_position`` and the
        # renamed ``input_embeds`` keywords (e.g. Kimi-Linear).
        result = getattr(masking_utils, function_name)(
            config="config",
            input_embeds="inputs",
            attention_mask="attention",
            past_key_values="cache",
            position_ids="positions",
            cache_position="removed-argument",
        )
        assert result == "mask"

    assert calls == [
        ("config", "inputs", "attention", "cache", "positions"),
        ("config", "inputs", "attention", "cache", "positions"),
    ]


def test_remote_masking_api_compatibility_preserves_supported_api(monkeypatch):
    import transformers.masking_utils as masking_utils

    def create_mask(config, input_embeds, attention_mask, past_key_values, cache_position=None):
        return cache_position

    monkeypatch.setattr(masking_utils, "create_causal_mask", create_mask)
    monkeypatch.setattr(masking_utils, "create_sliding_window_causal_mask", create_mask)

    _patch_remote_masking_api_compatibility()

    assert masking_utils.create_causal_mask is create_mask
    assert masking_utils.create_sliding_window_causal_mask is create_mask


def test_get_logits_pp_updates_pipeline_sequence_length():
    class _Schedule:
        def __init__(self):
            self._loss_fn = None
            self.ids = None
            self.attention_mask = None

        def eval(self, ids, *, target, losses, attention_mask):
            """Capture a pipeline batch and invoke the active loss callback.

            Args:
                ids: Tensor of shape [batch, sequence].
                target: Tensor of shape [batch, sequence].
                losses: Optional list populated by the pipeline loss callback.
                attention_mask: Tensor of shape [batch, sequence].
            """
            self.ids = ids
            self.attention_mask = attention_mask
            logits = torch.zeros(ids.shape[0], ids.shape[1], 7)
            assert self._loss_fn is not None
            self._loss_fn(logits, target)

    class _PipelineMesh:
        @staticmethod
        def get_group():
            return object()

        @staticmethod
        def size():
            return 1

    schedule = _Schedule()
    update_seq_len = Mock()
    trainer = SimpleNamespace(
        pp=SimpleNamespace(
            update_seq_len=update_seq_len,
            info=SimpleNamespace(
                schedule=schedule,
                has_first_stage=True,
                has_last_stage=True,
            ),
        ),
        pipeline_config=SimpleNamespace(pp_batch_size=1),
        model_parts=[SimpleNamespace(eval=lambda: None, config=SimpleNamespace(vocab_size=7))],
        device_mesh={"pp": _PipelineMesh()},
    )

    with (
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.dist.get_global_rank",
            return_value=0,
        ),
        patch("tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.dist.broadcast"),
    ):
        logits = _get_logits_pp(trainer, [11, 12, 13], torch.device("cpu"))

    update_seq_len.assert_called_once_with(3)
    assert schedule.ids.tolist() == [[11, 12, 13]]
    assert schedule.attention_mask.shape == (1, 3)
    assert schedule.attention_mask.tolist() == [[1, 1, 1]]
    assert logits.shape == (1, 3, 7)


@pytest.mark.parametrize(
    ("model_type", "expected_attn_implementation"),
    [
        ("deepseek_v4", "eager"),
        ("nemotron-nas", "eager"),
        ("nemotron_h", "eager"),
        ("step3p7", "eager"),
        ("nemotron_flash", "eager"),
    ],
)
def test_remote_code_attention_implementation(model_type, expected_attn_implementation):
    with patch(
        "transformers.PretrainedConfig.get_config_dict",
        return_value=({"model_type": model_type}, {}),
    ) as get_config_dict:
        hf_kwargs = _hf_source_load_kwargs(
            {"revision": "model-revision", "token": "model-token"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            hf_model_cls=AutoModelForCausalLM,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["attn_implementation"] == expected_attn_implementation
    get_config_dict.assert_called_once_with(
        "model-path",
        local_files_only=False,
        revision="model-revision",
        token="model-token",
    )


def test_explicit_attention_implementation_is_preserved():
    with patch(
        "transformers.PretrainedConfig.get_config_dict",
        return_value=({"model_type": "unknown_remote_model"}, {}),
    ):
        hf_kwargs = _hf_source_load_kwargs(
            {"attn_implementation": "eager"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            hf_model_cls=AutoModelForCausalLM,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["attn_implementation"] == "eager"


@pytest.mark.parametrize(("supported", "expected"), [(True, "sdpa"), (False, "eager")])
def test_builtin_attention_implementation_uses_supported_recipe_backend_or_eager(supported, expected):
    class FakeConfig:
        pass

    concrete_model_cls = SimpleNamespace(_supports_sdpa=supported)
    auto_model_cls = SimpleNamespace(_model_mapping={FakeConfig: concrete_model_cls})
    with patch("transformers.AutoConfig.from_pretrained", return_value=FakeConfig()):
        implementation = _resolve_hf_attn_implementation(
            "model-path",
            "sdpa",
            hf_model_cls=auto_model_cls,
            trust_remote_code=False,
        )

    assert implementation == expected


def test_hf_source_load_kwargs_explicit_false_disables_recipe_remote_code():
    hf_kwargs = _hf_source_load_kwargs(
        {"trust_remote_code": True},
        pretrained_model_name_or_path="model-path",
        source_dtype=torch.bfloat16,
        trust_remote_code=False,
        experts_implementation=None,
        hf_model_cls=AutoModelForCausalLM,
        device=torch.device("cpu"),
        hf_device_map_auto=False,
    )

    assert hf_kwargs["trust_remote_code"] is False


def test_hf_source_load_kwargs_passes_grouped_experts_implementation():
    hf_kwargs = _hf_source_load_kwargs(
        {},
        pretrained_model_name_or_path="model-path",
        source_dtype=torch.bfloat16,
        trust_remote_code=False,
        experts_implementation="grouped_mm",
        hf_model_cls=AutoModelForCausalLM,
        device=torch.device("cpu"),
        hf_device_map_auto=False,
    )

    assert hf_kwargs["experts_implementation"] == "grouped_mm"
    assert hf_kwargs["trust_remote_code"] is False


@pytest.mark.parametrize(
    ("trust_remote_code", "has_device_map", "expected_no_meta_calls"),
    [(True, True, 0), (False, False, 0), (True, False, 1)],
)
def test_hf_model_load_context_keeps_meta_for_device_map(
    trust_remote_code,
    has_device_map,
    expected_no_meta_calls,
):
    with patch("nemo_automodel._transformers.model_init.no_hf_meta_device") as no_hf_meta_device:
        no_hf_meta_device.return_value = nullcontext()
        with _hf_model_load_context(
            trust_remote_code=trust_remote_code,
            has_device_map=has_device_map,
        ):
            pass

    assert no_hf_meta_device.call_count == expected_no_meta_calls


def test_lm_head_alias_check_skips_nonstandard_embedding_accessor():
    class InputDependentEmbeddingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

        def get_input_embeddings(self, input_ids):
            raise AssertionError("input-dependent accessor must not be called")

    assert _lm_head_embedding_aliased(InputDependentEmbeddingModel()) is None


def test_lm_head_alias_check_skips_wrapper_around_input_dependent_accessor():
    class InputDependentEmbeddingModel(torch.nn.Module):
        def get_input_embeddings(self, input_ids):
            raise AssertionError("input-dependent accessor must not be called")

    class WrapperModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = InputDependentEmbeddingModel()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

    assert _lm_head_embedding_aliased(WrapperModel()) is None


def test_peft_no_split_modules_are_normalized_for_accelerate():
    model = SimpleNamespace(_no_split_modules={"SecondLayer", "FirstLayer"})

    _normalize_peft_no_split_modules(model)

    assert model._no_split_modules == ["FirstLayer", "SecondLayer"]


@pytest.mark.parametrize(("offline", "expected_local_files_only"), [(None, False), ("1", True)])
def test_hf_source_load_kwargs_respects_hf_offline(monkeypatch, offline, expected_local_files_only):
    if offline is None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", offline)

    hf_kwargs = _hf_source_load_kwargs(
        {},
        pretrained_model_name_or_path="model-path",
        source_dtype=torch.bfloat16,
        trust_remote_code=False,
        experts_implementation=None,
        hf_model_cls=AutoModelForCausalLM,
        device=torch.device("cpu"),
        hf_device_map_auto=False,
    )

    assert hf_kwargs["local_files_only"] is expected_local_files_only


@pytest.mark.parametrize(("offline", "expected_local_files_only"), [(None, False), ("1", True)])
def test_get_input_ids_respects_hf_offline(monkeypatch, offline, expected_local_files_only):
    if offline is None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", offline)
    tokenizer = Mock()
    tokenizer.encode.return_value = [11, 12, 13]

    with patch("nemo_automodel.NeMoAutoTokenizer.from_pretrained", return_value=tokenizer) as from_pretrained:
        input_ids = _get_input_ids("mistralai/Ministral-3-3B-Instruct-2512")

    assert input_ids == [11, 12, 13]
    from_pretrained.assert_called_once_with(
        "mistralai/Ministral-3-3B-Instruct-2512",
        trust_remote_code=True,
        local_files_only=expected_local_files_only,
    )
    tokenizer.encode.assert_called_once_with(_get_parity_document(), add_special_tokens=False)


@pytest.mark.parametrize(("offline", "expected_local_files_only"), [(None, False), ("1", True)])
def test_get_vlm_input_ids_uses_processor_tokenizer(monkeypatch, offline, expected_local_files_only):
    if offline is None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", offline)
    tokenizer = Mock()
    tokenizer.encode.return_value = [21, 22, 23]
    processor = SimpleNamespace(tokenizer=tokenizer)

    with patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as from_pretrained:
        input_ids = _get_vlm_input_ids("mistralai/Ministral-3-3B-Reasoning-2512")

    assert input_ids == [21, 22, 23]
    from_pretrained.assert_called_once_with(
        "mistralai/Ministral-3-3B-Reasoning-2512",
        trust_remote_code=True,
        local_files_only=expected_local_files_only,
    )
    tokenizer.encode.assert_called_once_with(_get_parity_document(), add_special_tokens=False)


def test_parity_document_is_the_frozen_long_form_finetuning_guide_snapshot():
    document = _get_parity_document()

    assert "Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT)" in document[:200]
    assert "## Configure Your Training Recipe" in document
    assert "## Next Steps" in document
    assert len(document.split()) > 6_000


@pytest.mark.parametrize("input_ids_loader", [_get_input_ids, _get_vlm_input_ids])
def test_parity_input_requires_a_model_tokenizer(input_ids_loader):
    with pytest.raises(ValueError, match="tokenizer_name is required"):
        input_ids_loader(None)


def test_load_input_ids_once_shares_rank0_result(tmp_path, monkeypatch):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"))
    rank0_loader = Mock(return_value=[31, 32, 33])
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "input-id-test")
    monkeypatch.setenv("RANK", "0")

    assert _load_input_ids_once(cfg, rank0_loader, "model/tokenizer", sequence_length=3) == [31, 32, 33]
    rank0_loader.assert_called_once_with("model/tokenizer")

    rank1_loader = Mock(side_effect=AssertionError("nonzero rank must not load the tokenizer"))
    monkeypatch.setenv("RANK", "1")

    assert _load_input_ids_once(cfg, rank1_loader, "model/tokenizer", sequence_length=3) == [31, 32, 33]
    rank1_loader.assert_not_called()

    rank0_reuse_loader = Mock(side_effect=AssertionError("rank 0 must reuse the published input IDs"))
    monkeypatch.setenv("RANK", "0")

    assert _load_input_ids_once(cfg, rank0_reuse_loader, "model/tokenizer", sequence_length=3) == [31, 32, 33]
    rank0_reuse_loader.assert_not_called()


def test_load_input_ids_once_rejects_short_document_for_parity_length(tmp_path):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"))

    with pytest.raises(ValueError, match="contains 3 tokens, but parity_sequence_length requires 8"):
        _load_input_ids_once(cfg, Mock(return_value=[7, 8, 9]), None, sequence_length=8)


def test_load_input_ids_once_truncates_long_document_to_parity_length(tmp_path):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"))

    input_ids = _load_input_ids_once(cfg, Mock(return_value=[7, 8, 9, 10, 11]), None, sequence_length=3)

    assert input_ids == [7, 8, 9]


def test_load_input_ids_once_waits_for_payload_visibility(tmp_path, monkeypatch):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "input-id-visibility-test")
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    monkeypatch.setenv("RANK", "1")
    sync_dir = tmp_path / ".checkpoint_robustness_input_ids_slurm_input-id-visibility-test_step_0"
    sync_dir.mkdir()
    (sync_dir / "done").write_text("ok\n")

    def publish_payload(_seconds):
        (sync_dir / "input_ids.json").write_text("[41, 42, 43]")

    loader = Mock(side_effect=AssertionError("nonzero rank must not load the tokenizer"))
    with patch(
        "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.time.sleep",
        side_effect=publish_payload,
    ):
        assert _load_input_ids_once(cfg, loader, "model/tokenizer", sequence_length=3) == [41, 42, 43]

    loader.assert_not_called()


def test_load_input_ids_once_propagates_rank0_failure(tmp_path, monkeypatch):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "input-id-failure-test")
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(ValueError, match="tokenizer failed"):
        _load_input_ids_once(
            cfg,
            Mock(side_effect=ValueError("tokenizer failed")),
            "model/tokenizer",
            sequence_length=3,
        )

    monkeypatch.setenv("RANK", "1")
    with pytest.raises(RuntimeError, match="Rank 0 input-ID loading failed"):
        _load_input_ids_once(cfg, Mock(), "model/tokenizer", sequence_length=3)


def test_vllm_deploy_tokenization_omits_token_type_ids():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    default_inputs = tokenizer("hello world", return_tensors="pt")
    generation_inputs = _tokenize_for_generation(tokenizer, "hello world", torch.device("cpu"))

    assert "token_type_ids" in default_inputs
    assert set(generation_inputs) == {"input_ids", "attention_mask"}
    torch.testing.assert_close(generation_inputs["input_ids"], default_inputs["input_ids"])
    torch.testing.assert_close(generation_inputs["attention_mask"], default_inputs["attention_mask"])
    assert generation_inputs["input_ids"].device.type == "cpu"


def test_extract_custom_args_accepts_hf_source_post_load_dequantize():
    custom, remaining = _extract_custom_args(["--hf_source_post_load_dequantize", "--other-arg"])

    assert custom["hf_source_post_load_dequantize"] is True
    assert remaining == ["--other-arg"]


def test_extract_custom_args_accepts_isolated_phase():
    custom, remaining = _extract_custom_args(["--isolated_phase", "train_and_save", "--other-arg"])

    assert custom["isolated_phase"] == "train_and_save"
    assert remaining == ["--other-arg"]


def test_extract_custom_args_enables_core_source_and_resume_checks_by_default():
    custom, remaining = _extract_custom_args(["--other-arg"])

    assert custom["source_load_parity_enabled"] is True
    assert custom["resume_enabled"] is True
    assert remaining == ["--other-arg"]


def test_extract_custom_args_reads_semantic_skips_and_parity_settings(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "ci:\n"
        "  checkpoint_robustness:\n"
        "    skip_source_load_parity: true\n"
        "    skip_source_load_logit_parity: true\n"
        "    skip_resume: true\n"
        "    skip_automodel_reload_logit_parity: true\n"
        "    skip_hf_reload_logit_parity: true\n"
        "    trust_remote_code: false\n"
        "    parity_sequence_length: 1024\n"
        "    parity_tolerance_profile: relaxed\n"
        "    parity_tolerance_profile_overrides:\n"
        "      source_load: strict\n"
        "      hf_reload: standard\n"
        "    parity_threshold_overrides:\n"
        "      source_load: {mean_kl: 0.01}\n"
        "      automodel_reload: {mean_kl: 0.04, cosine_similarity: 0.99}\n"
        "      hf_reload: {p95_kl: 0.08}\n"
        "      cross_tp: {cosine_similarity: 0.997}\n"
        "    hf_reload_timeout_seconds: 3600\n"
    )

    custom, remaining = _extract_custom_args(["--config", str(config_path)])

    assert custom["source_load_parity_enabled"] is False
    assert custom["resume_enabled"] is False
    assert custom["skip_source_load_logit_parity"] is True
    assert custom["skip_automodel_reload_logit_parity"] is True
    assert custom["skip_hf_reload_logit_parity"] is True
    assert custom["trust_remote_code"] is False
    assert custom["parity_sequence_length"] == "1024"
    assert custom["parity_tolerance_profile"] == "relaxed"
    assert custom["parity_tolerance_profile_overrides"] == {
        "source_load": "strict",
        "hf_reload": "standard",
    }
    assert custom["parity_threshold_overrides"] == {
        "source_load": {"mean_kl": 0.01},
        "automodel_reload": {"mean_kl": 0.04, "cosine_similarity": 0.99},
        "hf_reload": {"p95_kl": 0.08},
        "cross_tp": {"cosine_similarity": 0.997},
    }
    assert custom["hf_reload_timeout_seconds"] == "3600"
    assert remaining == ["--config", str(config_path)]


def test_extract_custom_args_rejects_removed_config_fields(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text("ci:\n  checkpoint_robustness:\n    hf_kl_threshold: 0.01\n")

    with pytest.raises(ValueError, match="Removed checkpoint-robustness fields.*hf_kl_threshold"):
        _extract_custom_args(["--config", str(config_path)])


def test_extract_custom_args_accepts_resume_tolerance_profile_and_numeric_override():
    custom, remaining = _extract_custom_args(
        ["--resume_tolerance_profile", "relaxed", "--resume_loss_threshold", "0.02", "--other-arg"]
    )

    assert custom["resume_tolerance_profile"] == "relaxed"
    assert custom["resume_loss_threshold"] == "0.02"
    assert remaining == ["--other-arg"]


def test_extract_custom_args_accepts_cli_profile_overrides():
    custom, remaining = _extract_custom_args(
        ["--parity_tolerance_profile_overrides", "{hf_reload: relaxed}", "--other-arg"]
    )

    assert custom["parity_tolerance_profile_overrides"] == {"hf_reload": "relaxed"}
    assert remaining == ["--other-arg"]


def test_extract_custom_args_accepts_hf_adapter_ignored_key_prefix():
    custom, remaining = _extract_custom_args(
        ["--hf_adapter_ignored_key_prefix", "base_model.model.mtp.", "--other-arg"]
    )

    assert custom["hf_adapter_ignored_key_prefix"] == "base_model.model.mtp."
    assert remaining == ["--other-arg"]


def test_distributed_failure_prints_stable_phase_marker(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "0")
    failure = (
        "CHECKPOINT_ROBUSTNESS_PHASE_FAILURE phase=automodel_reload check=logit_kl\n"
        "max per-token KL exceeded its threshold"
    )

    with pytest.raises(AssertionError, match="max per-token KL exceeded"):
        _raise_distributed_failure(failure)

    assert capsys.readouterr().err == (
        "[checkpoint_robustness][phase-error] "
        "CHECKPOINT_ROBUSTNESS_PHASE_FAILURE phase=automodel_reload check=logit_kl\n"
    )


def test_process_isolated_hf_reload_runs_rank0_hf_loader(tmp_path):
    artifact_dir = tmp_path / ".checkpoint_robustness"
    artifact_dir.mkdir()
    (artifact_dir / "reference_logits.pt").write_bytes(b"reference")
    cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(checkpoint_dir=tmp_path),
        get=lambda key, default=None: default,
    )
    reference_logits = torch.randn(1, 2, 3)
    recipe_cls = Mock()
    hf_model_cls = Mock()
    custom_args = {"hf_device_map_auto": True, "skip_resume": True, "trust_remote_code": True}

    with (
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.parse_args_and_load_config",
            return_value=cfg,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm."
            "_disable_distributed_atexit_teardown"
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._load_input_ids_once",
            return_value=[11, 12],
        ),
        patch("torch.distributed.init_process_group") as init_process_group,
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._prepare_hf_reload_sync",
            return_value=None,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._finish_hf_reload_sync",
            side_effect=lambda paths, error: error,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._raise_distributed_failure"
        ) as raise_distributed_failure,
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._run_vanilla_hf_reload",
            return_value=None,
        ) as run_hf_reload,
        patch("torch.load", return_value=reference_logits),
    ):
        _run_process_isolated_checkpoint_phase(
            "hf_reload",
            custom_args=custom_args,
            recipe_cls=recipe_cls,
            hf_model_cls=hf_model_cls,
            input_ids_loader=Mock(),
        )

    init_process_group.assert_called_once()
    assert init_process_group.call_args.kwargs["backend"] == "gloo"
    assert init_process_group.call_args.kwargs["timeout"].total_seconds() == 60
    run_hf_reload.assert_called_once_with(
        cfg,
        [11, 12],
        reference_logits,
        hf_model_cls=hf_model_cls,
        custom_args=custom_args,
    )
    raise_distributed_failure.assert_called_once_with(None)
    recipe_cls.assert_not_called()


def test_hf_reload_applies_remote_code_compatibility_patches():
    """Phase 3 installs the same remote-code compatibility setup as Phase 0."""
    with patch(
        "nemo_automodel._transformers.utils.apply_cache_compatibility_patches",
        side_effect=RuntimeError("compatibility sentinel"),
    ) as apply_compatibility:
        error = _run_vanilla_hf_reload(
            SimpleNamespace(),
            [],
            torch.empty(1, 0, 0),
            hf_model_cls=Mock(),
            custom_args={},
        )

    apply_compatibility.assert_called_once_with()
    assert error is not None
    assert "RuntimeError: compatibility sentinel" in error


def test_process_isolated_cross_tp_reload_uses_exported_weights_and_reports_parity(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    consolidated_dir = checkpoint_dir / "epoch_0_step_5/model/consolidated"
    consolidated_dir.mkdir(parents=True)
    artifact_dir = checkpoint_dir / ".checkpoint_robustness"
    artifact_dir.mkdir()
    reference_logits = torch.randn(1, 2, 3)
    candidate_logits = reference_logits.clone()
    torch.save(reference_logits, artifact_dir / "reference_logits.pt")
    cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(checkpoint_dir=checkpoint_dir, enabled=True),
        model=SimpleNamespace(pretrained_model_name_or_path="source-model"),
        distributed=SimpleNamespace(tp_size=1, dp_size=8),
    )
    model_part = torch.nn.Linear(2, 2, bias=False)
    cross_tp_trainer = SimpleNamespace(model_parts=[model_part], setup=Mock())
    recipe_cls = Mock(return_value=cross_tp_trainer)
    custom_args = {"cross_tp_size": "2", "parity_sequence_length": "2"}

    with (
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.parse_args_and_load_config",
            return_value=cfg,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm."
            "_disable_distributed_atexit_teardown"
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._load_input_ids_once",
            return_value=[11, 12],
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm."
            "_prepare_consolidated_hf_cache_once"
        ) as prepare_cache,
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._get_logits",
            return_value=candidate_logits,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._compare_logits",
            return_value=None,
        ) as compare_logits,
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._raise_distributed_failure"
        ) as raise_distributed_failure,
        patch("tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._release_recipe_memory"),
    ):
        _run_process_isolated_checkpoint_phase(
            "cross_tp_reload",
            custom_args=custom_args,
            recipe_cls=recipe_cls,
            hf_model_cls=Mock(),
            input_ids_loader=Mock(),
        )

    prepare_cache.assert_called_once_with(cfg, consolidated_dir)
    assert cfg.model.pretrained_model_name_or_path == str(consolidated_dir)
    assert cfg.checkpoint.enabled is False
    assert cfg.distributed.tp_size == 2
    assert cfg.distributed.dp_size is None
    recipe_cls.assert_called_once_with(cfg)
    cross_tp_trainer.setup.assert_called_once_with()
    compare_args = compare_logits.call_args.args
    assert compare_args[0] == artifact_dir
    torch.testing.assert_close(compare_args[1], reference_logits)
    torch.testing.assert_close(compare_args[2], candidate_logits)
    assert compare_args[3].phase == "phase_5"
    assert compare_args[3].comparison == "cross_tp_reload"
    raise_distributed_failure.assert_called_once_with(None)


def test_process_isolated_resume_rejects_skip_resume():
    with pytest.raises(ValueError, match="conflicts with skip_resume=true"):
        _run_process_isolated_checkpoint_phase(
            "resume",
            custom_args={"skip_resume": True},
            recipe_cls=Mock(),
            hf_model_cls=Mock(),
            input_ids_loader=Mock(),
        )


def test_process_isolated_source_load_reference_persists_hf_artifacts(tmp_path):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path))
    reference_logits = torch.randn(1, 2, 3)
    source_reference = (reference_logits, False, False)
    recipe_cls = Mock()
    hf_model_cls = Mock()
    custom_args = {
        "source_load_parity_enabled": True,
        "hf_device_map_auto": True,
        "trust_remote_code": True,
    }

    with (
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.parse_args_and_load_config",
            return_value=cfg,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm."
            "_disable_distributed_atexit_teardown"
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._load_input_ids_once",
            return_value=[11, 12],
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm."
            "_prepare_source_load_reference",
            return_value=source_reference,
        ) as prepare_source_load,
    ):
        _run_process_isolated_checkpoint_phase(
            "source_load_reference",
            custom_args=custom_args,
            recipe_cls=recipe_cls,
            hf_model_cls=hf_model_cls,
            input_ids_loader=Mock(),
        )

    prepare_source_load.assert_called_once_with(
        cfg,
        [11, 12],
        hf_model_cls=hf_model_cls,
        trust_remote_code=True,
        experts_implementation=None,
        hf_device_map_auto=True,
        hf_source_post_load_dequantize=False,
        parity_tolerance_profile="standard",
    )
    persisted_logits = torch.load(
        tmp_path / ".checkpoint_robustness" / "source_load_reference_logits.pt",
        map_location="cpu",
        weights_only=True,
    )
    torch.testing.assert_close(persisted_logits, reference_logits)
    assert (
        tmp_path / ".checkpoint_robustness" / "source_load_reference_metadata.json"
    ).read_text() == '{"explicit_tie_word_embeddings": false, "hf_aliased": false}'
    recipe_cls.assert_not_called()


def test_wait_for_source_load_artifacts_waits_for_both_files(tmp_path):
    reference_path = tmp_path / "reference.pt"
    metadata_path = tmp_path / "metadata.json"
    fail_path = tmp_path / "fail"
    sleep_calls = 0

    def publish_artifacts(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            reference_path.write_bytes(b"reference")
        else:
            metadata_path.write_text("{}")

    with patch(
        "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.time.sleep",
        side_effect=publish_artifacts,
    ):
        _wait_for_source_load_artifacts(reference_path, metadata_path, fail_path)

    assert sleep_calls == 2


def test_prepare_consolidated_hf_cache_once_serializes_preinit_workers(tmp_path, monkeypatch):
    consolidated_dir = tmp_path / "checkpoint/model/consolidated"
    consolidated_dir.mkdir(parents=True)
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoint"))
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")

    with (
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm."
            "_prepopulate_hf_dynamic_modules_cache"
        ) as prepopulate,
        patch("transformers.AutoConfig.from_pretrained") as auto_config,
    ):
        _prepare_consolidated_hf_cache_once(cfg, consolidated_dir)

    prepopulate.assert_called_once_with(consolidated_dir)
    auto_config.assert_called_once_with(str(consolidated_dir), trust_remote_code=True)

    monkeypatch.setenv("RANK", "1")
    with patch(
        "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm."
        "_prepopulate_hf_dynamic_modules_cache",
        side_effect=AssertionError("nonzero rank must reuse the completed cache marker"),
    ) as nonzero_prepopulate:
        _prepare_consolidated_hf_cache_once(cfg, consolidated_dir)

    nonzero_prepopulate.assert_not_called()


def test_process_isolated_source_load_parity_compares_persisted_reference(tmp_path):
    artifact_dir = tmp_path / ".checkpoint_robustness"
    artifact_dir.mkdir()
    reference_logits = torch.randn(1, 2, 3)
    torch.save(reference_logits, artifact_dir / "source_load_reference_logits.pt")
    (artifact_dir / "source_load_reference_metadata.json").write_text(
        '{"explicit_tie_word_embeddings": false, "hf_aliased": false}'
    )
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path))
    candidate_logits = torch.randn(1, 2, 3)
    model_part = torch.nn.Linear(2, 2, bias=False)
    source_trainer = SimpleNamespace(model_parts=[model_part], setup=Mock())
    recipe_cls = Mock(return_value=source_trainer)
    custom_args = {
        "source_load_parity_enabled": True,
        "parity_tolerance_profile": "standard",
    }

    with (
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.parse_args_and_load_config",
            return_value=cfg,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm."
            "_disable_distributed_atexit_teardown"
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._load_input_ids_once",
            return_value=[11, 12],
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._get_logits",
            return_value=candidate_logits,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._lm_head_embedding_aliased",
            return_value=False,
        ),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._compare_source_load_parity",
            return_value=None,
        ) as compare_source_load,
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._cleanup_source_load_sync"
        ) as cleanup_source_load,
        patch("tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._barrier"),
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm._raise_distributed_failure"
        ) as raise_distributed_failure,
    ):
        _run_process_isolated_checkpoint_phase(
            "source_load_parity",
            custom_args=custom_args,
            recipe_cls=recipe_cls,
            hf_model_cls=Mock(),
            input_ids_loader=Mock(),
        )

    recipe_cls.assert_called_once_with(cfg)
    source_trainer.setup.assert_called_once_with()
    compare_args = compare_source_load.call_args
    torch.testing.assert_close(compare_args.args[0][0], reference_logits)
    assert compare_args.args[0][1:] == (False, False)
    assert compare_args.args[1:] == (candidate_logits, False)
    assert compare_args.kwargs == {
        "artifact_dir": artifact_dir,
        "policy": _source_load_parity_policy(custom_args),
    }
    cleanup_source_load.assert_called_once_with(cfg)
    raise_distributed_failure.assert_called_once_with(None)


def test_trainable_parameter_digests_hash_only_trainable_parameters():
    first_part = torch.nn.Linear(2, 2, bias=False)
    second_part = torch.nn.Linear(2, 1, bias=False)
    second_part.weight.requires_grad_(False)
    with torch.no_grad():
        first_part.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    before = _trainable_parameter_digests([first_part, second_part])
    with torch.no_grad():
        first_part.weight[0, 0] = 5.0
    after = _trainable_parameter_digests([first_part, second_part])

    assert set(before) == {"part_0:weight"}
    assert before["part_0:weight"]["dtype"] == "torch.float32"
    assert before["part_0:weight"]["shape"] == [2, 2]
    assert before["part_0:weight"]["sha256"] != after["part_0:weight"]["sha256"]


def test_keep_hf_modules_in_fp32_uses_strict_dtype_plan_and_restores_class_state(tmp_path):
    from transformers import PretrainedConfig, PreTrainedModel

    class TinyConfig(PretrainedConfig):
        model_type = "checkpoint-robustness-gdn-dtype-test"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._experts_implementation_internal = "eager"

    class TinyModel(PreTrainedModel):
        config_class = TinyConfig

        def __init__(self, config):
            super().__init__(config)
            self.A_log = torch.nn.Parameter(torch.tensor([1.234567]))
            self.dt_bias = torch.nn.Parameter(torch.tensor([0.25]))
            self.post_init()

    previous = getattr(PreTrainedModel, "_keep_in_fp32_modules_strict", None)
    TinyModel(TinyConfig()).save_pretrained(tmp_path)
    plain = TinyModel.from_pretrained(tmp_path, dtype=torch.bfloat16)
    hf_config = SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"])
    with patch(
        "nemo_automodel._transformers.model_init._resolve_custom_model_cls_for_config",
        return_value=None,
    ):
        assert _hf_fp32_module_names(hf_config) == ("A_log", "dt_bias")
    with _keep_hf_modules_in_fp32(hf_config):
        assert set(PreTrainedModel._keep_in_fp32_modules_strict) >= {"A_log", "dt_bias"}
        strict = TinyModel.from_pretrained(tmp_path, dtype=torch.bfloat16)

    assert PreTrainedModel._keep_in_fp32_modules_strict == previous
    assert plain.A_log.dtype == torch.bfloat16
    assert plain.dt_bias.dtype == torch.bfloat16
    assert strict.A_log.dtype == torch.float32
    assert strict.dt_bias.dtype == torch.float32


def test_hf_fp32_module_names_includes_generic_model_strict_contract():
    class TinyAutoModel:
        _keep_in_fp32_modules_strict = ["rotary_emb", "router.e_score_correction_bias"]

    hf_config = SimpleNamespace(architectures=["TinyForCausalLM"])
    with patch(
        "nemo_automodel._transformers.model_init._resolve_custom_model_cls_for_config",
        return_value=TinyAutoModel,
    ):
        assert _hf_fp32_module_names(hf_config) == ("rotary_emb", "router.e_score_correction_bias")


def test_hf_fp32_module_names_combines_gdn_and_generic_contracts_without_duplicates():
    class TinyAutoModel:
        _keep_in_fp32_modules_strict = ["A_log", "rotary_emb"]

    hf_config = SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"])
    with patch(
        "nemo_automodel._transformers.model_init._resolve_custom_model_cls_for_config",
        return_value=TinyAutoModel,
    ):
        assert _hf_fp32_module_names(hf_config) == ("A_log", "dt_bias", "rotary_emb")


def test_hf_fp32_module_names_is_empty_without_model_contract():
    with patch(
        "nemo_automodel._transformers.model_init._resolve_custom_model_cls_for_config",
        return_value=None,
    ):
        assert _hf_fp32_module_names(SimpleNamespace(architectures=["LlamaForCausalLM"])) == ()


def test_source_load_parity_failure_is_returned_for_later_reporting(tmp_path):
    reference_logits = torch.tensor([[[2.0, -2.0], [1.0, -1.0]]])
    candidate_logits = -reference_logits

    failure = _compare_source_load_parity(
        (reference_logits, None, None),
        candidate_logits,
        None,
        artifact_dir=tmp_path,
        policy=_source_load_parity_policy({"parity_tolerance_profile": "strict"}),
    )

    assert failure is not None
    assert "source_load parity failed" in failure


def test_source_load_parity_success_returns_no_deferred_failure(tmp_path):
    logits = torch.tensor([[[2.0, -2.0], [1.0, -1.0]]])

    failure = _compare_source_load_parity(
        (logits, None, None),
        logits.clone(),
        None,
        artifact_dir=tmp_path,
        policy=_source_load_parity_policy({"parity_tolerance_profile": "strict"}),
    )

    assert failure is None


def test_source_load_logit_skip_keeps_metrics_informational():
    policy = _source_load_parity_policy({"skip_source_load_logit_parity": True})

    assert policy.enforce is False


def test_repeatability_policy_is_same_implementation_and_informational():
    policy = _repeatability_policy(
        phase="phase_2",
        comparison="automodel_reload_self_repeat",
        profile="relaxed",
    )

    assert policy.comparison_kind == "same_implementation"
    assert policy.profile == "relaxed"
    assert policy.enforce is False


def test_parity_policies_use_structured_per_comparison_profile_and_threshold_overrides():
    custom_args = {
        "parity_tolerance_profile": "standard",
        "parity_tolerance_profile_overrides": {
            "source_load": "strict",
            "hf_reload": "relaxed",
            "cross_tp": "relaxed",
        },
        "parity_threshold_overrides": {
            "source_load": {"mean_kl": 0.01},
            "automodel_reload": {"mean_kl": 0.04, "cosine_similarity": 0.99},
            "hf_reload": {"p95_kl": 0.08},
            "cross_tp": {"cosine_similarity": 0.997},
        },
    }

    source = _source_load_parity_policy(custom_args)
    automodel = _automodel_reload_parity_policy(custom_args)
    hf = _hf_reload_parity_policy(custom_args)
    cross_tp = _cross_tp_parity_policy(custom_args)

    assert source.profile == "strict"
    assert source.mean_kl_threshold_override == 0.01
    assert automodel.profile == "standard"
    assert automodel.mean_kl_threshold_override == 0.04
    assert automodel.p95_kl_threshold_override is None
    assert automodel.cosine_threshold_override == 0.99
    assert hf.profile == "relaxed"
    assert hf.p95_kl_threshold_override == 0.08
    assert cross_tp.profile == "relaxed"
    assert cross_tp.cosine_threshold_override == 0.997


def test_compare_logits_persists_machine_readable_metrics(tmp_path):
    logits = torch.tensor([[[2.0, -2.0], [1.0, -1.0]]])
    policy = _LogitParityPolicy(
        phase="phase_2",
        comparison="automodel_model_reload",
        comparison_kind="same_implementation",
        profile="standard",
    )

    failure = _compare_logits(tmp_path, logits, logits.clone(), policy)

    assert failure is None
    payload = json.loads((tmp_path / "parity_metrics/phase_2_automodel_model_reload.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["parity_document_sha256"] == _PARITY_DOCUMENT_SHA256
    assert payload["threshold_mode"] == "profile"
    assert payload["passed"] is True
    assert payload["within_active_thresholds"] is True
    assert payload["would_pass_profile"] is True
    assert payload["reference_logits"] == {"dtype": "torch.float32", "shape": [1, 2, 2]}
    assert payload["candidate_logits"] == {"dtype": "torch.float32", "shape": [1, 2, 2]}
    assert payload["metrics"]["token_count"] == 2
    assert payload["metrics"]["mean_kl"] == pytest.approx(0.0, abs=1e-8)
    assert payload["metrics"]["mean_jsd"] == pytest.approx(0.0, abs=5e-8)
    assert payload["metrics"]["p95_jsd"] == pytest.approx(0.0, abs=5e-8)
    assert payload["metrics"]["max_jsd"] == pytest.approx(0.0, abs=5e-8)


def test_compare_logits_marks_skipped_gate_as_informational(tmp_path):
    reference_logits = torch.tensor([[[20.0, -20.0]]])
    candidate_logits = -reference_logits
    policy = _LogitParityPolicy(
        phase="phase_3",
        comparison="hf_export_reload",
        comparison_kind="cross_framework",
        profile="strict",
        enforce=False,
    )

    failure = _compare_logits(tmp_path, reference_logits, candidate_logits, policy)

    assert failure is None
    payload = json.loads((tmp_path / "parity_metrics/phase_3_hf_export_reload.json").read_text())
    assert payload["enforced"] is False
    assert payload["passed"] is True
    assert payload["within_active_thresholds"] is False
    assert payload["failures"] == []
    assert payload["threshold_failures"]


def test_compare_logits_reports_and_applies_targeted_profile_threshold_overrides(tmp_path):
    reference_logits = torch.tensor([[[20.0, -20.0]]])
    candidate_logits = -reference_logits
    policy = _LogitParityPolicy(
        phase="phase_2",
        comparison="automodel_model_reload",
        comparison_kind="same_implementation",
        profile="strict",
        mean_kl_threshold_override=100.0,
        p95_kl_threshold_override=100.0,
        cosine_threshold_override=-1.0,
    )

    failure = _compare_logits(tmp_path, reference_logits, candidate_logits, policy)

    assert failure is None
    payload = json.loads((tmp_path / "parity_metrics/phase_2_automodel_model_reload.json").read_text())
    assert payload["threshold_mode"] == "profile_with_numeric_overrides"
    assert payload["profile_failures"]
    assert payload["threshold_failures"] == []
    assert payload["threshold_overrides"] == {
        "mean_kl": 100.0,
        "p95_kl": 100.0,
        "cosine_similarity": -1.0,
    }


def test_dequantize_hf_fp8_weights_in_place_handles_linear_and_expert_parameters():
    class FakeFP8Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts_implementation = "grouped_mm"
            self.weight = torch.nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
            self.gate_up_proj = torch.nn.Parameter(
                torch.tensor(
                    [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]],
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_scale_inv = torch.nn.Parameter(
                torch.tensor([0.25, 0.5]).view(2, 1, 1),
                requires_grad=False,
            )

        def set_experts_implementation(self, experts_implementation: str) -> None:
            self.experts_implementation = experts_implementation

    model = FakeFP8Module()
    expected_weight = model.weight.float() * model.weight_scale_inv.float()
    expected_experts = model.gate_up_proj.float() * model.gate_up_proj_scale_inv.float()

    assert _dequantize_hf_fp8_weights_in_place(model, torch.bfloat16) == 2
    assert model.weight.dtype == torch.bfloat16
    assert model.gate_up_proj.dtype == torch.bfloat16
    assert model.experts_implementation == "eager"
    torch.testing.assert_close(model.weight.float(), expected_weight, rtol=0, atol=1e-2)
    torch.testing.assert_close(model.gate_up_proj.float(), expected_experts, rtol=0, atol=1e-2)


def test_dequantize_hf_fp8_weights_in_place_restores_eager_expert_forward():
    from transformers import Mistral4Config
    from transformers.integrations.finegrained_fp8 import ALL_FP8_EXPERTS_FUNCTIONS, FP8Experts
    from transformers.integrations.moe import use_experts_implementation

    class TestFP8Experts(FP8Experts):
        pass

    wrapped_experts_class = use_experts_implementation(
        experts_class=TestFP8Experts,
        experts_interface=ALL_FP8_EXPERTS_FUNCTIONS,
    )
    config = Mistral4Config(
        hidden_size=4,
        moe_intermediate_size=3,
        n_routed_experts=2,
        num_experts_per_tok=1,
    )
    config._experts_implementation_internal = "grouped_mm"

    class FakeFP8Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = wrapped_experts_class(config=config, activation_scheme="static")
            with torch.no_grad():
                self.experts.gate_up_proj.fill_(0.25)
                self.experts.down_proj.fill_(0.25)
            self.experts.gate_up_proj_scale_inv = torch.nn.Parameter(
                torch.ones(config.n_routed_experts, 1, 1),
                requires_grad=False,
            )
            self.experts.down_proj_scale_inv = torch.nn.Parameter(
                torch.ones(config.n_routed_experts, 1, 1),
                requires_grad=False,
            )

        def set_experts_implementation(self, experts_implementation: str) -> None:
            config._experts_implementation_internal = experts_implementation

    model = FakeFP8Model()
    hidden_states = torch.ones(2, config.hidden_size, dtype=torch.bfloat16)
    top_k_index = torch.tensor([[0], [1]])
    top_k_weights = torch.ones(2, 1, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="activation_scheme='static'"):
        model.experts(hidden_states, top_k_index, top_k_weights)

    assert _dequantize_hf_fp8_weights_in_place(model, torch.bfloat16) == 2
    assert config._experts_implementation == "eager"
    output = model.experts(hidden_states, top_k_index, top_k_weights)
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_post_load_dequant_max_memory_reserves_fp8_expansion_headroom():
    properties = SimpleNamespace(total_memory=80 * 1024**3)
    with (
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.get_device_properties", return_value=properties),
    ):
        max_memory = _post_load_dequant_max_memory()

    assert max_memory == {0: int(properties.total_memory * 0.35), 1: int(properties.total_memory * 0.35)}


def test_load_hf_fp8_dequantized_config_preserves_checkpoint_quantization_settings(monkeypatch):
    source_config = SimpleNamespace(
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "static",
            "weight_block_size": None,
            "dequantize": False,
        }
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with patch("transformers.AutoConfig.from_pretrained", return_value=source_config) as from_pretrained:
        config = _load_hf_fp8_dequantized_config(
            "mistralai/Ministral-3-3B-Instruct-2512",
            trust_remote_code=False,
        )

    assert config.quantization_config == {
        "quant_method": "fp8",
        "activation_scheme": "static",
        "weight_block_size": None,
        "dequantize": True,
    }
    from_pretrained.assert_called_once_with(
        "mistralai/Ministral-3-3B-Instruct-2512",
        trust_remote_code=False,
        local_files_only=True,
    )


def test_load_hf_fp8_dequantized_config_ignores_non_fp8_checkpoint():
    source_config = SimpleNamespace(quantization_config={"quant_method": "awq"})

    with patch("transformers.AutoConfig.from_pretrained", return_value=source_config):
        assert _load_hf_fp8_dequantized_config("model-path", trust_remote_code=False) is None


def test_hf_reload_wait_returns_after_rank0_marker(tmp_path):
    done_path = tmp_path / "done"
    done_path.write_text("ok\n")

    _wait_for_hf_reload_rank0(done_path)


def test_hf_reload_wait_has_separate_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_RELOAD_TIMEOUT_SECONDS", "0")

    with pytest.raises(TimeoutError, match="rank 0 vanilla-HF reload"):
        _wait_for_hf_reload_rank0(tmp_path / "done")


def test_hf_reload_wait_accepts_explicit_timeout(tmp_path):
    with pytest.raises(TimeoutError, match="Timed out waiting 0s"):
        _wait_for_hf_reload_rank0(tmp_path / "done", timeout_s=0)


def test_hf_reload_finish_returns_error_without_distributed_sync():
    assert _finish_hf_reload_sync(None, "HF parity failed") == "HF parity failed"


def test_biencoder_robustness_reads_current_settings_from_config(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "ci:\n"
        "  checkpoint_robustness:\n"
        "    skip_hf_reload: true\n"
        "    skip_resume: true\n"
        "    parity_tolerance_profile: relaxed\n"
        "    parity_tolerance_profile_overrides:\n"
        "      hf_reload: standard\n"
        "    parity_threshold_overrides:\n"
        "      automodel_reload: {cosine_similarity: 0.997}\n"
        "      hf_reload: {cosine_similarity: 0.996}\n"
        "    resume_tolerance_profile: relaxed\n"
        "    dataloader.num_workers: 0\n"
    )

    custom, remaining = _extract_biencoder_custom_args(["--config", str(config_path)])

    assert custom == {
        "skip_hf_reload": True,
        "skip_resume": True,
        "parity_tolerance_profile": "relaxed",
        "parity_tolerance_profile_overrides": {"hf_reload": "standard"},
        "parity_threshold_overrides": {
            "automodel_reload": {"cosine_similarity": 0.997},
            "hf_reload": {"cosine_similarity": 0.996},
        },
        "resume_tolerance_profile": "relaxed",
    }
    assert remaining == ["--config", str(config_path)]


def test_biencoder_robustness_defaults_to_standard_profile_and_default_on_phases():
    custom, remaining = _extract_biencoder_custom_args(["--other-arg"])

    assert custom.get("parity_tolerance_profile", "standard") == "standard"
    assert custom.get("skip_hf_reload", False) is False
    assert custom.get("skip_resume", False) is False
    assert remaining == ["--other-arg"]


def test_biencoder_robustness_rejects_removed_config_fields(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text("ci:\n  checkpoint_robustness:\n    cosine_threshold: 0.999\n")

    with pytest.raises(ValueError, match="Removed retrieval checkpoint-robustness fields.*cosine_threshold"):
        _extract_biencoder_custom_args(["--config", str(config_path)])


def test_biencoder_robustness_rejects_non_cosine_threshold_overrides(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "ci:\n  checkpoint_robustness:\n    parity_threshold_overrides:\n      hf_reload: {mean_kl: 0.01}\n"
    )

    with pytest.raises(ValueError, match="hf_reload supports only cosine_similarity"):
        _extract_biencoder_custom_args(["--config", str(config_path)])


def test_biencoder_robustness_rejects_unsupported_profile_comparison(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "ci:\n  checkpoint_robustness:\n    parity_tolerance_profile_overrides:\n      source_load: relaxed\n"
    )

    with pytest.raises(ValueError, match="supports only automodel_reload and hf_reload"):
        _extract_biencoder_custom_args(["--config", str(config_path)])


def test_record_deferred_failure_preserves_all_comparison_failures():
    failures = []

    _record_deferred_failure(failures, "Phase 3 AutoModel reload parity", None)
    _record_deferred_failure(failures, "Phase 4 HF reload parity", "HF parity failed")

    assert failures == ["Phase 4 HF reload parity:\nHF parity failed"]


def _legacy_partial_rotary_minimax_config(**overrides):
    """Tiny in-tree MiniMax-M2 config built from checkpoint-style legacy fields."""
    from transformers import AutoConfig

    kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rotary_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    kwargs.update(overrides)
    return AutoConfig.for_model("minimax_m2", **kwargs)


def test_repair_legacy_partial_rotary_derives_factor_from_rotary_dim():
    config = _legacy_partial_rotary_minimax_config()

    factor_missing_before = not config.rope_parameters.get("partial_rotary_factor")
    assert _repair_legacy_partial_rotary_config(config) is factor_missing_before
    assert config.rope_parameters["partial_rotary_factor"] == pytest.approx(0.5)
    # A second pass finds the factor present and must not report a repair.
    assert _repair_legacy_partial_rotary_config(config) is False


def test_repaired_minimax_m2_config_rotates_only_rotary_dim():
    from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM

    config = _legacy_partial_rotary_minimax_config()
    _repair_legacy_partial_rotary_config(config)

    model = HFAutoModelForCausalLM.from_config(config)
    # inv_freq carries one frequency per rotated dim pair: rotary_dim // 2,
    # not head_dim // 2 (the full-rotation failure mode from AMINT-286).
    assert model.model.rotary_emb.inv_freq.shape[0] == config.rotary_dim // 2


def test_repair_legacy_partial_rotary_is_noop_without_legacy_spec():
    from transformers import AutoConfig

    llama = AutoConfig.for_model(
        "llama",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    rope_before = deepcopy(getattr(llama, "rope_parameters", None))
    assert _repair_legacy_partial_rotary_config(llama) is False
    assert getattr(llama, "rope_parameters", None) == rope_before

    full_rotary = _legacy_partial_rotary_minimax_config(rotary_dim=32)
    rope_before = deepcopy(full_rotary.rope_parameters)
    assert _repair_legacy_partial_rotary_config(full_rotary) is False
    assert full_rotary.rope_parameters == rope_before

class _FakeNemoOwnedConfig(PretrainedConfig):
    """Stands in for an AutoModel component config registered into AutoConfig."""

    model_type = "stub_reference"


# The helper discriminates on the class's owning package, not its bases.
_FakeNemoOwnedConfig.__module__ = "nemo_automodel.components.models.test_only.config"

_STUB_AUTO_MAP = {"AutoConfig": "configuration_stub.StubReferenceConfig"}


def _write_stub_remote_checkpoint(tmp_path, *, quantization_config=None):
    """Write a minimal remote-code checkpoint dir exposing its own config class."""
    (tmp_path / "configuration_stub.py").write_text(
        "from transformers import PretrainedConfig\n"
        "\n"
        "\n"
        "class StubReferenceConfig(PretrainedConfig):\n"
        '    model_type = "stub_reference"\n'
    )
    config = {"model_type": "stub_reference", "auto_map": _STUB_AUTO_MAP, "hidden_size": 8}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_replace_nemo_owned_reference_config_resolves_checkpoint_auto_map(tmp_path):
    ckpt = _write_stub_remote_checkpoint(tmp_path)
    hijacked = _FakeNemoOwnedConfig(auto_map=dict(_STUB_AUTO_MAP))

    replaced, did_replace = _replace_nemo_owned_reference_config(hijacked, ckpt, trust_remote_code=True)

    assert did_replace is True
    assert type(replaced).__name__ == "StubReferenceConfig"
    assert type(replaced).__module__.startswith("transformers_modules")


def test_replace_nemo_owned_reference_config_preserves_dequantize_request(tmp_path):
    ckpt = _write_stub_remote_checkpoint(tmp_path, quantization_config={"quant_method": "fp8"})
    hijacked = _FakeNemoOwnedConfig(
        auto_map=dict(_STUB_AUTO_MAP),
        quantization_config={"quant_method": "fp8", "dequantize": True},
    )

    replaced, did_replace = _replace_nemo_owned_reference_config(hijacked, ckpt, trust_remote_code=True)

    assert did_replace is True
    assert replaced.quantization_config["dequantize"] is True


def test_replace_nemo_owned_reference_config_noop_cases(tmp_path):
    from transformers import AutoConfig

    hf_config = AutoConfig.for_model("llama", vocab_size=64, hidden_size=32, num_hidden_layers=1)
    assert _replace_nemo_owned_reference_config(hf_config, tmp_path, trust_remote_code=True) == (hf_config, False)

    no_auto_map = _FakeNemoOwnedConfig()
    assert _replace_nemo_owned_reference_config(no_auto_map, tmp_path, trust_remote_code=True) == (no_auto_map, False)

    untrusted = _FakeNemoOwnedConfig(auto_map=dict(_STUB_AUTO_MAP))
    assert _replace_nemo_owned_reference_config(untrusted, tmp_path, trust_remote_code=False) == (untrusted, False)


def test_hf_source_load_kwargs_drops_nemo_owned_recipe_config():
    with patch(
        "transformers.PretrainedConfig.get_config_dict",
        return_value=({"model_type": "unknown_remote_model"}, {}),
    ):
        hf_kwargs = _hf_source_load_kwargs(
            {"config": _FakeNemoOwnedConfig(), "attn_implementation": "eager"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            hf_model_cls=AutoModelForCausalLM,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert "config" not in hf_kwargs


def test_hf_source_load_kwargs_keeps_hf_recipe_config():
    from transformers import AutoConfig

    hf_config = AutoConfig.for_model("llama", vocab_size=64, hidden_size=32, num_hidden_layers=1)
    with patch(
        "transformers.PretrainedConfig.get_config_dict",
        return_value=({"model_type": "unknown_remote_model"}, {}),
    ):
        hf_kwargs = _hf_source_load_kwargs(
            {"config": hf_config, "attn_implementation": "eager"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            hf_model_cls=AutoModelForCausalLM,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["config"] is hf_config


def _install_fake_fla(monkeypatch, gate_function):
    """Register a minimal fake fla.ops.kda[.gate] module tree exposing gate_function."""
    import sys
    from types import ModuleType

    fla = ModuleType("fla")
    ops = ModuleType("fla.ops")
    kda = ModuleType("fla.ops.kda")
    gate = ModuleType("fla.ops.kda.gate")
    gate.fused_kda_gate = gate_function
    kda.fused_kda_gate = gate_function
    kda.gate = gate
    ops.kda = kda
    fla.ops = ops
    for name, module in (("fla", fla), ("fla.ops", ops), ("fla.ops.kda", kda), ("fla.ops.kda.gate", gate)):
        monkeypatch.setitem(sys.modules, name, module)
    return kda, gate


def test_remote_fla_api_compatibility_translates_legacy_kda_gate(monkeypatch):
    """Legacy fused_kda_gate(g, A, head_k_dim, g_bias=...) calls must reach the renamed API.

    The fake installed API mirrors fla-core 0.4.2: g arrives pre-reshaped to
    [..., heads, head_k_dim] and the bias keyword is dt_bias.
    """
    calls = {}

    def fused_kda_gate(g, A_log, dt_bias=None, lower_bound=None, output_dtype=torch.float32):
        calls["g_shape"] = tuple(g.shape)
        calls["dt_bias"] = dt_bias
        return g

    kda, gate = _install_fake_fla(monkeypatch, fused_kda_gate)
    _patch_remote_fla_api_compatibility()

    # Legacy call: flat g of shape [batch, sequence, heads * head_k_dim].
    g = torch.zeros(2, 3, 8)
    bias = torch.ones(8)
    gate.fused_kda_gate(g, torch.zeros(2), 4, g_bias=bias)
    assert calls["g_shape"] == (2, 3, 2, 4)
    assert calls["dt_bias"] is bias

    # New-style calls pass through untouched.
    gate.fused_kda_gate(torch.zeros(2, 3, 2, 4), torch.zeros(2), dt_bias=None)
    assert calls["g_shape"] == (2, 3, 2, 4)
    assert calls["dt_bias"] is None

    # The package-level re-export is patched consistently, and re-patching no-ops.
    assert kda.fused_kda_gate is gate.fused_kda_gate
    patched = gate.fused_kda_gate
    _patch_remote_fla_api_compatibility()
    assert gate.fused_kda_gate is patched

    with pytest.raises(TypeError, match="beta/threshold"):
        gate.fused_kda_gate(g, torch.zeros(2), 4, g_bias=bias, beta=2.0)


def test_remote_fla_api_compatibility_preserves_legacy_capable_api(monkeypatch):
    def fused_kda_gate(g, A, head_k_dim, g_bias=None, beta=1.0, threshold=20.0):
        return g

    kda, gate = _install_fake_fla(monkeypatch, fused_kda_gate)
    _patch_remote_fla_api_compatibility()

    assert gate.fused_kda_gate is fused_kda_gate
    assert kda.fused_kda_gate is fused_kda_gate
