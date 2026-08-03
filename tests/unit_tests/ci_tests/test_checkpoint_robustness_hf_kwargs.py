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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_biencoder import (
    _extract_custom_args as _extract_biencoder_custom_args,
)
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _assert_peft_adapter_matches_checkpoint,
    _compare_source_load_parity,
    _dequantize_hf_fp8_weights_in_place,
    _diagnostic_sample_indices,
    _extract_custom_args,
    _finish_hf_reload_sync,
    _get_input_ids,
    _get_logits_pp,
    _get_logits_with_diagnostics,
    _hf_device_map_max_memory,
    _hf_fp32_module_names,
    _hf_model_load_context,
    _hf_reload_kl_error,
    _hf_source_load_kwargs,
    _keep_hf_modules_in_fp32,
    _lm_head_embedding_aliased,
    _load_hf_fp8_dequantized_config,
    _load_input_ids_once,
    _normalize_peft_no_split_modules,
    _patch_remote_masking_api_compatibility,
    _peft_adapter_load_kwargs,
    _post_load_dequant_max_memory,
    _raise_distributed_failure,
    _record_deferred_failure,
    _resolve_hf_model_class,
    _run_process_isolated_checkpoint_phase,
    _trainable_parameter_digests,
    _wait_for_hf_reload_rank0,
    _wait_for_source_load_artifacts,
)
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_vlm import _get_vlm_input_ids
from tests.functional_tests.checkpoint_robustness.test_checkpoint_vllm_deploy import _tokenize_for_generation


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


def test_peft_adapter_fingerprints_ignore_reported_base_layer_tensor(tmp_path):
    from safetensors.torch import save_file

    adapter_path = tmp_path / "adapter_model.safetensors"
    adapter_key = "base_model.model.lm_head.lora_A.weight"
    saved_state = {adapter_key: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}
    loaded_state = {
        adapter_key: saved_state[adapter_key].clone(),
        "base_model.model.lm_head.base_layer.weight": torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
    }
    save_file(saved_state, adapter_path)

    with patch("peft.get_peft_model_state_dict", return_value=loaded_state):
        matched, ignored = _assert_peft_adapter_matches_checkpoint(Mock(), adapter_path)

    assert matched == 1
    assert ignored == 0


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


def test_get_logits_verbose_diagnostics_repeats_forward(monkeypatch, capsys):
    class RepeatModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.config = SimpleNamespace(_commit_hash="test-revision", model_type="test")

        def forward(self, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            self.calls += 1
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 4), device=input_ids.device))

    monkeypatch.setenv("CHECKPOINT_ROBUSTNESS_VERBOSE_DIAGNOSTICS", "1")
    model = RepeatModel()

    logits = _get_logits_with_diagnostics(model, [1, 2], torch.device("cpu"))

    assert model.calls == 2
    assert logits.shape == (1, 2, 4)
    output = capsys.readouterr().out
    assert "[checkpoint-diagnostic][repeat-forward]" in output
    assert '"max_abs_diff": 0.0' in output
    assert '"max_self_kl": 0.0' in output


def test_diagnostic_sample_indices_are_exact_for_large_tensors():
    numel = 100_000_003

    indices = _diagnostic_sample_indices(numel, 64, torch.device("cpu"))

    assert indices.dtype == torch.int64
    assert indices[0].item() == 0
    assert indices[-1].item() == numel - 1
    assert torch.all(indices[1:] > indices[:-1])


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
        result = getattr(masking_utils, function_name)(
            "config",
            "inputs",
            "attention",
            "cache",
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

    def create_mask(config, inputs_embeds, attention_mask, past_key_values, cache_position=None):
        return cache_position

    monkeypatch.setattr(masking_utils, "create_causal_mask", create_mask)
    monkeypatch.setattr(masking_utils, "create_sliding_window_causal_mask", create_mask)

    _patch_remote_masking_api_compatibility()

    assert masking_utils.create_causal_mask is create_mask
    assert masking_utils.create_sliding_window_causal_mask is create_mask


@pytest.mark.parametrize("metadata_api", ["legacy", "user"])
def test_get_logits_pp_pads_prompt_to_static_stage_sequence_length(metadata_api):
    class _Schedule:
        def __init__(self):
            self._loss_fn = None
            self.ids = None
            self.attention_mask = None

        def eval(self, ids, *, target, losses, attention_mask):
            """Capture a padded pipeline batch and invoke the active loss callback.

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

    if metadata_api == "legacy":
        stage = SimpleNamespace(inputs_meta=(torch.empty(1, 16),))
    else:
        tensor_meta = SimpleNamespace(shape=torch.Size([1, 16]))
        stage = SimpleNamespace(_user_meta=SimpleNamespace(inputs=(tensor_meta,), outputs=()))

    schedule = _Schedule()
    trainer = SimpleNamespace(
        pp=SimpleNamespace(
            pp_seq_len=None,
            info=SimpleNamespace(
                schedule=schedule,
                stages=[stage],
                has_first_stage=True,
                has_last_stage=True,
            ),
        ),
        pipeline_config=SimpleNamespace(pp_batch_size=1),
        model_parts=[SimpleNamespace(eval=lambda: None, config=SimpleNamespace(vocab_size=7))],
        device_mesh={"pp": _PipelineMesh()},
        cfg=SimpleNamespace(get=lambda *_args: None),
    )

    with (
        patch(
            "tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.dist.get_global_rank",
            return_value=0,
        ),
        patch("tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm.dist.broadcast"),
    ):
        logits = _get_logits_pp(trainer, [11, 12, 13], torch.device("cpu"))

    assert schedule.ids.tolist() == [[11, 12, 13] + [0] * 13]
    assert schedule.attention_mask.shape == (1, 16)
    assert schedule.attention_mask.tolist() == [[1, 1, 1] + [0] * 13]
    assert logits.shape == (1, 3, 7)


@pytest.mark.parametrize(
    ("model_type", "expected_attn_implementation"),
    [("nemotron_h", "eager"), ("step3p7", "eager"), ("nemotron_flash", "flash_attention_2")],
)
def test_remote_code_attention_implementation(model_type, expected_attn_implementation):
    with patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type=model_type),
    ) as from_pretrained:
        hf_kwargs = _hf_source_load_kwargs(
            {"revision": "model-revision", "token": "model-token"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["attn_implementation"] == expected_attn_implementation
    from_pretrained.assert_called_once_with(
        "model-path",
        trust_remote_code=True,
        revision="model-revision",
        token="model-token",
    )


def test_explicit_attention_implementation_is_preserved():
    with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("must not probe config")):
        hf_kwargs = _hf_source_load_kwargs(
            {"attn_implementation": "eager"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["attn_implementation"] == "eager"


def test_hf_source_load_kwargs_passes_grouped_experts_implementation():
    hf_kwargs = _hf_source_load_kwargs(
        {},
        pretrained_model_name_or_path="model-path",
        source_dtype=torch.bfloat16,
        trust_remote_code=False,
        experts_implementation="grouped_mm",
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
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [11, 12, 13])

    with patch("nemo_automodel.NeMoAutoTokenizer.from_pretrained", return_value=tokenizer) as from_pretrained:
        input_ids = _get_input_ids("mistralai/Ministral-3-3B-Instruct-2512")

    assert input_ids == [11, 12, 13]
    from_pretrained.assert_called_once_with(
        "mistralai/Ministral-3-3B-Instruct-2512",
        trust_remote_code=True,
        local_files_only=expected_local_files_only,
    )


@pytest.mark.parametrize(("offline", "expected_local_files_only"), [(None, False), ("1", True)])
def test_get_vlm_input_ids_uses_processor_tokenizer(monkeypatch, offline, expected_local_files_only):
    if offline is None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", offline)
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [21, 22, 23])
    processor = SimpleNamespace(tokenizer=tokenizer)

    with patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as from_pretrained:
        input_ids = _get_vlm_input_ids("mistralai/Ministral-3-3B-Reasoning-2512")

    assert input_ids == [21, 22, 23]
    from_pretrained.assert_called_once_with(
        "mistralai/Ministral-3-3B-Reasoning-2512",
        trust_remote_code=True,
        local_files_only=expected_local_files_only,
    )


def test_load_input_ids_once_shares_rank0_result(tmp_path, monkeypatch):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"))
    rank0_loader = Mock(return_value=[31, 32, 33])
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "input-id-test")
    monkeypatch.setenv("RANK", "0")

    assert _load_input_ids_once(cfg, rank0_loader, "model/tokenizer") == [31, 32, 33]
    rank0_loader.assert_called_once_with("model/tokenizer")

    rank1_loader = Mock(side_effect=AssertionError("nonzero rank must not load the tokenizer"))
    monkeypatch.setenv("RANK", "1")

    assert _load_input_ids_once(cfg, rank1_loader, "model/tokenizer") == [31, 32, 33]
    rank1_loader.assert_not_called()

    rank0_reuse_loader = Mock(side_effect=AssertionError("rank 0 must reuse the published input IDs"))
    monkeypatch.setenv("RANK", "0")

    assert _load_input_ids_once(cfg, rank0_reuse_loader, "model/tokenizer") == [31, 32, 33]
    rank0_reuse_loader.assert_not_called()


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
        assert _load_input_ids_once(cfg, loader, "model/tokenizer") == [41, 42, 43]

    loader.assert_not_called()


def test_load_input_ids_once_propagates_rank0_failure(tmp_path, monkeypatch):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "input-id-failure-test")
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(ValueError, match="tokenizer failed"):
        _load_input_ids_once(cfg, Mock(side_effect=ValueError("tokenizer failed")), "model/tokenizer")

    monkeypatch.setenv("RANK", "1")
    with pytest.raises(RuntimeError, match="Rank 0 input-ID loading failed"):
        _load_input_ids_once(cfg, Mock(), "model/tokenizer")


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


def test_extract_custom_args_accepts_skip_hf_logit_parity():
    custom, remaining = _extract_custom_args(["--skip_hf_logit_parity", "--other-arg"])

    assert custom["skip_hf_logit_parity"] is True
    assert remaining == ["--other-arg"]


def test_extract_custom_args_accepts_skip_automodel_logit_parity():
    custom, remaining = _extract_custom_args(["--skip_automodel_logit_parity", "--other-arg"])

    assert custom["skip_automodel_logit_parity"] is True
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
    custom_args = {"hf_device_map_auto": True, "no_check_resume": True, "trust_remote_code": True}

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


def test_process_isolated_resume_rejects_no_check_resume():
    with pytest.raises(ValueError, match="conflicts with no_check_resume=true"):
        _run_process_isolated_checkpoint_phase(
            "resume",
            custom_args={"no_check_resume": True},
            recipe_cls=Mock(),
            hf_model_cls=Mock(),
            input_ids_loader=Mock(),
        )


@pytest.mark.parametrize("non_finite_kl", [float("nan"), float("inf"), float("-inf")])
def test_hf_reload_rejects_non_finite_kl(non_finite_kl):
    assert "non-finite KL divergence" in _hf_reload_kl_error(non_finite_kl, 7e-2)


def test_process_isolated_source_load_reference_persists_hf_artifacts(tmp_path):
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=tmp_path))
    reference_logits = torch.randn(1, 2, 3)
    source_reference = (reference_logits, False, False)
    recipe_cls = Mock()
    hf_model_cls = Mock()
    custom_args = {
        "check_source_load_parity": True,
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
        "check_source_load_parity": True,
        "source_load_kl_threshold": "4e-2",
        "source_load_mean_kl_threshold": "1e-2",
        "source_load_cosine_threshold": "0.9985",
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
        "source_load_kl_threshold": 4e-2,
        "source_load_mean_kl_threshold": 1e-2,
        "source_load_cosine_threshold": 0.9985,
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


def test_source_load_parity_failure_is_returned_for_later_reporting():
    reference_logits = torch.tensor([[[2.0, -2.0], [1.0, -1.0]]])
    candidate_logits = -reference_logits

    failure = _compare_source_load_parity(
        (reference_logits, None, None),
        candidate_logits,
        None,
        source_load_kl_threshold=0.0,
        source_load_mean_kl_threshold=0.0,
        source_load_cosine_threshold=1.0,
    )

    assert failure is not None
    assert "KL divergence between original HF source load and constructed trainer model too large" in failure


def test_source_load_parity_success_returns_no_deferred_failure():
    logits = torch.tensor([[[2.0, -2.0], [1.0, -1.0]]])

    failure = _compare_source_load_parity(
        (logits, None, None),
        logits.clone(),
        None,
        source_load_kl_threshold=0.0,
        source_load_mean_kl_threshold=0.0,
        source_load_cosine_threshold=1.0,
    )

    assert failure is None


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


def test_hf_reload_finish_returns_error_without_distributed_sync():
    assert _finish_hf_reload_sync(None, "HF parity failed") == "HF parity failed"


def test_biencoder_robustness_reads_hf_reload_settings_from_config(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "ci:\n"
        "  checkpoint_robustness:\n"
        "    check_hf_reload: true\n"
        "    check_resume: true\n"
        "    cosine_threshold: 0.998\n"
        "    hf_cosine_threshold: 0.997\n"
        "    dataloader.num_workers: 0\n"
    )

    custom, remaining = _extract_biencoder_custom_args(["--config", str(config_path)])

    assert custom == {
        "check_hf_reload": True,
        "check_resume": True,
        "cosine_threshold": "0.998",
        "hf_cosine_threshold": "0.997",
    }
    assert remaining == ["--config", str(config_path)]


def test_record_deferred_failure_preserves_all_comparison_failures():
    failures = []

    _record_deferred_failure(failures, "Phase 3 AutoModel reload parity", None)
    _record_deferred_failure(failures, "Phase 4 HF reload parity", "HF parity failed")

    assert failures == ["Phase 4 HF reload parity:\nHF parity failed"]
