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

import os
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_biencoder import (
    _extract_custom_args as _extract_biencoder_custom_args,
)
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _compare_source_load_parity,
    _dequantize_hf_fp8_weights_in_place,
    _extract_custom_args,
    _finish_hf_reload_sync,
    _get_input_ids,
    _hf_source_load_kwargs,
    _load_hf_fp8_dequantized_config,
    _post_load_dequant_max_memory,
    _prepare_hf_reload_sync,
    _record_deferred_failure,
    _wait_for_hf_reload_rank0,
)
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_vlm import _get_vlm_input_ids


def _run_hf_reload_sync_rank(rank, init_path, checkpoint_dir):
    os.environ["TORCHELASTIC_RUN_ID"] = "checkpoint-robustness-hf-sync-test"
    os.environ["HF_RELOAD_TIMEOUT_SECONDS"] = "15"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=3),
    )
    try:
        cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=checkpoint_dir))
        sync_paths = _prepare_hf_reload_sync(cfg)
        if rank == 0:
            time.sleep(4)
        _finish_hf_reload_sync(sync_paths)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("model_type", "expected_attn_implementation"),
    [("nemotron_h", "eager"), ("nemotron_flash", "flash_attention_2")],
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


def test_extract_custom_args_accepts_hf_source_post_load_dequantize():
    custom, remaining = _extract_custom_args(["--hf_source_post_load_dequantize", "--other-arg"])

    assert custom["hf_source_post_load_dequantize"] is True
    assert remaining == ["--other-arg"]


def test_source_load_parity_failure_is_returned_for_later_reporting():
    reference_logits = torch.tensor([[[2.0, -2.0], [1.0, -1.0]]])
    candidate_logits = -reference_logits

    failure = _compare_source_load_parity(
        (reference_logits, None, None),
        candidate_logits,
        SimpleNamespace(),
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
        SimpleNamespace(),
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


def test_record_deferred_failure_preserves_all_comparison_failures():
    failures = []

    _record_deferred_failure(failures, "Phase 3 AutoModel reload parity", None)
    _record_deferred_failure(failures, "Phase 4 HF reload parity", "HF parity failed")

    assert failures == ["Phase 4 HF reload parity:\nHF parity failed"]


def test_hf_reload_wait_does_not_start_collective_during_rank0_work(tmp_path):
    mp.spawn(
        _run_hf_reload_sync_rank,
        args=(str(tmp_path / "dist-init"), str(tmp_path / "checkpoints")),
        nprocs=2,
        join=True,
    )


def test_biencoder_robustness_reads_hf_reload_settings_from_config(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "ci:\n"
        "  checkpoint_robustness:\n"
        "    check_hf_reload: true\n"
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
    assert remaining == ["--config", str(config_path), "--dataloader.num_workers", "0"]


def test_biencoder_robustness_can_disable_resume_from_config(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text("ci:\n  checkpoint_robustness:\n    no_check_resume: true\n")

    custom, remaining = _extract_biencoder_custom_args(["--config", str(config_path)])

    assert "check_resume" not in custom
    assert "no_check_resume" not in custom
    assert remaining == ["--config", str(config_path)]
