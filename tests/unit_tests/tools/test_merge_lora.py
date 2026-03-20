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

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


# ---------------------------------------------------------------------------
# _resolve_auto_cls
# ---------------------------------------------------------------------------


class TestResolveAutoCls:
    def test_explicit_model_class(self, tmp_path):
        from transformers import AutoModel

        from tools.merge_lora import _resolve_auto_cls

        cls = _resolve_auto_cls(str(tmp_path), model_class="AutoModel")
        assert cls is AutoModel

    def test_explicit_model_class_causal_lm(self, tmp_path):
        from transformers import AutoModelForCausalLM

        from tools.merge_lora import _resolve_auto_cls

        cls = _resolve_auto_cls(str(tmp_path), model_class="AutoModelForCausalLM")
        assert cls is AutoModelForCausalLM

    def test_explicit_invalid_class_raises(self, tmp_path):
        from tools.merge_lora import _resolve_auto_cls

        with pytest.raises(ValueError, match="Unknown model class"):
            _resolve_auto_cls(str(tmp_path), model_class="NoSuchAutoModel")

    def test_task_type_causal_lm(self, tmp_path):
        from transformers import AutoModelForCausalLM

        from tools.merge_lora import _resolve_auto_cls

        config = {"task_type": "CAUSAL_LM", "r": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModelForCausalLM

    def test_task_type_feature_extraction(self, tmp_path):
        from transformers import AutoModel

        from tools.merge_lora import _resolve_auto_cls

        config = {"task_type": "FEATURE_EXTRACTION", "r": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModel

    def test_task_type_seq_cls(self, tmp_path):
        from transformers import AutoModelForSequenceClassification

        from tools.merge_lora import _resolve_auto_cls

        config = {"task_type": "SEQ_CLS", "r": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModelForSequenceClassification

    def test_task_type_seq_2_seq_lm(self, tmp_path):
        from transformers import AutoModelForSeq2SeqLM

        from tools.merge_lora import _resolve_auto_cls

        config = {"task_type": "SEQ_2_SEQ_LM", "r": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModelForSeq2SeqLM

    def test_task_type_token_cls(self, tmp_path):
        from transformers import AutoModelForTokenClassification

        from tools.merge_lora import _resolve_auto_cls

        config = {"task_type": "TOKEN_CLS", "r": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModelForTokenClassification

    def test_task_type_question_ans(self, tmp_path):
        from transformers import AutoModelForQuestionAnswering

        from tools.merge_lora import _resolve_auto_cls

        config = {"task_type": "QUESTION_ANS", "r": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModelForQuestionAnswering

    def test_unknown_task_type_falls_back(self, tmp_path):
        from transformers import AutoModelForCausalLM

        from tools.merge_lora import _resolve_auto_cls

        config = {"task_type": "SOME_FUTURE_TASK", "r": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModelForCausalLM

    def test_no_adapter_config_falls_back(self, tmp_path):
        from transformers import AutoModelForCausalLM

        from tools.merge_lora import _resolve_auto_cls

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModelForCausalLM

    def test_no_task_type_key_falls_back(self, tmp_path):
        from transformers import AutoModelForCausalLM

        from tools.merge_lora import _resolve_auto_cls

        config = {"r": 8, "lora_alpha": 16}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path))
        assert cls is AutoModelForCausalLM

    def test_explicit_overrides_adapter_config(self, tmp_path):
        from transformers import AutoModel

        from tools.merge_lora import _resolve_auto_cls

        config = {"task_type": "CAUSAL_LM", "r": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        cls = _resolve_auto_cls(str(tmp_path), model_class="AutoModel")
        assert cls is AutoModel


# ---------------------------------------------------------------------------
# _clean_quantization_config
# ---------------------------------------------------------------------------


class TestCleanQuantizationConfig:
    def test_removes_quantization_config(self, tmp_path):
        from tools.merge_lora import _clean_quantization_config

        config = {"hidden_size": 64, "quantization_config": {"bits": 4}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        _clean_quantization_config(str(tmp_path))

        result = json.loads(config_path.read_text())
        assert "quantization_config" not in result
        assert result["hidden_size"] == 64

    def test_removes_pretraining_tp(self, tmp_path):
        from tools.merge_lora import _clean_quantization_config

        config = {"hidden_size": 64, "pretraining_tp": 2}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        _clean_quantization_config(str(tmp_path))

        result = json.loads(config_path.read_text())
        assert "pretraining_tp" not in result

    def test_removes_both_keys(self, tmp_path):
        from tools.merge_lora import _clean_quantization_config

        config = {"hidden_size": 64, "quantization_config": {}, "pretraining_tp": 1}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        _clean_quantization_config(str(tmp_path))

        result = json.loads(config_path.read_text())
        assert "quantization_config" not in result
        assert "pretraining_tp" not in result
        assert result["hidden_size"] == 64

    def test_no_op_when_keys_absent(self, tmp_path):
        from tools.merge_lora import _clean_quantization_config

        config = {"hidden_size": 64, "num_layers": 2}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        _clean_quantization_config(str(tmp_path))

        result = json.loads(config_path.read_text())
        assert result == config

    def test_no_op_when_config_missing(self, tmp_path):
        from tools.merge_lora import _clean_quantization_config

        _clean_quantization_config(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------------
# dequantize_model
# ---------------------------------------------------------------------------


class _FakeLinear4bit(nn.Module):
    """Mimics ``bnb.nn.Linear4bit`` as a real ``nn.Module`` subclass."""

    def __init__(self, in_features, out_features, bias=True, quant_state=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.quant_state = quant_state or [None, None, torch.float16]
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None


class TestDequantizeModel:
    def _patch_bnb(self, fake_dequant):
        """Context manager that patches bitsandbytes so ``dequantize_model`` can import it."""
        mock_bnb = MagicMock()
        mock_bnb.nn.Linear4bit = _FakeLinear4bit
        mock_functional = MagicMock()
        mock_functional.dequantize_4bit = fake_dequant
        return patch.dict(
            "sys.modules",
            {
                "bitsandbytes": mock_bnb,
                "bitsandbytes.nn": mock_bnb.nn,
                "bitsandbytes.functional": mock_functional,
            },
        )

    def test_replaces_linear4bit_with_linear(self):
        from tools.merge_lora import dequantize_model

        model = nn.Module()
        model.add_module("layer", _FakeLinear4bit(16, 32, bias=False))

        fake_dequant = MagicMock(return_value=torch.randn(32, 16).to(torch.float16))
        with self._patch_bnb(fake_dequant):
            result = dequantize_model(model, dtype=torch.float16, device="cpu")

        assert isinstance(result.layer, nn.Linear)
        assert result.layer.weight.dtype == torch.float16
        assert result.layer.bias is None
        assert result.is_loaded_in_4bit is False

    def test_handles_nested_module_replacement(self):
        """Verify that a 4-bit module nested inside a submodule gets replaced."""
        from tools.merge_lora import dequantize_model

        parent = nn.Module()
        child = nn.Module()
        parent.add_module("sub", child)
        child.add_module("proj", _FakeLinear4bit(8, 8, bias=False))

        fake_dequant = MagicMock(return_value=torch.randn(8, 8).to(torch.float16))
        with self._patch_bnb(fake_dequant):
            dequantize_model(parent, dtype=torch.float16, device="cpu")

        assert isinstance(parent.sub.proj, nn.Linear)
        assert parent.sub.proj.weight.dtype == torch.float16

    def test_preserves_bias(self):
        """When the 4-bit module has a bias, the replacement keeps it."""
        from tools.merge_lora import dequantize_model

        model = nn.Module()
        model.add_module("biased", _FakeLinear4bit(4, 4, bias=True))

        fake_dequant = MagicMock(return_value=torch.randn(4, 4).to(torch.float32))
        with self._patch_bnb(fake_dequant):
            dequantize_model(model, dtype=torch.float32, device="cpu")

        assert isinstance(model.biased, nn.Linear)
        assert model.biased.bias is not None
        assert model.biased.weight.shape == (4, 4)

    def test_skips_non_4bit_modules(self):
        """Regular ``nn.Linear`` modules should not be touched."""
        from tools.merge_lora import dequantize_model

        model = nn.Module()
        original = nn.Linear(4, 4)
        model.add_module("regular", original)

        fake_dequant = MagicMock()
        with self._patch_bnb(fake_dequant):
            dequantize_model(model, dtype=torch.float16, device="cpu")

        fake_dequant.assert_not_called()
        assert model.regular is original

    def test_top_level_module_replacement(self):
        """A 4-bit module registered at the top level (no dot in name) is replaced."""
        from tools.merge_lora import dequantize_model

        model = nn.Module()
        model.add_module("top", _FakeLinear4bit(4, 8, bias=False))

        fake_dequant = MagicMock(return_value=torch.randn(8, 4).to(torch.float16))
        with self._patch_bnb(fake_dequant):
            dequantize_model(model, dtype=torch.float16, device="cpu")

        assert isinstance(model.top, nn.Linear)
        assert model.top.in_features == 4
        assert model.top.out_features == 8


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_args(self, monkeypatch):
        from tools.merge_lora import parse_args

        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--base-model", "my-model", "--adapter-path", "/a", "--output-dir", "/o"],
        )
        args = parse_args()
        assert args.base_model == "my-model"
        assert args.adapter_path == "/a"
        assert args.output_dir == "/o"

    def test_defaults(self, monkeypatch):
        from tools.merge_lora import parse_args

        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "-m", "m", "-a", "/a", "-o", "/o"],
        )
        args = parse_args()
        assert args.qlora is False
        assert args.dtype == "float16"
        assert args.device == "auto"
        assert args.no_save_tokenizer is False
        assert args.trust_remote_code is False
        assert args.model_class is None

    def test_all_flags(self, monkeypatch):
        from tools.merge_lora import parse_args

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "-m",
                "model",
                "-a",
                "/adapter",
                "-o",
                "/out",
                "--qlora",
                "--dtype",
                "bfloat16",
                "--device",
                "cpu",
                "--no-save-tokenizer",
                "--trust-remote-code",
                "--model-class",
                "AutoModel",
            ],
        )
        args = parse_args()
        assert args.qlora is True
        assert args.dtype == "bfloat16"
        assert args.device == "cpu"
        assert args.no_save_tokenizer is True
        assert args.trust_remote_code is True
        assert args.model_class == "AutoModel"

    def test_short_flags(self, monkeypatch):
        from tools.merge_lora import parse_args

        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "-m", "base", "-a", "adapter", "-o", "output"],
        )
        args = parse_args()
        assert args.base_model == "base"
        assert args.adapter_path == "adapter"
        assert args.output_dir == "output"

    def test_missing_required_arg_exits(self, monkeypatch):
        from tools.merge_lora import parse_args

        monkeypatch.setattr(sys, "argv", ["prog", "--base-model", "m"])
        with pytest.raises(SystemExit):
            parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_delegates_to_merge_lora(self, monkeypatch):
        from tools import merge_lora as merge_mod

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "-m",
                "/model",
                "-a",
                "/adapter",
                "-o",
                "/out",
                "--qlora",
                "--dtype",
                "bfloat16",
                "--device",
                "cpu",
                "--no-save-tokenizer",
                "--trust-remote-code",
                "--model-class",
                "AutoModel",
            ],
        )

        captured = {}

        def fake_merge(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(merge_mod, "merge_lora", fake_merge)
        merge_mod.main()

        assert captured["base_model"] == "/model"
        assert captured["adapter_path"] == "/adapter"
        assert captured["output_dir"] == "/out"
        assert captured["qlora"] is True
        assert captured["dtype"] == "bfloat16"
        assert captured["device"] == "cpu"
        assert captured["save_tokenizer"] is False
        assert captured["trust_remote_code"] is True
        assert captured["model_class"] == "AutoModel"

    def test_main_default_save_tokenizer_true(self, monkeypatch):
        from tools import merge_lora as merge_mod

        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "-m", "/m", "-a", "/a", "-o", "/o"],
        )

        captured = {}

        def fake_merge(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(merge_mod, "merge_lora", fake_merge)
        merge_mod.main()

        assert captured["save_tokenizer"] is True
        assert captured["model_class"] is None


# ---------------------------------------------------------------------------
# merge_lora (mocked)
# ---------------------------------------------------------------------------


class TestMergeLoraFunction:
    def _make_mock_model(self):
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []
        mock_model.named_modules.return_value = []
        return mock_model

    @patch("tools.merge_lora.gc")
    @patch("tools.merge_lora.torch")
    def test_lora_merge_basic_path(self, mock_torch, mock_gc, tmp_path, monkeypatch):
        from tools.merge_lora import merge_lora

        mock_torch.float16 = torch.float16
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_model = self._make_mock_model()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model

        mock_auto = MagicMock()
        mock_auto.__name__ = "AutoModelForCausalLM"
        mock_auto.from_pretrained.return_value = mock_model
        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        with patch("tools.merge_lora._resolve_auto_cls", return_value=mock_auto):
            with patch.dict(
                "sys.modules",
                {
                    "peft": MagicMock(PeftModel=mock_peft_cls),
                    "transformers": MagicMock(
                        AutoTokenizer=MagicMock(),
                    ),
                },
            ):
                merge_lora(
                    base_model="/fake/model",
                    adapter_path="/fake/adapter",
                    output_dir=str(tmp_path / "out"),
                    qlora=False,
                    dtype="float16",
                    device="cpu",
                    save_tokenizer=False,
                )

        mock_auto.from_pretrained.assert_called_once()
        mock_peft_cls.from_pretrained.assert_called_once()
        mock_peft_model.merge_and_unload.assert_called_once()
        mock_model.save_pretrained.assert_called_once()

    @patch("tools.merge_lora.gc")
    @patch("tools.merge_lora.torch")
    def test_lora_merge_qlora_path(self, mock_torch, mock_gc, tmp_path):
        from tools.merge_lora import merge_lora

        mock_torch.float16 = torch.float16
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_model = self._make_mock_model()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model

        mock_auto = MagicMock()
        mock_auto.__name__ = "AutoModelForCausalLM"
        mock_auto.from_pretrained.return_value = mock_model
        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model
        mock_tokenizer_cls = MagicMock()
        mock_bnb_config_cls = MagicMock()
        mock_dequantize = MagicMock(return_value=mock_model)

        with patch("tools.merge_lora._resolve_auto_cls", return_value=mock_auto):
            with patch.dict(
                "sys.modules",
                {
                    "peft": MagicMock(PeftModel=mock_peft_cls),
                    "transformers": MagicMock(
                        AutoTokenizer=mock_tokenizer_cls,
                        BitsAndBytesConfig=mock_bnb_config_cls,
                    ),
                },
            ):
                with patch("tools.merge_lora.dequantize_model", mock_dequantize):
                    merge_lora(
                        base_model="/fake/model",
                        adapter_path="/fake/adapter",
                        output_dir=str(tmp_path / "out"),
                        qlora=True,
                        dtype="float16",
                        device="cpu",
                        save_tokenizer=False,
                    )

        mock_dequantize.assert_called_once()
        load_kwargs = mock_auto.from_pretrained.call_args
        assert "quantization_config" in load_kwargs.kwargs or any(
            "quantization_config" in str(a) for a in load_kwargs.args
        )

    @patch("tools.merge_lora.gc")
    @patch("tools.merge_lora.torch")
    def test_lora_merge_saves_tokenizer(self, mock_torch, mock_gc, tmp_path):
        from tools.merge_lora import merge_lora

        mock_torch.float16 = torch.float16

        mock_model = self._make_mock_model()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model

        mock_auto = MagicMock()
        mock_auto.__name__ = "AutoModelForCausalLM"
        mock_auto.from_pretrained.return_value = mock_model
        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model
        mock_tokenizer_cls = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch("tools.merge_lora._resolve_auto_cls", return_value=mock_auto):
            with patch.dict(
                "sys.modules",
                {
                    "peft": MagicMock(PeftModel=mock_peft_cls),
                    "transformers": MagicMock(
                        AutoTokenizer=mock_tokenizer_cls,
                    ),
                },
            ):
                merge_lora(
                    base_model="/fake/model",
                    adapter_path="/fake/adapter",
                    output_dir=str(tmp_path / "out"),
                    dtype="float16",
                    device="cpu",
                    save_tokenizer=True,
                )

        mock_tokenizer_cls.from_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    @patch("tools.merge_lora.gc")
    @patch("tools.merge_lora.torch")
    def test_lora_merge_tokenizer_failure_is_warning(self, mock_torch, mock_gc, tmp_path):
        """If tokenizer save fails, merge_lora logs a warning but does not raise."""
        from tools.merge_lora import merge_lora

        mock_torch.float16 = torch.float16

        mock_model = self._make_mock_model()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model

        mock_auto = MagicMock()
        mock_auto.__name__ = "AutoModelForCausalLM"
        mock_auto.from_pretrained.return_value = mock_model
        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model
        mock_tokenizer_cls = MagicMock()
        mock_tokenizer_cls.from_pretrained.side_effect = RuntimeError("no tokenizer")

        with patch("tools.merge_lora._resolve_auto_cls", return_value=mock_auto):
            with patch.dict(
                "sys.modules",
                {
                    "peft": MagicMock(PeftModel=mock_peft_cls),
                    "transformers": MagicMock(
                        AutoTokenizer=mock_tokenizer_cls,
                    ),
                },
            ):
                merge_lora(
                    base_model="/fake/model",
                    adapter_path="/fake/adapter",
                    output_dir=str(tmp_path / "out"),
                    dtype="float16",
                    device="cpu",
                    save_tokenizer=True,
                )

    @patch("tools.merge_lora.gc")
    @patch("tools.merge_lora.torch")
    def test_qlora_cleans_quantization_config(self, mock_torch, mock_gc, tmp_path):
        from tools.merge_lora import merge_lora

        mock_torch.float16 = torch.float16

        mock_model = self._make_mock_model()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model

        output_dir = str(tmp_path / "out")

        def fake_save(path, **kwargs):
            import os

            os.makedirs(path, exist_ok=True)
            config = {"hidden_size": 64, "quantization_config": {"bits": 4}}
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(config, f)

        mock_model.save_pretrained.side_effect = fake_save

        mock_auto = MagicMock()
        mock_auto.__name__ = "AutoModelForCausalLM"
        mock_auto.from_pretrained.return_value = mock_model
        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model
        mock_bnb_config = MagicMock()
        mock_dequantize = MagicMock(return_value=mock_model)

        with patch("tools.merge_lora._resolve_auto_cls", return_value=mock_auto):
            with patch.dict(
                "sys.modules",
                {
                    "peft": MagicMock(PeftModel=mock_peft_cls),
                    "transformers": MagicMock(
                        AutoTokenizer=MagicMock(),
                        BitsAndBytesConfig=mock_bnb_config,
                    ),
                },
            ):
                with patch("tools.merge_lora.dequantize_model", mock_dequantize):
                    merge_lora(
                        base_model="/fake/model",
                        adapter_path="/fake/adapter",
                        output_dir=output_dir,
                        qlora=True,
                        dtype="float16",
                        device="cpu",
                        save_tokenizer=False,
                    )

        config_path = Path(output_dir) / "config.json"
        result = json.loads(config_path.read_text())
        assert "quantization_config" not in result

    @patch("tools.merge_lora.gc")
    @patch("tools.merge_lora.torch")
    def test_trust_remote_code_forwarded(self, mock_torch, mock_gc, tmp_path):
        from tools.merge_lora import merge_lora

        mock_torch.float16 = torch.float16

        mock_model = self._make_mock_model()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model

        mock_auto = MagicMock()
        mock_auto.__name__ = "AutoModelForCausalLM"
        mock_auto.from_pretrained.return_value = mock_model
        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        with patch("tools.merge_lora._resolve_auto_cls", return_value=mock_auto):
            with patch.dict(
                "sys.modules",
                {
                    "peft": MagicMock(PeftModel=mock_peft_cls),
                    "transformers": MagicMock(
                        AutoTokenizer=MagicMock(),
                    ),
                },
            ):
                merge_lora(
                    base_model="/fake/model",
                    adapter_path="/fake/adapter",
                    output_dir=str(tmp_path / "out"),
                    dtype="float16",
                    device="cpu",
                    save_tokenizer=False,
                    trust_remote_code=True,
                )

        call_kwargs = mock_auto.from_pretrained.call_args
        assert call_kwargs.kwargs.get("trust_remote_code") is True or call_kwargs[1].get("trust_remote_code") is True

    @patch("tools.merge_lora.gc")
    @patch("tools.merge_lora.torch")
    def test_dtype_bfloat16(self, mock_torch, mock_gc, tmp_path):
        from tools.merge_lora import merge_lora

        mock_torch.bfloat16 = torch.bfloat16

        mock_model = self._make_mock_model()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model

        mock_auto = MagicMock()
        mock_auto.__name__ = "AutoModelForCausalLM"
        mock_auto.from_pretrained.return_value = mock_model
        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        with patch("tools.merge_lora._resolve_auto_cls", return_value=mock_auto):
            with patch.dict(
                "sys.modules",
                {
                    "peft": MagicMock(PeftModel=mock_peft_cls),
                    "transformers": MagicMock(
                        AutoTokenizer=MagicMock(),
                    ),
                },
            ):
                merge_lora(
                    base_model="/fake/model",
                    adapter_path="/fake/adapter",
                    output_dir=str(tmp_path / "out"),
                    dtype="bfloat16",
                    device="cpu",
                    save_tokenizer=False,
                )

        call_kwargs = mock_auto.from_pretrained.call_args
        assert (
            call_kwargs.kwargs.get("torch_dtype") == torch.bfloat16
            or call_kwargs[1].get("torch_dtype") == torch.bfloat16
        )
