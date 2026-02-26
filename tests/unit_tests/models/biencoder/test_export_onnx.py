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

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_automodel.components.models.llama_bidirectional.export_onnx import (
    EmbeddingModelForExport,
    _Pooling,
    _parse_args,
    export_to_onnx,
    main,
    verify_onnx,
)


# _Pooling


class TestPooling:
    @pytest.fixture()
    def hidden_and_mask(self):
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]],
                [[2.0, 1.0], [4.0, 3.0], [1.5, 0.5]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
        return hidden, mask

    def test_avg_pooling(self, hidden_and_mask):
        hidden, mask = hidden_and_mask
        pool = _Pooling("avg")
        out = pool(hidden, mask)
        expected_0 = torch.tensor([(1.0 + 3.0) / 2, (2.0 + 4.0) / 2])
        assert torch.allclose(out[0], expected_0, atol=1e-6)
        assert out.shape == (2, 2)

    def test_cls_pooling(self, hidden_and_mask):
        hidden, mask = hidden_and_mask
        pool = _Pooling("cls")
        out = pool(hidden, mask)
        assert torch.allclose(out, hidden[:, 0])

    def test_last_pooling(self, hidden_and_mask):
        hidden, mask = hidden_and_mask
        pool = _Pooling("last")
        out = pool(hidden, mask)
        # mask[0] = [1,1,0] so position 2 is zeroed; last slice picks that zero
        assert torch.allclose(out[0], torch.tensor([0.0, 0.0]))
        # mask[1] = [1,1,1] so position 2 is kept
        assert torch.allclose(out[1], hidden[1, -1])

    def test_invalid_pool_type(self, hidden_and_mask):
        hidden, mask = hidden_and_mask
        pool = _Pooling("max")
        with pytest.raises(ValueError, match="Unsupported pool_type"):
            pool(hidden, mask)


# EmbeddingModelForExport

class TestEmbeddingModelForExport:
    @pytest.fixture()
    def fake_base_model(self):
        model = MagicMock()
        model.return_value = {"last_hidden_state": torch.randn(2, 5, 16)}
        return model

    def test_forward_with_normalize(self, fake_base_model):
        pooling = _Pooling("avg")
        wrapper = EmbeddingModelForExport(fake_base_model, pooling, normalize=True)
        ids = torch.ones(2, 5, dtype=torch.long)
        mask = torch.ones(2, 5, dtype=torch.long)
        out = wrapper(ids, mask)
        assert out.shape == (2, 16)
        norms = torch.linalg.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_forward_without_normalize(self, fake_base_model):
        pooling = _Pooling("avg")
        wrapper = EmbeddingModelForExport(fake_base_model, pooling, normalize=False)
        ids = torch.ones(2, 5, dtype=torch.long)
        mask = torch.ones(2, 5, dtype=torch.long)
        out = wrapper(ids, mask)
        assert out.shape == (2, 16)
        norms = torch.linalg.norm(out, dim=1)
        assert not torch.allclose(norms, torch.ones(2), atol=1e-5)


# Helpers to build mocks for export_to_onnx


def _make_fake_tokenizer(tmp_path):
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    tok.save_pretrained = MagicMock()
    return tok


def _make_fake_base_model():
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model
    model.return_value = {"last_hidden_state": torch.randn(2, 4, 8)}
    model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))
    return model


# export_to_onnx


class TestExportToOnnx:
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.torch.onnx.export")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoModel")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoTokenizer")
    def test_basic_export(self, mock_tokenizer_cls, mock_model_cls, mock_onnx_export, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_tokenizer_cls.from_pretrained.return_value = _make_fake_tokenizer(tmp_path)
        mock_model_cls.from_pretrained.return_value = _make_fake_base_model()

        result = export_to_onnx(
            model_path=str(model_dir),
            output_dir=str(output_dir),
            verify=False,
        )

        assert result.endswith("model.onnx")
        mock_onnx_export.assert_called_once()
        mock_tokenizer_cls.from_pretrained.return_value.save_pretrained.assert_called_once()

    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.torch.onnx.export")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoModel")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoTokenizer")
    def test_explicit_tokenizer_path(self, mock_tokenizer_cls, mock_model_cls, mock_onnx_export, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        tok_dir = tmp_path / "tokenizer_src"
        tok_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_tokenizer_cls.from_pretrained.return_value = _make_fake_tokenizer(tmp_path)
        mock_model_cls.from_pretrained.return_value = _make_fake_base_model()

        export_to_onnx(
            model_path=str(model_dir),
            output_dir=str(output_dir),
            tokenizer_path=str(tok_dir),
            verify=False,
        )

        mock_tokenizer_cls.from_pretrained.assert_called_once()
        call_args = mock_tokenizer_cls.from_pretrained.call_args
        assert str(tok_dir.resolve()) in call_args[0][0]

    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.torch.onnx.export")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoModel")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoTokenizer")
    def test_invalid_export_dtype(self, mock_tokenizer_cls, mock_model_cls, mock_onnx_export, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_tokenizer_cls.from_pretrained.return_value = _make_fake_tokenizer(tmp_path)
        mock_model_cls.from_pretrained.return_value = _make_fake_base_model()

        with pytest.raises(ValueError, match="Unsupported export_dtype"):
            export_to_onnx(
                model_path=str(model_dir),
                output_dir=str(tmp_path / "out"),
                export_dtype="bf16",
                verify=False,
            )

    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.verify_onnx")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.torch.onnx.export")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoModel")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoTokenizer")
    def test_verify_called_when_enabled(
        self, mock_tokenizer_cls, mock_model_cls, mock_onnx_export, mock_verify, tmp_path
    ):
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_tokenizer_cls.from_pretrained.return_value = _make_fake_tokenizer(tmp_path)
        mock_model_cls.from_pretrained.return_value = _make_fake_base_model()

        export_to_onnx(
            model_path=str(model_dir),
            output_dir=str(tmp_path / "out"),
            verify=True,
        )

        mock_verify.assert_called_once()

    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.torch.onnx.export")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoModel")
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.AutoTokenizer")
    def test_fp16_export(self, mock_tokenizer_cls, mock_model_cls, mock_onnx_export, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_tokenizer_cls.from_pretrained.return_value = _make_fake_tokenizer(tmp_path)
        mock_model_cls.from_pretrained.return_value = _make_fake_base_model()

        result = export_to_onnx(
            model_path=str(model_dir),
            output_dir=str(tmp_path / "out"),
            export_dtype="fp16",
            verify=False,
        )

        assert result.endswith("model.onnx")
        mock_onnx_export.assert_called_once()


# verify_onnx


class TestVerifyOnnx:
    def test_verify_runs_successfully(self, tmp_path):
        hidden_dim = 8
        batch = 2
        fake_embeddings = np.random.randn(batch, hidden_dim).astype(np.float32)

        fake_session = MagicMock()
        fake_input = MagicMock()
        fake_input.name = "input_ids"
        fake_input2 = MagicMock()
        fake_input2.name = "attention_mask"
        fake_output = MagicMock()
        fake_output.name = "embeddings"
        fake_session.get_inputs.return_value = [fake_input, fake_input2]
        fake_session.get_outputs.return_value = [fake_output]
        fake_session.run.return_value = [fake_embeddings]

        fake_tokenizer = MagicMock()
        fake_tokenizer.return_value = {
            "input_ids": np.ones((batch, 4), dtype=np.int64),
            "attention_mask": np.ones((batch, 4), dtype=np.int64),
        }

        with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
            import onnxruntime

            onnxruntime.InferenceSession = MagicMock(return_value=fake_session)
            verify_onnx(str(tmp_path / "model.onnx"), fake_tokenizer)

    def test_verify_skips_when_onnxruntime_missing(self, tmp_path, monkeypatch):
        saved_modules = sys.modules.copy()
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_raiser("onnxruntime", monkeypatch),
        )

        fake_tokenizer = MagicMock()
        verify_onnx(str(tmp_path / "model.onnx"), fake_tokenizer)


def _make_import_raiser(blocked_module, monkeypatch):
    """Return an __import__ wrapper that raises ImportError for *blocked_module*."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"mocked: {name} not available")
        return real_import(name, *args, **kwargs)

    return _import


# _parse_args


class TestParseArgs:
    def test_default_args(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["export_onnx", "--model-path", "/tmp/model", "--output-dir", "/tmp/out"],
        )
        args = _parse_args()
        assert args.model_path == "/tmp/model"
        assert args.output_dir == "/tmp/out"
        assert args.pooling == "avg"
        assert args.normalize is True
        assert args.opset == 17
        assert args.dtype == "fp32"
        assert args.no_verify is False
        assert args.tokenizer_path is None

    def test_all_custom_args(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "export_onnx",
                "--model-path",
                "/m",
                "--output-dir",
                "/o",
                "--tokenizer-path",
                "/t",
                "--pooling",
                "cls",
                "--no-normalize",
                "--opset",
                "14",
                "--dtype",
                "fp16",
                "--no-verify",
            ],
        )
        args = _parse_args()
        assert args.tokenizer_path == "/t"
        assert args.pooling == "cls"
        assert args.normalize is False
        assert args.opset == 14
        assert args.dtype == "fp16"
        assert args.no_verify is True


# main


class TestMain:
    @patch("nemo_automodel.components.models.llama_bidirectional.export_onnx.export_to_onnx")
    def test_main_invokes_export(self, mock_export, monkeypatch, capsys):
        mock_export.return_value = "/tmp/out/model.onnx"
        monkeypatch.setattr(
            sys,
            "argv",
            ["export_onnx", "--model-path", "/tmp/model", "--output-dir", "/tmp/out"],
        )

        main()

        mock_export.assert_called_once_with(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            tokenizer_path=None,
            pooling="avg",
            normalize=True,
            opset=17,
            export_dtype="fp32",
            verify=True,
        )
        captured = capsys.readouterr()
        assert "model.onnx" in captured.out
