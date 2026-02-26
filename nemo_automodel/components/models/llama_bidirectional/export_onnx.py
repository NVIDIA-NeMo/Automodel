#!/usr/bin/env python3
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

"""
Export a HuggingFace biencoder / embedding checkpoint to ONNX.

The resulting ONNX graph maps:
    (input_ids, attention_mask) -> embeddings   [batch, hidden_dim]

The export wraps the bare transformer with average-pooling and L2
normalisation so that the ONNX model produces ready-to-use embeddings.

Usage (standalone):
    python -m nemo_automodel.components.models.biencoder.export_onnx \
        --model-path /path/to/hf_checkpoint \
        --output-dir /path/to/onnx_output \
        [--pooling avg] [--normalize] [--opset 17] [--dtype fp32]

Usage (from Python):
    from nemo_automodel.components.models.llama_bidirectional.export_onnx import export_to_onnx
    onnx_path = export_to_onnx("/path/to/hf_checkpoint", "/path/to/onnx_output")
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin nn.Module wrappers used to build a single trace-able graph for ONNX
# ---------------------------------------------------------------------------


class _Pooling(nn.Module):
    """Pooling layer that reduces [batch, seq, hidden] -> [batch, hidden]."""

    def __init__(self, pool_type: str = "avg"):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.pool_type == "avg":
            eps = 1e-9
            return last_hidden.sum(dim=1) / (attention_mask.sum(dim=1)[..., None] + eps)
        elif self.pool_type == "cls":
            return last_hidden[:, 0]
        elif self.pool_type == "last":
            return last_hidden[:, -1]
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")


class EmbeddingModelForExport(nn.Module):
    """Wraps a base transformer with pooling + optional L2 normalisation.

    The ``forward`` signature is ``(input_ids, attention_mask) -> embeddings``
    which is the contract expected by downstream ONNX / TensorRT consumers.
    """

    def __init__(self, base_model: nn.Module, pooling: _Pooling, normalize: bool = True):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.normalize = normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["last_hidden_state"].to(torch.float32)
        embeddings = self.pooling(hidden_states, attention_mask)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


# ---------------------------------------------------------------------------
# Core export function
# ---------------------------------------------------------------------------


def export_to_onnx(
    model_path: str,
    output_dir: str,
    *,
    tokenizer_path: str | None = None,
    pooling: str = "avg",
    normalize: bool = True,
    opset: int = 17,
    export_dtype: str = "fp32",
    verify: bool = True,
) -> str:
    """Export a HuggingFace embedding model to ONNX.

    Args:
        model_path:      Path to the HuggingFace model directory (must contain
                         ``config.json`` and weight files).
        output_dir:      Directory where ``model.onnx`` and ``tokenizer/`` will
                         be written.
        tokenizer_path:  Path to load the tokenizer from.  Defaults to
                         *model_path* when not specified.  Useful when the
                         checkpoint directory does not contain tokenizer files.
        pooling:         Pooling strategy applied on top of transformer hidden
                         states.  One of ``"avg"``, ``"cls"``, ``"last"``.
        normalize:       If *True*, L2-normalise the pooled embeddings.
        opset:           ONNX opset version (default 17).
        export_dtype:    Export precision — ``"fp32"`` or ``"fp16"``.
        verify:          Run a quick onnxruntime round-trip after export.

    Returns:
        Absolute path to the exported ``model.onnx``.
    """
    model_path = str(Path(model_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    if tokenizer_path is None:
        tokenizer_path = model_path
    else:
        tokenizer_path = str(Path(tokenizer_path).resolve())

    # ------------------------------------------------------------------
    # 1. Load tokenizer + base transformer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    logger.info("Loading model from %s", model_path)
    base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()

    # ------------------------------------------------------------------
    # 2. Build the export wrapper (transformer + pool + normalise)
    # ------------------------------------------------------------------
    pooling_module = _Pooling(pool_type=pooling)
    export_model = EmbeddingModelForExport(base_model, pooling_module, normalize=normalize)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16}
    torch_dtype = dtype_map.get(export_dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported export_dtype={export_dtype!r}.  Choose fp32 or fp16.")
    export_model = export_model.to(torch_dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_model = export_model.to(device)

    # ------------------------------------------------------------------
    # 3. Build dummy inputs for the tracer
    # ------------------------------------------------------------------
    dummy_texts = ["hello world", "example sentence for tracing"]
    dummy = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)
    dummy_inputs = {
        "input_ids": dummy["input_ids"].to(device),
        "attention_mask": dummy["attention_mask"].to(device),
    }

    # ------------------------------------------------------------------
    # 4. Export
    # ------------------------------------------------------------------
    onnx_path = os.path.join(output_dir, "model.onnx")

    input_names = ["input_ids", "attention_mask"]
    output_names = ["embeddings"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "embeddings": {0: "batch_size", 1: "embedding_dim"},
    }

    logger.info("Exporting ONNX (opset=%d, dtype=%s) -> %s", opset, export_dtype, onnx_path)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch_dtype):
        torch.onnx.export(
            model=export_model,
            args=(dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            f=onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )
    logger.info("ONNX model saved to %s", onnx_path)

    # ------------------------------------------------------------------
    # 5. Save tokenizer alongside the ONNX model
    # ------------------------------------------------------------------
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    logger.info("Tokenizer saved to %s", tokenizer_dir)

    # ------------------------------------------------------------------
    # 6. Optional verification
    # ------------------------------------------------------------------
    if verify:
        verify_onnx(onnx_path, tokenizer)

    return onnx_path


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------


def verify_onnx(onnx_path: str, tokenizer) -> None:
    """Run a quick onnxruntime sanity check on the exported model."""
    try:
        import onnxruntime
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX verification.")
        return

    logger.info("Verifying ONNX model with onnxruntime ...")
    session = onnxruntime.InferenceSession(onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    logger.info("  inputs:  %s", input_names)
    logger.info("  outputs: %s", output_names)

    test_sentences = ["This is a test sentence.", "Another one for verification."]
    tokenized = tokenizer(test_sentences, return_tensors="np", padding=True, truncation=True)
    feed = {name: tokenized[name] for name in input_names if name in tokenized}

    outputs = session.run(output_names, feed)
    embeddings = outputs[0]
    logger.info("  output shape: %s", embeddings.shape)
    logger.info("  first embedding ([:8]): %s", embeddings[0][:8])

    # Basic sanity: shape should be [batch, hidden_dim]
    assert embeddings.ndim == 2, f"Expected 2-D output, got shape {embeddings.shape}"
    assert embeddings.shape[0] == len(test_sentences), (
        f"Batch dimension mismatch: expected {len(test_sentences)}, got {embeddings.shape[0]}"
    )
    logger.info("ONNX verification passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a HuggingFace biencoder/embedding model to ONNX.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for ONNX model and tokenizer output.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to load the tokenizer from (defaults to --model-path).",
    )
    parser.add_argument(
        "--pooling", type=str, default="avg", choices=["avg", "cls", "last"], help="Pooling strategy (default: avg)."
    )
    parser.add_argument(
        "--normalize", action="store_true", default=True, help="L2-normalise embeddings (default: True)."
    )
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="Disable L2 normalisation.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17).")
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Export precision (default: fp32)."
    )
    parser.add_argument(
        "--no-verify", action="store_true", default=False, help="Skip onnxruntime verification after export."
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args()
    onnx_path = export_to_onnx(
        model_path=args.model_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        pooling=args.pooling,
        normalize=args.normalize,
        opset=args.opset,
        export_dtype=args.dtype,
        verify=not args.no_verify,
    )
    print(f"Exported ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
