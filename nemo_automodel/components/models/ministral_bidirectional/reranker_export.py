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

"""Portable checkpoint export for the Mistral3 pooled reranker."""

import json
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
from transformers import PixtralProcessor

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .model import Mistral3VLBidirectionalForSequenceClassification


class Mistral3RerankerStateDictAdapter(StateDictAdapter):
    """Preserve backbone weights and expose the head at the native vLLM classifier path."""

    def to_hf(self, state_dict: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Rename the native scoring head without changing tensor values.

        Args:
            state_dict: Native parameters in their original layouts; score.weight has shape [labels, hidden].
            **kwargs: Unused adapter options.

        Returns:
            HF names mapped to the same tensors, with the head at language_model.score.weight.
        """
        return {"language_model.score.weight" if k == "score.weight" else k: v for k, v in state_dict.items()}

    def from_hf(
        self, hf_state_dict: dict[str, torch.Tensor], device_mesh: "DeviceMesh | None" = None, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Restore the native head name, accepting legacy score.weight checkpoints too.

        Args:
            hf_state_dict: HF tensors in native layouts; the head has shape [labels, hidden].
            device_mesh: Unused mesh; this adapter preserves placements and does not gather.
            **kwargs: Unused adapter options.

        Returns:
            Native names mapped to the same tensors, preserving device, dtype and distributed placements.
        """
        return {"score.weight" if k == "language_model.score.weight" else k: v for k, v in hf_state_dict.items()}

    def forced_hf_dtype_mapping(self, state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
        """Keep the trained head's dtype so its exported Dense copy describes the same weights.

        Args:
            state_dict: HF tensors, including language_model.score.weight of shape [labels, hidden].

        Returns:
            The scoring head's current dtype, overriding a potentially lower-precision source checkpoint.
        """
        weight = state_dict.get("language_model.score.weight")
        if weight is None:
            return {}
        dtype_names = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.float64: "F64"}
        return {"language_model.score.weight": dtype_names[weight.dtype]}

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: torch.Tensor, **kwargs: Any
    ) -> list[tuple[str, torch.Tensor]]:
        """Rename one parameter without gathering or copying it.

        Args:
            fqn: Native parameter name.
            tensor: Parameter in its native layout; the scoring head has shape [labels, hidden].
            **kwargs: Unused adapter options.

        Returns:
            One HF name and the original tensor with unchanged shape, dtype, device and placement.
        """
        return [("language_model.score.weight" if fqn == "score.weight" else fqn, tensor)]


class Mistral3RerankerMetadataExporter:
    """Snapshot the trained head on every rank, then write portable metadata on the writer rank."""

    def __init__(self, model: "Mistral3VLBidirectionalForSequenceClassification") -> None:
        self.model = model
        self.dense_weight: torch.Tensor | None = None
        self.processor: PixtralProcessor | None = None

    def validate(self, *, tokenizer: object, original_model_path: str | None) -> None:
        """Validate the export and collectively snapshot the small trained head before rank-zero I/O."""
        if self.model.config.pooling != "avg":
            raise ValueError("Portable Mistral3 reranker export requires mean pooling (pooling=avg).")
        temperature = self.model.effective_score_temperature
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Reranker score temperature must be finite and positive.")
        if not isinstance(tokenizer, PixtralProcessor) or not tokenizer.chat_template:
            raise ValueError("Mistral3 reranker export requires a Pixtral processor with a chat template.")
        if type(tokenizer) is PixtralProcessor:
            self.processor = tokenizer
        else:
            get_processor = getattr(tokenizer, "get_hf_export_processor", None)
            if not callable(get_processor):
                raise TypeError("Custom reranker processors must provide a verified stock get_hf_export_processor().")
            self.processor = get_processor()
        # full_tensor() is collective. validate() runs on every rank; save() does not.
        weight = self.model.score.weight.detach()
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        self.dense_weight = (weight.to(device="cpu", dtype=torch.float32) / temperature).contiguous()

    def save_model_assets(self, directory: str | Path) -> None:
        """Write stock configs and the small standalone Transformers scoring adapter."""
        if self.model.config.pooling != "avg":
            raise ValueError("Portable Mistral3 reranker export requires mean pooling (pooling=avg).")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = self.model.model.get_hf_export_config()
        config.architectures = ["Mistral3ForSequenceClassification"]
        config.text_config.architectures = ["Ministral3ForSequenceClassification"]
        config.auto_map = {"AutoModelForSequenceClassification": "model.Mistral3ForSequenceClassification"}
        config.score_temperature = self.model.effective_score_temperature
        # Generation temperature must never substitute for the scoring divisor.
        if hasattr(config, "temperature"):
            del config.temperature
        config.is_causal = config.text_config.is_causal
        config.save_pretrained(directory)
        shutil.copyfile(Path(__file__).with_name("reranker_model.py"), directory / "model.py")

    def save(
        self, *, hf_metadata_dir: str, tokenizer: object, original_model_path: str | None, v4_compatible: bool
    ) -> None:
        """Save a Transformer/Pooling/Dense chain using the snapshot from this checkpoint step."""
        if self.dense_weight is None or self.processor is None:
            raise RuntimeError("Reranker exporter must be validated on every rank before saving.")
        directory = Path(hf_metadata_dir)
        self.save_model_assets(directory)
        self.processor.save_pretrained(directory)
        forward = {"method": "forward", "method_output_name": "last_hidden_state"}
        configs = {
            "modules.json": [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.base.modules.transformer.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
                },
                {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.base.modules.dense.Dense"},
            ],
            "config_sentence_transformers.json": {
                "model_type": "CrossEncoder",
                "activation_fn": "torch.nn.modules.linear.Identity",
                "prompts": {},
                "default_prompt_name": None,
            },
            "sentence_bert_config.json": {
                "transformer_task": "feature-extraction",
                "module_output_name": "token_embeddings",
                "modality_config": {"text": forward, "message": {**forward, "format": "structured"}},
                "unpad_inputs": False,
                "model_kwargs": {"dtype": "auto", "attn_implementation": "sdpa"},
            },
            "1_Pooling/config.json": {
                "embedding_dimension": self.model.config.text_config.hidden_size,
                "pooling_mode": "mean",
                "include_prompt": True,
            },
            "2_Dense/config.json": {
                "in_features": self.model.config.text_config.hidden_size,
                "out_features": self.model.num_labels,
                "bias": False,
                "activation_function": "torch.nn.modules.linear.Identity",
                "module_input_name": "sentence_embedding",
                "module_output_name": "scores",
            },
        }
        for name, contents in configs.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(contents, indent=2) + "\n")
        save_file(
            {"linear.weight": self.dense_weight}, directory / "2_Dense/model.safetensors", metadata={"format": "pt"}
        )
        # Preserve notices from a local source snapshot without carrying stale modules or head weights.
        if original_model_path and Path(original_model_path).is_dir():
            for path in Path(original_model_path).iterdir():
                if path.is_file() and path.name.lower().startswith(("license", "notice")):
                    destination = directory / path.name
                    if path.resolve() != destination.resolve():
                        shutil.copyfile(path, destination)
