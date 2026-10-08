# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Attach Transformers adapter metadata used by generic distributed infrastructure."""

from __future__ import annotations

import importlib
from typing import Final

from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

LayerGroupPaths = dict[str, tuple[str, ...]]


_LANGUAGE_MODEL_LAYERS: LayerGroupPaths = {"language": ("model.language_model.layers",)}
_LLAVA_LAYERS: LayerGroupPaths = {
    "language": ("model.language_model.layers", "language_model.model.layers"),
    "vision": (
        "model.vision_tower.vision_model.encoder.layers",
        "model.vision_tower.encoder.layers",
        "vision_tower.vision_model.encoder.layers",
    ),
}
_MISTRAL3_LAYERS: LayerGroupPaths = {
    "language": ("model.language_model.layers",),
    "vision": (
        "model.vision_tower.encoder.layers",
        "model.vision_tower.vision_model.encoder.layers",
        "model.vision_tower.transformer.layers",
    ),
}
_QWEN2_VL_LAYERS: LayerGroupPaths = {
    "language": ("model.language_model.layers", "model.layers"),
    "vision": ("model.visual.blocks", "visual.blocks"),
}

_LAYER_GROUPS: Final[dict[str, LayerGroupPaths]] = {
    "BagelForUnifiedMultimodal": {
        "language": ("model.language_model.model.layers",),
        "vision": ("model.vit_model.vision_model.encoder.layers",),
    },
    "DeepseekV4ForCausalLM": {
        "language": ("model.layers",),
        "vision": ("model.vision.blocks",),
    },
    "Gemma3ForConditionalGeneration": _LLAVA_LAYERS,
    "Gemma4ForConditionalGeneration": {"language": ("model.language_model.layers",)},
    "KimiK25VLForConditionalGeneration": {
        **_LANGUAGE_MODEL_LAYERS,
        "vision": ("model.vision_tower.encoder.blocks",),
    },
    "KimiVLForConditionalGeneration": {
        **_LANGUAGE_MODEL_LAYERS,
        "vision": ("model.vision_tower.encoder.blocks",),
    },
    "Llama4ForConditionalGeneration": {
        "language": ("language_model.model.layers",),
        "vision": ("vision_model.model.layers",),
    },
    "LlamaNemotronVLModel": {
        "language": ("language_model.layers",),
        "vision": ("vision_model.vision_model.encoder.layers", "vision_model.encoder.layers"),
    },
    "LlavaForConditionalGeneration": _LLAVA_LAYERS,
    "LlavaNextForConditionalGeneration": _LLAVA_LAYERS,
    "LlavaNextVideoForConditionalGeneration": _LLAVA_LAYERS,
    "LlavaOnevisionForConditionalGeneration": _LLAVA_LAYERS,
    "MiniMaxM3SparseForConditionalGeneration": {
        "language": ("model.layers",),
        "vision": ("vision_tower.vision_model.encoder.layers",),
    },
    "Ministral3BidirectionalModel": {"language": ("layers",)},
    "Mistral3FP8VLMForConditionalGeneration": _MISTRAL3_LAYERS,
    "Mistral3ForConditionalGeneration": _MISTRAL3_LAYERS,
    "NemotronHForCausalLM": {"language": ("backbone.layers", "model.layers")},
    "GPT2LMHeadModel": {"language": ("transformer.h",)},
    "Qwen2VLForConditionalGeneration": _QWEN2_VL_LAYERS,
    "Qwen2_5_VLForConditionalGeneration": _QWEN2_VL_LAYERS,
    "Qwen3VLMoeForConditionalGeneration": {**_LANGUAGE_MODEL_LAYERS, "vision": ("model.visual.blocks",)},
    "Qwen3_5ForConditionalGeneration": {**_LANGUAGE_MODEL_LAYERS, "vision": ("model.visual.blocks",)},
    "Qwen3_5MoeForConditionalGeneration": {**_LANGUAGE_MODEL_LAYERS, "vision": ("model.visual.blocks",)},
    "SmolVLMForConditionalGeneration": {
        "language": ("model.text_model.layers",),
        "vision": ("model.vision_model.encoder.layers",),
    },
    "Step3p7ForConditionalGeneration": {
        "language": ("model.language_model.layers",),
        "vision": ("model.vision_model.transformer.resblocks",),
    },
}


def configure_parallelization_metadata(model_class: type) -> type:
    """Attach layer paths and explicitly compatible model-owned HF policies.

    Args:
        model_class: AutoModel's wrapper class for the selected implementation.
            Existing model-owned parallelizers take precedence.

    Returns:
        The same class with its parallelization metadata attached. Native model
        implementations are not imported to resolve an HF compatibility policy.
    """
    layer_groups = _LAYER_GROUPS.get(model_class.__name__)
    if layer_groups is not None:
        model_class.parallel_layer_groups = layer_groups
    if getattr(model_class, "parallelizer", None) is None:
        registration = MODEL_ARCH_MAPPING.get(model_class.__name__)
        # Only registrations that explicitly declare HF compatibility may load
        # a native sidecar. Importing other model packages can pull in kernels
        # that force_hf callers deliberately avoid.
        if registration is not None and len(registration) > 2 and "hf_parallelizer" in registration[2]:
            sidecar_module = f"{registration[0].rsplit('.', 1)[0]}.parallelization"
            model_class.parallelizer = importlib.import_module(sidecar_module).PARALLELIZER
    return model_class


__all__ = ["configure_parallelization_metadata"]
