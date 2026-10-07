# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Lazy index of model-owned sidecars; importing this module loads no models.

Qualified names preserve implementation-specific policies. Bare names are
explicit aliases for remote-code classes and metadata-only HF adapters.
Native classes and third-party classes can supply their own ``parallelizer``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.tensor.parallel import ParallelStyle

    from nemo_automodel.components.distributed import ModelParallelizer

MODEL_PARALLELIZERS: dict[str, str] = {
    "BagelForUnifiedMultimodal": "nemo_automodel.components.models.bagel.parallelization.HF_PARALLELIZER",
    "DeciLMForCausalLM": "nemo_automodel.components.models.nemotron_nas.parallelization.PARALLELIZER",
    "DeepseekV4ForCausalLM": "nemo_automodel.components.models.deepseek_v4.parallelization.PARALLELIZER",
    "FalconH1ForCausalLM": "nemo_automodel.components.models.falcon_h1.parallelization.PARALLELIZER",
    "GPT2LMHeadModel": "nemo_automodel.components.models.gpt2_parallelization.PARALLELIZER",
    "Gemma3ForConditionalGeneration": "nemo_automodel.components.models.gemma3.parallelization.LAYOUT_PARALLELIZER",
    "Gemma4ForConditionalGeneration": "nemo_automodel.components.models.gemma4_moe.parallelization.HF_PARALLELIZER",
    "KimiK25VLForConditionalGeneration": "nemo_automodel.components.models.kimi_k25_vl.parallelization.PARALLELIZER",
    "KimiVLForConditionalGeneration": "nemo_automodel.components.models.kimivl.parallelization.PARALLELIZER",
    "Llama4ForConditionalGeneration": "nemo_automodel.components.models.llama4.parallelization.PARALLELIZER",
    "LlamaNemotronVLModel": "nemo_automodel.components.models.llama_nemotron_vl.parallelization.PARALLELIZER",
    "LlavaForConditionalGeneration": "nemo_automodel.components.models.llava.parallelization.PARALLELIZER",
    "LlavaNextForConditionalGeneration": "nemo_automodel.components.models.llava.parallelization.PARALLELIZER",
    "LlavaNextVideoForConditionalGeneration": "nemo_automodel.components.models.llava.parallelization.PARALLELIZER",
    "LlavaOnevisionForConditionalGeneration": "nemo_automodel.components.models.llava.parallelization.PARALLELIZER",
    "MiniMaxM3SparseForConditionalGeneration": "nemo_automodel.components.models.minimax_m3_vl.parallelization.PARALLELIZER",
    "Ministral3BidirectionalModel": "nemo_automodel.components.models.ministral_bidirectional.parallelization.PARALLELIZER",
    "Mistral3FP8VLMForConditionalGeneration": "nemo_automodel.components.models.mistral3_vlm.parallelization.LAYOUT_PARALLELIZER",
    "Mistral3ForConditionalGeneration": "nemo_automodel.components.models.mistral3_vlm.parallelization.LAYOUT_PARALLELIZER",
    "NemotronFlashForCausalLM": "nemo_automodel.components.models.nemotron_flash.parallelization.PARALLELIZER",
    "NemotronHForCausalLM": "nemo_automodel.components.models.nemotron_v3.parallelization.PARALLELIZER",
    "NemotronLabsDiffusionModel": "nemo_automodel.components.models.nemotron_labs_diffusion.parallelization.PARALLELIZER",
    "Qwen2VLForConditionalGeneration": "nemo_automodel.components.models.qwen2_vl.parallelization.PARALLELIZER",
    "Qwen2_5_VLForConditionalGeneration": "nemo_automodel.components.models.qwen2_vl.parallelization.PARALLELIZER",
    "Qwen3VLMoeForConditionalGeneration": "nemo_automodel.components.models.qwen3_vl_moe.parallelization.PARALLELIZER",
    "Qwen3_5ForCausalLM": "nemo_automodel.components.models.qwen3_5.parallelization.PARALLELIZER",
    "Qwen3_5ForConditionalGeneration": "nemo_automodel.components.models.qwen3_5.parallelization.VLM_PARALLELIZER",
    "Qwen3_5MoeForConditionalGeneration": "nemo_automodel.components.models.qwen3_5_moe.parallelization.PARALLELIZER",
    "SmolVLMForConditionalGeneration": "nemo_automodel.components.models.smolvlm.parallelization.PARALLELIZER",
    "Step3p7ForConditionalGeneration": "nemo_automodel.components.models.step3p7.parallelization.PARALLELIZER",
    "nemo_automodel.components.models.baichuan.model.BaichuanForCausalLM": "nemo_automodel.components.models.baichuan.parallelization.PARALLELIZER",
    "nemo_automodel.components.models.llama.model.LlamaForCausalLM": "nemo_automodel.components.models.llama.parallelization.PARALLELIZER",
    "nemo_automodel.components.models.mistral3.model.Ministral3ForCausalLM": "nemo_automodel.components.models.mistral3.parallelization.PARALLELIZER",
    "nemo_automodel.components.models.mistral3_vlm.model.Mistral3FP8VLMForConditionalGeneration": "nemo_automodel.components.models.mistral3_vlm.parallelization.PARALLELIZER",
    "nemo_automodel.components.models.muse_glimmer.model.MuseGlimmerForConditionalGeneration": "nemo_automodel.components.models.muse_glimmer.parallelization.PARALLELIZER",
    "nemo_automodel.components.models.qwen2.model.Qwen2ForCausalLM": "nemo_automodel.components.models.qwen2.parallelization.PARALLELIZER",
    "nemo_automodel.components.models.qwen3.model.Qwen3ForCausalLM": "nemo_automodel.components.models.qwen3.parallelization.CAUSAL_PARALLELIZER",
    "nemo_automodel.components.models.qwen3_5.model.Qwen3_5ForCausalLM": "nemo_automodel.components.models.qwen3_5.parallelization.PARALLELIZER",
    "nemo_automodel.components.models.qwen3_5.model.Qwen3_5ForConditionalGeneration": "nemo_automodel.components.models.qwen3_5.parallelization.VLM_PARALLELIZER",
    "transformers.models.falcon_h1.modeling_falcon_h1.FalconH1ForCausalLM": "nemo_automodel.components.models.falcon_h1.parallelization.PARALLELIZER",
    "transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM": "nemo_automodel.components.models.gemma3.parallelization.PARALLELIZER",
    "transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration": "nemo_automodel.components.models.gemma3.parallelization.VLM_PARALLELIZER",
    "transformers.models.llama.modeling_llama.LlamaForCausalLM": "nemo_automodel.components.models.llama.parallelization.PARALLELIZER",
    "transformers.models.mistral3.modeling_mistral3.Mistral3ForConditionalGeneration": "nemo_automodel.components.models.mistral3_vlm.parallelization.PARALLELIZER",
    "transformers.models.phi.modeling_phi.PhiForCausalLM": "nemo_automodel.components.models.phi.parallelization.PARALLELIZER",
    "transformers.models.phi3.modeling_phi3.Phi3ForCausalLM": "nemo_automodel.components.models.phi3.parallelization.PARALLELIZER",
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM": "nemo_automodel.components.models.qwen2.parallelization.PARALLELIZER",
    "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM": "nemo_automodel.components.models.qwen3.parallelization.CAUSAL_PARALLELIZER",
    "transformers.models.qwen3.modeling_qwen3.Qwen3ForSequenceClassification": "nemo_automodel.components.models.qwen3.parallelization.PARALLELIZER",
    "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration": "nemo_automodel.components.models.qwen3_5.parallelization.VLM_PARALLELIZER",
}

NAMED_TP_PLANS = {
    "llama_nemotron_super_tp_plan": "nemo_automodel.components.models.nemotron_nas.parallelization.get_legacy_named_tp_plan",
}


@lru_cache(maxsize=None)
def _load_reference(reference: str):
    module, name = reference.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def resolve_model_parallelizer(model_class: type) -> ModelParallelizer | None:
    """Resolve an explicit class sidecar or a lazily indexed compatibility policy."""
    owned = getattr(model_class, "parallelizer", None)
    if owned is not None:
        return owned
    qualified_name = _get_class_qualname(model_class)
    reference = MODEL_PARALLELIZERS.get(qualified_name) or MODEL_PARALLELIZERS.get(model_class.__name__)
    return _load_reference(reference) if reference is not None else None


def resolve_named_tp_plan(name: str) -> Callable[[nn.Module, bool], dict[str, ParallelStyle]] | None:
    """Resolve a backwards-compatible plan name without loading model implementations."""
    reference = NAMED_TP_PLANS.get(name)
    return _load_reference(reference) if reference is not None else None


def _get_class_qualname(model_class: type) -> str:
    """Use stable names for HF mixin wrappers that preserve class metadata."""
    return f"{model_class.__module__}.{model_class.__qualname__}"
