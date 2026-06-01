# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the Gemma4 drafter wrapper class and its registry entry."""

import importlib

import pytest
import torch

from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin


def _gemma4_assistant_available() -> bool:
    """Return True when ``transformers.models.gemma4_assistant`` is importable."""
    try:
        importlib.import_module("transformers.models.gemma4_assistant")
        return True
    except (ModuleNotFoundError, ImportError):
        return False


_HAS_GEMMA4_ASSISTANT = _gemma4_assistant_available()
_SKIP_REASON = (
    "transformers.models.gemma4_assistant not available "
    "(requires transformers>=5.8.0.dev with the gemma4_assistant module)."
)


class TestRegistryEntry:
    """The drafter must be reachable via NeMo's MODEL_ARCH_MAPPING under the
    canonical HF architecture name ``Gemma4AssistantForCausalLM``."""

    def test_arch_name_present(self):
        assert "Gemma4AssistantForCausalLM" in MODEL_ARCH_MAPPING

    def test_arch_points_to_drafter_module(self):
        module_path, class_name = MODEL_ARCH_MAPPING["Gemma4AssistantForCausalLM"][:2]
        assert module_path == "nemo_automodel.components.models.gemma4_drafter.model"
        assert class_name == "Gemma4DrafterForCausalLM"


class TestDrafterModuleImports:
    """The drafter wrapper module must always import cleanly even when the
    underlying transformers symbol is missing — that is the entire point of
    routing the import through ``safe_import``-style guards."""

    def test_module_imports_regardless_of_transformers_version(self):
        # Should not raise even on transformers 5.0 (where gemma4_assistant
        # is missing). The placeholder path raises only at use time.
        from nemo_automodel.components.models.gemma4_drafter import model as drafter_module

        assert hasattr(drafter_module, "Gemma4DrafterForCausalLM")

    def test_init_reexports_class(self):
        from nemo_automodel.components.models.gemma4_drafter import (
            Gemma4DrafterForCausalLM,
            Gemma4JointOutput,
            Gemma4WithDrafter,
            ModelClass,
        )

        assert Gemma4DrafterForCausalLM is ModelClass
        assert Gemma4JointOutput is not None
        assert Gemma4WithDrafter is not None


@pytest.mark.skipif(not _HAS_GEMMA4_ASSISTANT, reason=_SKIP_REASON)
class TestDrafterClassStructure:
    """When transformers TOT is available, the wrapper inherits both
    HFCheckpointingMixin and the HF drafter and is constructible from a tiny
    config without going to the HF Hub."""

    def test_wrapper_is_hf_checkpointing_mixin(self):
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
            Gemma4AssistantForCausalLM,
        )

        from nemo_automodel.components.models.gemma4_drafter.model import (
            Gemma4DrafterForCausalLM,
        )

        assert issubclass(Gemma4DrafterForCausalLM, HFCheckpointingMixin)
        assert issubclass(Gemma4DrafterForCausalLM, Gemma4AssistantForCausalLM)

    def test_construct_from_tiny_config(self):
        """Build a tiny drafter on CPU and verify the expected submodules."""
        from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
            Gemma4AssistantConfig,
        )

        from nemo_automodel.components.models.gemma4_drafter.model import (
            Gemma4DrafterForCausalLM,
        )

        text_cfg = _make_tiny_drafter_text_config()
        cfg = Gemma4AssistantConfig(text_config=text_cfg, backbone_hidden_size=text_cfg.hidden_size)

        torch.manual_seed(0)
        model = Gemma4DrafterForCausalLM(cfg)

        # Expected submodules are the four pillars of the released drafter.
        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")
        assert hasattr(model, "pre_projection")
        assert hasattr(model, "post_projection")
        # masked_embedding is created only when use_ordered_embeddings=True;
        # default is False so it must be None.
        assert getattr(model, "masked_embedding", "missing") is None

        # Pre/post projection geometry.
        H_b = cfg.backbone_hidden_size
        H = text_cfg.hidden_size
        assert tuple(model.pre_projection.weight.shape) == (H, 2 * H_b)
        assert tuple(model.post_projection.weight.shape) == (H_b, H)
        assert model.pre_projection.bias is None
        assert model.post_projection.bias is None

    def test_lm_head_tied_to_embed_tokens(self):
        """The drafter's ``lm_head.weight`` is tied to ``model.embed_tokens.weight``
        per the released config (``tie_word_embeddings=True`` is the default)."""
        from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
            Gemma4AssistantConfig,
        )

        from nemo_automodel.components.models.gemma4_drafter.model import (
            Gemma4DrafterForCausalLM,
        )

        text_cfg = _make_tiny_drafter_text_config()
        cfg = Gemma4AssistantConfig(text_config=text_cfg, backbone_hidden_size=text_cfg.hidden_size)
        model = Gemma4DrafterForCausalLM(cfg)

        # tie_word_embeddings defaults to True in Gemma4AssistantConfig.
        assert cfg.tie_word_embeddings is True
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    def test_use_ordered_embeddings_creates_masked_embedder(self):
        from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
            Gemma4AssistantConfig,
        )

        from nemo_automodel.components.models.gemma4_drafter.model import (
            Gemma4DrafterForCausalLM,
        )

        text_cfg = _make_tiny_drafter_text_config()
        cfg = Gemma4AssistantConfig(
            text_config=text_cfg,
            backbone_hidden_size=text_cfg.hidden_size,
            use_ordered_embeddings=True,
            num_centroids=4,
            centroid_intermediate_top_k=2,
        )
        model = Gemma4DrafterForCausalLM(cfg)

        assert model.masked_embedding is not None
        # token_ordering must be a registered buffer of shape [vocab_size].
        assert model.masked_embedding.token_ordering.shape == (text_cfg.vocab_size,)
        assert model.masked_embedding.token_ordering.dtype == torch.long


def _make_tiny_drafter_text_config():
    """Build a tiny Gemma4 text config that satisfies Gemma4AssistantConfig.validate_architecture.

    Drafter constraints:
        - hidden_size_per_layer_input == 0
        - vocab_size_per_layer_input == 0
        - enable_moe_block is False
        - use_double_wide_mlp is False
        - num_kv_shared_layers == num_hidden_layers (set automatically by
          Gemma4AssistantConfig.__post_init__ if left at 0)
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=16,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        torch_dtype="float32",
    )
