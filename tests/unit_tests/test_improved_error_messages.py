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

"""Tests for user-friendly error messages when models are unavailable.

Covers three scenarios:
1. Importing a non-existent model submodule (e.g. ``nemo_automodel.components.models.qwen3_1024``)
2. Using ``get_hf_config`` / ``from_pretrained`` with a checkpoint whose model type
   is not recognized by the installed Transformers version.
3. The ``nemo_automodel.models`` → ``nemo_automodel.components.models`` alias.
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# ============================================================================
# 1. Import-time error for missing model submodules
# ============================================================================


class TestModelsImportError:
    """Importing a non-existent model subpackage should raise a helpful error."""

    def test_import_nonexistent_model_raises_module_not_found(self):
        """``import nemo_automodel.components.models.qwen3_1024`` should give a helpful error."""
        with pytest.raises(ModuleNotFoundError, match="has no submodule 'qwen3_1024'"):
            importlib.import_module("nemo_automodel.components.models.qwen3_1024")

    def test_import_error_suggests_upgrade(self):
        with pytest.raises(ModuleNotFoundError, match="pip install --upgrade nemo_automodel"):
            importlib.import_module("nemo_automodel.components.models.qwen3_1024")

    def test_import_error_suggests_install_from_source(self):
        with pytest.raises(
            ModuleNotFoundError, match=r"pip install git\+https://github\.com/NVIDIA-NeMo/Automodel\.git"
        ):
            importlib.import_module("nemo_automodel.components.models.qwen3_1024")

    def test_import_error_lists_available_submodules(self):
        with pytest.raises(ModuleNotFoundError, match="Available model submodules in this installation"):
            importlib.import_module("nemo_automodel.components.models.qwen3_1024")

    def test_import_error_mentions_requested_name(self):
        """The error message should contain the name the user asked for."""
        with pytest.raises(ModuleNotFoundError, match="nonexistent_model_xyz"):
            importlib.import_module("nemo_automodel.components.models.nonexistent_model_xyz")

    def test_existing_submodule_still_imports(self):
        """Sanity: importing a real model submodule should still work."""
        mod = importlib.import_module("nemo_automodel.components.models.common")
        assert mod is not None

    def test_getattr_import_error(self):
        """Direct attribute access on the package should also produce the friendly error."""
        import nemo_automodel.components.models as models_pkg

        with pytest.raises(ModuleNotFoundError, match="has no submodule 'does_not_exist'"):
            _ = models_pkg.does_not_exist

    def test_getattr_suggests_upgrade(self):
        import nemo_automodel.components.models as models_pkg

        with pytest.raises(ModuleNotFoundError, match="pip install --upgrade nemo_automodel"):
            _ = models_pkg.qwen3_1024


# ============================================================================
# 2. nemo_automodel.models → nemo_automodel.components.models alias
# ============================================================================


class TestModelsAlias:
    """``nemo_automodel.models`` should be a transparent alias for
    ``nemo_automodel.components.models``."""

    def test_alias_package_is_same_object(self):
        import nemo_automodel.components.models as real
        import nemo_automodel.models as alias

        assert alias is real

    def test_alias_submodule_import(self):
        """``import nemo_automodel.models.common`` should give the same module."""
        real = importlib.import_module("nemo_automodel.components.models.common")
        alias = importlib.import_module("nemo_automodel.models.common")
        assert alias is real

    def test_alias_submodule_is_same_object_in_sys_modules(self):
        importlib.import_module("nemo_automodel.models.common")
        importlib.import_module("nemo_automodel.components.models.common")
        assert sys.modules["nemo_automodel.models.common"] is sys.modules["nemo_automodel.components.models.common"]

    def test_alias_nonexistent_model_gives_helpful_error(self):
        """The nice error should also fire via the short path."""
        with pytest.raises(ModuleNotFoundError, match="has no submodule 'qwen3_1024'"):
            importlib.import_module("nemo_automodel.models.qwen3_1024")

    def test_alias_nonexistent_model_suggests_upgrade(self):
        with pytest.raises(ModuleNotFoundError, match="pip install --upgrade nemo_automodel"):
            importlib.import_module("nemo_automodel.models.qwen3_1024")

    def test_alias_accessible_from_top_level(self):
        """``nemo_automodel.models`` should be accessible as an attribute."""
        import nemo_automodel

        assert hasattr(nemo_automodel, "models")

    def test_models_in_top_level_dir(self):
        import nemo_automodel

        assert "models" in dir(nemo_automodel)


# ============================================================================
# 3. get_hf_config / from_pretrained / from_config error for unrecognized model types
# ============================================================================


_HF_UNRECOGNIZED_ERROR = ValueError(
    "The checkpoint you are trying to load has model type `qwen3_1024` "
    "but Transformers does not recognize this architecture. This could be "
    "because of an issue with the checkpoint, or because your version of "
    "Transformers is out of date."
)


class TestGetHfConfigUnrecognizedModelType:
    """get_hf_config should wrap HF's ValueError with upgrade instructions."""

    def test_unrecognized_model_type_suggests_upgrade(self):
        from nemo_automodel._transformers.model_init import get_hf_config

        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=_HF_UNRECOGNIZED_ERROR,
        ):
            with pytest.raises(ValueError, match="pip install --upgrade nemo_automodel"):
                get_hf_config("fake/qwen3_1024_model", attn_implementation="eager")

    def test_message_does_not_mention_transformers(self):
        from nemo_automodel._transformers.model_init import get_hf_config

        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=_HF_UNRECOGNIZED_ERROR,
        ):
            with pytest.raises(ValueError) as exc_info:
                get_hf_config("fake/qwen3_1024_model", attn_implementation="eager")
            msg = str(exc_info.value)
            assert "pip install --upgrade nemo_automodel" in msg
            assert "pip install --upgrade nemo_automodel transformers" not in msg

    def test_unrecognized_model_type_suggests_install_from_source(self):
        from nemo_automodel._transformers.model_init import get_hf_config

        hf_error = ValueError(
            "The checkpoint you are trying to load has model type `qwen3_1024` "
            "but Transformers does not recognize this architecture."
        )
        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=hf_error,
        ):
            with pytest.raises(ValueError, match=r"pip install git\+https://github\.com/NVIDIA-NeMo/Automodel\.git"):
                get_hf_config("fake/qwen3_1024_model", attn_implementation="eager")

    def test_unrecognized_model_type_includes_checkpoint_name(self):
        from nemo_automodel._transformers.model_init import get_hf_config

        hf_error = ValueError(
            "The checkpoint you are trying to load has model type `future_model` "
            "but Transformers does not recognize this architecture."
        )
        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=hf_error,
        ):
            with pytest.raises(ValueError, match="org/future_model_v2"):
                get_hf_config("org/future_model_v2", attn_implementation="eager")

    def test_unrecognized_model_type_starts_with_original_error(self):
        """The enhanced message should print the original HF error first."""
        from nemo_automodel._transformers.model_init import get_hf_config

        original_msg = (
            "The checkpoint you are trying to load has model type `qwen3_1024` "
            "but Transformers does not recognize this architecture."
        )
        hf_error = ValueError(original_msg)
        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=hf_error,
        ):
            with pytest.raises(ValueError) as exc_info:
                get_hf_config("fake/qwen3_1024_model", attn_implementation="eager")
            msg = str(exc_info.value)
            assert msg.startswith(original_msg)

    def test_other_value_errors_are_reraised_unchanged(self):
        """ValueErrors that are *not* about unrecognized architecture should pass through."""
        from nemo_automodel._transformers.model_init import get_hf_config

        hf_error = ValueError("some completely different error")
        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=hf_error,
        ):
            with pytest.raises(ValueError, match="some completely different error"):
                get_hf_config("fake/model", attn_implementation="eager")

    def test_non_value_errors_propagate(self):
        """Non-ValueError exceptions should propagate unchanged."""
        from nemo_automodel._transformers.model_init import get_hf_config

        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=OSError("network error"),
        ):
            with pytest.raises(OSError, match="network error"):
                get_hf_config("fake/model", attn_implementation="eager")

    def test_config_kwarg_bypasses_from_pretrained(self):
        """When a config is passed directly, AutoConfig.from_pretrained is not called."""
        from unittest.mock import MagicMock

        from nemo_automodel._transformers.model_init import get_hf_config

        sentinel_config = MagicMock()
        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
        ) as mock_from_pretrained:
            result = get_hf_config("ignored", attn_implementation="eager", config=sentinel_config)
            mock_from_pretrained.assert_not_called()
            assert result is sentinel_config


class TestCheckpointDtypeRestoration:
    def test_checkpoint_tensor_dtypes_come_from_provided_state_dict(self):
        from nemo_automodel.components.checkpoint.utils import _get_checkpoint_tensor_dtypes

        state_dict = {
            "linear.weight": torch.empty(2, 2, dtype=torch.bfloat16),
            "norm.weight": torch.empty(2, dtype=torch.float32),
        }

        result = _get_checkpoint_tensor_dtypes("ignored", object(), {"state_dict": state_dict})

        assert result == {
            "linear.weight": torch.bfloat16,
            "norm.weight": torch.float32,
        }

    def test_restore_loaded_model_dtype_uses_checkpoint_dtype_per_tensor(self):
        from nemo_automodel._transformers.model_init import _restore_loaded_model_dtype

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.norm = torch.nn.LayerNorm(2, elementwise_affine=True)

        model = DummyModel().to(torch.float32)
        with patch(
            "nemo_automodel.components.checkpoint.utils._get_checkpoint_tensor_dtypes",
            return_value={
                "linear.weight": torch.bfloat16,
                "norm.weight": torch.float32,
            },
        ):
            _restore_loaded_model_dtype(
                model,
                "fake/model",
                SimpleNamespace(),
                quantization_config=None,
                load_kwargs={},
            )

        assert model.linear.weight.dtype == torch.bfloat16
        assert model.norm.weight.dtype == torch.float32

    def test_restore_loaded_model_dtype_preserves_tied_weights(self):
        from nemo_automodel._transformers.model_init import _restore_loaded_model_dtype

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(4, 3)
                self.lm_head = torch.nn.Linear(3, 4, bias=False)
                self.lm_head.weight = self.embed_tokens.weight

        model = DummyModel().to(torch.float32)
        with patch(
            "nemo_automodel.components.checkpoint.utils._get_checkpoint_tensor_dtypes",
            return_value={"lm_head.weight": torch.bfloat16},
        ):
            _restore_loaded_model_dtype(
                model,
                "fake/model",
                SimpleNamespace(),
                quantization_config=None,
                load_kwargs={},
            )

        assert model.lm_head.weight is model.embed_tokens.weight
        assert model.lm_head.weight.dtype == torch.bfloat16
        assert model.embed_tokens.weight.dtype == torch.bfloat16


class TestFromConfigUnrecognizedModelType:
    """from_config(string_config) should surface the same helpful message."""

    def test_from_config_string_surfaces_helpful_error(self):
        """from_config('model_id') flows through get_hf_config and gets the enhanced error."""
        from nemo_automodel._transformers.model_init import get_hf_config

        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=_HF_UNRECOGNIZED_ERROR,
        ):
            with pytest.raises(ValueError, match="pip install --upgrade nemo_automodel"):
                get_hf_config("org/some_new_model", attn_implementation="eager")

    def test_from_config_string_does_not_mention_transformers(self):
        from nemo_automodel._transformers.model_init import get_hf_config

        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=_HF_UNRECOGNIZED_ERROR,
        ):
            with pytest.raises(ValueError) as exc_info:
                get_hf_config("org/some_new_model", attn_implementation="eager")
            msg = str(exc_info.value)
            assert "pip install --upgrade nemo_automodel" in msg
            assert "pip install --upgrade nemo_automodel transformers" not in msg

    def test_from_config_string_suggests_install_from_source(self):
        from nemo_automodel._transformers.model_init import get_hf_config

        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=_HF_UNRECOGNIZED_ERROR,
        ):
            with pytest.raises(ValueError, match=r"pip install git\+https://github\.com/NVIDIA-NeMo/Automodel\.git"):
                get_hf_config("org/some_new_model", attn_implementation="eager")

    def test_from_config_string_starts_with_original_error(self):
        """The enhanced message should print the original HF error first."""
        from nemo_automodel._transformers.model_init import get_hf_config

        with patch(
            "nemo_automodel._transformers.model_init.AutoConfig.from_pretrained",
            side_effect=_HF_UNRECOGNIZED_ERROR,
        ):
            with pytest.raises(ValueError) as exc_info:
                get_hf_config("org/some_new_model", attn_implementation="eager")
            msg = str(exc_info.value)
            assert msg.startswith(str(_HF_UNRECOGNIZED_ERROR))
