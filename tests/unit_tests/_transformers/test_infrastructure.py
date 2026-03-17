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

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from nemo_automodel._transformers.utils import _should_load_before_shard


class TestShouldLoadBeforeShard:
    """Tests for _should_load_before_shard.

    load_before_shard should be True only when ALL of these hold:
      - no pipeline parallelism (autopipeline is None)
      - no tensor parallelism (tp_size <= 1)
      - no expert parallelism (ep_size <= 1)
      - checkpoint needs loading (pretrained_model_name_or_path and load_base_model)
      - no PEFT (peft_config is None)
    """

    # Defaults that satisfy all conditions (single-GPU checkpoint load, no PEFT).
    _DEFAULTS = dict(
        autopipeline=None,
        tp_size=1,
        ep_size=1,
        pretrained_model_name_or_path="/some/path",
        load_base_model=True,
        peft_config=None,
    )

    def test_single_gpu_loads_before_shard(self):
        """With no parallelism and a valid checkpoint path, should load before shard."""
        assert _should_load_before_shard(**self._DEFAULTS) is True

    def test_ep_greater_than_1_skips_load_before_shard(self):
        """With EP > 1, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "ep_size": 2}) is False

    def test_tp_greater_than_1_skips_load_before_shard(self):
        """With TP > 1, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "tp_size": 2}) is False

    def test_pp_skips_load_before_shard(self):
        """With pipeline parallelism, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "autopipeline": object()}) is False

    def test_peft_skips_load_before_shard(self):
        """With PEFT config, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "peft_config": object()}) is False

    def test_no_pretrained_path_skips_load_before_shard(self):
        """Without a pretrained path, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "pretrained_model_name_or_path": ""}) is False

    def test_load_base_model_false_skips_load_before_shard(self):
        """With load_base_model=False, should NOT load before shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "load_base_model": False}) is False

    @pytest.mark.parametrize(
        "tp_size,ep_size",
        [
            (2, 2),
            (4, 1),
            (1, 4),
            (2, 4),
        ],
        ids=["tp2_ep2", "tp4_ep1", "tp1_ep4", "tp2_ep4"],
    )
    def test_any_parallelism_skips_load_before_shard(self, tp_size, ep_size):
        """Any TP or EP > 1 should skip load-before-shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "tp_size": tp_size, "ep_size": ep_size}) is False

    def test_all_conditions_false(self):
        """When every condition blocks, result is still False."""
        assert (
            _should_load_before_shard(
                tp_size=2,
                ep_size=4,
                autopipeline=object(),
                pretrained_model_name_or_path="",
                load_base_model=False,
                peft_config=object(),
            )
            is False
        )

    def test_ep_size_exactly_1_allows_load(self):
        """ep_size=1 should not block load-before-shard."""
        assert _should_load_before_shard(**{**self._DEFAULTS, "ep_size": 1}) is True


# =============================================================================
# Tests for apply_model_infrastructure: post-shard initialize_model_weights
# =============================================================================


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace()


_INFRA_MODULE = "nemo_automodel._transformers.infrastructure"


def _run_apply_model_infrastructure(*, is_meta_device, load_base_model, model_wrapper=None):
    """Helper that invokes apply_model_infrastructure with heavy dependencies stubbed out."""
    from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

    model = _DummyModel()

    with (
        patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
        patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
        patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
        patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
        patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
    ):
        mock_ckpt = MockCheckpointer.return_value
        mock_ckpt.config = MagicMock()
        mock_ckpt.config.dequantize_base_checkpoint = False

        result = apply_model_infrastructure(
            model=model,
            is_meta_device=is_meta_device,
            device=torch.device("cpu"),
            load_base_model=load_base_model,
            model_wrapper=model_wrapper,
            pretrained_model_name_or_path="test/model" if load_base_model else "",
        )

        return result, mock_ckpt


class TestApplyModelInfrastructurePostShardInit:
    """Tests for initialize_model_weights being called in apply_model_infrastructure."""

    def test_from_config_meta_calls_initialize_model_weights(self):
        """from_config path (load_base_model=False) on meta device should call initialize_model_weights."""
        _, mock_ckpt = _run_apply_model_infrastructure(is_meta_device=True, load_base_model=False)

        mock_ckpt.initialize_model_weights.assert_called_once()

    def test_from_config_meta_does_not_call_load_base_model(self):
        """from_config path should NOT call load_base_model (no checkpoint to load)."""
        _, mock_ckpt = _run_apply_model_infrastructure(is_meta_device=True, load_base_model=False)

        mock_ckpt.load_base_model.assert_not_called()

    def test_from_pretrained_meta_calls_both(self):
        """from_pretrained path on meta device should call both initialize_model_weights and load_base_model."""
        _, mock_ckpt = _run_apply_model_infrastructure(is_meta_device=True, load_base_model=True)

        mock_ckpt.initialize_model_weights.assert_called_once()
        mock_ckpt.load_base_model.assert_called_once()

    def test_non_meta_skips_initialize_model_weights(self):
        """Non-meta device model should not call initialize_model_weights."""
        _, mock_ckpt = _run_apply_model_infrastructure(is_meta_device=False, load_base_model=False)

        mock_ckpt.initialize_model_weights.assert_not_called()

    def test_megatron_fsdp_skips_post_shard_init(self):
        """MegatronFSDPManager wrapper should skip post-shard initialize_model_weights."""
        mock_wrapper = MagicMock(spec=["parallelize", "moe_mesh"])
        mock_wrapper.moe_mesh = None
        # Make isinstance check work for MegatronFSDPManager
        from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager

        mock_wrapper.__class__ = MegatronFSDPManager

        _, mock_ckpt = _run_apply_model_infrastructure(
            is_meta_device=True, load_base_model=False, model_wrapper=mock_wrapper
        )

        mock_ckpt.initialize_model_weights.assert_not_called()

    def test_peft_init_method_forwarded_to_initialize_model_weights(self):
        """peft_config.lora_A_init should be forwarded as peft_init_method."""
        from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

        model = _DummyModel()
        peft_config = SimpleNamespace(lora_A_init="xavier", use_triton=False)

        with (
            patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
            patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
            patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
            patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
            patch(f"{_INFRA_MODULE}._apply_peft_and_lower_precision", return_value=model),
            patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
        ):
            mock_ckpt = MockCheckpointer.return_value
            mock_ckpt.config = MagicMock()
            mock_ckpt.config.dequantize_base_checkpoint = False

            apply_model_infrastructure(
                model=model,
                is_meta_device=True,
                device=torch.device("cpu"),
                load_base_model=False,
                peft_config=peft_config,
                pretrained_model_name_or_path="",
            )

            mock_ckpt.initialize_model_weights.assert_called_once_with(
                model, torch.device("cpu"), peft_init_method="xavier"
            )


# =============================================================================
# Tests for load_before_shard path in apply_model_infrastructure
# =============================================================================


def _run_apply_model_infrastructure_load_before_shard(*, peft_config=None):
    """Helper that invokes apply_model_infrastructure with load_before_shard=True."""
    from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

    model = _DummyModel()

    with (
        patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
        patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
        patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
        patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=True),
        patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
    ):
        mock_ckpt = MockCheckpointer.return_value
        mock_ckpt.config = MagicMock()
        mock_ckpt.config.dequantize_base_checkpoint = False

        result = apply_model_infrastructure(
            model=model,
            is_meta_device=True,
            device=torch.device("cpu"),
            load_base_model=True,
            peft_config=peft_config,
            pretrained_model_name_or_path="test/model",
            cache_dir="/tmp/cache",
        )

        return result, mock_ckpt, model


class TestLoadBeforeShardPath:
    """Tests for the load_before_shard code path in apply_model_infrastructure."""

    def test_load_before_shard_calls_initialize_then_load(self):
        """load_before_shard should call initialize_model_weights before load_base_model."""
        _, mock_ckpt, model = _run_apply_model_infrastructure_load_before_shard()

        mock_ckpt.initialize_model_weights.assert_called_once_with(
            model, torch.device("cpu"), peft_init_method=None
        )
        mock_ckpt.load_base_model.assert_called_once_with(
            model, torch.device("cpu"), "/tmp/cache", "test/model", load_base_model=True
        )

        init_idx = mock_ckpt.method_calls.index(
            call.initialize_model_weights(model, torch.device("cpu"), peft_init_method=None)
        )
        load_idx = mock_ckpt.method_calls.index(
            call.load_base_model(model, torch.device("cpu"), "/tmp/cache", "test/model", load_base_model=True)
        )
        assert init_idx < load_idx

    def test_load_before_shard_skips_post_shard_init(self):
        """When load_before_shard is True, post-shard initialize_model_weights should not run again."""
        _, mock_ckpt, _ = _run_apply_model_infrastructure_load_before_shard()

        assert mock_ckpt.initialize_model_weights.call_count == 1

    def test_load_before_shard_does_not_call_load_base_model_with_peft_init_method(self):
        """load_base_model should NOT receive peft_init_method (moved to initialize_model_weights)."""
        _, mock_ckpt, _ = _run_apply_model_infrastructure_load_before_shard()

        _, kwargs = mock_ckpt.load_base_model.call_args
        assert "peft_init_method" not in kwargs


# =============================================================================
# Tests for from_config load_base_model kwarg forwarding
# =============================================================================


class TestFromConfigLoadBaseModelKwarg:
    """Tests for from_config accepting and forwarding load_base_model as a kwarg."""

    def test_from_config_defaults_load_base_model_to_false(self):
        """from_config should default load_base_model to False when not provided."""
        from nemo_automodel._transformers.auto_model import _BaseNeMoAutoModelClass

        with patch.object(_BaseNeMoAutoModelClass, "_build_model", return_value=MagicMock()) as mock_build:
            _BaseNeMoAutoModelClass.from_config(
                config=MagicMock(name_or_path="test"),
                trust_remote_code=False,
            )

        _, build_kwargs = mock_build.call_args
        assert build_kwargs["load_base_model"] is False

    def test_from_config_forwards_load_base_model_true(self):
        """from_config should forward load_base_model=True when provided as kwarg."""
        from nemo_automodel._transformers.auto_model import _BaseNeMoAutoModelClass

        with patch.object(_BaseNeMoAutoModelClass, "_build_model", return_value=MagicMock()) as mock_build:
            _BaseNeMoAutoModelClass.from_config(
                config=MagicMock(name_or_path="test"),
                trust_remote_code=False,
                load_base_model=True,
            )

        _, build_kwargs = mock_build.call_args
        assert build_kwargs["load_base_model"] is True
