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


def test_moe_infrastructure_forwards_fsdp2_tp_sequence_and_offload_settings():
    """EP's dedicated parallelizer must retain the FSDP2 manager's TP settings."""
    from nemo_automodel._transformers.infrastructure import instantiate_infrastructure
    from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager

    manager = object.__new__(FSDP2Manager)
    manager.tp_plan = {"lm_head": object()}
    manager.sequence_parallel = False
    manager.offload_policy = object()
    manager.mp_policy = object()
    manager.reshard_after_forward = True
    manager.enable_async_tensor_parallel = False
    mesh = SimpleNamespace(ep_size=2)

    with (
        patch(f"{_INFRA_MODULE}._instantiate_distributed", return_value=manager),
        patch(f"{_INFRA_MODULE}._instantiate_pipeline", return_value=None),
        patch(f"{_INFRA_MODULE}._instantiate_qat", return_value=None),
    ):
        model_wrapper, _, parallelize_fn, _ = instantiate_infrastructure(
            distributed_config=SimpleNamespace(activation_checkpointing=False),
            mesh=mesh,
        )

    assert model_wrapper is manager
    assert parallelize_fn.keywords["tp_shard_plan"] is manager.tp_plan
    assert parallelize_fn.keywords["sequence_parallel"] is False
    assert parallelize_fn.keywords["offload_policy"] is manager.offload_policy
    assert parallelize_fn.keywords["mp_policy"] is manager.mp_policy
    assert parallelize_fn.keywords["reshard_after_forward"] is True
    assert parallelize_fn.keywords["enable_async_tensor_parallel"] is False


# =============================================================================
# Tests for apply_model_infrastructure: post-shard initialize_model_weights
# =============================================================================


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace()


def test_safe_moe_tp_requires_real_checkpoint_source_and_rejects_peft():
    from nemo_automodel._transformers.infrastructure import (
        _validate_safe_moe_tp_weight_source,
        _verify_safe_moe_tp_weights_loaded,
    )

    model = _DummyModel()
    model._nemo_moe_tp_requires_pretrained_weights = True

    with pytest.raises(ValueError, match="from_config/random initialization"):
        _validate_safe_moe_tp_weight_source(
            model,
            checkpoint_source_available=False,
            peft_config=None,
        )
    with pytest.raises(ValueError, match="does not support PEFT"):
        _validate_safe_moe_tp_weight_source(
            model,
            checkpoint_source_available=True,
            peft_config=object(),
        )

    _validate_safe_moe_tp_weight_source(
        model,
        checkpoint_source_available=True,
        peft_config=None,
    )
    with pytest.raises(RuntimeError, match="without a completed checkpoint"):
        _verify_safe_moe_tp_weights_loaded(model, checkpoint_loaded=False)
    _verify_safe_moe_tp_weights_loaded(model, checkpoint_loaded=True)


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


def test_apply_model_infrastructure_handles_unwrapped_single_rank_ddp_model():
    """Single-rank DDP skips wrapping, so the returned model may not have ``.module``."""
    from nemo_automodel._transformers.infrastructure import apply_model_infrastructure
    from nemo_automodel.components.distributed.ddp import DDPManager

    model = _DummyModel()
    model_wrapper = object.__new__(DDPManager)

    with (
        patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
        patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
        patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
        patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
        patch(f"{_INFRA_MODULE}._shard_ep_fsdp", return_value=model),
        patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
    ):
        mock_ckpt = MockCheckpointer.return_value
        mock_ckpt.config = MagicMock()
        mock_ckpt.config.dequantize_base_checkpoint = False

        result = apply_model_infrastructure(
            model=model,
            is_meta_device=False,
            device=torch.device("cpu"),
            load_base_model=False,
            model_wrapper=model_wrapper,
        )

    assert result is model
    assert hasattr(model, "_pre_shard_hf_state_dict_keys")


class TestApplyModelInfrastructurePostShardInit:
    """Tests for initialize_model_weights being called in apply_model_infrastructure."""

    def test_load_only_checkpointer_disables_consolidated_export(self):
        """Infrastructure checkpointer loads base weights only and should not warn about consolidated saves."""
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

            apply_model_infrastructure(
                model=model,
                is_meta_device=True,
                device=torch.device("cpu"),
                load_base_model=True,
                pretrained_model_name_or_path="test/model",
            )

        checkpoint_config = MockCheckpointer.call_args.args[0]
        assert checkpoint_config.save_consolidated.value == "false"

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

    def test_calls_model_to_device_when_checkpoint_loaded_without_dtensor(self):
        """Unsharded post-shard checkpoint loads should still move buffers with model.to(device)."""
        from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

        model = _DummyModel()

        with (
            patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
            patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
            patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
            patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
            patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
            patch.object(model, "to", wraps=model.to) as mock_to,
        ):
            mock_ckpt = MockCheckpointer.return_value
            mock_ckpt.config = MagicMock()
            mock_ckpt.config.dequantize_base_checkpoint = False

            # from_pretrained on meta device: should_load_checkpoint = True
            apply_model_infrastructure(
                model=model,
                is_meta_device=True,
                device=torch.device("cpu"),
                load_base_model=True,
                pretrained_model_name_or_path="test/model",
            )

            mock_to.assert_called_once_with(torch.device("cpu"), non_blocking=True)

    def test_skips_model_to_device_when_checkpoint_loaded_with_dtensor(self, monkeypatch):
        """DTensor-sharded post-shard checkpoint loads should skip model.to(device)."""
        import torch.distributed.tensor as dist_tensor

        from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

        class FakeDTensor:
            pass

        class ModelWithShardedParameter(_DummyModel):
            def parameters(self, recurse=True):
                return iter([FakeDTensor()])

        monkeypatch.setattr(dist_tensor, "DTensor", FakeDTensor)
        model = ModelWithShardedParameter()

        with (
            patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
            patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
            patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
            patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
            patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
            patch.object(model, "to", wraps=model.to) as mock_to,
        ):
            mock_ckpt = MockCheckpointer.return_value
            mock_ckpt.config = MagicMock()
            mock_ckpt.config.dequantize_base_checkpoint = False

            apply_model_infrastructure(
                model=model,
                is_meta_device=True,
                device=torch.device("cpu"),
                load_base_model=True,
                pretrained_model_name_or_path="test/model",
            )

            mock_to.assert_not_called()

    def test_calls_model_to_device_when_from_config_meta(self):
        """model.to(device) should still be called on the from_config meta path (no checkpoint loaded)."""
        from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

        model = _DummyModel()

        with (
            patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
            patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
            patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
            patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
            patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
            patch.object(model, "to", wraps=model.to) as mock_to,
        ):
            mock_ckpt = MockCheckpointer.return_value
            mock_ckpt.config = MagicMock()
            mock_ckpt.config.dequantize_base_checkpoint = False

            # from_config on meta device: need_post_shard_init = True,
            # but should_load_checkpoint = False (no pretrained path).
            # model.to(device) should still be called to move buffers.
            apply_model_infrastructure(
                model=model,
                is_meta_device=True,
                device=torch.device("cpu"),
                load_base_model=False,
                pretrained_model_name_or_path="",
            )

            mock_to.assert_called_once_with(torch.device("cpu"), non_blocking=True)

    def test_model_to_falls_back_to_to_empty_on_meta_tensor_error(self):
        """model.to() raising 'Cannot copy out of meta tensor' should fall back to model.to_empty()."""
        from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

        model = _DummyModel()

        with (
            patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
            patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
            patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
            patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
            patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
            patch.object(model, "to", side_effect=NotImplementedError("Cannot copy out of meta tensor; no data!")),
            patch.object(model, "to_empty") as mock_to_empty,
        ):
            mock_ckpt = MockCheckpointer.return_value
            mock_ckpt.config = MagicMock()
            mock_ckpt.config.dequantize_base_checkpoint = False

            apply_model_infrastructure(
                model=model,
                is_meta_device=False,
                device=torch.device("cpu"),
                load_base_model=False,
                pretrained_model_name_or_path="",
            )

            mock_to_empty.assert_called_once_with(device=torch.device("cpu"))

    def test_model_to_reraises_other_not_implemented_error(self):
        """model.to() raising NotImplementedError without meta tensor message should re-raise."""
        from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

        model = _DummyModel()

        with (
            patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
            patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
            patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
            patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
            patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
            patch.object(model, "to", side_effect=NotImplementedError("Some other error")),
        ):
            mock_ckpt = MockCheckpointer.return_value
            mock_ckpt.config = MagicMock()
            mock_ckpt.config.dequantize_base_checkpoint = False

            with pytest.raises(NotImplementedError, match="Some other error"):
                apply_model_infrastructure(
                    model=model,
                    is_meta_device=False,
                    device=torch.device("cpu"),
                    load_base_model=False,
                    pretrained_model_name_or_path="",
                )

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

    def test_applies_rotary_fix_automatically_when_needed(self):
        """Nemotron Flash rotary workaround should run from shared infrastructure, not the train loop."""
        from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

        model = _DummyModel()

        with (
            patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
            patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
            patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
            patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
            patch(f"{_INFRA_MODULE}.should_fix_rotary_embeddings", return_value=True) as mock_should_fix,
            patch(f"{_INFRA_MODULE}.fix_rotary_embeddings") as mock_fix_rotary,
            patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
        ):
            mock_ckpt = MockCheckpointer.return_value
            mock_ckpt.config = MagicMock()
            mock_ckpt.config.dequantize_base_checkpoint = False

            apply_model_infrastructure(
                model=model,
                is_meta_device=False,
                device=torch.device("cpu"),
                load_base_model=False,
                pretrained_model_name_or_path="",
            )

            mock_should_fix.assert_called_once_with([model])
            mock_fix_rotary.assert_called_once_with([model])

    def test_skips_rotary_fix_when_not_needed(self):
        """Shared infrastructure should leave non-Nemotron models untouched."""
        from nemo_automodel._transformers.infrastructure import apply_model_infrastructure

        model = _DummyModel()

        with (
            patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
            patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
            patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
            patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
            patch(f"{_INFRA_MODULE}.should_fix_rotary_embeddings", return_value=False) as mock_should_fix,
            patch(f"{_INFRA_MODULE}.fix_rotary_embeddings") as mock_fix_rotary,
            patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
        ):
            mock_ckpt = MockCheckpointer.return_value
            mock_ckpt.config = MagicMock()
            mock_ckpt.config.dequantize_base_checkpoint = False

            apply_model_infrastructure(
                model=model,
                is_meta_device=False,
                device=torch.device("cpu"),
                load_base_model=False,
                pretrained_model_name_or_path="",
            )

            mock_should_fix.assert_called_once_with([model])
            mock_fix_rotary.assert_not_called()


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

        mock_ckpt.initialize_model_weights.assert_called_once_with(model, torch.device("cpu"), peft_init_method=None)
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


def test_apply_model_infrastructure_attaches_cp_hooks_for_non_te(monkeypatch):
    """When mesh.cp_size>1 and attention is non-TE, apply_model_infrastructure
    attaches the mask-strip CP hook to every model part.
    The DTensor-SDPA hook is gated on torch.compile; this case has no compile
    (model_wrapper=None), so it is not attached.
    Guards infrastructure.py:apply_model_infrastructure CP branch."""
    from nemo_automodel._transformers import infrastructure as infra

    model = _DummyModel()
    mesh = SimpleNamespace(
        cp_size=2,
        pp_size=1,
        tp_size=1,
        ep_size=1,
        dp_size=1,
        dp_shard_size=1,
        dp_replicate_size=1,
        device_mesh={"cp": SimpleNamespace(size=lambda: 2)},
        moe_mesh=None,
    )

    attached = {"ctx": 0, "attn": 0}

    with (
        patch(f"{_INFRA_MODULE}.get_world_size_safe", return_value=1),
        patch(f"{_INFRA_MODULE}._supports_logits_to_keep", return_value=True),
        patch(f"{_INFRA_MODULE}.print_trainable_parameters"),
        patch(f"{_INFRA_MODULE}._should_load_before_shard", return_value=False),
        patch(f"{_INFRA_MODULE}._uses_te_attention", return_value=False),
        patch(f"{_INFRA_MODULE}.Checkpointer") as MockCheckpointer,
        patch(
            "nemo_automodel.components.distributed.cp_utils.attach_context_parallel_hooks",
            side_effect=lambda mp: attached.__setitem__("ctx", attached["ctx"] + 1),
        ),
        patch(
            "nemo_automodel.components.distributed.cp_utils.attach_cp_sdpa_hooks",
            side_effect=lambda mp, cp_mesh: attached.__setitem__("attn", attached["attn"] + 1),
        ),
    ):
        mock_ckpt = MockCheckpointer.return_value
        mock_ckpt.config = MagicMock()
        mock_ckpt.config.dequantize_base_checkpoint = False
        infra.apply_model_infrastructure(
            model=model,
            is_meta_device=False,
            device=torch.device("cpu"),
            load_base_model=False,
            model_wrapper=None,
            mesh=mesh,
            pretrained_model_name_or_path="",
        )

    assert attached == {"ctx": 1, "attn": 0}


# =============================================================================
# Tests for instantiate_infrastructure: MoE parallelize_fn option threading
# =============================================================================


def test_instantiate_infrastructure_threads_ac_scope_into_moe_parallelize_fn():
    """Expert-parallel configs must inherit the strategy config's normalized AC scope."""
    from nemo_automodel._transformers.infrastructure import instantiate_infrastructure
    from nemo_automodel.components.distributed.config import DDPConfig
    from nemo_automodel.components.moe.parallelizer import parallelize_model

    distributed_config = DDPConfig(activation_checkpointing=True, activation_checkpointing_scope="vision")
    mesh = SimpleNamespace(ep_size=2, pp_size=1, device_mesh=None, moe_mesh=None)

    # The manager needs an initialized process group; the scope is read from the
    # strategy config, so the manager itself is irrelevant here.
    with patch(f"{_INFRA_MODULE}._instantiate_distributed", return_value=None):
        _, _, parallelize_fn, _ = instantiate_infrastructure(
            distributed_config=distributed_config,
            mesh=mesh,
        )

    assert parallelize_fn.func is parallelize_model
    assert parallelize_fn.keywords["activation_checkpointing"] is True
    # DDPConfig.__post_init__ normalizes the scope; the partial must carry it through.
    assert parallelize_fn.keywords["activation_checkpointing_scope"] == ("vision",)
