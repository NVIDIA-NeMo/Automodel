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

"""Tests for the parallelization strategy pattern."""

import logging
import sys
from abc import ABC
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel

from nemo_automodel._transformers.model_init import bind_model_specs
from nemo_automodel.components.distributed import parallelizer as parallelizer_mod
from nemo_automodel.components.distributed.activation_checkpointing import (
    apply_full_layer_checkpointing_to_layers,
    query_activation_checkpointing_spec,
    sdpa_backend_snapshot_context_fn,
)
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import (
    _DEFAULT_STRATEGY,
    DefaultParallelizationStrategy,
    ParallelizationStrategy,
    _extract_model_layers,
    fsdp2_strategy_parallelize,
    get_model_layer_groups,
    get_parallelization_strategy,
)
from nemo_automodel.components.models.hunyuan_video15.parallelization import HunyuanVideo15Transformer3DModel
from nemo_automodel.components.models.ltx2_video.parallelization import LTX2VideoTransformer3DModel
from nemo_automodel.components.models.nemotron_nas import parallelization as nas_parallelization
from nemo_automodel.components.models.nemotron_nas.parallelization import (
    NemotronNASParallelizationStrategy,
    validate_tp_mesh_for_nemotron_nas,
)
from nemo_automodel.components.models.nemotron_v3.parallelization import (
    NEMOTRON_H_ACTIVATION_CHECKPOINTING_SPEC,
    NEMOTRON_H_PARALLEL_SPEC,
    NEMOTRON_H_TP_PLAN,
    NemotronHParallelizationStrategy,
)
from nemo_automodel.components.models.qwen3_5.parallelization import (
    QWEN3_5_PARALLEL_SPEC,
    Qwen3_5ParallelizationStrategy,
)
from nemo_automodel.components.models.qwen_image.parallelization import QwenImageTransformer2DModel
from nemo_automodel.components.models.wan.parallelization import WAN_TP_PLAN, WanTransformer3DModel


class MockModel(nn.Module):
    """Mock model for testing purposes."""

    def __init__(self, model_name="MockModel", num_attention_heads=8, num_key_value_heads=8):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=num_attention_heads * 8,
        )

        # Create mock model structure
        class MockInnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([self._create_mock_layer() for _ in range(2)])

            def _create_mock_layer(self):
                """Create a mock transformer layer."""
                layer = nn.Module()
                layer.mlp = nn.Linear(10, 10)
                return layer

        self.model = MockInnerModel()

        # Set the class name for strategy selection
        self.__class__.__name__ = model_name

    def forward(self, x):
        return x


class MockNemotronHModel(nn.Module):
    """Mock NemotronH model for testing."""

    def __init__(self):
        super().__init__()

        class MockSupports:
            def __init__(self, model):
                self.model = model
                self.supports_mtp_cp = True
                self.supports_mtp_cp_pp = False

            @property
            def mtp_enabled(self):
                return bool(getattr(getattr(self.model, "mtp_config", None), "enabled", False))

        self.supports = MockSupports(self)
        self.config = SimpleNamespace(
            num_attention_heads=8,
            num_key_value_heads=8,
        )

        # Create backbone structure specific to NemotronH
        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([self._create_mock_layer() for _ in range(2)])

            def _create_mock_layer(self):
                layer = nn.Module()
                # Use setattr to avoid linter issues with dynamic attributes
                setattr(layer, "block_type", "mlp")  # Set block type for NemotronH
                layer.mixer = nn.Module()
                layer.mixer.up_proj = nn.Linear(10, 10)
                layer.mixer.down_proj = nn.Linear(10, 10)
                return layer

        self.backbone = MockBackbone()
        self.__class__.__name__ = "NemotronHForCausalLM"
        self.__class__.parallel_spec = NEMOTRON_H_PARALLEL_SPEC
        self.__class__.activation_checkpointing_spec = NEMOTRON_H_ACTIVATION_CHECKPOINTING_SPEC

    def forward(self, x):
        return x


class MockNemotronV3Model(nn.Module):
    """Mock of the native Nemotron-V3 model: decoder blocks live in ``model.model.layers``
    as a ``ModuleDict`` (keyed "0".."N-1") and there is no ``backbone`` attribute."""

    def __init__(self, num_layers=4):
        super().__init__()

        class MockInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict()
                for i in range(num_layers):
                    layer = nn.Module()
                    setattr(layer, "block_type", "mlp" if i % 2 == 0 else "attention")
                    self.layers[str(i)] = layer

        self.model = MockInner()
        self.__class__.__name__ = "NemotronHForCausalLM"
        self.__class__.parallel_spec = NEMOTRON_H_PARALLEL_SPEC
        self.__class__.activation_checkpointing_spec = NEMOTRON_H_ACTIVATION_CHECKPOINTING_SPEC

    def forward(self, x):
        return x


class TestNemotronHLayoutResolution:
    """Both classes named ``NemotronHForCausalLM`` must resolve their decoder blocks
    without an AttributeError (AM-448): the HF model exposes ``backbone.layers``
    (``ModuleList``) and the native Nemotron-V3 model exposes ``model.layers``
    (``ModuleDict``)."""

    def test_layer_groups_hf_backbone_modulelist(self):
        model = MockNemotronHModel()
        assert get_model_layer_groups(model) == {"language": list(model.backbone.layers)}

    def test_layer_groups_native_model_moduledict(self):
        model = MockNemotronV3Model(num_layers=4)
        assert get_model_layer_groups(model) == {"language": list(model.model.layers.values())}

    def test_extract_model_layers_native_has_no_backbone(self):
        # Regression for AM-448: the registry still lists "backbone.layers", but the native
        # model has no `backbone`; _reduce_attrs must skip it and resolve "model.layers"
        # instead of raising.
        assert len(_extract_model_layers(MockNemotronV3Model(num_layers=4))) == 4

    def test_extract_model_layers_hf_backbone(self):
        assert len(_extract_model_layers(MockNemotronHModel())) == 2


@pytest.fixture
def mock_device_mesh():
    """Create a mock device mesh for testing."""
    mesh = MagicMock(spec=DeviceMesh)
    mesh.device_type = "cuda"

    # Mock submeshes
    dp_replicate_mesh = MagicMock()
    dp_shard_mesh = MagicMock()
    tp_mesh = MagicMock()

    dp_replicate_mesh.size.return_value = 1
    dp_shard_mesh.size.return_value = 2
    tp_mesh.size.return_value = 1

    dp_replicate_mesh.ndim = 1
    dp_shard_mesh.ndim = 1
    tp_mesh.ndim = 1

    # Mesh dimension names used by parallelizer to check for optional submeshes (e.g. "cp")
    mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "tp")

    # Configure mesh access
    mesh.__getitem__.side_effect = lambda key: {
        "dp_replicate": dp_replicate_mesh,
        "dp_shard_cp": dp_shard_mesh,
        "tp": tp_mesh,
        ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,  # Combined mesh
    }[key]

    return mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh


@pytest.fixture
def mock_distributed_env(monkeypatch):
    """Mock the distributed environment for strategy tests."""
    # Mock FSDP functions
    fully_shard_mock = MagicMock(side_effect=lambda model, **kwargs: model)
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.fully_shard", fully_shard_mock, raising=False
    )

    # Mock tensor parallel functions
    parallelize_module_mock = MagicMock()
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.parallelize_module", parallelize_module_mock, raising=False
    )

    # Mock checkpoint wrapper. Sub-module/whole-block wrapping now lives in
    # activation_checkpointing.py and calls that module's checkpoint_wrapper, so
    # patch it there too.
    checkpoint_wrapper_mock = MagicMock(side_effect=lambda x, **_kwargs: x)
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.checkpoint_wrapper", checkpoint_wrapper_mock, raising=False
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.activation_checkpointing.checkpoint_wrapper",
        checkpoint_wrapper_mock,
        raising=False,
    )

    # Mock apply_fsdp2_sharding_recursively
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively",
        apply_fsdp_mock,
        raising=False,
    )

    # Mock grouped layer extraction, which is used for both sharding and AC scope filtering.
    extract_layer_groups_mock = MagicMock(return_value={"language": []})
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer._extract_model_layer_groups",
        extract_layer_groups_mock,
        raising=False,
    )

    # Mock _get_parallel_plan with a head-sharding plan (validation is skipped for plans that keep heads whole).
    get_plan_mock = MagicMock(return_value={"model.layers.*.self_attn.q_proj": ColwiseParallel()})
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer._get_parallel_plan", get_plan_mock, raising=False
    )

    # Mock validate_tp_mesh
    validate_tp_mock = MagicMock()
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.validate_tp_mesh", validate_tp_mock, raising=False
    )

    return {
        "fully_shard": fully_shard_mock,
        "parallelize_module": parallelize_module_mock,
        "checkpoint_wrapper": checkpoint_wrapper_mock,
        "apply_fsdp": apply_fsdp_mock,
        "extract_layer_groups": extract_layer_groups_mock,
        "get_plan": get_plan_mock,
        "validate_tp": validate_tp_mock,
    }


class TestParallelizationStrategy:
    """Test the abstract ParallelizationStrategy base class."""

    def test_is_abstract(self):
        """Test that ParallelizationStrategy is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ParallelizationStrategy()  # type: ignore

    def test_has_abstract_parallelize_method(self):
        """Test that the parallelize method is abstract."""
        assert hasattr(ParallelizationStrategy, "parallelize")
        assert getattr(ParallelizationStrategy.parallelize, "__isabstractmethod__", False)

    def test_inherits_from_abc(self):
        """Test that ParallelizationStrategy inherits from ABC."""
        assert issubclass(ParallelizationStrategy, ABC)


class TestDefaultParallelizationStrategy:
    """Test the DefaultParallelizationStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create a DefaultParallelizationStrategy instance."""
        return DefaultParallelizationStrategy()

    def test_can_be_instantiated(self, strategy):
        """Test that DefaultParallelizationStrategy can be instantiated."""
        assert isinstance(strategy, DefaultParallelizationStrategy)
        assert isinstance(strategy, ParallelizationStrategy)

    def test_parallelize_method_signature(self, strategy):
        """Test that parallelize method has the correct signature."""
        method = strategy.parallelize
        assert callable(method)

        # Check that all required parameters are supported
        import inspect

        sig = inspect.signature(method)
        required_params = [
            "model",
            "device_mesh",
            "mp_policy",
            "offload_policy",
            "sequence_parallel",
            "activation_checkpointing",
            "activation_checkpointing_scope",
            "tp_shard_plan",
            "dp_replicate_mesh_name",
            "dp_shard_cp_mesh_name",
            "tp_mesh_name",
            "frozen_multimodal_sharding",
            "reapply_trainability",
        ]

        for param in required_params:
            assert param in sig.parameters

    def test_parallelize_basic_flow(self, strategy, mock_device_mesh, mock_distributed_env):
        """Test the basic parallelization flow of DefaultParallelizationStrategy."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        model = MockModel()

        # Call the strategy
        result = strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        # Verify the strategy was called correctly
        assert result is model  # Should return the same model

        # Verify key functions were called
        mock_distributed_env["extract_layer_groups"].assert_called_once_with(model)
        mock_distributed_env["apply_fsdp"].assert_called_once()
        mock_distributed_env["fully_shard"].assert_called()

    @pytest.mark.parametrize(
        ("reshard_after_forward", "expected_input_reshard", "expected_output_reshard"),
        [(None, True, False), (True, True, False), (False, False, False)],
    )
    def test_parallelize_splits_untied_input_and_output_embeddings(
        self,
        strategy,
        mock_device_mesh,
        mock_distributed_env,
        reshard_after_forward,
        expected_input_reshard,
        expected_output_reshard,
    ):
        """Untied trainable tables become separate FSDP units before the root."""
        mesh, _, _, _ = mock_device_mesh
        model = MockModel()
        model.model.embed_tokens = nn.Embedding(32, 10)
        model.lm_head = nn.Linear(10, 32, bias=False)
        model.get_input_embeddings = lambda: model.model.embed_tokens
        model.get_output_embeddings = lambda: model.lm_head

        strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
            reshard_after_forward=reshard_after_forward,
        )

        calls = mock_distributed_env["fully_shard"].call_args_list
        modules = [item.args[0] for item in calls]
        assert model.model.embed_tokens in modules
        assert model.lm_head in modules
        assert modules[-1] is model
        embed_call = next(item for item in calls if item.args[0] is model.model.embed_tokens)
        head_call = next(item for item in calls if item.args[0] is model.lm_head)
        assert embed_call.kwargs["reshard_after_forward"] is expected_input_reshard
        assert head_call.kwargs["reshard_after_forward"] is expected_output_reshard

    def test_parallelize_keeps_tied_embeddings_in_root(self, strategy, mock_device_mesh, mock_distributed_env):
        """Input/output parameters sharing storage must not acquire two FSDP owners."""
        mesh, _, _, _ = mock_device_mesh
        model = MockModel()
        model.config.tie_word_embeddings = True
        model.model.embed_tokens = nn.Embedding(32, 10)
        model.lm_head = nn.Linear(10, 32, bias=False)
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach())
        assert model.lm_head.weight is not model.model.embed_tokens.weight
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        model.get_input_embeddings = lambda: model.model.embed_tokens
        model.get_output_embeddings = lambda: model.lm_head

        strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        modules = [item.args[0] for item in mock_distributed_env["fully_shard"].call_args_list]
        assert model.model.embed_tokens not in modules
        assert model.lm_head not in modules
        assert modules[-1] is model

    def test_parallelize_keeps_frozen_embeddings_in_root(self, strategy, mock_device_mesh, mock_distributed_env):
        """Frozen tables do not need standalone gradient communication units."""
        mesh, _, _, _ = mock_device_mesh
        model = MockModel()
        model.model.embed_tokens = nn.Embedding(32, 10).requires_grad_(False)
        model.lm_head = nn.Linear(10, 32, bias=False).requires_grad_(False)
        model.get_input_embeddings = lambda: model.model.embed_tokens
        model.get_output_embeddings = lambda: model.lm_head

        strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        modules = [item.args[0] for item in mock_distributed_env["fully_shard"].call_args_list]
        assert model.model.embed_tokens not in modules
        assert model.lm_head not in modules
        assert modules[-1] is model

    def test_parallelize_with_tensor_parallel(self, strategy, mock_device_mesh, mock_distributed_env):
        """Test parallelization with tensor parallelism enabled."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2  # Enable TP

        model = MockModel()

        strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        # Should call validate_tp_mesh, _get_parallel_plan, and parallelize_module
        mock_distributed_env["validate_tp"].assert_called_once_with(model, tp_mesh)
        mock_distributed_env["get_plan"].assert_called_once()
        mock_distributed_env["parallelize_module"].assert_called_once()

    def test_tp_validation_is_skipped_when_the_plan_keeps_heads_whole(
        self, strategy, mock_device_mesh, mock_distributed_env
    ):
        """Head counts constrain the TP size only when attention is sharded across heads."""
        mesh, _, _, tp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        mock_distributed_env["get_plan"].return_value = {"model.layers.*.mlp.up_proj": ColwiseParallel()}

        strategy.parallelize(model=MockModel(), device_mesh=mesh)

        mock_distributed_env["validate_tp"].assert_not_called()
        mock_distributed_env["parallelize_module"].assert_called_once()

    def test_trainability_rebind_runs_after_tp_and_before_fsdp(self, strategy, mock_device_mesh, mock_distributed_env):
        """FSDP captures the selector result on the post-TP hierarchy."""
        mesh, _, _, tp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        model = MockModel()
        events = []

        mock_distributed_env["parallelize_module"].side_effect = lambda *_args, **_kwargs: events.append("tp")
        mock_distributed_env["apply_fsdp"].side_effect = lambda *_args, **_kwargs: events.append("fsdp")

        strategy.parallelize(
            model=model,
            device_mesh=mesh,
            reapply_trainability=lambda _model: events.append("trainability"),
        )

        assert events == ["tp", "trainability", "fsdp"]

    def test_parallelize_with_activation_checkpointing(self, strategy, mock_device_mesh, mock_distributed_env):
        """Test parallelization with activation checkpointing enabled."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh

        # Mock layers with all the attributes that get checkpointed
        mock_layer = nn.Module()
        mock_layer.mlp = nn.Linear(10, 10)
        mock_layer.self_attn = nn.Linear(10, 10)
        mock_layer.input_layernorm = nn.Linear(10, 10)
        mock_layer.post_attention_layernorm = nn.Linear(10, 10)
        mock_distributed_env["extract_layer_groups"].return_value = {"language": [mock_layer]}

        model = MockModel()

        strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=True,
        )

        # Should apply checkpoint wrapper to all expected layer components
        checkpoint_wrapper_mock = mock_distributed_env["checkpoint_wrapper"]

        # Check that checkpoint_wrapper was called with all expected attributes
        expected_calls = [
            call(mock_layer.mlp, context_fn=sdpa_backend_snapshot_context_fn),
            call(mock_layer.self_attn, context_fn=sdpa_backend_snapshot_context_fn),
            call(mock_layer.input_layernorm),
            call(mock_layer.post_attention_layernorm),
        ]
        checkpoint_wrapper_mock.assert_has_calls(expected_calls, any_order=False)

    def test_parallelize_with_custom_mesh_names(self, strategy, mock_device_mesh, mock_distributed_env):
        """Test parallelization with custom mesh names."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh

        # Update mesh mock to support custom names
        mesh.mesh_dim_names = ("custom_dp_replicate", "custom_dp_shard", "custom_tp")
        mesh.__getitem__.side_effect = lambda key: {
            "custom_dp_replicate": dp_replicate_mesh,
            "custom_dp_shard": dp_shard_mesh,
            "custom_tp": tp_mesh,
            ("custom_dp_replicate", "custom_dp_shard"): dp_shard_mesh,
        }[key]

        model = MockModel()

        strategy.parallelize(
            model=model,
            device_mesh=mesh,
            dp_replicate_mesh_name="custom_dp_replicate",
            dp_shard_cp_mesh_name="custom_dp_shard",
            tp_mesh_name="custom_tp",
        )

        # Verify mesh access used custom names
        expected_calls = [
            call("custom_tp"),
            call(("custom_dp_replicate", "custom_dp_shard")),
        ]
        mesh.__getitem__.assert_has_calls(expected_calls, any_order=True)

    def test_explicit_reshard_true_warns_with_pipeline_parallelism(
        self, strategy, mock_device_mesh, mock_distributed_env, monkeypatch, caplog
    ):
        """Explicit layer resharding overrides the PP default and should warn."""
        mesh, _, _, _ = mock_device_mesh
        pp_mesh = MagicMock()
        pp_mesh.size.return_value = 2
        dp_mesh = MagicMock()
        dp_mesh.mesh_dim_names = ("pp",)
        dp_mesh.__getitem__.side_effect = lambda key: {"pp": pp_mesh}[key]
        monkeypatch.setattr(parallelizer_mod, "get_fsdp_dp_mesh", lambda *args, **kwargs: dp_mesh)

        with caplog.at_level(logging.WARNING, logger=parallelizer_mod.__name__):
            strategy.parallelize(
                model=MockModel(),
                device_mesh=mesh,
                sequence_parallel=False,
                activation_checkpointing=False,
                reshard_after_forward=True,
            )

        assert "reshard_after_forward=True overrides the pipeline-parallel default" in caplog.text


class TestNemotronHParallelizationStrategy:
    """Test the NemotronHParallelizationStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create a NemotronHParallelizationStrategy instance."""
        return NemotronHParallelizationStrategy()

    @pytest.fixture
    def nemotron_model(self):
        """Create a mock NemotronH model."""
        return MockNemotronHModel()

    def test_can_be_instantiated(self, strategy):
        """Test that NemotronHParallelizationStrategy can be instantiated."""
        assert isinstance(strategy, NemotronHParallelizationStrategy)
        assert isinstance(strategy, ParallelizationStrategy)

    @pytest.mark.parametrize("mtp_enabled", [True, False])
    def test_configures_only_enabled_mtp_attention_and_mamba_for_cp(
        self,
        strategy,
        mock_device_mesh,
        nemotron_model,
        monkeypatch,
        mtp_enabled,
    ):
        """The strategy installs CP collectives only on enabled MTP blocks."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        cp_group = object()
        cp_mesh = MagicMock()
        cp_mesh.size.return_value = 2
        cp_mesh.get_group.return_value = cp_group
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "cp", "tp")
        mesh.__getitem__.side_effect = lambda key: {
            "dp_replicate": dp_replicate_mesh,
            "dp_shard_cp": dp_shard_mesh,
            "cp": cp_mesh,
            "tp": tp_mesh,
            ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,
        }[key]

        class FakeDotProductAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.set_context_parallel_group = MagicMock()

        attention_module = FakeDotProductAttention()
        attention_layer = nn.Module()
        attention_layer.block_type = "attention"
        attention_layer.mixer = nn.Module()
        attention_layer.mixer.attn_module = attention_module
        attention_module_2 = FakeDotProductAttention()
        attention_layer_2 = nn.Module()
        attention_layer_2.block_type = "attention"
        attention_layer_2.mixer = nn.Module()
        attention_layer_2.mixer.attn_module = attention_module_2

        mamba_layer = nn.Module()
        mamba_layer.block_type = "mamba"
        mamba_layer.mixer = nn.Module()
        mamba_layer.mixer.num_heads = 8
        mamba_layer.mixer.head_dim = 16
        mamba_layer.mixer.n_groups = 2
        mamba_layer.mixer.ssm_state_size = 64

        nemotron_model.mtp_config = SimpleNamespace(enabled=mtp_enabled)
        nemotron_model.mtp = nn.Module()
        nemotron_model.mtp.layers = nn.ModuleList([attention_layer, attention_layer_2, mamba_layer])

        transformer_engine = ModuleType("transformer_engine")
        transformer_engine.__path__ = []
        transformer_engine_pytorch = ModuleType("transformer_engine.pytorch")
        transformer_engine_pytorch.__path__ = []
        transformer_engine_attention = ModuleType("transformer_engine.pytorch.attention")
        transformer_engine_attention.DotProductAttention = FakeDotProductAttention
        monkeypatch.setitem(sys.modules, "transformer_engine", transformer_engine)
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", transformer_engine_pytorch)
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.attention", transformer_engine_attention)

        from nemo_automodel.components.distributed.context_parallel import mamba as mamba_module

        mamba_cp = object()
        mamba_cp_ctor = MagicMock(return_value=mamba_cp)
        monkeypatch.setattr(mamba_module, "MambaContextParallel", mamba_cp_ctor)
        cp_ranks = [0, 1]
        get_cp_ranks = MagicMock(return_value=cp_ranks)
        cp_stream = object()
        monkeypatch.setattr(parallelizer_mod.torch.distributed, "get_process_group_ranks", get_cp_ranks)
        monkeypatch.setattr(parallelizer_mod.torch.cuda, "Stream", lambda: cp_stream)
        monkeypatch.setattr(parallelizer_mod, "fully_shard", lambda model, **_kwargs: model)
        monkeypatch.setattr(
            parallelizer_mod.parallelizer_utils,
            "fully_shard_by_dtype",
            lambda model, *_args, **_kwargs: model,
        )

        strategy.parallelize(model=nemotron_model, device_mesh=mesh)

        if not mtp_enabled:
            attention_module.set_context_parallel_group.assert_not_called()
            attention_module_2.set_context_parallel_group.assert_not_called()
            mamba_cp_ctor.assert_not_called()
            assert not hasattr(mamba_layer.mixer, "cp")
            return

        attention_module.set_context_parallel_group.assert_called_once_with(
            cp_group,
            cp_ranks,
            cp_stream,
            cp_comm_type="p2p",
        )
        attention_module_2.set_context_parallel_group.assert_called_once_with(
            cp_group,
            cp_ranks,
            cp_stream,
            cp_comm_type="p2p",
        )
        get_cp_ranks.assert_called_once_with(cp_group)
        mamba_cp_ctor.assert_called_once_with(
            cp_group=cp_group,
            num_heads=8,
            head_dim=16,
            n_groups=2,
            d_state=64,
            mixer=mamba_layer.mixer,
        )
        assert mamba_layer.mixer.cp is mamba_cp

    def test_cp_mtp_pipeline_stage_raises_explicit_unsupported_topology(
        self,
        strategy,
        mock_device_mesh,
        nemotron_model,
        monkeypatch,
    ):
        """Every trimmed PP stage must fail before stage-specific CP wiring."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        cp_group = object()
        cp_mesh = MagicMock()
        cp_mesh.size.return_value = 2
        cp_mesh.get_group.return_value = cp_group
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "cp", "tp")
        mesh.__getitem__.side_effect = lambda key: {
            "dp_replicate": dp_replicate_mesh,
            "dp_shard_cp": dp_shard_mesh,
            "cp": cp_mesh,
            "tp": tp_mesh,
            ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,
        }[key]
        nemotron_model.mtp_config = SimpleNamespace(enabled=True)
        nemotron_model.mtp = None
        nemotron_model._is_pipeline_parallel_stage = lambda: True
        monkeypatch.setattr(parallelizer_mod.torch.distributed, "get_process_group_ranks", lambda _group: [0, 1])

        with pytest.raises(NotImplementedError, match="MTP with context and pipeline parallelism"):
            strategy.parallelize(model=nemotron_model, device_mesh=mesh)

    def test_cp_raises_when_enabled_mtp_layers_are_unavailable(
        self,
        strategy,
        mock_device_mesh,
        nemotron_model,
        monkeypatch,
    ):
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        cp_group = object()
        cp_mesh = MagicMock()
        cp_mesh.size.return_value = 2
        cp_mesh.get_group.return_value = cp_group
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "cp", "tp")
        mesh.__getitem__.side_effect = lambda key: {
            "dp_replicate": dp_replicate_mesh,
            "dp_shard_cp": dp_shard_mesh,
            "cp": cp_mesh,
            "tp": tp_mesh,
            ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,
        }[key]
        nemotron_model.mtp_config = SimpleNamespace(enabled=True)
        nemotron_model.mtp = nn.Module()
        monkeypatch.setattr(parallelizer_mod.torch.distributed, "get_process_group_ranks", lambda _group: [0, 1])

        with pytest.raises(RuntimeError, match=r"MTP is enabled but model\.mtp\.layers is unavailable"):
            strategy.parallelize(model=nemotron_model, device_mesh=mesh)

    def test_cp_rejects_enabled_mtp_without_capability(
        self,
        strategy,
        mock_device_mesh,
        nemotron_model,
        monkeypatch,
    ):
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        cp_mesh = MagicMock()
        cp_mesh.size.return_value = 2
        cp_mesh.get_group.return_value = object()
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "cp", "tp")
        mesh.__getitem__.side_effect = lambda key: {
            "dp_replicate": dp_replicate_mesh,
            "dp_shard_cp": dp_shard_mesh,
            "cp": cp_mesh,
            "tp": tp_mesh,
            ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,
        }[key]
        nemotron_model.mtp_config = SimpleNamespace(enabled=True)
        nemotron_model.mtp = nn.Module()
        nemotron_model.mtp.layers = nn.ModuleList([nn.Module()])
        nemotron_model.supports.supports_mtp_cp = False
        monkeypatch.setattr(parallelizer_mod.torch.distributed, "get_process_group_ranks", lambda _group: [0, 1])

        with pytest.raises(RuntimeError, match="does not support MTP with context parallelism"):
            strategy.parallelize(model=nemotron_model, device_mesh=mesh)

    def test_sequence_parallel_not_supported(self, strategy, mock_device_mesh, nemotron_model):
        """Test that sequence parallelism raises assertion error."""
        mesh, _, _, _ = mock_device_mesh

        with pytest.raises(AssertionError, match="Sequence parallelism is not supported"):
            strategy.parallelize(
                model=nemotron_model,
                device_mesh=mesh,
                sequence_parallel=True,
            )

    def test_custom_tp_plan_is_applied_by_the_shared_flow(
        self, strategy, mock_device_mesh, nemotron_model, mock_distributed_env
    ):
        """A user plan is honoured exactly as for any other model."""
        mesh, _, _, tp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        custom_plan = {"test": ColwiseParallel()}

        result = strategy.parallelize(model=nemotron_model, device_mesh=mesh, tp_shard_plan=custom_plan)

        assert result is nemotron_model
        assert mock_distributed_env["get_plan"].call_args.args[2] is custom_plan

    @pytest.mark.parametrize("tp_size", [1, 2])
    @patch("nemo_automodel.components.distributed.parallelizer.parallelize_module")
    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    @patch("nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype")
    def test_nemotron_specific_parallelization(
        self,
        fully_shard_by_dtype,
        fully_shard,
        mock_parallelize_module,
        strategy,
        mock_device_mesh,
        nemotron_model,
        tp_size,
    ):
        """The declared MLP-only plan is applied once and every block is sharded dtype-aware."""
        mesh, _, dp_shard_mesh, tp_mesh = mock_device_mesh
        fully_shard.side_effect = lambda model, **kwargs: model
        fully_shard_by_dtype.side_effect = lambda model, *args, **kwargs: model
        tp_mesh.size.return_value = tp_size

        strategy.parallelize(
            model=nemotron_model,
            device_mesh=mesh,
            activation_checkpointing=False,
        )

        if tp_size == 1:
            assert mock_parallelize_module.call_count == 0
        else:
            mock_parallelize_module.assert_called_once()
            applied_plan = mock_parallelize_module.call_args.args[2]
            assert set(applied_plan) == set(NEMOTRON_H_TP_PLAN)
            assert "backbone.layers.*.mixer.up_proj" in applied_plan
            assert "model.layers.*.mixer.down_proj" in applied_plan

        # Every decoder block is sharded by dtype (``shard_by_dtype``), the root by ``fully_shard``.
        assert fully_shard_by_dtype.call_count == len(nemotron_model.backbone.layers)
        assert [c.args[0] for c in fully_shard_by_dtype.call_args_list] == list(nemotron_model.backbone.layers)
        assert fully_shard.call_count == 1

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    @patch("nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype")
    def test_threads_reshard_after_forward_to_layer_sharding(
        self,
        fully_shard_by_dtype,
        fully_shard,
        strategy,
        mock_device_mesh,
        nemotron_model,
    ):
        """Nemotron layers must honor explicit FSDP reshard overrides."""
        mesh, _, _, _ = mock_device_mesh
        fully_shard.side_effect = lambda model, **kwargs: model
        fully_shard_by_dtype.side_effect = lambda model, *args, **kwargs: model

        strategy.parallelize(
            model=nemotron_model,
            device_mesh=mesh,
            activation_checkpointing=False,
            reshard_after_forward=True,
        )

        assert fully_shard_by_dtype.call_count == len(nemotron_model.backbone.layers)
        for call_args in fully_shard_by_dtype.call_args_list:
            assert call_args.kwargs["reshard_after_forward"] is True

    @patch("nemo_automodel.components.distributed.activation_checkpointing.checkpoint_wrapper")
    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    @patch("nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype")
    def test_activation_checkpointing(
        self,
        mock_fully_shard_by_dtype,
        mock_fully_shard,
        mock_checkpoint,
        strategy,
        mock_device_mesh,
        nemotron_model,
    ):
        """Whole MLP and Mamba blocks are checkpointed; attention blocks are left unwrapped."""
        mesh, _, dp_shard_mesh, tp_mesh = mock_device_mesh
        mock_fully_shard.side_effect = lambda model, **kwargs: model
        mock_fully_shard_by_dtype.side_effect = lambda model, *args, **kwargs: model
        mock_checkpoint.side_effect = lambda x, **_kwargs: x

        mamba_layer = nn.Module()
        mamba_layer.block_type = "mamba"
        mamba_layer.mixer = nn.Linear(10, 10)
        attention_layer = nn.Module()
        attention_layer.block_type = "attention"
        attention_layer.mixer = nn.Linear(10, 10)
        nemotron_model.backbone.layers.append(mamba_layer)
        nemotron_model.backbone.layers.append(attention_layer)

        strategy.parallelize(
            model=nemotron_model,
            device_mesh=mesh,
            activation_checkpointing=True,
        )

        wrapped = [c.args[0] for c in mock_checkpoint.call_args_list]
        assert len(wrapped) == 3  # 2 MLP (from MockNemotronHModel) + 1 Mamba layer
        assert mamba_layer in wrapped
        assert attention_layer not in wrapped
        assert all(c.kwargs["checkpoint_impl"] is not None for c in mock_checkpoint.call_args_list)


class TestQwen3_5ParallelizationStrategy:
    """Test the Qwen3.5 dtype-based FSDP strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a Qwen3_5ParallelizationStrategy instance."""
        return Qwen3_5ParallelizationStrategy()

    @pytest.mark.parametrize(
        "frozen_multimodal_sharding, expected_ignored, expected_vision_sharded",
        [
            ("root", False, False),
            ("per_layer", False, True),
            ("replicate", True, False),
        ],
    )
    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    @patch("nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype")
    def test_frozen_multimodal_modules_are_not_separately_sharded(
        self,
        fully_shard_by_dtype,
        fully_shard,
        strategy,
        mock_device_mesh,
        frozen_multimodal_sharding,
        expected_ignored,
        expected_vision_sharded,
    ):
        """Qwen3.5 applies all three frozen multimodal policies."""

        class MockQwen35Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10)])
                self.vision_tower = nn.Module()
                self.vision_tower.layers = nn.ModuleList([nn.Linear(10, 10)])

        class MockQwen35Model(nn.Module):
            parallel_spec = QWEN3_5_PARALLEL_SPEC

            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8, hidden_size=64)
                self.model = MockQwen35Inner()

        mesh, _, _, _ = mock_device_mesh
        model = MockQwen35Model()
        for param in model.model.vision_tower.parameters():
            param.requires_grad_(False)
        frozen_vision_params = set(model.model.vision_tower.parameters())
        fully_shard.side_effect = lambda model, **kwargs: model
        fully_shard_by_dtype.side_effect = lambda model, *args, **kwargs: model

        result = strategy.parallelize(
            model=model,
            device_mesh=mesh,
            frozen_multimodal_sharding=frozen_multimodal_sharding,
        )

        sharded_by_dtype = [call_args.args[0] for call_args in fully_shard_by_dtype.call_args_list]
        assert result is model
        assert model.model.layers[0] in sharded_by_dtype
        assert (model.model.vision_tower.layers[0] in sharded_by_dtype) is expected_vision_sharded
        root_kwargs = fully_shard.call_args_list[-1].kwargs
        if expected_ignored:
            assert root_kwargs["ignored_params"] == frozen_vision_params
        else:
            assert "ignored_params" not in root_kwargs

    def test_spec_opts_into_dtype_aware_sharding_with_the_default_flow(self):
        assert QWEN3_5_PARALLEL_SPEC.shard_by_dtype is True
        assert QWEN3_5_PARALLEL_SPEC.tp_plan is None
        assert QWEN3_5_PARALLEL_SPEC.layer_groups is None
        assert isinstance(QWEN3_5_PARALLEL_SPEC.strategy, DefaultParallelizationStrategy)

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    @patch("nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype")
    def test_hands_the_cp_mesh_to_the_model_after_the_default_flow(
        self, fully_shard_by_dtype, fully_shard, strategy, mock_device_mesh, monkeypatch
    ):
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        cp_mesh = MagicMock()
        cp_mesh.size.return_value = 2
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "cp", "tp")
        mesh.__getitem__.side_effect = lambda key: {
            "dp_replicate": dp_replicate_mesh,
            "dp_shard_cp": dp_shard_mesh,
            "cp": cp_mesh,
            "tp": tp_mesh,
            ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,
        }[key]
        fully_shard.side_effect = lambda model, **kwargs: model
        fully_shard_by_dtype.side_effect = lambda model, *args, **kwargs: model
        monkeypatch.setattr(parallelizer_mod.parallelizer_utils, "configure_fsdp_unused_param_reduction", lambda m: 0)

        model = MockModel("Qwen3_5ForCausalLM")
        strategy.parallelize(model=model, device_mesh=mesh)

        assert model.cp_mesh is cp_mesh


class TestStrategyRegistry:
    """Test the strategy registry functionality."""

    def test_registry_contains_nemotron_strategy(self):
        """Test that the registry contains NemotronH strategy."""
        assert isinstance(NEMOTRON_H_PARALLEL_SPEC.strategy, NemotronHParallelizationStrategy)

    def test_default_strategy_exists(self):
        """Test that the default strategy exists."""
        assert _DEFAULT_STRATEGY is not None
        assert isinstance(_DEFAULT_STRATEGY, DefaultParallelizationStrategy)

    def test_get_parallelization_strategy_for_nemotron(self):
        """Test strategy selection for NemotronH model."""
        model = MockNemotronHModel()
        strategy = get_parallelization_strategy(model)

        assert isinstance(strategy, NemotronHParallelizationStrategy)

    def test_get_parallelization_strategy_for_regular_model(self):
        """Test strategy selection for regular models."""
        model = MockModel("RegularModel")
        strategy = get_parallelization_strategy(model)

        assert isinstance(strategy, DefaultParallelizationStrategy)
        assert strategy is _DEFAULT_STRATEGY

    def test_get_parallelization_strategy_unknown_model(self):
        """Test strategy selection for unknown model types."""
        model = MockModel("UnknownModelType")
        strategy = get_parallelization_strategy(model)

        assert isinstance(strategy, DefaultParallelizationStrategy)
        assert strategy is _DEFAULT_STRATEGY


class TestWanDeclaration:
    """The diffusers Wan transformer declares a TP plan and whole-block checkpointing; the shared flow applies them."""

    @pytest.fixture
    def wan_model(self):
        class ConditionEmbedder(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_embedder = nn.Linear(8, 8)
                self.time_embedder = nn.Linear(8, 8)
                self.time_proj = nn.Linear(8, 8)

        class WanTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.ffn = nn.Linear(8, 8)

        class WanModel(nn.Module):
            _no_split_modules = ["WanTransformerBlock"]
            parallel_spec = WanTransformer3DModel.parallel_spec
            activation_checkpointing_spec = WanTransformer3DModel.activation_checkpointing_spec

            def __init__(self):
                super().__init__()
                self.condition_embedder = ConditionEmbedder()
                self.blocks = nn.ModuleList([WanTransformerBlock(), WanTransformerBlock()])
                self.proj_out = nn.Linear(8, 8)

        return WanModel()

    @pytest.fixture
    def mesh_tp2(self):
        mesh = MagicMock()
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "tp")
        tp_mesh = MagicMock()
        tp_mesh.size.return_value = 2
        dp_mesh = MagicMock()
        dp_mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        mesh.__getitem__.side_effect = lambda key: {
            "tp": tp_mesh,
            ("dp_replicate", "dp_shard_cp"): dp_mesh,
        }[key]
        return mesh, dp_mesh, tp_mesh

    def test_declares_the_plan_and_whole_block_checkpointing_only(self):
        spec = WanTransformer3DModel.parallel_spec
        assert spec.tp_plan is WAN_TP_PLAN
        assert spec.strategy is None and spec.layer_groups is None
        assert set(WAN_TP_PLAN) == {
            "condition_embedder.text_embedder.linear_1",
            "condition_embedder.text_embedder.linear_2",
            "condition_embedder.time_embedder.linear_1",
            "condition_embedder.time_embedder.linear_2",
            "condition_embedder.time_proj",
            "blocks.*.ffn.net.0.proj",
            "blocks.*.ffn.net.2",
            "proj_out",
        }
        assert WanTransformer3DModel.activation_checkpointing_spec.granularity == "layer"

    def test_blocks_form_the_backbone_layer_group(self, wan_model):
        assert get_model_layer_groups(wan_model) == {"backbone": list(wan_model.blocks)}

    def test_shared_flow_applies_the_declared_plan_without_head_validation(self, wan_model, mesh_tp2, monkeypatch):
        mesh, dp_mesh, tp_mesh = mesh_tp2
        parallelize_module_mock = MagicMock()
        validate_tp_mock = MagicMock()
        apply_fsdp_mock = MagicMock()
        monkeypatch.setattr(parallelizer_mod, "parallelize_module", parallelize_module_mock)
        monkeypatch.setattr(parallelizer_mod, "validate_tp_mesh", validate_tp_mock)
        monkeypatch.setattr(parallelizer_mod, "apply_fsdp2_sharding_recursively", apply_fsdp_mock)
        monkeypatch.setattr(parallelizer_mod, "fully_shard", lambda model, **_kwargs: model)

        result = fsdp2_strategy_parallelize(model=wan_model, device_mesh=mesh)

        assert result is wan_model
        parallelize_module_mock.assert_called_once()
        applied_model, applied_mesh, applied_plan = parallelize_module_mock.call_args.args
        assert applied_model is wan_model and applied_mesh is tp_mesh
        assert set(applied_plan) == set(WAN_TP_PLAN)
        # The plan shards no attention heads, so the head-count check does not apply.
        validate_tp_mock.assert_not_called()
        assert apply_fsdp_mock.call_args.args[:2] == (wan_model, dp_mesh)


class TestDiffusersBlockCheckpointingDeclarations:
    """Hunyuan-1.5 and LTX-2 only declare whole-block checkpointing; the shared flow wraps their blocks."""

    def test_hunyuan_declares_only_whole_block_checkpointing(self):
        assert not hasattr(HunyuanVideo15Transformer3DModel, "parallel_spec")
        assert HunyuanVideo15Transformer3DModel.activation_checkpointing_spec.granularity == "layer"

    def test_ltx2_names_its_block_container_and_whole_block_checkpointing(self):
        # diffusers declares no ``_no_split_modules`` for LTX-2, so the container is declared.
        assert LTX2VideoTransformer3DModel.parallel_spec.layer_groups == {"backbone": ("transformer_blocks",)}
        assert LTX2VideoTransformer3DModel.activation_checkpointing_spec.granularity == "layer"

    def test_shared_flow_wraps_whole_blocks_and_threads_prefetch_options(self, monkeypatch):
        class HunyuanVideo15TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(2, 2)

        class HunyuanModel(nn.Module):
            _no_split_modules = ["HunyuanVideo15TransformerBlock"]
            activation_checkpointing_spec = HunyuanVideo15Transformer3DModel.activation_checkpointing_spec

            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [HunyuanVideo15TransformerBlock(), HunyuanVideo15TransformerBlock()]
                )

        model = HunyuanModel()
        mesh = MagicMock()
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "tp")
        tp_mesh = MagicMock()
        tp_mesh.size.return_value = 1
        dp_mesh = MagicMock()
        dp_mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        mesh.__getitem__.side_effect = lambda key: {"tp": tp_mesh, ("dp_replicate", "dp_shard_cp"): dp_mesh}[key]
        checkpoint_wrapper_mock = MagicMock(side_effect=lambda module, **_kwargs: module)
        apply_fsdp_mock = MagicMock()
        fully_shard_mock = MagicMock(side_effect=lambda model, **_kwargs: model)
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.activation_checkpointing.checkpoint_wrapper",
            checkpoint_wrapper_mock,
        )
        monkeypatch.setattr(parallelizer_mod, "apply_fsdp2_sharding_recursively", apply_fsdp_mock)
        monkeypatch.setattr(parallelizer_mod, "fully_shard", fully_shard_mock)

        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            activation_checkpointing=True,
            enable_fsdp2_prefetch=False,
            fsdp2_backward_prefetch_depth=5,
            fsdp2_forward_prefetch_depth=4,
        )

        assert result is model
        assert [c.args[0] for c in checkpoint_wrapper_mock.call_args_list] == list(model.transformer_blocks)
        apply_fsdp_mock.assert_called_once()
        assert apply_fsdp_mock.call_args.args[:2] == (model, dp_mesh)
        assert apply_fsdp_mock.call_args.args[4:7] == (False, 5, 4)
        fully_shard_mock.assert_called_once()


class TestFsdp2StrategyParallelizeIntegration:
    """Test the main fsdp2_strategy_parallelize function with the new strategy pattern."""

    def test_delegates_to_strategy(self, mock_device_mesh, mock_distributed_env):
        """Test that fsdp2_strategy_parallelize delegates to the appropriate strategy."""
        mesh, _, _, _ = mock_device_mesh

        # Test with regular model (should use default strategy)
        model = MockModel("RegularModel")

        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        assert result is model
        # Verify that default strategy functions were called
        mock_distributed_env["extract_layer_groups"].assert_called_once_with(model)

    @patch("nemo_automodel.components.distributed.parallelizer.parallelize_module")
    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    @patch("nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype")
    def test_delegates_to_nemotron_strategy(
        self, fully_shard_by_dtype, fully_shard, mock_parallelize_module, mock_device_mesh
    ):
        """fsdp2_strategy_parallelize uses the NemotronH strategy, which runs the shared flow dtype-aware."""
        mesh, _, _, _ = mock_device_mesh
        fully_shard.side_effect = lambda model, **kwargs: model
        fully_shard_by_dtype.side_effect = lambda model, *args, **kwargs: model
        model = MockNemotronHModel()

        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        assert result is model
        assert fully_shard_by_dtype.call_count == len(model.backbone.layers)

    def test_backward_compatibility_arguments(self, mock_device_mesh, mock_distributed_env):
        """Test that all original function arguments are still supported."""
        mesh, _, _, _ = mock_device_mesh
        model = MockModel("RegularModel")

        # Test with all possible arguments
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            mp_policy=None,
            offload_policy=None,
            sequence_parallel=False,
            activation_checkpointing=True,
            tp_shard_plan=None,
            dp_replicate_mesh_name="dp_replicate",
            dp_shard_cp_mesh_name="dp_shard_cp",
            tp_mesh_name="tp",
        )

        assert result is model

    def test_preserves_function_signature(self):
        """Test that the main function preserves its original signature."""
        import inspect

        sig = inspect.signature(fsdp2_strategy_parallelize)

        # Check that all expected parameters are present
        expected_params = [
            "model",
            "device_mesh",
            "mp_policy",
            "offload_policy",
            "sequence_parallel",
            "activation_checkpointing",
            "activation_checkpointing_scope",
            "tp_shard_plan",
            "dp_replicate_mesh_name",
            "dp_shard_cp_mesh_name",
            "tp_mesh_name",
            "frozen_multimodal_sharding",
            "reapply_trainability",
        ]

        for param in expected_params:
            assert param in sig.parameters

        # Check default values are preserved
        assert sig.parameters["sequence_parallel"].default is False
        assert sig.parameters["activation_checkpointing"].default is False
        assert sig.parameters["dp_replicate_mesh_name"].default == "dp_replicate"
        assert sig.parameters["dp_shard_cp_mesh_name"].default == "dp_shard_cp"
        assert sig.parameters["tp_mesh_name"].default == "tp"
        assert sig.parameters["frozen_multimodal_sharding"].default == "root"


class TestStrategyExtensibility:
    """Test the extensibility of the strategy pattern."""

    def test_can_add_new_strategy_to_registry(self):
        """Test that new strategies can be added to the registry."""

        # Create a custom strategy
        class CustomStrategy(ParallelizationStrategy):
            def parallelize(self, model, device_mesh, **kwargs):
                return model

        custom_strategy = CustomStrategy()

        # Declare it on the model class
        model = MockModel("CustomModel")
        with patch.object(type(model), "parallel_spec", ParallelSpec(strategy=custom_strategy), create=True):
            strategy = get_parallelization_strategy(model)

        assert strategy is custom_strategy
        assert isinstance(strategy, CustomStrategy)

    def test_strategy_isolation(self):
        """Test that strategies are isolated and don't interfere with each other."""
        # Get strategies for different models
        regular_model = MockModel("RegularModel")
        nemotron_model = MockNemotronHModel()

        regular_strategy = get_parallelization_strategy(regular_model)
        nemotron_strategy = get_parallelization_strategy(nemotron_model)

        # Strategies should be different instances
        assert regular_strategy is not nemotron_strategy
        assert type(regular_strategy) != type(nemotron_strategy)

        # Both should be proper strategy objects
        assert isinstance(regular_strategy, ParallelizationStrategy)
        assert isinstance(nemotron_strategy, ParallelizationStrategy)


class TestDeciLMNemotronNASValidation:
    """Tests for DeciLM nemotron-nas special validation path in validate_tp_mesh."""

    def _make_decilm_nas_model(
        self,
        *,
        num_attention_heads=8,
        num_hidden_layers=3,
        block_kinds=("linear", "group", "noop"),
        n_heads_in_group=2,
        num_key_value_heads=3,
    ):
        """Create a minimal mock model/config for DeciLM nemotron-nas branch.

        num_key_value_heads is intentionally allowed to be incompatible with TP so
        that the generic path would fail if reached; the DeciLM branch should bypass it.
        """
        # Build block_configs with attention attributes
        blocks = []
        for kind in block_kinds[:num_hidden_layers]:
            if kind == "linear":
                attn = SimpleNamespace(replace_with_linear=True, n_heads_in_group=None, no_op=False)
            elif kind == "group":
                attn = SimpleNamespace(replace_with_linear=False, n_heads_in_group=n_heads_in_group, no_op=False)
            elif kind == "noop":
                attn = SimpleNamespace(replace_with_linear=False, n_heads_in_group=None, no_op=True)
            else:
                attn = SimpleNamespace(replace_with_linear=False, n_heads_in_group=None, no_op=True)
            blocks.append(SimpleNamespace(attention=attn))

        config = SimpleNamespace(
            architectures=["DeciLMForCausalLM"],
            model_type="nemotron-nas",
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            block_configs=blocks,
            num_key_value_heads=num_key_value_heads,
        )

        class _M(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.config = cfg

        return _M(config)

    def test_nemotron_nas_strategy_validates_before_the_default_flow(self, monkeypatch):
        """The per-layer block-config validation runs in the strategy, then the default flow is delegated to."""
        model = self._make_decilm_nas_model()
        validated = []
        monkeypatch.setattr(
            nas_parallelization, "validate_tp_mesh_for_nemotron_nas", lambda m, tp_size: validated.append((m, tp_size))
        )
        delegated = {}

        def fake_parallelize(self, m, device_mesh, **kwargs):
            delegated["model"] = m
            delegated["tp_mesh_name"] = kwargs.get("tp_mesh_name")
            return m

        monkeypatch.setattr(parallelizer_mod.DefaultParallelizationStrategy, "parallelize", fake_parallelize)
        device_mesh = MagicMock()
        device_mesh.mesh_dim_names = ("dp", "tp")
        device_mesh.__getitem__.return_value.size.return_value = 2

        assert NemotronNASParallelizationStrategy().parallelize(model, device_mesh) is model
        assert validated == [(model, 2)]
        assert delegated == {"model": model, "tp_mesh_name": "tp"}

        device_mesh.__getitem__.return_value.size.return_value = 1
        NemotronNASParallelizationStrategy().parallelize(model, device_mesh)
        assert len(validated) == 1  # nothing to validate at tp_size=1

    def test_validate_tp_mesh_for_nemotron_nas_valid_config_passes(self):
        # a valid config covering linear, grouped, and noop attention cases
        model = self._make_decilm_nas_model(
            num_attention_heads=8,
            num_hidden_layers=3,
            block_kinds=("linear", "group", "noop"),
            n_heads_in_group=2,
        )

        validate_tp_mesh_for_nemotron_nas(model, tp_size=2)


class TestQwenImageDeclaration:
    """The diffusers Qwen image transformer declares whole-block checkpointing; the shared flow applies it."""

    @staticmethod
    def _tiny_transformer():
        """Build a one-block upstream Qwen transformer without downloading weights."""
        diffusers = pytest.importorskip("diffusers")
        return bind_model_specs(
            diffusers.QwenImageTransformer2DModel(
                patch_size=2,
                in_channels=16,
                out_channels=4,
                num_layers=1,
                attention_head_dim=8,
                num_attention_heads=1,
                joint_attention_dim=12,
                axes_dims_rope=(2, 2, 4),
                zero_cond_t=True,
            )
        )

    def test_declaration_is_whole_block_checkpointing_on_the_default_strategy(self):
        assert not hasattr(QwenImageTransformer2DModel, "parallel_spec")
        assert QwenImageTransformer2DModel.activation_checkpointing_spec.granularity == "layer"

        import torch

        upstream = type("QwenImageTransformer2DModel", (torch.nn.Module,), {"config_name": "config.json"})
        module = bind_model_specs(upstream())
        assert query_activation_checkpointing_spec(module).granularity == "layer"
        assert get_parallelization_strategy(module) is _DEFAULT_STRATEGY

    def test_blocks_form_the_backbone_layer_group(self):
        model = self._tiny_transformer()
        assert get_model_layer_groups(model) == {"backbone": list(model.transformer_blocks)}

    def test_whole_block_checkpointing_preserves_canonical_state_dict(self):
        """Keep upstream Diffusers keys and every dual-stream branch parameter."""
        import torch
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        torch.manual_seed(9)
        model = self._tiny_transformer()
        expected_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

        apply_full_layer_checkpointing_to_layers(model, get_model_layer_groups(model)["backbone"])
        apply_full_layer_checkpointing_to_layers(model, get_model_layer_groups(model)["backbone"])

        assert isinstance(model.transformer_blocks[0], CheckpointWrapper)
        actual_state = model.state_dict()
        assert actual_state.keys() == expected_state.keys()
        for name, expected in expected_state.items():
            torch.testing.assert_close(actual_state[name], expected)

        expected_branch_parameters = {
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
            "transformer_blocks.0.txt_mlp.net.0.proj.weight",
        }
        assert expected_branch_parameters <= set(actual_state)

    def test_shared_flow_checkpoints_complete_blocks(self, monkeypatch):
        """Every dual-stream block is one checkpoint unit covering attention and both MLPs."""
        import torch
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        model = self._tiny_transformer()
        mesh = MagicMock()
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp", "tp")
        tp_mesh = MagicMock()
        tp_mesh.size.return_value = 1
        dp_mesh = MagicMock()
        dp_mesh.mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        mesh.__getitem__.side_effect = lambda key: {"tp": tp_mesh, ("dp_replicate", "dp_shard_cp"): dp_mesh}[key]
        monkeypatch.setattr(parallelizer_mod, "apply_fsdp2_sharding_recursively", MagicMock())
        monkeypatch.setattr(parallelizer_mod, "fully_shard", lambda model, **_kwargs: model)

        result = fsdp2_strategy_parallelize(model=model, device_mesh=mesh, activation_checkpointing=True)

        assert result is model
        wrapped_block = model.transformer_blocks[0]
        assert isinstance(wrapped_block, CheckpointWrapper)
        inner_block = wrapped_block._checkpoint_wrapped_module
        assert isinstance(inner_block.attn, torch.nn.Module)
        assert isinstance(inner_block.img_mlp, torch.nn.Module)
        assert isinstance(inner_block.txt_mlp, torch.nn.Module)
