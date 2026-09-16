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

"""CPU coverage for the existing qat_config boundary and preparation ordering."""

from collections import OrderedDict
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from torch import nn

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.quantization.qat import QATConfig, QATRule
from nemo_automodel.components.quantization.weight_qat import WeightFakeQuantizer, WeightQuantizationConfig


@pytest.fixture
def infrastructure():
    from nemo_automodel._transformers import infrastructure

    return infrastructure


@pytest.fixture
def recipe():
    from nemo_automodel.recipes.llm import train_ft

    return train_ft


def _qat_config():
    return QATConfig(rules=(QATRule(("projection",), WeightQuantizationConfig("mxfp4")),))


def _peft_config():
    return PeftConfig(match_all_linear=True, dim=4, alpha=8, use_memory_efficient_lora=False)


def _model(device="cpu"):
    return nn.Sequential(OrderedDict(projection=nn.Linear(32, 32, device=device), activation=nn.Tanh()))


def _qat_node(*, key="qat_config", delay=0):
    return ConfigNode(
        {
            "enabled": True,
            "fake_quant_after_n_steps": delay,
            key: {
                "_target_": "nemo_automodel.components.quantization.qat.QATConfig",
                "rules": [
                    {
                        "target_modules": ["projection"],
                        "weight": {
                            "format": "mxfp4",
                        },
                    }
                ],
            },
        }
    )


def test_nested_config_boundary_instantiates_typed_rules():
    config = _qat_node().qat_config.instantiate()
    assert config == _qat_config()
    assert isinstance(config.rules, tuple)
    assert isinstance(config.rules[0].weight, WeightQuantizationConfig)
    assert isinstance(config.build(), tuple)
    assert config.build()[0].weight.block_size == (1, 32)


@pytest.mark.parametrize("form", ["simple", "rules"])
def test_infrastructure_builds_independent_lora_controllers(infrastructure, form):
    rule = _qat_config().rules[0]
    config = (
        QATConfig(target_modules=rule.target_modules, weight=rule.weight)
        if form == "simple" else QATConfig(rules=(rule,))
    )
    with patch.object(QATConfig, "build", autospec=True, return_value=config.build()) as build:
        runtime = infrastructure._instantiate_qat(config)
        build.assert_called_once_with(config)
        assert isinstance(runtime, QAT)
    first = infrastructure.instantiate_infrastructure(qat_config=config)
    second = infrastructure.instantiate_infrastructure(qat_config=config)
    assert first[:3] == second[:3] == (None, None, None)
    assert isinstance(first[3], QAT)
    assert first[3] is not second[3]
    assert config.build() == _qat_config().build()
    assert rule.weight.block_size is None
    assert infrastructure._instantiate_qat(None) is None


@pytest.mark.parametrize("kind", [None, "int8_dynact_int4weight", "int4_weight_only"])
def test_legacy_qat_uses_config_owned_build(infrastructure, kind):
    config = QATConfig(kind)
    sentinel = object()
    with patch.object(config, "build", return_value=sentinel) as create:
        assert infrastructure._instantiate_qat(config) is sentinel
    create.assert_called_once_with()


@pytest.mark.parametrize("kind", [None, "int8_dynact_int4weight", "int4_weight_only"])
def test_infrastructure_preserves_legacy_quantizer_return_type(infrastructure, kind):
    config = QATConfig(kind)
    first = infrastructure._instantiate_qat(config)
    second = infrastructure._instantiate_qat(config)
    assert type(first) is type(second) is type(config.build())
    assert not isinstance(first, (QAT, tuple))
    assert first is not second
    assert first.precision == first.scales_precision == torch.bfloat16
    assert config.quantizer_kwargs == {}


@pytest.mark.parametrize("config", [object(), {}, ()])
def test_infrastructure_rejects_non_qat_config(infrastructure, config):
    with pytest.raises(TypeError, match="qat_config must be a QATConfig"):
        infrastructure._instantiate_qat(config)


def test_infrastructure_preserves_unknown_legacy_quantizer_error(infrastructure):
    with pytest.raises(ValueError, match="Unknown quantizer_type: unknown"):
        infrastructure._instantiate_qat(QATConfig("unknown"))


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_prepare_runs_after_real_peft_without_torchao_mode(infrastructure, device):
    model = _model(device)
    base = model[0].weight
    with (
        torch.device(device),
        patch("nemo_automodel.components.quantization.qat.prepare_qat_model") as old_prepare,
    ):
        result = infrastructure._apply_peft_and_lower_precision(
            model, 1, None, _peft_config(), None, None, QAT(_qat_config())
        )
    assert result is model
    assert isinstance(model[0], LinearLoRA)
    assert isinstance(model[0].weight_fake_quantizer, WeightFakeQuantizer)
    assert model[0].weight is base
    assert not base.requires_grad
    assert all(parameter.device.type == device for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert not hasattr(model, "_qat_mode")
    old_prepare.assert_not_called()
    if device == "cpu":
        model(torch.randn(2, 32)).square().mean().backward()
        assert base.grad is None
        assert torch.isfinite(model[0].lora_B.weight.grad).all()
        assert model[0].lora_B.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "override, message",
    [
        ({"peft_config": None}, "requires peft_config"),
        ({"fp8_config": object()}, "fp8_config"),
        ({"quantization_config": object()}, "quantization_config"),
        ({"tp_size": 2}, "tensor parallelism"),
        ({"autopipeline": object()}, "autopipeline"),
    ],
)
def test_rejects_incompatible_precision_before_peft_mutation(infrastructure, override, message):
    model = _model()
    base = model[0].weight
    peft = _peft_config()
    peft.use_triton = True
    kwargs = dict(
        tp_size=1,
        autopipeline=None,
        peft_config=peft,
        quantization_config=None,
        fp8_config=None,
        qat_quantizer=QAT(_qat_config()),
    )
    kwargs.update(override)
    with patch.object(infrastructure, "apply_lora_to_linear_modules") as apply_peft:
        with pytest.raises(ValueError, match=message):
            infrastructure._apply_peft_and_lower_precision(model, **kwargs)
    apply_peft.assert_not_called()
    assert type(model[0]) is nn.Linear
    assert model[0].weight is base and base.requires_grad
    assert peft.use_triton


@pytest.mark.parametrize("axis", ["tp_size", "cp_size", "pp_size"])
def test_mesh_axes_rejected_before_model_mutation(infrastructure, axis):
    model = _model()
    with (
        patch.object(infrastructure.MeshContext, axis, new_callable=PropertyMock, return_value=2),
        patch.object(infrastructure, "Checkpointer") as checkpointer,
        pytest.raises(ValueError, match="TP=CP=PP=1"),
    ):
        infrastructure.apply_model_infrastructure(
            model,
            is_meta_device=False,
            device=torch.device("cpu"),
            peft_config=_peft_config(),
            qat_quantizer=QAT(_qat_config()),
        )
    checkpointer.assert_not_called()
    assert type(model[0]) is nn.Linear


@pytest.mark.parametrize("manager_name", ["DDPManager", "MegatronFSDPManager"])
def test_unverified_managers_rejected_even_at_world_one(infrastructure, manager_name):
    manager = object.__new__(getattr(infrastructure, manager_name))
    with pytest.raises(ValueError, match="Megatron FSDP or DDP"):
        infrastructure.apply_model_infrastructure(
            _model(),
            is_meta_device=False,
            device=torch.device("cpu"),
            model_wrapper=manager,
            peft_config=_peft_config(),
            qat_quantizer=QAT(_qat_config()),
        )


def test_multi_rank_without_explicit_mesh_is_rejected(infrastructure):
    with (
        patch.object(infrastructure, "get_world_size_safe", return_value=2),
        pytest.raises(ValueError, match="FSDP2Manager and an explicit MeshContext"),
    ):
        infrastructure.apply_model_infrastructure(
            _model(),
            is_meta_device=False,
            device=torch.device("cpu"),
            peft_config=_peft_config(),
            qat_quantizer=QAT(_qat_config()),
        )


def _topology(world_size=2, ep_size=1, dp_replicate=1):
    """CPU-only topology doubles; these do not claim collective/parity coverage."""
    def named_mesh(axes):
        result = MagicMock()
        result.mesh_dim_names = tuple(axes)
        result.size.return_value = world_size
        result.__getitem__.side_effect = lambda axis: MagicMock(size=lambda: axes[axis])
        return result

    mesh = MeshContext(
        device_mesh=named_mesh({
            "pp": 1, "dp_replicate": dp_replicate, "dp_shard": world_size // dp_replicate, "cp": 1, "tp": 1,
        }),
        moe_mesh=named_mesh({"ep_shard": world_size // ep_size, "ep": ep_size}) if ep_size > 1 else None,
    )
    manager = FSDP2Manager(FSDP2Config(enable_compile=False), mesh.device_mesh, mesh.moe_mesh)
    return mesh, manager


@pytest.mark.parametrize("world_size, ep_size", [(2, 1), (32, 1), (2, 2), (32, 32), (64, 64)])
def test_supported_topologies_prepare_before_sharding(infrastructure, world_size, ep_size):
    mesh, _ = _topology(world_size, ep_size)
    manager, pipeline, parallelize_fn, quantizer = infrastructure.instantiate_infrastructure(
        distributed_config=FSDP2Config(enable_compile=False), mesh=mesh, qat_config=_qat_config(),
    )
    assert isinstance(manager, FSDP2Manager)
    assert pipeline is None
    model = _model()
    # Stop at the sharding boundary: real 2-H100 parity is covered separately.
    class ReachedSharding(Exception):
        pass

    def shard(prepared, *args):
        assert prepared is model
        assert isinstance(prepared[0], LinearLoRA)
        assert isinstance(prepared[0].weight_fake_quantizer, WeightFakeQuantizer)
        assert not prepared[0].weight.requires_grad
        raise ReachedSharding

    with (
        patch.object(infrastructure, "get_world_size_safe", return_value=world_size),
        patch.object(infrastructure, "_shard_ep_fsdp", side_effect=shard),
        pytest.raises(ReachedSharding),
    ):
        infrastructure.apply_model_infrastructure(
            model, is_meta_device=False, device=torch.device("cpu"), mesh=mesh, model_wrapper=manager,
            parallelize_fn=parallelize_fn,
            peft_config=_peft_config(), qat_quantizer=quantizer,
        )


@pytest.mark.parametrize("case, message", [
    ("hsdp", "dp_replicate_size must be 1"),
    ("extra_expert_shard", "ep_size=world_size"),
    ("missing_ep_parallelizer", "FSDP2 MoE parallelizer"),
    ("custom_dense_parallelizer", "not a custom parallelize_fn"),
    ("missing_manager", "FSDP2Manager"),
    ("mismatched_manager_mesh", "same meshes"),
    ("mismatched_manager_moe_mesh", "same meshes"),
    ("subset_mesh", "dp_shard_size=world_size"),
    ("oversized_moe_mesh", "additional ep_shard"),
    ("autopipeline", "no autopipeline"),
])
def test_unsupported_compositions_fail_before_preparation(infrastructure, case, message):
    world_size = 4 if case in ("hsdp", "extra_expert_shard") else 2
    ep_size = 2 if case in ("extra_expert_shard", "missing_ep_parallelizer", "oversized_moe_mesh") else 1
    mesh, manager = _topology(world_size, ep_size, dp_replicate=2 if case == "hsdp" else 1)
    kwargs = dict(mesh=mesh, model_wrapper=manager, parallelize_fn=MagicMock() if ep_size > 1 else None)
    if case == "missing_ep_parallelizer":
        kwargs["parallelize_fn"] = None
    elif case == "custom_dense_parallelizer":
        kwargs["parallelize_fn"] = MagicMock()
    elif case == "missing_manager":
        kwargs["model_wrapper"] = None
    elif case == "mismatched_manager_mesh":
        manager.device_mesh = _topology()[0].device_mesh
    elif case == "mismatched_manager_moe_mesh":
        manager.moe_mesh = _topology(ep_size=2)[0].moe_mesh
    elif case == "subset_mesh":
        world_size = 4
    elif case == "oversized_moe_mesh":
        mesh.moe_mesh.size.return_value = 4
    elif case == "autopipeline":
        kwargs["autopipeline"] = object()
    with (
        patch.object(infrastructure, "get_world_size_safe", return_value=world_size),
        patch.object(infrastructure, "_apply_peft_and_lower_precision") as prepare,
        patch.object(infrastructure, "_shard_ep_fsdp") as shard,
        patch.object(infrastructure, "Checkpointer") as checkpointer,
        pytest.raises(ValueError, match=message),
    ):
        infrastructure.apply_model_infrastructure(
            _model(), is_meta_device=False, device=torch.device("cpu"),
            peft_config=_peft_config(), qat_quantizer=QAT(_qat_config()), **kwargs,
        )
    prepare.assert_not_called()
    shard.assert_not_called()
    checkpointer.assert_not_called()


@pytest.mark.parametrize("expert_kind", ["ordinary", "qat"])
@pytest.mark.parametrize("backend", ["torch", "torch_mm", "mxfp8", "deepep", "hybridep"])
def test_ep_checks_actual_experts_before_sharding(infrastructure, expert_kind, backend):
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP

    config = MoEConfig(
        dim=32, inter_dim=32, moe_inter_dim=32, n_routed_experts=2, n_shared_experts=0,
        n_activated_experts=2, n_expert_groups=1, n_limited_groups=1, train_gate=False,
        gate_bias_update_factor=0.0, aux_loss_coeff=0.0, score_func="softmax", route_scale=1.0,
        norm_topk_prob=True, dtype=torch.float32,
    )
    if backend in ("deepep", "hybridep"):
        # No dispatcher initialization: rejection must happen before any collective.
        experts = GroupedExpertsDeepEP(config, dispatcher_backend=backend)
    else:
        experts = GroupedExperts(config, BackendConfig(
            experts="torch_mm_mxfp8" if backend == "mxfp8" else backend, dispatcher="torch",
        ))
    model = nn.ModuleDict({"projection": nn.Linear(32, 32), "experts": experts})
    targets = ["projection", "experts"] if expert_kind == "qat" else ["projection"]
    quantizer = QAT(QATConfig(target_modules=tuple(targets), weight=WeightQuantizationConfig("mxfp4")))
    mesh, manager = _topology(ep_size=2)
    with (
        patch.object(infrastructure, "get_world_size_safe", return_value=2),
        patch.object(infrastructure, "_shard_ep_fsdp", side_effect=lambda model, *args: model) as shard,
        # QAT.prepare permits this local DeepEP kernel; the high-level distributed
        # gate must independently reject the dispatcher even when prepare succeeds.
        patch("nemo_automodel._transformers.qat._HAS_GROUPED_GEMM", True),
    ):
        kwargs = dict(
            is_meta_device=False, device=torch.device("cpu"), mesh=mesh, model_wrapper=manager,
            parallelize_fn=MagicMock(), qat_quantizer=quantizer,
            peft_config=PeftConfig(target_modules=targets, dim=4, use_memory_efficient_lora=False),
        )
        if backend == "torch":
            assert infrastructure.apply_model_infrastructure(model, **kwargs) is model
            shard.assert_called_once()
            assert isinstance(model["experts"], GroupedExpertsLoRA if expert_kind == "qat" else GroupedExperts)
            assert isinstance(model["projection"].weight_fake_quantizer, WeightFakeQuantizer)
        else:
            with pytest.raises(ValueError, match="reference loop|MXFP8 expert backend"):
                infrastructure.apply_model_infrastructure(model, **kwargs)
            shard.assert_not_called()


def test_real_cpu_infrastructure_prepares_before_sharding(infrastructure):
    original_shard = infrastructure._shard_ep_fsdp

    def shard(model, *args):
        assert isinstance(model[0], LinearLoRA)
        assert isinstance(model[0].weight_fake_quantizer, WeightFakeQuantizer)
        return original_shard(model, *args)

    with patch.object(infrastructure, "_shard_ep_fsdp", side_effect=shard):
        model = infrastructure.apply_model_infrastructure(
            _model(),
            is_meta_device=False,
            device=torch.device("cpu"),
            peft_config=_peft_config(),
            qat_quantizer=QAT(_qat_config()),
        )
    base = model[0].weight.detach().clone()
    adapter = model[0].lora_B.weight.detach().clone()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    model(torch.randn(3, 32)).square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(model[0].weight, base, rtol=0, atol=0)
    assert not torch.equal(model[0].lora_B.weight, adapter)
    assert not hasattr(model, "_qat_mode")


def _trainability_model(kind, device="cpu"):
    if kind == "linear":
        return _model(device)

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP

    config = MoEConfig(
        dim=32, inter_dim=32, moe_inter_dim=32, n_routed_experts=2, n_shared_experts=0,
        n_activated_experts=2, n_expert_groups=1, n_limited_groups=1, train_gate=False,
        gate_bias_update_factor=0.0, aux_loss_coeff=0.0, score_func="softmax", route_scale=1.0,
        norm_topk_prob=True, dtype=torch.float32, expert_bias=True,
    )
    with torch.device(device):
        if kind == "deepep":
            experts = GroupedExpertsDeepEP(config, BackendConfig(experts="torch_mm", dispatcher="deepep"))
        else:
            experts = GroupedExperts(config, BackendConfig(experts="torch", dispatcher="torch"))
    return nn.ModuleDict({"projection": experts})


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("kind", ["linear", "grouped", "deepep"])
def test_qat_unfreeze_rejected_before_sharding(infrastructure, kind, device):
    model = _trainability_model(kind, device)
    quantizer = QAT(_qat_config())
    with (
        patch.object(infrastructure, "Checkpointer") as checkpointer,
        patch.object(infrastructure, "_shard_ep_fsdp", wraps=infrastructure._shard_ep_fsdp) as shard,
        patch.object(quantizer, "prepare", wraps=quantizer.prepare) as prepare,
        pytest.raises(ValueError, match="projection.*[Ff]ull-parameter.*QAT.*unsupported"),
    ):
        infrastructure.apply_model_infrastructure(
            model, is_meta_device=device == "meta", device=torch.device("cpu"),
            peft_config=PeftConfig(target_modules=["projection"], dim=4, use_memory_efficient_lora=False),
            qat_quantizer=quantizer,
            freeze_config={
                "freeze_modules": [{"path": "projection"}],
                "unfreeze_modules": [{"path": "projection"}],
            },
        )
    shard.assert_not_called()
    checkpointer.return_value.initialize_model_weights.assert_not_called()
    checkpointer.return_value.load_base_model.assert_not_called()
    prepare.assert_called_once()
    assert all(p.device.type == device for p in model.parameters())


@pytest.mark.parametrize("kind, parameter_name", [
    ("linear", "weight"), ("linear", "bias"),
    ("grouped", "gate_and_up_projs"), ("grouped", "down_projs"),
    ("grouped", "gate_up_proj_bias"), ("grouped", "down_proj_bias"),
    ("deepep", "gate_and_up_projs"), ("deepep", "down_projs"),
    ("deepep", "gate_up_proj_bias"), ("deepep", "down_proj_bias"),
])
def test_qat_validates_each_direct_base_parameter(infrastructure, kind, parameter_name):
    model = infrastructure._apply_peft_and_lower_precision(
        _trainability_model(kind), 1, None,
        PeftConfig(target_modules=["projection"], dim=4, use_memory_efficient_lora=False),
        None, None, QAT(_qat_config()),
    )
    model.get_parameter(f"projection.{parameter_name}").requires_grad_(True)
    # Preserve the rebound state: no PEFT reset or explicit freeze masks the violation.
    with pytest.raises(ValueError, match=rf"projection\.{parameter_name}.*[Ff]ull-parameter.*QAT.*unsupported"):
        infrastructure._apply_trainability_policy(model, peft_enabled=False, freeze_config=None, strict=False)


@pytest.mark.parametrize("stage", ["shard_callback", "checkpoint"])
def test_qat_revalidates_after_module_rebinding(infrastructure, stage):
    model = _model()
    model.add_module("later", nn.Linear(32, 32))
    quantizer = QAT(_qat_config())

    def rebind(current):
        current.later = current.projection
        del current.projection

    def shard(current, wrapper, parallelize_fn, mesh, reapply_trainability):
        if stage == "shard_callback":
            rebind(current)
            reapply_trainability(current)
        return current

    with (
        patch.object(infrastructure, "Checkpointer") as checkpointer,
        patch.object(infrastructure, "_shard_ep_fsdp", side_effect=shard),
        patch.object(quantizer, "prepare", wraps=quantizer.prepare) as prepare,
    ):
        checkpointer.return_value.load_base_model.side_effect = lambda current, *a, **kw: rebind(current)
        with pytest.raises(ValueError, match="later.weight.*[Ff]ull-parameter.*QAT.*unsupported"):
            infrastructure.apply_model_infrastructure(
                model, is_meta_device=False, device=torch.device("cpu"),
                peft_config=PeftConfig(target_modules=["projection"], dim=4, use_memory_efficient_lora=False),
                qat_quantizer=quantizer,
                freeze_config={"unfreeze_modules": [{"path": "later"}]},
                load_base_model=True, pretrained_model_name_or_path="unused-local-model",
            )
    prepare.assert_called_once()
    assert checkpointer.return_value.load_base_model.call_count == (stage == "checkpoint")


@pytest.mark.parametrize("kind", ["linear", "grouped", "deepep"])
@pytest.mark.parametrize("freeze_target", [False, True])
def test_qat_safe_trainability_policy_preserves_adapters(infrastructure, kind, freeze_target):
    model = _trainability_model(kind)
    with patch.object(infrastructure, "Checkpointer"):
        result = infrastructure.apply_model_infrastructure(
            model, is_meta_device=False, device=torch.device("cpu"),
            peft_config=PeftConfig(target_modules=["projection"], dim=4, use_memory_efficient_lora=False),
            qat_quantizer=QAT(_qat_config()),
            freeze_config={"freeze_modules": [{"path": "projection"}]} if freeze_target else None,
        )
    assert result is model
    for name, parameter in result.named_parameters():
        assert parameter.requires_grad == ("lora_" in name and not freeze_target)


@pytest.mark.parametrize("qat_enabled", [False, True])
def test_unfreeze_untargeted_modules_and_adapter_only_remains_supported(infrastructure, qat_enabled):
    model = _model()
    model.add_module("ordinary", nn.LayerNorm(32))
    model.add_module("unquantized", nn.Linear(32, 32))
    with patch.object(infrastructure, "Checkpointer"):
        infrastructure.apply_model_infrastructure(
            model, is_meta_device=False, device=torch.device("cpu"),
            peft_config=_peft_config(), qat_quantizer=QAT(_qat_config()) if qat_enabled else None,
            freeze_config={
                "freeze_modules": [{"path": "projection"}],
                "unfreeze_modules": [
                    {"path": "projection.lora_B"}, {"path": "ordinary"}, {"path": "unquantized"},
                ],
            },
        )
    assert not model.projection.weight.requires_grad
    assert not model.projection.bias.requires_grad
    assert not model.projection.lora_A.weight.requires_grad
    assert model.projection.lora_B.weight.requires_grad
    assert all(p.requires_grad for p in model.ordinary.parameters())
    assert isinstance(model.unquantized, LinearLoRA)
    assert model.unquantized.weight_fake_quantizer is None
    assert all(p.requires_grad for p in model.unquantized.parameters())


def test_legacy_qat_retains_bfloat16_requirement_and_mode(infrastructure):
    quantizer = object()
    model = _model()
    with pytest.raises(NotImplementedError, match="bfloat16"):
        infrastructure._apply_peft_and_lower_precision(model, 1, None, None, None, None, quantizer)
    model.bfloat16()
    with patch(
        "nemo_automodel.components.quantization.qat.prepare_qat_model", return_value=(model, "4w-qat")
    ) as prepare:
        result = infrastructure._apply_peft_and_lower_precision(model, 1, None, None, None, None, quantizer)
    prepare.assert_called_once_with(model, quantizer)
    assert result._qat_mode == "4w-qat"


@pytest.mark.parametrize("method", ["from_config", "from_pretrained"])
def test_auto_model_routes_lora_qat_without_distributed_strategy(infrastructure, method):
    from transformers import PretrainedConfig

    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM

    module = "nemo_automodel._transformers.auto_model"
    hf_config = PretrainedConfig()
    sentinel = _model()
    with (
        patch(f"{module}.get_hf_config", return_value=hf_config),
        patch(f"{module}.get_is_hf_model", return_value=True),
        patch(f"{module}.resolve_sdpa_method", return_value=[]),
        patch("torch.cuda.current_device", return_value=0),
        patch.object(NeMoAutoModelForCausalLM, "_build_model", return_value=sentinel) as build,
    ):
        result = getattr(NeMoAutoModelForCausalLM, method)(
            hf_config if method == "from_config" else "unused-local-model",
            qat_config=_qat_config(),
            peft_config=_peft_config(),
            trust_remote_code=False,
        )
    assert result is sentinel
    assert isinstance(build.call_args.kwargs["qat_quantizer"], QAT)
    assert build.call_args.kwargs["model_wrapper"] is None
    assert "qat_config" not in build.call_args.kwargs


@pytest.mark.parametrize("key", ["qat_config", "quantizer"])
@pytest.mark.parametrize("typed", [False, True])
def test_recipe_forwards_typed_qat_config_with_peft(recipe, key, typed):
    qat = ConfigNode({"enabled": True, key: _qat_config()}) if typed else _qat_node(key=key)
    cfg_model = ConfigNode({"_target_": recipe.NeMoAutoModelForCausalLM.from_config})
    peft = _peft_config()
    sentinel = _model()
    with patch.object(cfg_model, "instantiate", return_value=sentinel) as instantiate:
        assert recipe.build_model(cfg_model, peft, seed=42, cfg_qat=qat) is sentinel
    assert instantiate.call_args.kwargs["qat_config"] == _qat_config()
    assert instantiate.call_args.kwargs["peft_config"] is peft


@pytest.mark.parametrize("delay", [1, -1, None])
def test_recipe_rejects_lora_delay_before_model_build(recipe, delay):
    cfg_model = ConfigNode({"_target_": recipe.NeMoAutoModelForCausalLM.from_config})
    with patch.object(cfg_model, "instantiate") as instantiate:
        with pytest.raises(ValueError, match="fake_quant_after_n_steps=0"):
            recipe.build_model(cfg_model, _peft_config(), seed=42, cfg_qat=_qat_node(delay=delay))
    instantiate.assert_not_called()


@pytest.mark.parametrize("delay", [0, 1, -1, None])
def test_recipe_setup_does_not_silently_delay_lora_qat(recipe, delay):
    cfg = ConfigNode({"qat": _qat_node(delay=delay)})
    setup = recipe.TrainFinetuneRecipeForNextTokenPrediction._setup_qat
    # An empty parts list proves the new mode never reads old TorchAO markers.
    if delay == 0:
        assert setup(None, cfg, []) == (None, None, None)
    else:
        with pytest.raises(ValueError, match="fake_quant_after_n_steps=0"):
            setup(None, cfg, [])


@pytest.mark.parametrize("invalid", ["missing", "unsupported", "legacy_peft"])
def test_recipe_rejects_invalid_enabled_qat_before_build(recipe, invalid):
    from nemo_automodel.components.quantization.qat import QATConfig

    values = {"enabled": True}
    error, message = ValueError, "qat.enabled requires"
    if invalid == "unsupported":
        values["qat_config"] = object()
        error, message = TypeError, "QATConfig"
    elif invalid == "legacy_peft":
        values["qat_config"] = QATConfig()
        message = "QAT with PEFT"
    cfg_model = ConfigNode({"_target_": recipe.NeMoAutoModelForCausalLM.from_config})
    with patch.object(cfg_model, "instantiate") as instantiate:
        with pytest.raises(error, match=message):
            recipe.build_model(cfg_model, _peft_config(), seed=42, cfg_qat=ConfigNode(values))
    instantiate.assert_not_called()
