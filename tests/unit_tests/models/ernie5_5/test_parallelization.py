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

"""Tests of the ERNIE 5.5 parallelization sidecar in one process.

The admission tests call ``parallelize_model`` with the registered model class, built on the meta device with
three released layers (two HCA layers and one DSA layer), and with every executor of the shared dispatcher
replaced by a recorder, so a rejection is shown to happen before any of them runs. The sharding tests run the
real FSDP stage of the shared expert-parallel flow with stand-ins for what needs several processes:

* ``fully_shard`` is a recorder that marks a module as an FSDP unit and notes the parameters it newly owns;
* the mesh is an object with the members that the sidecar and ``apply_fsdp`` read;
* the mesh context is a namespace with the sizes, the strategy and the precision policies;
* the number of ranks of the job is a value that the test sets;
* the expert-parallel executor is reduced to its FSDP stage.

The last test runs a small real model on CPU. FSDP2 over several ranks is covered by
``tests/functional_tests/moe/test_ernie5_5_parallelization.py``.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy

from nemo_automodel.components.distributed import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
    MoEParallelizerConfig,
    model_parallelizer,
    parallelizer_utils,
)
from nemo_automodel.components.distributed import parallelizer as distributed_parallelizer
from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer, parallelize_model
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5 import parallelization
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.hca import Ernie5_5HCAAttention
from nemo_automodel.components.models.ernie5_5.layers import Ernie5_5HyperHead
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.parallelization import PARALLELIZER
from nemo_automodel.components.moe import parallelizer as moe_parallelizer
from tests.unit_tests.models.ernie5_5.conftest import checkpoint_style_config

pytestmark = pytest.mark.cpu_only

RANKS = 16
# Compression ratios of the three layers: 128 selects an HCA layer, -2 a DSA layer.
WITH_DSA, HCA_ONLY = (128, 128, -2), (128, 128, 128)
BULK_POLICY = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
FP16_POLICY = MixedPrecisionPolicy(param_dtype=torch.float16)
HYPER_HEAD_PARAMETERS = ("hc_fn", "hc_base", "hc_scale")
EXECUTORS = ("_parallelize_moe", "_parallelize_fsdp2", "_parallelize_ddp", "_parallelize_megatron_fsdp")
Recorded = list[tuple[nn.Module, set[nn.Parameter], MixedPrecisionPolicy | None]]


class _Mesh:
    """Stand-in for the data-parallel mesh of the FSDP units."""

    def __init__(self, ndim: int = 1, ranks: int = RANKS) -> None:
        self.ndim = ndim
        self.ranks = ranks
        self.mesh_dim_names = ("dp_replicate", "dp_shard")[-ndim:]
        self.group = object()

    def get_group(self) -> object:
        """Return the process group of a one-dimensional mesh."""
        return self.group

    def size(self) -> int:
        """Return the number of ranks."""
        return self.ranks


def _set_world_size(monkeypatch: pytest.MonkeyPatch, ranks: int = RANKS) -> None:
    """Report a job of ``ranks`` processes to the sidecar; no process group exists in these tests."""
    monkeypatch.setattr(parallelization.dist, "get_world_size", lambda group=None: ranks)


def _backend() -> BackendConfig:
    return BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")


def _model(ratios: tuple[int, ...] = WITH_DSA) -> Ernie5_5_MoeForCausalLM:
    """Build the registered class on the meta device with three layers of the released shapes.

    Args:
        ratios: Compression ratio of each layer. The default gives the kinds of the released layers 0, 1 and 7.

    Returns:
        A model whose first layer has a dense MLP and whose other layers have an MoE; with the default ratios the
        layers are HCA, HCA and DSA.
    """
    values = checkpoint_style_config()
    values.update(num_hidden_layers=3, compress_ratios=list(ratios), num_nextn_predict_layers=0)
    with torch.device("meta"):
        return Ernie5_5_MoeForCausalLM(Ernie5_5Config(**values), _backend())


def _hybridep_model(dtype: str = "bfloat16") -> Ernie5_5_MoeForCausalLM:
    """Build the three released layers with the grouped-GEMM experts and the HybridEP dispatcher.

    The model is built in one process, where the shared MoE warns that HybridEP needs several and holds the
    experts without a token dispatcher. The sidecar reads the backend of the model, not the expert class.

    Args:
        dtype: Model dtype, ``"bfloat16"`` or ``"float32"``.

    Returns:
        A model on the meta device with an HCA layer and a dense MLP, an HCA layer and an MoE, and a DSA layer and
        an MoE.
    """
    values = checkpoint_style_config()
    values.update(num_hidden_layers=3, compress_ratios=list(WITH_DSA), num_nextn_predict_layers=0, torch_dtype=dtype)
    backend = BackendConfig(attn="eager", linear="torch", experts="torch_mm", dispatcher="hybridep")
    with torch.device("meta"), pytest.warns(UserWarning, match="'hybridep' dispatcher is enabled .* world size is 1"):
        return Ernie5_5_MoeForCausalLM(Ernie5_5Config(**values), backend)


def _context(**changes: object) -> SimpleNamespace:
    """Build the members of a mesh context that the sidecar and the dispatcher read before an executor runs."""
    values = {
        "strategy_config": FSDP2Config(),
        "moe_parallel_config": None,
        "device_mesh": _Mesh(),
        "moe_mesh": _Mesh(),
        "activation_checkpointing": False,
        "reapply_trainability": None,
        "ep_size": 4,
        "tp_size": 1,
        "cp_size": 1,
        "pp_size": 1,
        "dp_replicate_size": 1,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _record_executors(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    """Replace every executor of the shared dispatcher with a recorder."""
    executors = {name: Mock(name=name) for name in EXECUTORS}
    for name, executor in executors.items():
        monkeypatch.setattr(model_parallelizer, name, executor)
    return executors


def _record_sharding(monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """Replace ``fully_shard`` with a recorder of each unit's newly owned parameters and policy."""
    calls: Recorded = []

    def shard(module, *, mesh, mp_policy=None, ignored_params=None, **kwargs):
        """Record one ``fully_shard`` call and mark ``module`` as an FSDP unit.

        Args:
            module: Module that becomes a unit, with parameters of any shape.
            mesh: Stand-in mesh of the call; not read.
            mp_policy: Precision policy of the unit.
            ignored_params: Parameters of any shape that another unit already owns; they do not count as newly
                owned by this unit.
            **kwargs: Other arguments of ``fully_shard``; not read.

        Returns:
            ``module``, whose class is now a subclass of ``FSDPModule``.
        """
        assert not isinstance(module, FSDPModule), "An existing FSDP unit was sharded twice"
        owned = set(module.parameters()) - set(ignored_params or ())
        for child in module.modules():
            if child is not module and isinstance(child, FSDPModule):
                owned.difference_update(child.parameters())
        calls.append((module, owned, mp_policy))
        # ``fully_shard`` turns a module into a subclass of ``FSDPModule`` in place; the recorder does the same.
        module.__class__ = type(f"Recorded{type(module).__name__}", (FSDPModule, type(module)), {})
        return module

    for target in (parallelizer_utils, moe_parallelizer, distributed_parallelizer):
        monkeypatch.setattr(target, "fully_shard", shard)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_accumulated_grad_guard", lambda: None)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_uniform_reduce_dtype", lambda: None)
    # The recorder has no real mesh: the router correction-bias buffers stay local tensors.
    monkeypatch.setattr(moe_parallelizer, "distribute_tensor", lambda tensor, **kwargs: tensor)
    return calls


def _fsdp_stage(mesh: _Mesh) -> Mock:
    """Build a stand-in for the expert-parallel executor that runs only its FSDP stage, on ``mesh``."""

    def run(model, mesh_context, *, parallelizer):
        moe_parallelizer.apply_fsdp(
            model,
            mesh,
            ep_enabled=False,
            ep_shard_enabled=False,
            mp_policy=BULK_POLICY,
            reshard_after_forward=False,
            model_parallelizer=parallelizer if parallelizer._customizes_moe_fsdp else None,
        )
        return model

    return Mock(side_effect=run)


def _sync_groups(model: nn.Module) -> list[object]:
    return [
        module.compressor._hca_param_sync_group
        for module in model.modules()
        if isinstance(module, Ernie5_5HCAAttention)
    ]


def _assert_untouched(model: nn.Module, hca_layers: int = 2) -> None:
    assert not any(isinstance(module, FSDPModule) for module in model.modules())
    assert _sync_groups(model) == [None] * hca_layers


def test_registered_class_attaches_the_sidecar():
    assert Ernie5_5_MoeForCausalLM.parallelizer is PARALLELIZER
    assert get_model_parallelizer(_model()) is PARALLELIZER
    # The sidecar keeps the inherited entry point and uses the hooks of the contract only.
    assert "parallelize" not in vars(type(PARALLELIZER))


def test_admitted_topology_dispatches_once_and_builds_both_boundaries(monkeypatch):
    model, mesh = _model(), _Mesh()
    context = _context()
    _set_world_size(monkeypatch)
    calls = _record_sharding(monkeypatch)
    executor = _fsdp_stage(mesh)
    monkeypatch.setattr(model_parallelizer, "_parallelize_moe", executor)

    assert parallelize_model(model, context) is model

    executor.assert_called_once_with(model, context, parallelizer=PARALLELIZER)
    assert _sync_groups(model) == [mesh.group, mesh.group]
    head = model.model.hyper_head
    head_parameters = {getattr(head, name) for name in HYPER_HEAD_PARAMETERS}
    units = [module for module, _, _ in calls]
    owners = [(module, policy) for module, owned, policy in calls if owned & head_parameters]
    assert len(owners) == 1
    owner, policy = owners[0]
    assert owner is head and calls[units.index(head)][1] == head_parameters
    assert policy.param_dtype is torch.float32 and policy.reduce_dtype is torch.float32
    assert policy.cast_forward_inputs is False
    # The FP32 unit exists before the inner root, and each root is sharded once under the bulk policy.
    assert units.index(head) < units.index(model.model) < units.index(model)
    assert units.count(model.model) == units.count(model) == 1
    assert calls[units.index(model.model)][2].param_dtype is torch.bfloat16


def test_shared_flow_without_the_sidecar_leaves_the_hyper_head_in_the_root_unit(monkeypatch):
    """Control for the test above: the boundary and the groups come from the sidecar, not from the shared flow."""
    model, mesh = _model(), _Mesh()
    calls = _record_sharding(monkeypatch)
    moe_parallelizer.apply_fsdp(
        model, mesh, ep_enabled=False, ep_shard_enabled=False, mp_policy=BULK_POLICY, reshard_after_forward=False
    )
    owners = [(module, policy) for module, owned, policy in calls if model.model.hyper_head.hc_fn in owned]
    assert len(owners) == 1 and owners[0][0] is model.model
    assert owners[0][1].param_dtype is torch.bfloat16
    assert _sync_groups(model) == [None, None]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"strategy_config": DDPConfig(), "ep_size": 1}, "requires distributed.strategy fsdp2.*got DDPConfig"),
        (
            {"strategy_config": MegatronFSDPConfig(), "ep_size": 1},
            "requires distributed.strategy fsdp2.*got MegatronFSDPConfig",
        ),
        ({"strategy_config": DDPConfig()}, "requires distributed.strategy fsdp2.*got DDPConfig"),
        ({"tp_size": 2}, r"set these distributed sizes to 1: \{'tp_size': 2\}"),
        ({"cp_size": 2}, r"set these distributed sizes to 1: \{'cp_size': 2\}"),
        ({"pp_size": 2}, r"set these distributed sizes to 1: \{'pp_size': 2\}"),
        ({"dp_replicate_size": 2}, r"set these distributed sizes to 1: \{'dp_replicate_size': 2\}"),
        ({"tp_size": 2, "pp_size": 4}, r"\{'tp_size': 2, 'pp_size': 4\}"),
        ({"ep_size": 1}, "set distributed.ep_size to a value greater than 1, got 1"),
        ({"activation_checkpointing": True}, "set distributed.activation_checkpointing to false, got True"),
        (
            {"activation_checkpointing": "selective"},
            "set distributed.activation_checkpointing to false, got 'selective'",
        ),
        (
            {"strategy_config": FSDP2Config(mp_policy=FP16_POLICY)},
            r"set distributed\.mp_policy\.param_dtype to one of them, got torch\.float16",
        ),
        (
            {"moe_parallel_config": MoEParallelizerConfig(mp_policy=FP16_POLICY)},
            r"set distributed\.moe\.mp_policy\.param_dtype to one of them, got torch\.float16",
        ),
        ({"device_mesh": None}, "requires a data-parallel device mesh"),
        ({"device_mesh": _Mesh(ranks=8)}, "mesh over all 16 ranks of the job, got a mesh of 8 ranks"),
    ],
)
def test_unsupported_topology_is_rejected_before_any_executor(monkeypatch, changes, message):
    model = _model()
    _set_world_size(monkeypatch)
    executors = _record_executors(monkeypatch)
    with pytest.raises(ValueError, match=message):
        parallelize_model(model, _context(**changes))
    for executor in executors.values():
        executor.assert_not_called()
    _assert_untouched(model)


def test_rank_count_that_splits_dsa_heads_is_rejected_before_any_executor(monkeypatch):
    """The released shapes have 64 DSA heads, so 128 ranks cannot each hold whole heads in a checkpoint."""
    model = _model()
    _set_world_size(monkeypatch, 128)
    executors = _record_executors(monkeypatch)
    message = "divides hybrid_mla_num_attention_heads=64, .* got a job of 128 ranks"
    with pytest.raises(ValueError, match=message):
        parallelize_model(model, _context(device_mesh=_Mesh(ranks=128)))
    for executor in executors.values():
        executor.assert_not_called()
    _assert_untouched(model)


@pytest.mark.parametrize("trim", ["layers", "embed_tokens", "hyper_head", "norm", "lm_head", "multimax"])
def test_pipeline_stage_is_rejected_before_any_executor(monkeypatch, trim):
    model = _model()
    if trim == "layers":
        del model.model.layers[2]
    else:
        setattr(model.model, trim, None)
    executors = _record_executors(monkeypatch)
    # The mesh reports no pipeline axis, as in a call that passes a stage with a mesh of its own.
    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        parallelize_model(model, _context())
    for executor in executors.values():
        executor.assert_not_called()
    assert not any(isinstance(module, FSDPModule) for module in model.modules())


@pytest.mark.parametrize(
    ("ranks", "ratios", "changes"),
    [
        pytest.param(RANKS, WITH_DSA, {"ep_size": RANKS}, id="one_rank_per_expert_group"),
        pytest.param(64, WITH_DSA, {}, id="64_ranks_hold_whole_dsa_heads"),
        pytest.param(128, HCA_ONLY, {}, id="128_ranks_without_a_dsa_layer"),
        pytest.param(
            RANKS,
            WITH_DSA,
            {"strategy_config": FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32))},
            id="fp32_compute",
        ),
        pytest.param(
            RANKS,
            WITH_DSA,
            {"strategy_config": FSDP2Config(mp_policy=MixedPrecisionPolicy())},
            id="compute_in_the_storage_dtype",
        ),
    ],
)
def test_supported_topology_is_admitted_and_leaves_the_model_untouched(monkeypatch, ranks, ratios, changes):
    model = _model(ratios)
    _set_world_size(monkeypatch, ranks)
    context = _context(device_mesh=_Mesh(ranks=ranks), **changes)
    assert PARALLELIZER._prepare_model(model, context) is None
    _assert_untouched(model, hca_layers=ratios.count(128))


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({}, id="bf16_compute_of_the_default_policy"),
        pytest.param(
            {"strategy_config": FSDP2Config(mp_policy=MixedPrecisionPolicy())}, id="compute_in_the_storage_dtype"
        ),
        pytest.param({"moe_parallel_config": MoEParallelizerConfig(mp_policy=BULK_POLICY)}, id="bf16_moe_policy"),
    ],
)
def test_hybridep_dispatcher_with_bf16_compute_is_admitted(monkeypatch, changes):
    model = _hybridep_model()
    _set_world_size(monkeypatch)
    assert PARALLELIZER._prepare_model(model, _context(**changes)) is None
    _assert_untouched(model)


@pytest.mark.parametrize(
    ("dtype", "changes", "message"),
    [
        pytest.param(
            "bfloat16",
            {"strategy_config": FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32))},
            r"hybridep computes in bfloat16 only: set distributed\.mp_policy\.param_dtype to null or bfloat16, "
            r".* got torch\.float32",
            id="fp32_compute_of_the_strategy_policy",
        ),
        pytest.param(
            "bfloat16",
            {"moe_parallel_config": MoEParallelizerConfig(mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32))},
            r"hybridep computes in bfloat16 only: set distributed\.moe\.mp_policy\.param_dtype to null or bfloat16, "
            r".* got torch\.float32",
            id="fp32_compute_of_the_moe_policy",
        ),
        pytest.param(
            "float32",
            {},
            r"hybridep is validated with model\.dtype bfloat16 only: set model\.dtype to bfloat16, .* got "
            r"model\.dtype torch\.float32",
            id="fp32_model",
        ),
    ],
)
def test_hybridep_dispatcher_without_bf16_compute_is_rejected_before_any_executor(monkeypatch, dtype, changes, message):
    """HybridEP exchanges 16-bit activations only; the torch pair admits the same FP32 policy (``fp32_compute``)."""
    model = _hybridep_model(dtype)
    _set_world_size(monkeypatch)
    executors = _record_executors(monkeypatch)
    with pytest.raises(ValueError, match=message):
        parallelize_model(model, _context(**changes))
    for executor in executors.values():
        executor.assert_not_called()
    _assert_untouched(model)


def test_executor_that_bypasses_the_hook_is_reported(monkeypatch):
    model = _model()
    _set_world_size(monkeypatch)
    monkeypatch.setattr(model_parallelizer, "_parallelize_moe", Mock(return_value=model))
    with pytest.raises(RuntimeError, match="did not shard ERNIE 5.5 through its sidecar"):
        parallelize_model(model, _context())


def test_two_dimensional_mesh_is_rejected_by_the_hook(monkeypatch):
    model = _model()
    calls = _record_sharding(monkeypatch)
    with pytest.raises(ValueError, match=r"got a mesh with the dimensions \(dp_replicate, dp_shard\)"):
        PARALLELIZER._fully_shard_module(model.model, mesh=_Mesh(ndim=2), mp_policy=BULK_POLICY)
    assert calls == []
    _assert_untouched(model)


def test_hook_shards_a_hyper_head_passed_directly_once(monkeypatch):
    head, mesh = Ernie5_5HyperHead(2, 4), _Mesh()
    calls = _record_sharding(monkeypatch)
    assert PARALLELIZER._fully_shard_module(head, mesh=mesh, mp_policy=BULK_POLICY) is head
    assert PARALLELIZER._fully_shard_module(head, mesh=mesh, mp_policy=BULK_POLICY) is head
    assert [module for module, _, _ in calls] == [head]
    assert calls[0][2].param_dtype is torch.float32


SHORT, LONG, VOCAB = 64, 160, 64


def _tiny_model() -> Ernie5_5_MoeForCausalLM:
    """Build an FP32 model small enough for CPU: HCA with a dense MLP, HCA with an MoE and DSA with an MoE."""
    config = Ernie5_5Config(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        max_position_embeddings=1024,
        compress_ratios=[128, 128, -2],
        num_nextn_predict_layers=0,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=6,
        o_lora_rank=4,
        o_groups=2,
        sliding_window=8,
        hybrid_mla_num_attention_heads=2,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=4,
        hybrid_mla_qk_nope_head_dim=4,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        rope_interleave=True,
        hc_mult=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        moe_latent_size=8,
        pad_token_id=0,
        torch_dtype="float32",
    )
    model = Ernie5_5_MoeForCausalLM(config, _backend())
    torch.manual_seed(5583)
    model.initialize_weights(dtype=torch.float32)
    return model


def _parameters_without_gradient(model: Ernie5_5_MoeForCausalLM, length: int) -> set[str]:
    """Run one row of ``length`` tokens forward and backward and name the parameters that receive no gradient.

    Args:
        model: Model whose logits have shape [1, length, vocab].
        length: Number of tokens of the row.

    Returns:
        Names of the parameters whose gradient is None after backward.
    """
    model.zero_grad(set_to_none=True)
    tokens = (torch.arange(length) % (VOCAB - 1) + 1)[None]
    model(tokens).logits.square().sum().backward()
    return {name for name, parameter in model.named_parameters() if parameter.grad is None}


def test_only_the_parameters_the_sidecar_synchronizes_depend_on_the_row():
    """Guard the assumption under which the sidecar leaves FSDP's reduction of unused parameters off.

    FSDP reduces a parameter only on the ranks that hold a gradient for it, so a parameter that one row uses and
    another row does not would make ranks issue different collectives. The HCA compressor is such a module (a row
    shorter than one compression window does not use it), and the sidecar gives it a synchronization group. This
    test fails when any other trainable parameter receives a gradient for one row and none for another.
    """
    model = _tiny_model()
    synchronized = {
        f"{prefix}.compressor.{name}"
        for prefix, module in model.named_modules()
        if isinstance(module, Ernie5_5HCAAttention)
        for name, parameter in module.compressor.named_parameters()
        if parameter.requires_grad
    }
    assert len(synchronized) == 8
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    unused_short = _parameters_without_gradient(model, SHORT)
    unused_long = _parameters_without_gradient(model, LONG)
    assert unused_long == frozen
    assert unused_short - unused_long == synchronized
