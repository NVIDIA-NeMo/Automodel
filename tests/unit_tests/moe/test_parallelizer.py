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

import sys
import types
from unittest.mock import MagicMock

import pytest


class DummyExperts:
    def __init__(self):
        self._params = {"weight": object()}

    def named_parameters(self, recurse=False):
        for name, param in self._params.items():
            yield name, param

    def register_parameter(self, name, param):
        self._params[name] = param

    def parameters(self):
        for p in self._params.values():
            yield p


class DummyMoE:
    def __init__(self):
        self.experts = DummyExperts()


class DummyBlock:
    def __init__(self, mlp=None):
        self.mlp = mlp if mlp is not None else DummyMoE()


class LayerContainer:
    def __init__(self, blocks):
        self._blocks = blocks
        self.registered = {}

    def named_children(self):
        return [(str(i), b) for i, b in enumerate(self._blocks)]

    def register_module(self, name, module):
        self.registered[name] = module


class DummyModel:
    def __init__(self, blocks, embed_tokens=None, lm_head=None):
        self.layers = LayerContainer(blocks)
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head


def _install_torch_and_layers_stubs(monkeypatch):
    # Build minimal torch stub hierarchy
    torch_stub = types.ModuleType("torch")

    # nn submodule
    nn_stub = types.ModuleType("torch.nn")

    class Parameter:
        def __init__(self, data=None):
            self.data = data

    class Module:
        pass

    nn_stub.Parameter = Parameter
    nn_stub.Module = Module
    torch_stub.nn = nn_stub

    # cuda submodule
    cuda_stub = types.ModuleType("torch.cuda")

    class Stream:
        def __init__(self):
            pass

    cuda_stub.Stream = Stream
    torch_stub.cuda = cuda_stub

    # distributed submodules and symbols
    dist_stub = types.ModuleType("torch.distributed")

    # device_mesh
    device_mesh_stub = types.ModuleType("torch.distributed.device_mesh")

    class DeviceMesh:
        def __init__(self, *args, **kwargs):
            pass

    device_mesh_stub.DeviceMesh = DeviceMesh

    # fsdp
    fsdp_stub = types.ModuleType("torch.distributed.fsdp")

    def fully_shard(*args, **kwargs):
        return None

    fsdp_stub.fully_shard = fully_shard

    fsdp_fully_stub = types.ModuleType("torch.distributed.fsdp._fully_shard")

    class MixedPrecisionPolicy:
        def __init__(self, *args, **kwargs):
            pass

    class OffloadPolicy:
        def __init__(self, *args, **kwargs):
            pass

    fsdp_fully_stub.MixedPrecisionPolicy = MixedPrecisionPolicy
    fsdp_fully_stub.OffloadPolicy = OffloadPolicy

    # tensor
    tensor_stub = types.ModuleType("torch.distributed.tensor")

    def distribute_module(*args, **kwargs):
        return "DISTRIBUTED"

    def distribute_tensor(*args, **kwargs):
        return object()

    class Shard:
        def __init__(self, *args, **kwargs):
            pass

    tensor_stub.distribute_module = distribute_module
    tensor_stub.distribute_tensor = distribute_tensor
    tensor_stub.Shard = Shard

    # tensor.parallel
    tp_stub = types.ModuleType("torch.distributed.tensor.parallel")

    class ParallelStyle:
        pass

    def parallelize_module(*args, **kwargs):
        return None

    tp_stub.ParallelStyle = ParallelStyle
    tp_stub.parallelize_module = parallelize_module

    # algorithms._checkpoint.checkpoint_wrapper
    alg_stub = types.ModuleType("torch.distributed.algorithms")
    alg_cp_stub = types.ModuleType("torch.distributed.algorithms._checkpoint")
    cpw_stub = types.ModuleType(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
    )

    def checkpoint_wrapper(*args, **kwargs):
        return args[0]

    cpw_stub.checkpoint_wrapper = checkpoint_wrapper

    # utils.checkpoint
    utils_checkpoint_stub = types.ModuleType("torch.utils.checkpoint")

    class CheckpointPolicy:
        MUST_SAVE = 1
        PREFER_RECOMPUTE = 2

    def create_selective_checkpoint_contexts(policy_factory):
        return "CTX"

    utils_checkpoint_stub.CheckpointPolicy = CheckpointPolicy
    utils_checkpoint_stub.create_selective_checkpoint_contexts = (
        create_selective_checkpoint_contexts
    )

    # ops.aten.mm.default sentinel
    aten = types.SimpleNamespace(mm=types.SimpleNamespace(default=object()))
    torch_stub.ops = types.SimpleNamespace(aten=aten)

    # dtype and device classes for type annotations
    class dtype:
        pass

    class device:
        pass

    torch_stub.dtype = dtype
    torch_stub.device = device

    # common dtypes referenced by code
    torch_stub.bfloat16 = object()
    torch_stub.float32 = object()

    # register into sys.modules via monkeypatch
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_stub)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed.device_mesh", device_mesh_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fsdp_stub)
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.fsdp._fully_shard",
        fsdp_fully_stub,
    )
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", tensor_stub)
    monkeypatch.setitem(
        sys.modules, "torch.distributed.tensor.parallel", tp_stub
    )
    monkeypatch.setitem(sys.modules, "torch.distributed.algorithms", alg_stub)
    monkeypatch.setitem(
        sys.modules, "torch.distributed.algorithms._checkpoint", alg_cp_stub
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
        cpw_stub,
    )
    monkeypatch.setitem(sys.modules, "torch.utils.checkpoint", utils_checkpoint_stub)

    # Stub heavy layers import as well to avoid real dependencies
    layers_stub = types.ModuleType(
        "nemo_automodel.components.moe.layers"
    )

    class GroupedExpertsDeepEP:
        pass

    class MoE:
        pass

    layers_stub.GroupedExpertsDeepEP = GroupedExpertsDeepEP
    layers_stub.MoE = MoE
    monkeypatch.setitem(
        sys.modules, "nemo_automodel.components.moe.layers", layers_stub
    )


def _import_parallelizer_with_stubs(monkeypatch):
    import importlib

    # ensure fresh import of parallelizer
    for mod in [
        "nemo_automodel.components.moe.parallelizer",
        "nemo_automodel.components.moe.layers",
    ]:
        if mod in sys.modules:
            sys.modules.pop(mod)

    _install_torch_and_layers_stubs(monkeypatch)
    return importlib.import_module(
        "nemo_automodel.components.moe.parallelizer"
    )


def test_expert_parallel_apply_calls_distribute_module(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    ep = P.ExpertParallel()
    module = DummyBlock().mlp.experts
    device_mesh = object()

    distribute_module_mock = MagicMock(return_value="DISTRIBUTED")
    monkeypatch.setattr(P, "distribute_module", distribute_module_mock)

    result = ep._apply(module, device_mesh)

    assert result == "DISTRIBUTED"
    assert distribute_module_mock.call_count == 1
    args, kwargs = distribute_module_mock.call_args
    # (module, device_mesh, partition_fn)
    assert args[0] is module
    assert args[1] is device_mesh
    assert callable(args[2])
    # ensure bound to same instance
    assert isinstance(args[2], types.MethodType) and args[2].__self__ is ep


def test_expert_parallel_partition_fn_shards_and_dispatcher(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    # make the target module also look like GroupedExpertsDeepEP
    class DummyGrouped(DummyExperts):
        def __init__(self):
            super().__init__()
            self.dispatch_called_with = None

        def init_token_dispatcher(self, ep_mesh):
            self.dispatch_called_with = ep_mesh

        # override register_parameter to avoid strict type checks
        def register_parameter(self, name, param):
            setattr(self, name, param)

    # patch GroupedExpertsDeepEP symbol used in isinstance checks
    monkeypatch.setattr(P, "GroupedExpertsDeepEP", DummyGrouped)

    # mock distribute_tensor and Shard
    shard_sentinel = object()

    def fake_shard(dim):
        assert dim == 0
        return shard_sentinel

    distributed_obj = object()
    distribute_tensor_mock = MagicMock(return_value=distributed_obj)
    monkeypatch.setattr(P, "Shard", fake_shard)
    monkeypatch.setattr(P, "distribute_tensor", distribute_tensor_mock)

    ep = P.ExpertParallel()
    module = DummyGrouped()
    device_mesh = type("Mesh", (), {"ndim": 1})()

    # original parameter should exist
    assert any(True for _ in module.named_parameters(recurse=False))
    ep._partition_fn("any", module, device_mesh)

    # verify distribute_tensor was called for each top-level parameter with Shard(0)
    for _, param in module.named_parameters(recurse=False):
        pass  # push iterator once for coverage; we validate calls below

    assert distribute_tensor_mock.call_count >= 1
    for args, kwargs in distribute_tensor_mock.call_args_list:
        assert args[1] is device_mesh
        assert isinstance(args[2], list) and args[2][0] is shard_sentinel

    # dispatcher must be initialized
    assert module.dispatch_called_with is device_mesh


def test_apply_ep_parallelizes_moe_experts(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    # Patch MoE symbol for isinstance
    monkeypatch.setattr(P, "MoE", DummyMoE)
    parallelize_module_mock = MagicMock()
    monkeypatch.setattr(P, "parallelize_module", parallelize_module_mock)

    block = DummyBlock(mlp=DummyMoE())
    model = DummyModel([block])
    ep_mesh = type("Mesh", (), {"size": lambda self: 2})()

    P.apply_ep(model, ep_mesh)

    assert parallelize_module_mock.call_count == 1
    args, kwargs = parallelize_module_mock.call_args
    assert kwargs["module"] is block.mlp.experts
    assert kwargs["device_mesh"] is ep_mesh
    assert isinstance(kwargs["parallelize_plan"], P.ExpertParallel)


def test_apply_ac_wraps_blocks_with_and_without_context(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    wrapper_returns = [object(), object()]

    def fake_wrapper(block, preserve_rng_state, context_fn=None):
        assert preserve_rng_state is True
        # if ignore_router=True, context_fn should be provided
        return wrapper_returns.pop(0)

    wrapper_mock = MagicMock(side_effect=fake_wrapper)
    ctx_mock = MagicMock(return_value="CTX")
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", wrapper_mock)
    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", ctx_mock)

    blocks = [DummyBlock(), DummyBlock()]
    model = DummyModel(blocks)

    # ignore_router=True path
    P.apply_ac(model, ignore_router=True)
    assert wrapper_mock.call_count == 2
    # registration should replace both blocks
    assert len(model.layers.registered) == 2

    # reset for ignore_router=False path
    wrapper_returns.extend([object(), object()])
    model = DummyModel([DummyBlock(), DummyBlock()])
    wrapper_mock.reset_mock()
    model.layers.registered.clear()

    P.apply_ac(model, ignore_router=False)
    # context_fn should not be passed (3rd arg remains default None)
    for _, kwargs in wrapper_mock.call_args_list:
        assert "context_fn" not in kwargs or kwargs["context_fn"] is None
    assert len(model.layers.registered) == 2


def test_apply_ac_custom_policy_respects_hidden_and_expert_dims(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_policy = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_policy
        captured_policy = policy_cb
        return "CTX"

    def fake_wrapper(block, preserve_rng_state, context_fn=None):
        assert preserve_rng_state is True
        assert callable(context_fn)
        assert context_fn() == "CTX"
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    hidden_size = 17
    num_experts = 31
    model = DummyModel([DummyBlock(), DummyBlock()])

    P.apply_ac(model, ignore_router=True, hidden_size=hidden_size, num_experts=num_experts)

    assert captured_policy is not None

    torch_stub = sys.modules["torch"]
    rhs_match = type("Mat", (), {"shape": (hidden_size, num_experts)})()
    rhs_mismatch = type("Mat", (), {"shape": (hidden_size, num_experts + 1)})()

    policy = captured_policy
    must_save = policy(None, torch_stub.ops.aten.mm.default, object(), rhs_match)
    prefer_recompute_shape = policy(None, torch_stub.ops.aten.mm.default, object(), rhs_mismatch)
    prefer_recompute_func = policy(None, object(), object(), rhs_match)

    assert must_save == P.CheckpointPolicy.MUST_SAVE
    assert prefer_recompute_shape == P.CheckpointPolicy.PREFER_RECOMPUTE
    assert prefer_recompute_func == P.CheckpointPolicy.PREFER_RECOMPUTE


def _find_call_by_first_arg(mock_obj, target_first_arg):
    for args, kwargs in mock_obj.call_args_list:
        if args and args[0] is target_first_arg:
            return args, kwargs
    return None


def test_apply_fsdp_calls_with_ignored_params_and_shard_for_experts(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    # Patch MoE symbol for isinstance
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    mp_policy_mock = MagicMock(return_value="MP_POLICY")
    shard_sentinel = object()

    def fake_shard(dim):
        assert dim == 1
        return shard_sentinel

    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", mp_policy_mock)
    monkeypatch.setattr(P, "Shard", fake_shard)

    block = DummyBlock(mlp=DummyMoE())
    embed = object()
    lm = object()
    model = DummyModel([block], embed_tokens=embed, lm_head=lm)

    fsdp_mesh = type("Mesh", (), {"size": lambda self: 2})()
    ep_shard_mesh = type("Mesh", (), {"size": lambda self: 2})()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        pp_enabled=True,
        ep_enabled=True,
        ep_shard_enabled=True,
        ep_shard_mesh=ep_shard_mesh,
    )

    # Experts should have a dedicated shard call
    experts = block.mlp.experts
    experts_call = _find_call_by_first_arg(fully_shard_mock, experts)
    assert experts_call is not None
    _, experts_kwargs = experts_call
    assert experts_kwargs["mesh"] is ep_shard_mesh
    assert experts_kwargs["reshard_after_forward"] is False  # pp_enabled=True -> not pp == False
    assert callable(experts_kwargs["shard_placement_fn"])  # lambda _: Shard(1)

    # Block should be sharded with ignored_params when ep_enabled
    block_call = _find_call_by_first_arg(fully_shard_mock, block)
    assert block_call is not None
    _, block_kwargs = block_call
    assert block_kwargs["mesh"] is fsdp_mesh
    assert block_kwargs["mp_policy"] == "MP_POLICY"
    ignored = block_kwargs.get("ignored_params")
    assert isinstance(ignored, set) and len(ignored) == len(list(experts.parameters()))

    # embed, lm_head and model should also be sharded on fsdp_mesh
    embed_call = _find_call_by_first_arg(fully_shard_mock, embed)
    assert embed_call is not None and embed_call[1]["mesh"] is fsdp_mesh

    lm_call = _find_call_by_first_arg(fully_shard_mock, lm)
    assert lm_call is not None and lm_call[1]["mesh"] is fsdp_mesh

    model_call = _find_call_by_first_arg(fully_shard_mock, model)
    assert model_call is not None and model_call[1]["mesh"] is fsdp_mesh


def test_apply_fsdp_without_ep_enabled_has_no_ignored_params(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    block = DummyBlock(mlp=DummyMoE())
    model = DummyModel([block])
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        pp_enabled=False,
        ep_enabled=False,
        ep_shard_enabled=False,
        ep_shard_mesh=None,
    )

    block_call = _find_call_by_first_arg(fully_shard_mock, block)
    assert block_call is not None
    _, block_kwargs = block_call
    assert block_kwargs["mesh"] is fsdp_mesh
    assert block_kwargs.get("ignored_params") is None


class MeshView:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class FakeWorldMesh:
    def __init__(self, sizes_by_key, mesh_dim_names):
        self._sizes = sizes_by_key
        self.mesh_dim_names = set(mesh_dim_names)

    def __getitem__(self, key):
        return MeshView(self._sizes[key])


class FakeMoeMesh:
    def __init__(self, sizes_by_key):
        self._sizes = sizes_by_key

    def __getitem__(self, key):
        return MeshView(self._sizes[key])


def test_parallelize_model_calls_subsystems_and_validates(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_ep_mock = MagicMock()
    apply_ac_mock = MagicMock()
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_ep", apply_ep_mock)
    monkeypatch.setattr(P, "apply_ac", apply_ac_mock)
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)

    world_mesh = FakeWorldMesh({("dp",): 2, "tp": 1, "cp": 1}, mesh_dim_names=["dp", "tp", "cp"])
    moe_mesh = FakeMoeMesh({"ep": 2, ("es1", "es2"): 2})

    # model.model.moe_config.n_routed_experts must be divisible by ep size (2)
    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        pp_enabled=True,
        dp_axis_names=("dp",),
        cp_axis_name=None,
        tp_axis_name=None,
        ep_axis_name="ep",
        ep_shard_axis_names=("es1", "es2"),
        activation_checkpointing=True,
    )
    apply_ep_mock.assert_called_once()
    # AC enabled
    apply_ac_mock.assert_called_once_with(model)
    # FSDP called with combined flags and derived meshes
    args, kwargs = apply_fsdp_mock.call_args
    # handle positional or keyword invocations
    fsdp_model = kwargs.get("model", args[0] if args else None)
    fsdp_mesh_arg = kwargs.get("fsdp_mesh", args[1] if len(args) > 1 else None)
    pp_enabled = kwargs.get("pp_enabled", args[2] if len(args) > 2 else None)
    ep_enabled = kwargs.get("ep_enabled", args[3] if len(args) > 3 else None)
    ep_shard_enabled = kwargs.get("ep_shard_enabled", args[4] if len(args) > 4 else None)
    ep_shard_mesh_arg = kwargs.get("ep_shard_mesh", args[5] if len(args) > 5 else None)

    assert fsdp_model is model
    assert fsdp_mesh_arg.size() == 2
    assert pp_enabled is True
    assert ep_enabled is True
    assert ep_shard_enabled is True
    assert ep_shard_mesh_arg.size() == 2


def test_parallelize_model_asserts_on_invalid_tp_cp_and_ep_divisibility(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    world_mesh_bad_tp = FakeWorldMesh({"tp": 2, "cp": 1}, mesh_dim_names=["tp", "cp"])
    moe_mesh = FakeMoeMesh({"ep": 2})

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 3})()  # not divisible by 2

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    # TP size != 1 -> assertion
    with pytest.raises(AssertionError):
        P.parallelize_model(
            model=model,
            world_mesh=world_mesh_bad_tp,
            moe_mesh=moe_mesh,
            pp_enabled=False,
            dp_axis_names=None,
            cp_axis_name=None,
            tp_axis_name="tp",
            ep_axis_name=None,
            ep_shard_axis_names=None,
            activation_checkpointing=False,
        )

    # EP enabled but divisibility violated -> assertion
    world_mesh_ok = FakeWorldMesh({("dp",): 1, "tp": 1, "cp": 1}, mesh_dim_names=["dp", "tp", "cp"])
    moe_mesh_ep = FakeMoeMesh({"ep": 2})
    with pytest.raises(AssertionError):
        P.parallelize_model(
            model=model,
            world_mesh=world_mesh_ok,
            moe_mesh=moe_mesh_ep,
            pp_enabled=False,
            dp_axis_names=("dp",),
            cp_axis_name=None,
            tp_axis_name=None,
            ep_axis_name="ep",
            ep_shard_axis_names=None,
            activation_checkpointing=False,
        )


def test_apply_fsdp_with_lm_head_precision_fp32(monkeypatch):
    """Test that apply_fsdp applies custom MixedPrecisionPolicy to lm_head when lm_head_precision is fp32."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    mp_policy_mock = MagicMock(return_value="MP_POLICY")
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", mp_policy_mock)

    torch_stub = sys.modules["torch"]
    block = DummyBlock(mlp=DummyMoE())
    lm = object()
    model = DummyModel([block], lm_head=lm)
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        pp_enabled=False,
        ep_enabled=False,
        ep_shard_enabled=False,
        lm_head_precision=torch_stub.float32,
    )

    # Find the lm_head call
    lm_call = _find_call_by_first_arg(fully_shard_mock, lm)
    assert lm_call is not None
    _, lm_kwargs = lm_call

    # Verify custom MixedPrecisionPolicy was created with fp32 for all dtypes
    assert mp_policy_mock.call_count >= 2  # default + lm_head
    # Find the call for lm_head's custom policy
    fp32_policy_calls = [
        call for call in mp_policy_mock.call_args_list
        if call[1].get("param_dtype") == torch_stub.float32
        and call[1].get("reduce_dtype") == torch_stub.float32
        and call[1].get("output_dtype") == torch_stub.float32
    ]
    assert len(fp32_policy_calls) == 1


def test_apply_fsdp_without_lm_head_precision_uses_default_policy(monkeypatch):
    """Test that apply_fsdp uses default MixedPrecisionPolicy for lm_head when lm_head_precision is None."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    mp_policy_mock = MagicMock(return_value="MP_POLICY")
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", mp_policy_mock)

    block = DummyBlock(mlp=DummyMoE())
    lm = object()
    model = DummyModel([block], lm_head=lm)
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        pp_enabled=False,
        ep_enabled=False,
        ep_shard_enabled=False,
        lm_head_precision=None,
    )

    # Find the lm_head call
    lm_call = _find_call_by_first_arg(fully_shard_mock, lm)
    assert lm_call is not None

    # Should only have one MixedPrecisionPolicy call (the default one)
    assert mp_policy_mock.call_count == 1


def test_parallelize_model_passes_lm_head_precision_to_apply_fsdp(monkeypatch):
    """Test that parallelize_model passes lm_head_precision to apply_fsdp."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_ac", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])
    moe_mesh = None

    torch_stub = sys.modules["torch"]

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        pp_enabled=False,
        dp_axis_names=("dp",),
        lm_head_precision=torch_stub.float32,
    )

    # Verify apply_fsdp was called with lm_head_precision
    apply_fsdp_mock.assert_called_once()
    _, kwargs = apply_fsdp_mock.call_args
    assert kwargs.get("lm_head_precision") == torch_stub.float32
