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

import hashlib
import importlib
import inspect
import time
from datetime import timedelta
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.distributed import megatron_fsdp_compat as compat


def make_fsdp_dtensor(
    local_tensor,
    param,
    dist_index,
    is_sharded_param=True,
    is_expert_param=False,
    run_check=False,
    update_uneven_dtensor_chunk_meta=False,
    force_sync_tp_duplicated_param=False,
):
    """Minimal source fixture containing the 0.5.0 reshape defect."""
    del (
        dist_index,
        is_sharded_param,
        is_expert_param,
        run_check,
        update_uneven_dtensor_chunk_meta,
        force_sync_tp_duplicated_param,
    )
    orig_param = param
    if len(orig_param.shape) > 1:
        local_shape = (-1, *orig_param.shape[1:])
    else:
        local_shape = (-1,)
    return local_tensor.view(local_shape)


class _StorageResizeBasedBucketAllocatorFixture:
    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)


class _MegatronFSDPFixture:
    def forward(self, *inputs, **kwargs):
        """
        Wrapped forward pass of the model managed by FSDP.
        """
        self._replace_param_with_raw_if_needed()
        with torch.autograd.profiler.record_function("CustomFSDP.forward"):
            # Call the forward pass of the wrapped module.
            output = self.module.forward(*inputs, **kwargs)
            return output


class _ParamAndGradBufferFixture:
    def update_main_grads(self):
        """Minimal cached-gradient fixture containing the 0.5.0 reshape defect."""
        for name, orig_param, optimizer_grad in self.cached_grad_updates:
            if name in self.dist_main_grad:
                if len(orig_param.shape) > 1:
                    local_shape = (-1, *orig_param.shape[1:])
                else:
                    local_shape = (-1,)
                self.dist_main_grad[name]._local_tensor = optimizer_grad.view(local_shape)


def _free_storage(_tensor):
    """Fixture binding replaced on the synthetic module when semantics are tested."""


class _FakeDTensor:
    def __init__(self, global_shape, local_shape):
        self.shape = torch.Size(global_shape)
        self._local = torch.empty(local_shape)

    def to_local(self):
        return self._local


def _fixture_allocator_type():
    original = _StorageResizeBasedBucketAllocatorFixture.free
    free = FunctionType(
        original.__code__,
        original.__globals__,
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    free.__annotations__ = dict(original.__annotations__)
    free.__kwdefaults__ = original.__kwdefaults__
    free.__qualname__ = "StorageResizeBasedBucketAllocator.free"
    return type("StorageResizeBasedBucketAllocator", (), {"free": free})


def _fixture_wrapper_type():
    original = _MegatronFSDPFixture.forward
    forward = FunctionType(
        original.__code__,
        original.__globals__,
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    forward.__annotations__ = dict(original.__annotations__)
    forward.__kwdefaults__ = original.__kwdefaults__
    forward.__qualname__ = "MegatronFSDP.forward"
    return type("MegatronFSDP", (), {"forward": forward})


def _fixture_param_and_grad_buffer_type():
    original = _ParamAndGradBufferFixture.update_main_grads
    update_main_grads = FunctionType(
        original.__code__,
        original.__globals__,
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    update_main_grads.__annotations__ = dict(original.__annotations__)
    update_main_grads.__kwdefaults__ = original.__kwdefaults__
    update_main_grads.__qualname__ = "ParamAndGradBuffer.update_main_grads"
    return type("ParamAndGradBuffer", (), {"update_main_grads": update_main_grads})


def _fixture_module(function=make_fsdp_dtensor):
    module = ModuleType("megatron_fsdp.param_and_grad_buffer_fixture")
    module.DTensor = _FakeDTensor
    module.make_fsdp_dtensor = function
    module.StorageResizeBasedBucketAllocator = _fixture_allocator_type()
    module.ParamAndGradBuffer = _fixture_param_and_grad_buffer_type()
    module.torch = torch
    module._free_storage = _free_storage
    root_module = ModuleType("megatron_fsdp.megatron_fsdp_fixture")
    root_module.MegatronFSDP = _fixture_wrapper_type()
    root_module.torch = torch
    module._root_forward_module = root_module
    return module


def _accept_fixture_structure(monkeypatch, module, function=make_fsdp_dtensor):
    monkeypatch.setattr(compat, "_MEGATRON_FSDP_TP_DTENSOR_MODULE", module.__name__)
    monkeypatch.setattr(compat, "_MEGATRON_FSDP_TP_DTENSOR_QUALNAME", function.__qualname__)
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_TP_DTENSOR_SOURCE_BASENAME",
        Path(inspect.getsourcefile(function)).name,
    )
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_TP_DTENSOR_FIRSTLINENO",
        function.__code__.co_firstlineno,
    )
    monkeypatch.setattr(function, "__module__", module.__name__)
    allocator_type = module.StorageResizeBasedBucketAllocator
    allocator_free = allocator_type.free
    monkeypatch.setattr(allocator_type, "__module__", module.__name__)
    monkeypatch.setattr(allocator_free, "__module__", module.__name__)
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_STREAM_LIFETIME_QUALNAME",
        allocator_free.__qualname__,
    )
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_STREAM_LIFETIME_FIRSTLINENO",
        allocator_free.__code__.co_firstlineno,
    )
    root_module = module._root_forward_module
    wrapper_type = root_module.MegatronFSDP
    wrapper_forward = wrapper_type.forward
    monkeypatch.setattr(compat, "_MEGATRON_FSDP_ROOT_FORWARD_MODULE", root_module.__name__)
    monkeypatch.setattr(compat, "_MEGATRON_FSDP_ROOT_FORWARD_CLASS", wrapper_type.__qualname__)
    monkeypatch.setattr(compat, "_MEGATRON_FSDP_ROOT_FORWARD_QUALNAME", wrapper_forward.__qualname__)
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_ROOT_FORWARD_SOURCE_BASENAME",
        Path(inspect.getsourcefile(wrapper_forward)).name,
    )
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_ROOT_FORWARD_FIRSTLINENO",
        wrapper_forward.__code__.co_firstlineno,
    )
    monkeypatch.setattr(wrapper_type, "__module__", root_module.__name__)
    monkeypatch.setattr(wrapper_forward, "__module__", root_module.__name__)
    buffer_type = module.ParamAndGradBuffer
    update_main_grads = buffer_type.update_main_grads
    monkeypatch.setattr(compat, "_MEGATRON_FSDP_UPDATE_MAIN_GRADS_CLASS", buffer_type.__qualname__)
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_UPDATE_MAIN_GRADS_QUALNAME",
        update_main_grads.__qualname__,
    )
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIRSTLINENO",
        update_main_grads.__code__.co_firstlineno,
    )
    monkeypatch.setattr(buffer_type, "__module__", module.__name__)
    monkeypatch.setattr(update_main_grads, "__module__", module.__name__)
    return allocator_free


def _lock_fixture_sources(monkeypatch, module):
    source = inspect.getsource(make_fsdp_dtensor)
    original = module.make_fsdp_dtensor
    original_free = _accept_fixture_structure(monkeypatch, module)
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256",
        hashlib.sha256(source.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256",
        hashlib.sha256(
            source.replace(
                compat._MEGATRON_FSDP_TP_DTENSOR_BROKEN_BLOCK,
                compat._MEGATRON_FSDP_TP_DTENSOR_FIXED_BLOCK,
            ).encode()
        ).hexdigest(),
    )
    free_source = inspect.getsource(original_free)
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256",
        hashlib.sha256(free_source.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256",
        hashlib.sha256(
            free_source.replace(
                compat._MEGATRON_FSDP_STREAM_LIFETIME_BROKEN_BLOCK,
                compat._MEGATRON_FSDP_STREAM_LIFETIME_FIXED_BLOCK,
            ).encode()
        ).hexdigest(),
    )
    root_module = module._root_forward_module
    original_forward = root_module.MegatronFSDP.forward
    forward_source = inspect.getsource(original_forward)
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256",
        hashlib.sha256(forward_source.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256",
        hashlib.sha256(
            forward_source.replace(
                compat._MEGATRON_FSDP_ROOT_FORWARD_BROKEN_BLOCK,
                compat._MEGATRON_FSDP_ROOT_FORWARD_FIXED_BLOCK,
            ).encode()
        ).hexdigest(),
    )
    original_update_main_grads = module.ParamAndGradBuffer.update_main_grads
    update_main_grads_source = inspect.getsource(original_update_main_grads)
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256",
        hashlib.sha256(update_main_grads_source.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        compat,
        "_MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256",
        hashlib.sha256(
            update_main_grads_source.replace(
                compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_BROKEN_BLOCK,
                compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIXED_BLOCK,
            ).encode()
        ).hexdigest(),
    )
    real_import_module = compat.importlib.import_module

    def import_fixture(name):
        if name == compat._MEGATRON_FSDP_ROOT_FORWARD_MODULE:
            return root_module
        return real_import_module(name)

    monkeypatch.setattr(compat.importlib, "import_module", import_fixture)
    return original, original_free, original_forward, original_update_main_grads


def _patch_fixture(monkeypatch):
    module = _fixture_module()
    original, original_free, _original_forward, _original_update_main_grads = _lock_fixture_sources(
        monkeypatch,
        module,
    )
    compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
        package_version="0.5.0",
        param_and_grad_buffer=module,
        megatron_fsdp_module=module._root_forward_module,
    )
    return module, original, original_free


def test_050_patch_uses_tp_local_trailing_shape(monkeypatch):
    module, _original, _original_free = _patch_fixture(monkeypatch)

    # Rowwise TP exposes global (8, 16), local (8, 8). Cover both observed
    # no-unit slices (0/16/40) and the Linear-unit 24-element slice. The old
    # global-tail view either fails or silently produces the wrong row count.
    for numel, expected_shape in (
        (0, (0, 8)),
        (16, (2, 8)),
        (24, (3, 8)),
        (40, (5, 8)),
    ):
        result = module.make_fsdp_dtensor(
            torch.arange(numel),
            _FakeDTensor(global_shape=(8, 16), local_shape=(8, 8)),
            object(),
        )
        assert result.shape == expected_shape

    patched = module.make_fsdp_dtensor
    patched_update_main_grads = module.ParamAndGradBuffer.update_main_grads
    assert compat._megatron_fsdp_050_tp_dtensor_patch_status(
        package_version="0.5.0",
        param_and_grad_buffer=module,
    ) == {
        "package_version": "0.5.0",
        "official_source_sha256": compat._MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256,
        "patched_source_sha256": compat._MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256,
        "patch_marker": compat._MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256,
        "tp_local_shape_active": True,
        "stream_lifetime_official_source_sha256": (compat._MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256),
        "stream_lifetime_patched_source_sha256": (compat._MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256),
        "stream_lifetime_patch_marker": (compat._MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256),
        "stream_lifetime_active": True,
        "root_forward_official_source_sha256": compat._MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256,
        "root_forward_patched_source_sha256": compat._MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256,
        "root_forward_patch_marker": compat._MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256,
        "root_forward_hooks_active": True,
        "update_main_grads_official_source_sha256": compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256,
        "update_main_grads_patched_source_sha256": (compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256),
        "update_main_grads_patch_marker": compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256,
        "update_main_grads_local_shape_active": True,
        "active": True,
    }

    # Installation is idempotent and keeps the same patched callable.
    compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
        package_version="0.5.0",
        param_and_grad_buffer=module,
    )
    assert module.make_fsdp_dtensor is patched
    assert module.ParamAndGradBuffer.update_main_grads is patched_update_main_grads


def test_050_cached_main_grad_update_uses_tp_local_trailing_shape(monkeypatch):
    module, _original, _original_free = _patch_fixture(monkeypatch)
    buffer = module.ParamAndGradBuffer()
    buffer.dist_main_grad = {
        "rowwise": SimpleNamespace(_local_tensor=None),
        "plain_matrix": SimpleNamespace(_local_tensor=None),
        "vector": SimpleNamespace(_local_tensor=None),
    }
    buffer.cached_grad_updates = [
        (
            "rowwise",
            _FakeDTensor(global_shape=(8, 16), local_shape=(8, 8)),
            torch.arange(24),
        ),
        ("plain_matrix", torch.empty(4, 6), torch.arange(12)),
        ("vector", torch.empty(7), torch.arange(3)),
    ]

    buffer.update_main_grads()

    assert buffer.dist_main_grad["rowwise"]._local_tensor.shape == (3, 8)
    assert buffer.dist_main_grad["plain_matrix"]._local_tensor.shape == (2, 6)
    assert buffer.dist_main_grad["vector"]._local_tensor.shape == (3,)


def test_050_stream_lifetime_records_current_stream_before_storage_free(monkeypatch):
    module, _original, _original_free = _patch_fixture(monkeypatch)
    events = []
    current_stream = object()

    class _Cuda:
        @staticmethod
        def current_stream():
            events.append(("current_stream", current_stream))
            return current_stream

    class _BucketData:
        def record_stream(self, stream):
            events.append(("record_stream", stream))

    data = _BucketData()
    module.torch = SimpleNamespace(cuda=_Cuda())
    module._free_storage = lambda tensor: events.append(("free_storage", tensor))
    allocator = module.StorageResizeBasedBucketAllocator()
    allocator.buckets = {7: SimpleNamespace(data=data)}

    allocator.free(7)

    assert events == [
        ("current_stream", current_stream),
        ("record_stream", current_stream),
        ("free_storage", data),
    ]
    assert allocator.buckets[7].data is data


def test_050_root_forward_invokes_registered_root_hooks(monkeypatch):
    module, _original, _original_free = _patch_fixture(monkeypatch)
    events = []

    class _Root(torch.nn.Module):
        def forward(self, value, *, offset):
            events.append(("forward", value.item(), offset.item()))
            return value + offset

    root = _Root()

    def record_pre_hook(_module, args, kwargs):
        events.append(("pre_hook", args[0].item(), kwargs["offset"].item()))

    root.register_forward_pre_hook(record_pre_hook, with_kwargs=True)
    wrapper = module._root_forward_module.MegatronFSDP.__new__(module._root_forward_module.MegatronFSDP)
    wrapper.module = root
    wrapper._replace_param_with_raw_if_needed = lambda: events.append(("replace_raw",))

    first = wrapper.forward(torch.tensor(2), offset=torch.tensor(3))
    second = wrapper.forward(torch.tensor(5), offset=torch.tensor(7))

    assert (first.item(), second.item()) == (5, 12)
    assert events == [
        ("replace_raw",),
        ("pre_hook", 2, 3),
        ("forward", 2, 3),
        ("replace_raw",),
        ("pre_hook", 5, 7),
        ("forward", 5, 7),
    ]


def test_050_root_forward_no_grad_does_not_trigger_backward_hooks(monkeypatch):
    module, _original, _original_free = _patch_fixture(monkeypatch)
    events = []

    class _Root(torch.nn.Module):
        def forward(self, value):
            events.append("forward")
            return value.sin() * value.cos()

    root = _Root()
    root.register_forward_pre_hook(lambda *_unused: events.append("pre_hook"))
    root.register_full_backward_pre_hook(lambda *_unused: events.append("backward_pre_hook"))
    root.register_full_backward_hook(lambda *_unused: events.append("backward_hook"))
    wrapper = module._root_forward_module.MegatronFSDP.__new__(module._root_forward_module.MegatronFSDP)
    wrapper.module = root
    wrapper._replace_param_with_raw_if_needed = lambda: events.append("replace_raw")
    value = torch.tensor([0.2, 0.3], requires_grad=True)

    with torch.no_grad():
        inference = wrapper.forward(value)

    assert inference.grad_fn is None
    assert not inference.requires_grad
    assert value.grad is None
    assert events == ["replace_raw", "pre_hook", "forward"]

    events.clear()
    training = wrapper.forward(value)
    training.sum().backward()

    assert value.grad is not None
    assert events == [
        "replace_raw",
        "pre_hook",
        "forward",
        "backward_pre_hook",
        "backward_hook",
    ]


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_050_root_forward_checkpoint_recompute_notifies_backward_once(monkeypatch, use_reentrant):
    from torch.utils.checkpoint import checkpoint

    module, _original, _original_free = _patch_fixture(monkeypatch)
    events = []

    class _Root(torch.nn.Module):
        def forward(self, value):
            events.append(("forward", torch.is_grad_enabled()))
            return value.sin() * value.cos()

    root = _Root()
    root.register_forward_pre_hook(lambda *_unused: events.append(("pre_hook", torch.is_grad_enabled())))
    root.register_full_backward_pre_hook(lambda *_unused: events.append(("backward_pre_hook",)))
    root.register_full_backward_hook(lambda *_unused: events.append(("backward_hook",)))
    wrapper = module._root_forward_module.MegatronFSDP.__new__(module._root_forward_module.MegatronFSDP)
    wrapper.module = root
    wrapper._replace_param_with_raw_if_needed = lambda: events.append(("replace_raw",))
    value = torch.tensor([0.2, 0.3], requires_grad=True)

    output = checkpoint(wrapper.forward, value, use_reentrant=use_reentrant)
    output.sum().backward()

    assert value.grad is not None
    assert sum(event[0] == "replace_raw" for event in events) == 2
    assert sum(event[0] == "pre_hook" for event in events) == 2
    assert sum(event[0] == "forward" for event in events) == 2
    assert sum(event[0] == "backward_pre_hook" for event in events) == 1
    assert sum(event[0] == "backward_hook" for event in events) == 1


def test_050_status_and_idempotent_patch_reject_stream_free_code_rollback(monkeypatch):
    module, _original, original_free = _patch_fixture(monkeypatch)
    module.StorageResizeBasedBucketAllocator.free.__code__ = original_free.__code__

    with pytest.raises(RuntimeError, match="code differs from the locked stream-lifetime"):
        compat._megatron_fsdp_050_tp_dtensor_patch_status(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )
    with pytest.raises(RuntimeError, match="code differs from the locked stream-lifetime"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )


def test_050_status_and_idempotent_patch_reject_root_forward_code_rollback(monkeypatch):
    module = _fixture_module()
    _original, _original_free, original_forward, _original_update_main_grads = _lock_fixture_sources(
        monkeypatch,
        module,
    )
    compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
        package_version="0.5.0",
        param_and_grad_buffer=module,
        megatron_fsdp_module=module._root_forward_module,
    )
    module._root_forward_module.MegatronFSDP.forward.__code__ = original_forward.__code__

    with pytest.raises(RuntimeError, match="code differs from the locked root-hook"):
        compat._megatron_fsdp_050_tp_dtensor_patch_status(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )
    with pytest.raises(RuntimeError, match="code differs from the locked root-hook"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )


def test_050_status_and_idempotent_patch_reject_update_main_grads_code_rollback(monkeypatch):
    module = _fixture_module()
    _original, _original_free, _original_forward, original_update_main_grads = _lock_fixture_sources(
        monkeypatch,
        module,
    )
    compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
        package_version="0.5.0",
        param_and_grad_buffer=module,
        megatron_fsdp_module=module._root_forward_module,
    )
    module.ParamAndGradBuffer.update_main_grads.__code__ = original_update_main_grads.__code__

    with pytest.raises(RuntimeError, match="code differs from the locked cached main-gradient"):
        compat._megatron_fsdp_050_tp_dtensor_patch_status(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )
    with pytest.raises(RuntimeError, match="code differs from the locked cached main-gradient"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )


def test_050_status_rejects_update_main_grads_global_rebinding(monkeypatch):
    module, _original, _original_free = _patch_fixture(monkeypatch)
    patched = module.ParamAndGradBuffer.update_main_grads
    rebound = FunctionType(
        patched.__code__,
        dict(patched.__globals__),
        name=patched.__name__,
        argdefs=patched.__defaults__,
        closure=patched.__closure__,
    )
    rebound.__kwdefaults__ = patched.__kwdefaults__
    rebound.__module__ = patched.__module__
    rebound.__qualname__ = patched.__qualname__
    setattr(
        rebound,
        compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCH_MARKER,
        compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256,
    )
    setattr(
        rebound,
        compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_ORIGINAL_SOURCE_MARKER,
        compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256,
    )
    module.ParamAndGradBuffer.update_main_grads = rebound

    with pytest.raises(RuntimeError, match="update_main_grads has mutated global bindings"):
        compat._megatron_fsdp_050_tp_dtensor_patch_status(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )


def test_050_set_installation_rolls_back_all_callables_on_status_failure(monkeypatch):
    module = _fixture_module()
    original, original_free, original_forward, original_update_main_grads = _lock_fixture_sources(
        monkeypatch,
        module,
    )

    def reject_status(**_kwargs):
        raise RuntimeError("injected post-install status failure")

    monkeypatch.setattr(
        compat,
        "_megatron_fsdp_050_tp_dtensor_patch_status",
        reject_status,
    )
    with pytest.raises(RuntimeError, match="injected post-install status failure"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )

    assert module.make_fsdp_dtensor is original
    assert module.StorageResizeBasedBucketAllocator.free is original_free
    assert module._root_forward_module.MegatronFSDP.forward is original_forward
    assert module.ParamAndGradBuffer.update_main_grads is original_update_main_grads
    assert not hasattr(module, compat._MEGATRON_FSDP_TP_DTENSOR_HELPER)


def test_050_status_and_idempotent_patch_reject_function_code_rollback(monkeypatch):
    module, original, _original_free = _patch_fixture(monkeypatch)
    module.make_fsdp_dtensor.__code__ = original.__code__

    with pytest.raises(RuntimeError, match="code differs from the locked"):
        compat._megatron_fsdp_050_tp_dtensor_patch_status(
            package_version="0.5.0",
            param_and_grad_buffer=module,
        )
    with pytest.raises(RuntimeError, match="code differs from the locked"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
        )


def _global_shape_helper_code(dtensor_type):
    def global_shape(param):
        if dtensor_type is None:
            raise AssertionError("unreachable")
        return tuple(param.shape)

    return global_shape.__code__


def test_050_status_and_idempotent_patch_reject_helper_code_mutation(monkeypatch):
    module, _original, _original_free = _patch_fixture(monkeypatch)
    helper = getattr(module, compat._MEGATRON_FSDP_TP_DTENSOR_HELPER)
    helper.__code__ = _global_shape_helper_code(module.DTensor)

    with pytest.raises(RuntimeError, match="mutated local-shape helper"):
        compat._megatron_fsdp_050_tp_dtensor_patch_status(
            package_version="0.5.0",
            param_and_grad_buffer=module,
        )
    with pytest.raises(RuntimeError, match="mutated local-shape helper"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
        )


def test_050_status_and_idempotent_patch_reject_global_helper_replacement(monkeypatch):
    module, _original, _original_free = _patch_fixture(monkeypatch)

    def global_shape(param):
        return tuple(param.shape)

    setattr(module, compat._MEGATRON_FSDP_TP_DTENSOR_HELPER, global_shape)
    with pytest.raises(RuntimeError, match="mutated local-shape helper"):
        compat._megatron_fsdp_050_tp_dtensor_patch_status(
            package_version="0.5.0",
            param_and_grad_buffer=module,
        )
    with pytest.raises(RuntimeError, match="mutated local-shape helper"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
        )


def test_050_status_is_read_only_and_rejects_an_unpatched_function():
    module = _fixture_module()
    original = module.make_fsdp_dtensor
    with pytest.raises(RuntimeError, match="patch is not active"):
        compat._megatron_fsdp_050_tp_dtensor_patch_status(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )
    assert module.make_fsdp_dtensor is original


def test_050_patch_rejects_unknown_abi_before_source_rewrite():
    def unknown_make_fsdp_dtensor(local_tensor, param):
        return local_tensor, param

    module = _fixture_module(unknown_make_fsdp_dtensor)
    with pytest.raises(RuntimeError, match="unsupported Megatron-FSDP 0.5.0 make_fsdp_dtensor ABI"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )


def test_050_patch_rejects_unknown_update_main_grads_abi(monkeypatch):
    module = _fixture_module()
    _lock_fixture_sources(monkeypatch, module)

    def unknown_update_main_grads(self, extra):
        del self, extra

    unknown_update_main_grads.__module__ = module.__name__
    unknown_update_main_grads.__qualname__ = compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_QUALNAME
    module.ParamAndGradBuffer.update_main_grads = unknown_update_main_grads

    with pytest.raises(RuntimeError, match="unsupported .*ParamAndGradBuffer.update_main_grads ABI"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )


def test_050_patch_rejects_unknown_source_with_matching_signature(monkeypatch):
    module = _fixture_module()
    _accept_fixture_structure(monkeypatch, module)
    with pytest.raises(RuntimeError, match="source does not match the official wheel"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.0",
            param_and_grad_buffer=module,
            megatron_fsdp_module=module._root_forward_module,
        )


def test_tp_patch_rejects_unqualified_package_version():
    with pytest.raises(RuntimeError, match="refusing to patch an unknown ABI"):
        compat._patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version="0.5.1",
            param_and_grad_buffer=_fixture_module(),
        )


def _configure_process_local_fixture():
    source = inspect.getsource(make_fsdp_dtensor)
    module = _fixture_module()
    function = module.make_fsdp_dtensor
    compat._MEGATRON_FSDP_TP_DTENSOR_MODULE = module.__name__
    compat._MEGATRON_FSDP_TP_DTENSOR_QUALNAME = function.__qualname__
    compat._MEGATRON_FSDP_TP_DTENSOR_SOURCE_BASENAME = Path(inspect.getsourcefile(function)).name
    compat._MEGATRON_FSDP_TP_DTENSOR_FIRSTLINENO = function.__code__.co_firstlineno
    compat._MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256 = hashlib.sha256(source.encode()).hexdigest()
    compat._MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256 = hashlib.sha256(
        source.replace(
            compat._MEGATRON_FSDP_TP_DTENSOR_BROKEN_BLOCK,
            compat._MEGATRON_FSDP_TP_DTENSOR_FIXED_BLOCK,
        ).encode()
    ).hexdigest()
    function.__module__ = module.__name__
    allocator_type = module.StorageResizeBasedBucketAllocator
    allocator_free = allocator_type.free
    allocator_type.__module__ = module.__name__
    allocator_free.__module__ = module.__name__
    free_source = inspect.getsource(allocator_free)
    compat._MEGATRON_FSDP_STREAM_LIFETIME_QUALNAME = allocator_free.__qualname__
    compat._MEGATRON_FSDP_STREAM_LIFETIME_FIRSTLINENO = allocator_free.__code__.co_firstlineno
    compat._MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256 = hashlib.sha256(free_source.encode()).hexdigest()
    compat._MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256 = hashlib.sha256(
        free_source.replace(
            compat._MEGATRON_FSDP_STREAM_LIFETIME_BROKEN_BLOCK,
            compat._MEGATRON_FSDP_STREAM_LIFETIME_FIXED_BLOCK,
        ).encode()
    ).hexdigest()
    root_module = module._root_forward_module
    wrapper_type = root_module.MegatronFSDP
    wrapper_forward = wrapper_type.forward
    compat._MEGATRON_FSDP_ROOT_FORWARD_MODULE = root_module.__name__
    compat._MEGATRON_FSDP_ROOT_FORWARD_CLASS = wrapper_type.__qualname__
    compat._MEGATRON_FSDP_ROOT_FORWARD_QUALNAME = wrapper_forward.__qualname__
    compat._MEGATRON_FSDP_ROOT_FORWARD_SOURCE_BASENAME = Path(inspect.getsourcefile(wrapper_forward)).name
    compat._MEGATRON_FSDP_ROOT_FORWARD_FIRSTLINENO = wrapper_forward.__code__.co_firstlineno
    wrapper_type.__module__ = root_module.__name__
    wrapper_forward.__module__ = root_module.__name__
    forward_source = inspect.getsource(wrapper_forward)
    compat._MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256 = hashlib.sha256(forward_source.encode()).hexdigest()
    compat._MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256 = hashlib.sha256(
        forward_source.replace(
            compat._MEGATRON_FSDP_ROOT_FORWARD_BROKEN_BLOCK,
            compat._MEGATRON_FSDP_ROOT_FORWARD_FIXED_BLOCK,
        ).encode()
    ).hexdigest()
    buffer_type = module.ParamAndGradBuffer
    update_main_grads = buffer_type.update_main_grads
    compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_CLASS = buffer_type.__qualname__
    compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_QUALNAME = update_main_grads.__qualname__
    compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIRSTLINENO = update_main_grads.__code__.co_firstlineno
    buffer_type.__module__ = module.__name__
    update_main_grads.__module__ = module.__name__
    update_main_grads_source = inspect.getsource(update_main_grads)
    compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256 = hashlib.sha256(
        update_main_grads_source.encode()
    ).hexdigest()
    compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256 = hashlib.sha256(
        update_main_grads_source.replace(
            compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_BROKEN_BLOCK,
            compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIXED_BLOCK,
        ).encode()
    ).hexdigest()
    return module


def _gloo_rank_skew_worker(rank, world_size, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    try:
        tp_groups = (
            dist.new_group((0, 1)),
            dist.new_group((2, 3)),
        )
        dp_groups = (
            dist.new_group((0, 2)),
            dist.new_group((1, 3)),
        )
        module = _configure_process_local_fixture()
        if rank == 0:
            setattr(
                module.StorageResizeBasedBucketAllocator.free,
                compat._MEGATRON_FSDP_STREAM_LIFETIME_PATCH_MARKER,
                "rank-zero-invalid-marker",
            )
        try:
            compat._patch_megatron_fsdp_050_tp_dtensor_reshape_with_consensus(
                tp_group=tp_groups[rank // 2],
                dp_group=dp_groups[rank % 2],
                package_version="0.5.0",
                param_and_grad_buffer=module,
                megatron_fsdp_module=module._root_forward_module,
            )
        except RuntimeError as error:
            local = str(error)
        else:
            local = "NO_ERROR"
        messages = [None] * world_size
        dist.all_gather_object(messages, local)
        assert "NO_ERROR" not in messages
        assert len(set(messages)) == 1
        assert "rank=0 RuntimeError" in messages[0]
        assert "unknown or partial compatibility marker set" in messages[0]
    finally:
        dist.destroy_process_group()


def test_gloo_rank_skew_patch_error_reaches_full_tp_dp_cartesian_mesh(tmp_path):
    world_size = 4
    rendezvous = str(tmp_path / "gloo-rendezvous")
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_gloo_rank_skew_worker,
            args=(rank, world_size, rendezvous),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 30
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
    for process in hung:
        process.join(5)
    assert not hung, "rank-skew consensus test hung instead of synchronously failing"
    assert [process.exitcode for process in processes] == [0] * world_size


def _installed_megatron_fsdp_version():
    try:
        return importlib_metadata.version("megatron-fsdp")
    except importlib_metadata.PackageNotFoundError:
        return None


# Unlike the fixture-based tests above, which monkeypatch the fingerprint
# constants, these tests validate the *shipped* constants against the actually
# installed wheel, so a typo'd constant cannot pass CI while hard-failing every
# dp>1 startup. They skip on any other installed version (e.g. containers still
# shipping 0.2.3) and run wherever the pinned megatron-fsdp==0.5.0 is installed.
requires_real_megatron_fsdp_050 = pytest.mark.skipif(
    _installed_megatron_fsdp_version() != compat._MEGATRON_FSDP_TP_DTENSOR_VERSION,
    reason=(
        "requires the installed megatron-fsdp to be exactly "
        f"{compat._MEGATRON_FSDP_TP_DTENSOR_VERSION} (the release the shipped fingerprint "
        f"constants target); found {_installed_megatron_fsdp_version()!r}"
    ),
)


@requires_real_megatron_fsdp_050
def test_050_shipped_fingerprints_match_installed_wheel_and_patch_installs():
    param_and_grad_buffer = importlib.import_module(compat._MEGATRON_FSDP_TP_DTENSOR_MODULE)
    megatron_fsdp_module = importlib.import_module(compat._MEGATRON_FSDP_ROOT_FORWARD_MODULE)
    targets = (
        (
            param_and_grad_buffer.make_fsdp_dtensor,
            compat._MEGATRON_FSDP_TP_DTENSOR_PATCH_MARKER,
            compat._MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256,
            compat._MEGATRON_FSDP_TP_DTENSOR_QUALNAME,
            compat._MEGATRON_FSDP_TP_DTENSOR_FIRSTLINENO,
        ),
        (
            param_and_grad_buffer.StorageResizeBasedBucketAllocator.free,
            compat._MEGATRON_FSDP_STREAM_LIFETIME_PATCH_MARKER,
            compat._MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256,
            compat._MEGATRON_FSDP_STREAM_LIFETIME_QUALNAME,
            compat._MEGATRON_FSDP_STREAM_LIFETIME_FIRSTLINENO,
        ),
        (
            megatron_fsdp_module.MegatronFSDP.forward,
            compat._MEGATRON_FSDP_ROOT_FORWARD_PATCH_MARKER,
            compat._MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256,
            compat._MEGATRON_FSDP_ROOT_FORWARD_QUALNAME,
            compat._MEGATRON_FSDP_ROOT_FORWARD_FIRSTLINENO,
        ),
        (
            param_and_grad_buffer.ParamAndGradBuffer.update_main_grads,
            compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCH_MARKER,
            compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256,
            compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_QUALNAME,
            compat._MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIRSTLINENO,
        ),
    )
    for function, marker_name, official_sha256, qualname, firstlineno in targets:
        if getattr(function, marker_name, None) is not None:
            # An earlier test in this process already installed the compat set;
            # the idempotent re-install below still re-verifies every constant.
            continue
        source = inspect.getsource(function)
        assert hashlib.sha256(source.encode()).hexdigest() == official_sha256, qualname
        assert function.__qualname__ == qualname
        assert function.__code__.co_firstlineno == firstlineno, qualname

    # The install path fail-closes on any mismatch between the shipped official
    # and patched-source fingerprints, structure constants, and the wheel.
    compat._patch_megatron_fsdp_050_tp_dtensor_reshape(package_version="0.5.0")
    status = compat._megatron_fsdp_050_tp_dtensor_patch_status(package_version="0.5.0")
    assert status["active"] is True
    assert status["package_version"] == "0.5.0"
    assert compat._MEGATRON_FSDP_TP_DTENSOR_HELPER in param_and_grad_buffer.make_fsdp_dtensor.__code__.co_names


@requires_real_megatron_fsdp_050
def test_050_compat_kwargs_translate_precision_controls_for_installed_wheel():
    megatron_fsdp = importlib.import_module("megatron_fsdp")
    from nemo_automodel.components.distributed.parallelizer import _megatron_fsdp_compat_kwargs

    kwargs = _megatron_fsdp_compat_kwargs(
        megatron_fsdp.fully_shard,
        grad_reduce_in_fp32=True,
        preserve_fp32_weights=False,
        check_for_nan_in_grad=False,
        report_nan_in_param_grad=True,
    )
    assert set(kwargs) == {"mixed_precision_policy", "report_nan_in_param_grad"}
    assert isinstance(kwargs["mixed_precision_policy"], megatron_fsdp.MixedPrecisionPolicy)
    assert kwargs["report_nan_in_param_grad"] is True
