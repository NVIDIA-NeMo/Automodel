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

"""Fail-closed compatibility fixes for published Megatron-FSDP releases.

The four rewrite pipelines (``make_fsdp_dtensor``,
``StorageResizeBasedBucketAllocator.free``, ``MegatronFSDP.forward``, and
``ParamAndGradBuffer.update_main_grads``) intentionally repeat the same
validate/rewrite/compile/verify structure so each stays independently
auditable against its wheel fingerprints. Consolidating them behind a single
patch-spec dataclass is a planned follow-up, expected to land together with
retiring the 0.5.0 pin rather than in this compatibility pass.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import struct
import textwrap
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import CodeType, ModuleType, SimpleNamespace
from typing import Any

import torch.distributed as dist

logger = logging.getLogger(__name__)

_MEGATRON_FSDP_TP_DTENSOR_VERSION = "0.5.0"
_MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256 = "45d638136716026e8d3f93a4d29f350529f48d235c5599d438a7ff1d5ebb76e3"
_MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256 = "3ff5165cf6b55d83c297bd564cccdd3efe3fe79629e854c47831552abcef68d5"
_MEGATRON_FSDP_TP_DTENSOR_PATCH_MARKER = "_nemo_automodel_megatron_fsdp_050_tp_local_shape"
_MEGATRON_FSDP_TP_DTENSOR_ORIGINAL_SOURCE_MARKER = "_nemo_automodel_megatron_fsdp_050_official_source"
_MEGATRON_FSDP_TP_DTENSOR_HELPER = "_nemo_automodel_megatron_fsdp_local_param_shape"
_MEGATRON_FSDP_TP_DTENSOR_MODULE = "megatron_fsdp.param_and_grad_buffer"
_MEGATRON_FSDP_TP_DTENSOR_QUALNAME = "make_fsdp_dtensor"
_MEGATRON_FSDP_TP_DTENSOR_SOURCE_BASENAME = "param_and_grad_buffer.py"
_MEGATRON_FSDP_TP_DTENSOR_FIRSTLINENO = 4588
_MEGATRON_FSDP_TP_DTENSOR_PARAMETERS = (
    ("local_tensor", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("param", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("dist_index", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("is_sharded_param", inspect.Parameter.POSITIONAL_OR_KEYWORD, True),
    ("is_expert_param", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    ("run_check", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    (
        "update_uneven_dtensor_chunk_meta",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        False,
    ),
    (
        "force_sync_tp_duplicated_param",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        False,
    ),
)
_MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256 = "f3b415af19bb05dfaabf0554912649d419ecf988bd0b33ba343b4e111704c7fe"
_MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256 = (
    "cf09e406110776cc52a6363147a69f0959fa7de1acecc347949e4dc6d4a38324"
)
_MEGATRON_FSDP_STREAM_LIFETIME_PATCH_MARKER = "_nemo_automodel_megatron_fsdp_050_stream_lifetime"
_MEGATRON_FSDP_STREAM_LIFETIME_ORIGINAL_SOURCE_MARKER = (
    "_nemo_automodel_megatron_fsdp_050_stream_lifetime_official_source"
)
_MEGATRON_FSDP_STREAM_LIFETIME_QUALNAME = "StorageResizeBasedBucketAllocator.free"
_MEGATRON_FSDP_STREAM_LIFETIME_FIRSTLINENO = 550
_MEGATRON_FSDP_STREAM_LIFETIME_PARAMETERS = (
    ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("bucket_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
)
_MEGATRON_FSDP_STREAM_LIFETIME_BROKEN_BLOCK = """\
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)
"""
_MEGATRON_FSDP_STREAM_LIFETIME_FIXED_BLOCK = """\
        if bucket_id in self.buckets:
            self.buckets[bucket_id].data.record_stream(torch.cuda.current_stream())
            _free_storage(self.buckets[bucket_id].data)
"""
_MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256 = "91536b19302a1835813ad29adfddf00d6fc8fed885e4acf753098bb15964b7b3"
_MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256 = "973faf4ddc3568f59b1e21472b9a599442033835c68d144461dc51f934f12687"
_MEGATRON_FSDP_ROOT_FORWARD_PATCH_MARKER = "_nemo_automodel_megatron_fsdp_050_root_forward_hooks"
_MEGATRON_FSDP_ROOT_FORWARD_ORIGINAL_SOURCE_MARKER = (
    "_nemo_automodel_megatron_fsdp_050_root_forward_hooks_official_source"
)
_MEGATRON_FSDP_ROOT_FORWARD_MODULE = "megatron_fsdp.megatron_fsdp"
_MEGATRON_FSDP_ROOT_FORWARD_CLASS = "MegatronFSDP"
_MEGATRON_FSDP_ROOT_FORWARD_QUALNAME = "MegatronFSDP.forward"
_MEGATRON_FSDP_ROOT_FORWARD_SOURCE_BASENAME = "megatron_fsdp.py"
_MEGATRON_FSDP_ROOT_FORWARD_FIRSTLINENO = 1452
_MEGATRON_FSDP_ROOT_FORWARD_PARAMETERS = (
    ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("inputs", inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.empty),
    ("kwargs", inspect.Parameter.VAR_KEYWORD, inspect.Parameter.empty),
)
_MEGATRON_FSDP_ROOT_FORWARD_BROKEN_BLOCK = """\
            output = self.module.forward(*inputs, **kwargs)
"""
_MEGATRON_FSDP_ROOT_FORWARD_FIXED_BLOCK = """\
            output = self.module(*inputs, **kwargs)
"""
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256 = "985769209960fe6ce6f79b23418305e1540ae23f8e265b355cb6d5cae51f7c67"
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256 = (
    "4e4f4a51505e27b813fd574c8db6ef6ce0c8655d380f443edfe7752f3b655e7e"
)
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCH_MARKER = "_nemo_automodel_megatron_fsdp_050_update_main_grads_local_shape"
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_ORIGINAL_SOURCE_MARKER = (
    "_nemo_automodel_megatron_fsdp_050_update_main_grads_official_source"
)
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_CLASS = "ParamAndGradBuffer"
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_QUALNAME = "ParamAndGradBuffer.update_main_grads"
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIRSTLINENO = 2938
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_PARAMETERS = (
    ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
)
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_BROKEN_BLOCK = """\
                if len(orig_param.shape) > 1:
                    local_shape = (-1, *orig_param.shape[1:])
                else:
                    local_shape = (-1,)
"""
_MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIXED_BLOCK = f"""\
                local_param_shape = {_MEGATRON_FSDP_TP_DTENSOR_HELPER}(orig_param)
                if len(local_param_shape) > 1:
                    local_shape = (-1, *local_param_shape[1:])
                else:
                    local_shape = (-1,)
"""
_MEGATRON_FSDP_TP_DTENSOR_BROKEN_BLOCK = """\
    if len(orig_param.shape) > 1:
        local_shape = (-1, *orig_param.shape[1:])
    else:
        local_shape = (-1,)
"""
_MEGATRON_FSDP_TP_DTENSOR_FIXED_BLOCK = f"""\
    local_param_shape = {_MEGATRON_FSDP_TP_DTENSOR_HELPER}(orig_param)
    if len(local_param_shape) > 1:
        local_shape = (-1, *local_param_shape[1:])
    else:
        local_shape = (-1,)
"""


def _make_megatron_fsdp_local_param_shape_helper(dtensor_type: type):
    """Create the exact closure installed in Megatron-FSDP's module globals."""

    def local_param_shape(param: Any) -> tuple[int, ...]:
        """Return the rank-local shape used to view a rank-local flat buffer.

        Args:
            param: Parameter whose flat-buffer view shape is required. For a
                ``DTensor``, ``param.shape`` is the global shape (for rowwise
                TP the trailing dims differ per rank), so the rank-local shard
                is read via ``to_local()``. A plain tensor's ``shape`` is
                already rank-local and is used as-is.

        Returns:
            The rank-local shape as a plain tuple of ints.
        """
        tensor = param.to_local() if isinstance(param, dtensor_type) else param
        return tuple(tensor.shape)

    return local_param_shape


def _code_constant_summary(value: Any) -> tuple[Any, ...]:
    """Normalize a code constant without relying on cross-version bytecode hashes."""
    if isinstance(value, CodeType):
        return ("code", _code_object_summary(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_code_constant_summary(item) for item in value))
    if isinstance(value, frozenset):
        return (
            "frozenset",
            tuple(
                sorted(
                    (_code_constant_summary(item) for item in value),
                    key=repr,
                )
            ),
        )
    if isinstance(value, float):
        return ("float", struct.pack("!d", value))
    if isinstance(value, complex):
        return (
            "complex",
            struct.pack("!d", value.real),
            struct.pack("!d", value.imag),
        )
    if value is None or value is Ellipsis:
        return (type(value).__name__,)
    if isinstance(value, (bool, int, str, bytes)):
        return (type(value).__name__, value)
    return (
        "other",
        type(value).__module__,
        type(value).__qualname__,
        repr(value),
    )


def _code_object_summary(code: CodeType) -> tuple[Any, ...]:
    """Return a recursive, same-interpreter semantic summary of a code tree."""
    return (
        code.co_argcount,
        getattr(code, "co_posonlyargcount", 0),
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        tuple(_code_constant_summary(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        getattr(code, "co_exceptiontable", b""),
    )


def _megatron_fsdp_make_fsdp_dtensor_abi(function: Any) -> tuple[tuple[str, Any, Any], ...]:
    """Return the ABI fields that this compatibility fix relies on."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "cannot inspect Megatron-FSDP make_fsdp_dtensor; refusing the TP compatibility patch"
        ) from exc
    return tuple((parameter.name, parameter.kind, parameter.default) for parameter in parameters)


def _megatron_fsdp_storage_resize_free_abi(function: Any) -> tuple[tuple[str, Any, Any], ...]:
    """Return the allocator ABI that the stream-lifetime fix relies on."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "cannot inspect Megatron-FSDP StorageResizeBasedBucketAllocator.free; "
            "refusing the stream-lifetime compatibility patch"
        ) from exc
    return tuple((parameter.name, parameter.kind, parameter.default) for parameter in parameters)


def _megatron_fsdp_root_forward_abi(function: Any) -> tuple[tuple[str, Any, Any], ...]:
    """Return the wrapper-forward ABI that the root-hook fix relies on."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "cannot inspect Megatron-FSDP MegatronFSDP.forward; refusing the root-hook compatibility patch"
        ) from exc
    return tuple((parameter.name, parameter.kind, parameter.default) for parameter in parameters)


def _megatron_fsdp_update_main_grads_abi(function: Any) -> tuple[tuple[str, Any, Any], ...]:
    """Return the buffer-method ABI that the cached main-gradient fix relies on."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "cannot inspect Megatron-FSDP ParamAndGradBuffer.update_main_grads; "
            "refusing the cached main-gradient compatibility patch"
        ) from exc
    return tuple((parameter.name, parameter.kind, parameter.default) for parameter in parameters)


def _validate_megatron_fsdp_make_fsdp_dtensor_structure(module: ModuleType, function: Any) -> None:
    """Reject functions that are not the locked wheel's top-level implementation."""
    source_file = inspect.getsourcefile(function)
    if (
        getattr(module, "__name__", None) != _MEGATRON_FSDP_TP_DTENSOR_MODULE
        or getattr(function, "__module__", None) != _MEGATRON_FSDP_TP_DTENSOR_MODULE
        or getattr(function, "__qualname__", None) != _MEGATRON_FSDP_TP_DTENSOR_QUALNAME
        or function.__code__.co_firstlineno != _MEGATRON_FSDP_TP_DTENSOR_FIRSTLINENO
        or source_file is None
        or Path(source_file).name != _MEGATRON_FSDP_TP_DTENSOR_SOURCE_BASENAME
    ):
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 make_fsdp_dtensor module/source structure differs; "
            "refusing to patch an unknown implementation"
        )


def _validate_megatron_fsdp_storage_resize_free_structure(
    module: ModuleType,
    allocator_type: type,
    function: Any,
) -> None:
    """Reject allocator methods that are not the locked wheel's implementation."""
    source_file = inspect.getsourcefile(function)
    if (
        getattr(module, "__name__", None) != _MEGATRON_FSDP_TP_DTENSOR_MODULE
        or getattr(allocator_type, "__module__", None) != _MEGATRON_FSDP_TP_DTENSOR_MODULE
        or getattr(allocator_type, "__qualname__", None) != "StorageResizeBasedBucketAllocator"
        or getattr(function, "__module__", None) != _MEGATRON_FSDP_TP_DTENSOR_MODULE
        or getattr(function, "__qualname__", None) != _MEGATRON_FSDP_STREAM_LIFETIME_QUALNAME
        or function.__code__.co_firstlineno != _MEGATRON_FSDP_STREAM_LIFETIME_FIRSTLINENO
        or getattr(allocator_type, "free", None) is not function
        or source_file is None
        or Path(source_file).name != _MEGATRON_FSDP_TP_DTENSOR_SOURCE_BASENAME
    ):
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 StorageResizeBasedBucketAllocator.free module/source "
            "structure differs; refusing to patch an unknown implementation"
        )


def _validate_megatron_fsdp_root_forward_structure(
    module: ModuleType,
    wrapper_type: type,
    function: Any,
) -> None:
    """Reject wrapper forwards that are not the locked wheel implementation."""
    source_file = inspect.getsourcefile(function)
    if (
        getattr(module, "__name__", None) != _MEGATRON_FSDP_ROOT_FORWARD_MODULE
        or getattr(wrapper_type, "__module__", None) != _MEGATRON_FSDP_ROOT_FORWARD_MODULE
        or getattr(wrapper_type, "__qualname__", None) != _MEGATRON_FSDP_ROOT_FORWARD_CLASS
        or getattr(function, "__module__", None) != _MEGATRON_FSDP_ROOT_FORWARD_MODULE
        or getattr(function, "__qualname__", None) != _MEGATRON_FSDP_ROOT_FORWARD_QUALNAME
        or function.__code__.co_firstlineno != _MEGATRON_FSDP_ROOT_FORWARD_FIRSTLINENO
        or getattr(wrapper_type, "forward", None) is not function
        or source_file is None
        or Path(source_file).name != _MEGATRON_FSDP_ROOT_FORWARD_SOURCE_BASENAME
    ):
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 MegatronFSDP.forward module/source structure differs; "
            "refusing to patch an unknown implementation"
        )


def _validate_megatron_fsdp_update_main_grads_structure(
    module: ModuleType,
    buffer_type: type,
    function: Any,
) -> None:
    """Reject main-gradient methods that are not the locked wheel implementation."""
    source_file = inspect.getsourcefile(function)
    if (
        getattr(module, "__name__", None) != _MEGATRON_FSDP_TP_DTENSOR_MODULE
        or getattr(buffer_type, "__module__", None) != _MEGATRON_FSDP_TP_DTENSOR_MODULE
        or getattr(buffer_type, "__qualname__", None) != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_CLASS
        or getattr(function, "__module__", None) != _MEGATRON_FSDP_TP_DTENSOR_MODULE
        or getattr(function, "__qualname__", None) != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_QUALNAME
        or function.__code__.co_firstlineno != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIRSTLINENO
        or getattr(buffer_type, "update_main_grads", None) is not function
        or source_file is None
        or Path(source_file).name != _MEGATRON_FSDP_TP_DTENSOR_SOURCE_BASENAME
    ):
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 ParamAndGradBuffer.update_main_grads module/source "
            "structure differs; refusing to patch an unknown implementation"
        )


def _locked_megatron_fsdp_patched_source(function: Any) -> str:
    """Read and rewrite the exact official function source, without mutation."""
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "cannot read Megatron-FSDP 0.5.0 make_fsdp_dtensor source; refusing an unverified runtime patch"
        ) from exc
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if source_sha256 != _MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 make_fsdp_dtensor source does not match the "
            "official wheel; refusing to patch an unknown implementation "
            f"(source_sha256={source_sha256})"
        )
    if source.count(_MEGATRON_FSDP_TP_DTENSOR_BROKEN_BLOCK) != 1:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 make_fsdp_dtensor reshape structure changed; refusing an ambiguous runtime patch"
        )
    patched_source = source.replace(
        _MEGATRON_FSDP_TP_DTENSOR_BROKEN_BLOCK,
        _MEGATRON_FSDP_TP_DTENSOR_FIXED_BLOCK,
    )
    patched_source_sha256 = hashlib.sha256(patched_source.encode()).hexdigest()
    if patched_source_sha256 != _MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 patched source fingerprint differs; refusing an unverified runtime patch"
        )
    return patched_source


def _locked_megatron_fsdp_stream_lifetime_patched_source(function: Any) -> str:
    """Read and rewrite the exact official allocator method, without mutation."""
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "cannot read Megatron-FSDP 0.5.0 StorageResizeBasedBucketAllocator.free "
            "source; refusing an unverified runtime patch"
        ) from exc
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if source_sha256 != _MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 StorageResizeBasedBucketAllocator.free source does "
            "not match the official wheel; refusing to patch an unknown implementation "
            f"(source_sha256={source_sha256})"
        )
    if source.count(_MEGATRON_FSDP_STREAM_LIFETIME_BROKEN_BLOCK) != 1:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 temporary-bucket release structure changed; refusing an ambiguous runtime patch"
        )
    patched_source = source.replace(
        _MEGATRON_FSDP_STREAM_LIFETIME_BROKEN_BLOCK,
        _MEGATRON_FSDP_STREAM_LIFETIME_FIXED_BLOCK,
    )
    patched_source_sha256 = hashlib.sha256(patched_source.encode()).hexdigest()
    if patched_source_sha256 != _MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 stream-lifetime patched source fingerprint differs; "
            "refusing an unverified runtime patch"
        )
    return patched_source


def _locked_megatron_fsdp_root_forward_patched_source(function: Any) -> str:
    """Read and rewrite the exact official wrapper forward, without mutation."""
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "cannot read Megatron-FSDP 0.5.0 MegatronFSDP.forward source; refusing an unverified runtime patch"
        ) from exc
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if source_sha256 != _MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 MegatronFSDP.forward source does not match the official wheel; "
            "refusing to patch an unknown implementation "
            f"(source_sha256={source_sha256})"
        )
    if source.count(_MEGATRON_FSDP_ROOT_FORWARD_BROKEN_BLOCK) != 1:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 root-forward call structure changed; refusing an ambiguous runtime patch"
        )
    patched_source = source.replace(
        _MEGATRON_FSDP_ROOT_FORWARD_BROKEN_BLOCK,
        _MEGATRON_FSDP_ROOT_FORWARD_FIXED_BLOCK,
    )
    patched_source_sha256 = hashlib.sha256(patched_source.encode()).hexdigest()
    if patched_source_sha256 != _MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 root-forward patched source fingerprint differs; refusing an unverified runtime patch"
        )
    return patched_source


def _locked_megatron_fsdp_update_main_grads_patched_source(function: Any) -> str:
    """Read and rewrite the exact official main-gradient method, without mutation."""
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "cannot read Megatron-FSDP 0.5.0 ParamAndGradBuffer.update_main_grads "
            "source; refusing an unverified runtime patch"
        ) from exc
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if source_sha256 != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 ParamAndGradBuffer.update_main_grads source does "
            "not match the official wheel; refusing to patch an unknown implementation "
            f"(source_sha256={source_sha256})"
        )
    if source.count(_MEGATRON_FSDP_UPDATE_MAIN_GRADS_BROKEN_BLOCK) != 1:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 cached main-gradient reshape structure changed; refusing an ambiguous runtime patch"
        )
    patched_source = source.replace(
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_BROKEN_BLOCK,
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_FIXED_BLOCK,
    )
    patched_source_sha256 = hashlib.sha256(patched_source.encode()).hexdigest()
    if patched_source_sha256 != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 cached main-gradient patched source fingerprint differs; "
            "refusing an unverified runtime patch"
        )
    return patched_source


def _compile_megatron_fsdp_patched_function(module: ModuleType, function: Any, patched_source: str) -> Any:
    """Compile the expected function in the running interpreter and module ABI."""
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "\n" * (function.__code__.co_firstlineno - 1) + patched_source,
            function.__code__.co_filename,
            "exec",
            dont_inherit=True,
        ),
        vars(module),
        namespace,
    )
    expected = namespace["make_fsdp_dtensor"]
    if _megatron_fsdp_make_fsdp_dtensor_abi(expected) != _MEGATRON_FSDP_TP_DTENSOR_PARAMETERS:
        raise RuntimeError("compiled Megatron-FSDP compatibility function changed its ABI")
    return expected


def _compile_megatron_fsdp_stream_lifetime_patched_function(
    module: ModuleType,
    function: Any,
    patched_source: str,
) -> Any:
    """Compile the expected allocator method in the running interpreter and module ABI."""
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "\n" * (function.__code__.co_firstlineno - 1) + textwrap.dedent(patched_source),
            function.__code__.co_filename,
            "exec",
            dont_inherit=True,
        ),
        vars(module),
        namespace,
    )
    expected = namespace["free"]
    expected.__module__ = _MEGATRON_FSDP_TP_DTENSOR_MODULE
    expected.__qualname__ = _MEGATRON_FSDP_STREAM_LIFETIME_QUALNAME
    if _megatron_fsdp_storage_resize_free_abi(expected) != _MEGATRON_FSDP_STREAM_LIFETIME_PARAMETERS:
        raise RuntimeError("compiled Megatron-FSDP stream-lifetime compatibility method changed its ABI")
    return expected


def _compile_megatron_fsdp_root_forward_patched_function(
    module: ModuleType,
    function: Any,
    patched_source: str,
) -> Any:
    """Compile the expected wrapper forward in the running interpreter and module ABI."""
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "\n" * (function.__code__.co_firstlineno - 1) + textwrap.dedent(patched_source),
            function.__code__.co_filename,
            "exec",
            dont_inherit=True,
        ),
        vars(module),
        namespace,
    )
    expected = namespace["forward"]
    expected.__module__ = _MEGATRON_FSDP_ROOT_FORWARD_MODULE
    expected.__qualname__ = _MEGATRON_FSDP_ROOT_FORWARD_QUALNAME
    if _megatron_fsdp_root_forward_abi(expected) != _MEGATRON_FSDP_ROOT_FORWARD_PARAMETERS:
        raise RuntimeError("compiled Megatron-FSDP root-forward compatibility method changed its ABI")
    return expected


def _compile_megatron_fsdp_update_main_grads_patched_function(
    module: ModuleType,
    function: Any,
    patched_source: str,
) -> Any:
    """Compile the expected main-gradient method in the running interpreter and module ABI."""
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "\n" * (function.__code__.co_firstlineno - 1) + textwrap.dedent(patched_source),
            function.__code__.co_filename,
            "exec",
            dont_inherit=True,
        ),
        vars(module),
        namespace,
    )
    expected = namespace["update_main_grads"]
    expected.__module__ = _MEGATRON_FSDP_TP_DTENSOR_MODULE
    expected.__qualname__ = _MEGATRON_FSDP_UPDATE_MAIN_GRADS_QUALNAME
    if _megatron_fsdp_update_main_grads_abi(expected) != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PARAMETERS:
        raise RuntimeError("compiled Megatron-FSDP cached main-gradient compatibility method changed its ABI")
    return expected


def _validate_megatron_fsdp_local_shape_helper(module: ModuleType, function: Any) -> None:
    """Verify the installed helper's code, closure, binding, and plain semantics."""
    helper = getattr(module, _MEGATRON_FSDP_TP_DTENSOR_HELPER, None)
    dtensor_type = getattr(module, "DTensor", None)
    if not isinstance(dtensor_type, type):
        raise RuntimeError("patched Megatron-FSDP module lost the expected DTensor type")
    expected_helper = _make_megatron_fsdp_local_param_shape_helper(dtensor_type)
    helper_code = getattr(helper, "__code__", None)
    if (
        not callable(helper)
        or not isinstance(helper_code, CodeType)
        or function.__globals__.get(_MEGATRON_FSDP_TP_DTENSOR_HELPER) is not helper
        or _code_object_summary(helper_code) != _code_object_summary(expected_helper.__code__)
        or helper.__defaults__ != expected_helper.__defaults__
        or helper.__kwdefaults__ != expected_helper.__kwdefaults__
    ):
        raise RuntimeError("patched Megatron-FSDP make_fsdp_dtensor has a mutated local-shape helper")
    closure = {name: cell.cell_contents for name, cell in zip(helper_code.co_freevars, helper.__closure__ or ())}
    if closure != {"dtensor_type": dtensor_type}:
        raise RuntimeError("patched Megatron-FSDP local-shape helper captured the wrong DTensor type")
    probe = SimpleNamespace(shape=(3, 5))
    if helper(probe) != (3, 5):
        raise RuntimeError("patched Megatron-FSDP local-shape helper failed its semantic probe")


def _patch_megatron_fsdp_050_tp_dtensor_reshape(
    *,
    package_version: str | None = None,
    param_and_grad_buffer: ModuleType | None = None,
    megatron_fsdp_module: ModuleType | None = None,
) -> None:
    """Install the locked Megatron-FSDP 0.5.0 TP compatibility set.

    The published 0.5.0 implementation views a rank-local flat buffer with
    ``orig_param.shape[1:]``.  For a Torch-native rowwise TP DTensor this is
    the *global* trailing shape, so only some DP/CP shards can perform the
    view; the remaining ranks enter uneven-DTensor metadata collectives and
    deadlock.  The correct view uses ``orig_param.to_local().shape[1:]``.

    This is deliberately an exact, fail-closed source compatibility patch.
    It runs before TP mutates the model, accepts only the locked 0.5.0 ABI and
    official function fingerprints, and refuses unknown releases or source
    changes instead of executing a guessed rewrite. It also records the
    current compute stream before a temporary all-gather bucket is resized to
    zero. The wheel allocates those buckets on a dedicated parameter-gather
    stream; without ``record_stream`` the CUDA caching allocator may recycle
    storage while a TP/FSDP compute kernel still consumes a parameter view.

    The published wrapper also invokes ``self.module.forward(...)`` directly,
    bypassing the root module's registered pre-forward hook. Root-owned shallow
    parameters therefore remain attached to the storage that FSDP freed after
    initialization, while child FSDP units are gathered normally. Invoke the
    wrapped module through ``nn.Module.__call__`` so its root parameter-gather
    hook executes before model code consumes those parameters.

    The cached ``dist_main_grad`` update path repeats the global-shape view on
    every optimizer step after the first. Reuse the same rank-local shape
    helper there so rowwise TP gradients retain their local trailing dimension.

    All four rewrites are validated and compiled before any is installed. Any
    installation/status failure restores every original callable and removes
    the temporary helper, so a process can never continue with a partial set.
    """
    if package_version is None:
        try:
            package_version = importlib_metadata.version("megatron-fsdp")
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError("Megatron-FSDP metadata is unavailable; refusing the TP compatibility patch") from exc
    if package_version != _MEGATRON_FSDP_TP_DTENSOR_VERSION:
        raise RuntimeError(
            "unsupported Megatron-FSDP version for Torch-native TP + Megatron-FSDP: "
            f"expected {_MEGATRON_FSDP_TP_DTENSOR_VERSION!r}, got {package_version!r}; "
            "refusing to patch an unknown ABI"
        )

    if param_and_grad_buffer is None:
        param_and_grad_buffer = importlib.import_module("megatron_fsdp.param_and_grad_buffer")
    if megatron_fsdp_module is None:
        megatron_fsdp_module = importlib.import_module(_MEGATRON_FSDP_ROOT_FORWARD_MODULE)
    function = getattr(param_and_grad_buffer, "make_fsdp_dtensor", None)
    if function is None:
        raise RuntimeError("Megatron-FSDP 0.5.0 has no make_fsdp_dtensor; refusing to patch an unknown ABI")
    allocator_type = getattr(param_and_grad_buffer, "StorageResizeBasedBucketAllocator", None)
    if not isinstance(allocator_type, type):
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 has no StorageResizeBasedBucketAllocator; refusing to patch an unknown ABI"
        )
    allocator_free = getattr(allocator_type, "free", None)
    if allocator_free is None:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 has no StorageResizeBasedBucketAllocator.free; refusing to patch an unknown ABI"
        )
    wrapper_type = getattr(megatron_fsdp_module, _MEGATRON_FSDP_ROOT_FORWARD_CLASS, None)
    if not isinstance(wrapper_type, type):
        raise RuntimeError("Megatron-FSDP 0.5.0 has no MegatronFSDP class; refusing to patch an unknown ABI")
    wrapper_forward = getattr(wrapper_type, "forward", None)
    if wrapper_forward is None:
        raise RuntimeError("Megatron-FSDP 0.5.0 has no MegatronFSDP.forward; refusing to patch an unknown ABI")
    buffer_type = getattr(param_and_grad_buffer, _MEGATRON_FSDP_UPDATE_MAIN_GRADS_CLASS, None)
    if not isinstance(buffer_type, type):
        raise RuntimeError("Megatron-FSDP 0.5.0 has no ParamAndGradBuffer class; refusing to patch an unknown ABI")
    update_main_grads = getattr(buffer_type, "update_main_grads", None)
    if update_main_grads is None:
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 has no ParamAndGradBuffer.update_main_grads; refusing to patch an unknown ABI"
        )

    marker = getattr(function, _MEGATRON_FSDP_TP_DTENSOR_PATCH_MARKER, None)
    stream_marker = getattr(allocator_free, _MEGATRON_FSDP_STREAM_LIFETIME_PATCH_MARKER, None)
    root_forward_marker = getattr(wrapper_forward, _MEGATRON_FSDP_ROOT_FORWARD_PATCH_MARKER, None)
    update_main_grads_marker = getattr(
        update_main_grads,
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCH_MARKER,
        None,
    )
    if (
        marker == _MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256
        and stream_marker == _MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256
        and root_forward_marker == _MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256
        and update_main_grads_marker == _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256
    ):
        _megatron_fsdp_050_tp_dtensor_patch_status(
            package_version=package_version,
            param_and_grad_buffer=param_and_grad_buffer,
            megatron_fsdp_module=megatron_fsdp_module,
        )
        return
    if (
        marker is not None
        or stream_marker is not None
        or root_forward_marker is not None
        or update_main_grads_marker is not None
    ):
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 has an unknown or partial compatibility marker set; refusing to stack runtime patches"
        )

    abi = _megatron_fsdp_make_fsdp_dtensor_abi(function)
    if abi != _MEGATRON_FSDP_TP_DTENSOR_PARAMETERS:
        raise RuntimeError(
            "unsupported Megatron-FSDP 0.5.0 make_fsdp_dtensor ABI; "
            f"expected {_MEGATRON_FSDP_TP_DTENSOR_PARAMETERS!r}, got {abi!r}"
        )
    _validate_megatron_fsdp_make_fsdp_dtensor_structure(param_and_grad_buffer, function)
    patched_source = _locked_megatron_fsdp_patched_source(function)
    stream_abi = _megatron_fsdp_storage_resize_free_abi(allocator_free)
    if stream_abi != _MEGATRON_FSDP_STREAM_LIFETIME_PARAMETERS:
        raise RuntimeError(
            "unsupported Megatron-FSDP 0.5.0 StorageResizeBasedBucketAllocator.free ABI; "
            f"expected {_MEGATRON_FSDP_STREAM_LIFETIME_PARAMETERS!r}, got {stream_abi!r}"
        )
    _validate_megatron_fsdp_storage_resize_free_structure(
        param_and_grad_buffer,
        allocator_type,
        allocator_free,
    )
    stream_patched_source = _locked_megatron_fsdp_stream_lifetime_patched_source(allocator_free)
    root_forward_abi = _megatron_fsdp_root_forward_abi(wrapper_forward)
    if root_forward_abi != _MEGATRON_FSDP_ROOT_FORWARD_PARAMETERS:
        raise RuntimeError(
            "unsupported Megatron-FSDP 0.5.0 MegatronFSDP.forward ABI; "
            f"expected {_MEGATRON_FSDP_ROOT_FORWARD_PARAMETERS!r}, got {root_forward_abi!r}"
        )
    _validate_megatron_fsdp_root_forward_structure(
        megatron_fsdp_module,
        wrapper_type,
        wrapper_forward,
    )
    root_forward_patched_source = _locked_megatron_fsdp_root_forward_patched_source(wrapper_forward)
    update_main_grads_abi = _megatron_fsdp_update_main_grads_abi(update_main_grads)
    if update_main_grads_abi != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PARAMETERS:
        raise RuntimeError(
            "unsupported Megatron-FSDP 0.5.0 ParamAndGradBuffer.update_main_grads ABI; "
            f"expected {_MEGATRON_FSDP_UPDATE_MAIN_GRADS_PARAMETERS!r}, got {update_main_grads_abi!r}"
        )
    _validate_megatron_fsdp_update_main_grads_structure(
        param_and_grad_buffer,
        buffer_type,
        update_main_grads,
    )
    update_main_grads_patched_source = _locked_megatron_fsdp_update_main_grads_patched_source(update_main_grads)

    module_globals = vars(param_and_grad_buffer)
    if _MEGATRON_FSDP_TP_DTENSOR_HELPER in module_globals:
        raise RuntimeError(
            "Megatron-FSDP already defines the compatibility helper name; refusing to overwrite unknown runtime state"
        )
    dtensor_type = getattr(param_and_grad_buffer, "DTensor", None)
    if not isinstance(dtensor_type, type):
        raise RuntimeError(
            "Megatron-FSDP 0.5.0 does not expose the expected DTensor type; refusing to patch an unknown ABI"
        )

    local_param_shape = _make_megatron_fsdp_local_param_shape_helper(dtensor_type)
    patched = _compile_megatron_fsdp_patched_function(
        param_and_grad_buffer,
        function,
        patched_source,
    )
    patched_free = _compile_megatron_fsdp_stream_lifetime_patched_function(
        param_and_grad_buffer,
        allocator_free,
        stream_patched_source,
    )
    patched_forward = _compile_megatron_fsdp_root_forward_patched_function(
        megatron_fsdp_module,
        wrapper_forward,
        root_forward_patched_source,
    )
    patched_update_main_grads = _compile_megatron_fsdp_update_main_grads_patched_function(
        param_and_grad_buffer,
        update_main_grads,
        update_main_grads_patched_source,
    )
    if _megatron_fsdp_make_fsdp_dtensor_abi(patched) != abi:
        raise RuntimeError("patched Megatron-FSDP make_fsdp_dtensor changed its ABI")
    if _megatron_fsdp_storage_resize_free_abi(patched_free) != stream_abi:
        raise RuntimeError("patched Megatron-FSDP StorageResizeBasedBucketAllocator.free changed its ABI")
    if _megatron_fsdp_root_forward_abi(patched_forward) != root_forward_abi:
        raise RuntimeError("patched Megatron-FSDP MegatronFSDP.forward changed its ABI")
    if _megatron_fsdp_update_main_grads_abi(patched_update_main_grads) != update_main_grads_abi:
        raise RuntimeError("patched Megatron-FSDP ParamAndGradBuffer.update_main_grads changed its ABI")
    setattr(
        patched,
        _MEGATRON_FSDP_TP_DTENSOR_PATCH_MARKER,
        _MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256,
    )
    setattr(
        patched,
        _MEGATRON_FSDP_TP_DTENSOR_ORIGINAL_SOURCE_MARKER,
        _MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256,
    )
    setattr(
        patched_free,
        _MEGATRON_FSDP_STREAM_LIFETIME_PATCH_MARKER,
        _MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256,
    )
    setattr(
        patched_free,
        _MEGATRON_FSDP_STREAM_LIFETIME_ORIGINAL_SOURCE_MARKER,
        _MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256,
    )
    setattr(
        patched_forward,
        _MEGATRON_FSDP_ROOT_FORWARD_PATCH_MARKER,
        _MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256,
    )
    setattr(
        patched_forward,
        _MEGATRON_FSDP_ROOT_FORWARD_ORIGINAL_SOURCE_MARKER,
        _MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256,
    )
    setattr(
        patched_update_main_grads,
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCH_MARKER,
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256,
    )
    setattr(
        patched_update_main_grads,
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_ORIGINAL_SOURCE_MARKER,
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256,
    )

    try:
        module_globals[_MEGATRON_FSDP_TP_DTENSOR_HELPER] = local_param_shape
        param_and_grad_buffer.make_fsdp_dtensor = patched
        allocator_type.free = patched_free
        wrapper_type.forward = patched_forward
        buffer_type.update_main_grads = patched_update_main_grads
        _megatron_fsdp_050_tp_dtensor_patch_status(
            package_version=package_version,
            param_and_grad_buffer=param_and_grad_buffer,
            megatron_fsdp_module=megatron_fsdp_module,
        )
    except Exception:
        param_and_grad_buffer.make_fsdp_dtensor = function
        allocator_type.free = allocator_free
        wrapper_type.forward = wrapper_forward
        buffer_type.update_main_grads = update_main_grads
        module_globals.pop(_MEGATRON_FSDP_TP_DTENSOR_HELPER, None)
        raise

    logger.warning(
        "Applied the fingerprint-guarded Megatron-FSDP 0.5.0 Torch-native TP "
        "local-shape, temporary-bucket stream-lifetime, root-forward-hook, and cached-main-gradient compatibility fixes"
    )


def _megatron_fsdp_050_tp_dtensor_patch_status(
    *,
    package_version: str | None = None,
    param_and_grad_buffer: ModuleType | None = None,
    megatron_fsdp_module: ModuleType | None = None,
) -> dict[str, str | bool]:
    """Verify, without mutating state, that the exact compatibility set is active."""
    if package_version is None:
        try:
            package_version = importlib_metadata.version("megatron-fsdp")
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Megatron-FSDP metadata is unavailable while verifying the TP compatibility patch"
            ) from exc
    if package_version != _MEGATRON_FSDP_TP_DTENSOR_VERSION:
        raise RuntimeError(
            f"Megatron-FSDP TP compatibility evidence has the wrong package version: {package_version!r}"
        )
    if param_and_grad_buffer is None:
        param_and_grad_buffer = importlib.import_module("megatron_fsdp.param_and_grad_buffer")
    if megatron_fsdp_module is None:
        megatron_fsdp_module = importlib.import_module(_MEGATRON_FSDP_ROOT_FORWARD_MODULE)
    function = getattr(param_and_grad_buffer, "make_fsdp_dtensor", None)
    if function is None:
        raise RuntimeError("Megatron-FSDP TP compatibility evidence has no make_fsdp_dtensor")
    marker = getattr(function, _MEGATRON_FSDP_TP_DTENSOR_PATCH_MARKER, None)
    original_source = getattr(function, _MEGATRON_FSDP_TP_DTENSOR_ORIGINAL_SOURCE_MARKER, None)
    if (
        marker != _MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256
        or original_source != _MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256
    ):
        raise RuntimeError("Megatron-FSDP 0.5.0 TP local-shape compatibility patch is not active")
    if _megatron_fsdp_make_fsdp_dtensor_abi(function) != _MEGATRON_FSDP_TP_DTENSOR_PARAMETERS:
        raise RuntimeError("patched Megatron-FSDP make_fsdp_dtensor no longer has the verified ABI")
    _validate_megatron_fsdp_make_fsdp_dtensor_structure(param_and_grad_buffer, function)
    patched_source = _locked_megatron_fsdp_patched_source(function)
    expected_function = _compile_megatron_fsdp_patched_function(
        param_and_grad_buffer,
        function,
        patched_source,
    )
    if _code_object_summary(function.__code__) != _code_object_summary(expected_function.__code__):
        raise RuntimeError("patched Megatron-FSDP make_fsdp_dtensor code differs from the locked local-shape rewrite")
    _validate_megatron_fsdp_local_shape_helper(param_and_grad_buffer, function)

    allocator_type = getattr(param_and_grad_buffer, "StorageResizeBasedBucketAllocator", None)
    if not isinstance(allocator_type, type):
        raise RuntimeError(
            "Megatron-FSDP stream-lifetime compatibility evidence has no StorageResizeBasedBucketAllocator"
        )
    allocator_free = getattr(allocator_type, "free", None)
    if allocator_free is None:
        raise RuntimeError("Megatron-FSDP stream-lifetime compatibility evidence has no allocator free method")
    stream_marker = getattr(
        allocator_free,
        _MEGATRON_FSDP_STREAM_LIFETIME_PATCH_MARKER,
        None,
    )
    stream_original_source = getattr(
        allocator_free,
        _MEGATRON_FSDP_STREAM_LIFETIME_ORIGINAL_SOURCE_MARKER,
        None,
    )
    if (
        stream_marker != _MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256
        or stream_original_source != _MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256
    ):
        raise RuntimeError("Megatron-FSDP 0.5.0 temporary-bucket stream-lifetime compatibility patch is not active")
    if _megatron_fsdp_storage_resize_free_abi(allocator_free) != _MEGATRON_FSDP_STREAM_LIFETIME_PARAMETERS:
        raise RuntimeError(
            "patched Megatron-FSDP StorageResizeBasedBucketAllocator.free no longer has the verified ABI"
        )
    _validate_megatron_fsdp_storage_resize_free_structure(
        param_and_grad_buffer,
        allocator_type,
        allocator_free,
    )
    stream_patched_source = _locked_megatron_fsdp_stream_lifetime_patched_source(allocator_free)
    expected_allocator_free = _compile_megatron_fsdp_stream_lifetime_patched_function(
        param_and_grad_buffer,
        allocator_free,
        stream_patched_source,
    )
    if _code_object_summary(allocator_free.__code__) != _code_object_summary(expected_allocator_free.__code__):
        raise RuntimeError(
            "patched Megatron-FSDP StorageResizeBasedBucketAllocator.free code differs "
            "from the locked stream-lifetime rewrite"
        )
    if (
        allocator_free.__globals__ is not vars(param_and_grad_buffer)
        or allocator_free.__globals__.get("torch") is not getattr(param_and_grad_buffer, "torch", None)
        or allocator_free.__globals__.get("_free_storage") is not getattr(param_and_grad_buffer, "_free_storage", None)
    ):
        raise RuntimeError("patched Megatron-FSDP StorageResizeBasedBucketAllocator.free has mutated global bindings")

    wrapper_type = getattr(megatron_fsdp_module, _MEGATRON_FSDP_ROOT_FORWARD_CLASS, None)
    if not isinstance(wrapper_type, type):
        raise RuntimeError("Megatron-FSDP root-hook compatibility evidence has no MegatronFSDP class")
    wrapper_forward = getattr(wrapper_type, "forward", None)
    if wrapper_forward is None:
        raise RuntimeError("Megatron-FSDP root-hook compatibility evidence has no wrapper forward method")
    root_forward_marker = getattr(
        wrapper_forward,
        _MEGATRON_FSDP_ROOT_FORWARD_PATCH_MARKER,
        None,
    )
    root_forward_original_source = getattr(
        wrapper_forward,
        _MEGATRON_FSDP_ROOT_FORWARD_ORIGINAL_SOURCE_MARKER,
        None,
    )
    if (
        root_forward_marker != _MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256
        or root_forward_original_source != _MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256
    ):
        raise RuntimeError("Megatron-FSDP 0.5.0 root-forward-hook compatibility patch is not active")
    if _megatron_fsdp_root_forward_abi(wrapper_forward) != _MEGATRON_FSDP_ROOT_FORWARD_PARAMETERS:
        raise RuntimeError("patched Megatron-FSDP MegatronFSDP.forward no longer has the verified ABI")
    _validate_megatron_fsdp_root_forward_structure(
        megatron_fsdp_module,
        wrapper_type,
        wrapper_forward,
    )
    root_forward_patched_source = _locked_megatron_fsdp_root_forward_patched_source(wrapper_forward)
    expected_wrapper_forward = _compile_megatron_fsdp_root_forward_patched_function(
        megatron_fsdp_module,
        wrapper_forward,
        root_forward_patched_source,
    )
    if _code_object_summary(wrapper_forward.__code__) != _code_object_summary(expected_wrapper_forward.__code__):
        raise RuntimeError("patched Megatron-FSDP MegatronFSDP.forward code differs from the locked root-hook rewrite")
    if wrapper_forward.__globals__ is not vars(megatron_fsdp_module) or wrapper_forward.__globals__.get(
        "torch"
    ) is not getattr(megatron_fsdp_module, "torch", None):
        raise RuntimeError("patched Megatron-FSDP MegatronFSDP.forward has mutated global bindings")

    buffer_type = getattr(param_and_grad_buffer, _MEGATRON_FSDP_UPDATE_MAIN_GRADS_CLASS, None)
    if not isinstance(buffer_type, type):
        raise RuntimeError("Megatron-FSDP cached main-gradient compatibility evidence has no ParamAndGradBuffer class")
    update_main_grads = getattr(buffer_type, "update_main_grads", None)
    if update_main_grads is None:
        raise RuntimeError("Megatron-FSDP cached main-gradient compatibility evidence has no update_main_grads method")
    update_main_grads_marker = getattr(
        update_main_grads,
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCH_MARKER,
        None,
    )
    update_main_grads_original_source = getattr(
        update_main_grads,
        _MEGATRON_FSDP_UPDATE_MAIN_GRADS_ORIGINAL_SOURCE_MARKER,
        None,
    )
    if (
        update_main_grads_marker != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256
        or update_main_grads_original_source != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256
    ):
        raise RuntimeError("Megatron-FSDP 0.5.0 cached main-gradient compatibility patch is not active")
    if _megatron_fsdp_update_main_grads_abi(update_main_grads) != _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PARAMETERS:
        raise RuntimeError("patched Megatron-FSDP ParamAndGradBuffer.update_main_grads no longer has the verified ABI")
    _validate_megatron_fsdp_update_main_grads_structure(
        param_and_grad_buffer,
        buffer_type,
        update_main_grads,
    )
    update_main_grads_patched_source = _locked_megatron_fsdp_update_main_grads_patched_source(update_main_grads)
    expected_update_main_grads = _compile_megatron_fsdp_update_main_grads_patched_function(
        param_and_grad_buffer,
        update_main_grads,
        update_main_grads_patched_source,
    )
    if _code_object_summary(update_main_grads.__code__) != _code_object_summary(expected_update_main_grads.__code__):
        raise RuntimeError(
            "patched Megatron-FSDP ParamAndGradBuffer.update_main_grads code differs "
            "from the locked cached main-gradient rewrite"
        )
    if (
        update_main_grads.__globals__ is not vars(param_and_grad_buffer)
        or update_main_grads.__globals__.get(_MEGATRON_FSDP_TP_DTENSOR_HELPER)
        is not getattr(param_and_grad_buffer, _MEGATRON_FSDP_TP_DTENSOR_HELPER, None)
        or update_main_grads.__globals__.get("make_fsdp_dtensor") is not function
    ):
        raise RuntimeError("patched Megatron-FSDP ParamAndGradBuffer.update_main_grads has mutated global bindings")
    return {
        "package_version": package_version,
        "official_source_sha256": _MEGATRON_FSDP_TP_DTENSOR_SOURCE_SHA256,
        "patched_source_sha256": _MEGATRON_FSDP_TP_DTENSOR_PATCHED_SOURCE_SHA256,
        "patch_marker": str(marker),
        "tp_local_shape_active": True,
        "stream_lifetime_official_source_sha256": _MEGATRON_FSDP_STREAM_LIFETIME_SOURCE_SHA256,
        "stream_lifetime_patched_source_sha256": (_MEGATRON_FSDP_STREAM_LIFETIME_PATCHED_SOURCE_SHA256),
        "stream_lifetime_patch_marker": str(stream_marker),
        "stream_lifetime_active": True,
        "root_forward_official_source_sha256": _MEGATRON_FSDP_ROOT_FORWARD_SOURCE_SHA256,
        "root_forward_patched_source_sha256": _MEGATRON_FSDP_ROOT_FORWARD_PATCHED_SOURCE_SHA256,
        "root_forward_patch_marker": str(root_forward_marker),
        "root_forward_hooks_active": True,
        "update_main_grads_official_source_sha256": _MEGATRON_FSDP_UPDATE_MAIN_GRADS_SOURCE_SHA256,
        "update_main_grads_patched_source_sha256": _MEGATRON_FSDP_UPDATE_MAIN_GRADS_PATCHED_SOURCE_SHA256,
        "update_main_grads_patch_marker": str(update_main_grads_marker),
        "update_main_grads_local_shape_active": True,
        "active": True,
    }


def _normalized_patch_errors(records: list[Any]) -> list[dict[str, Any]]:
    """Flatten, deduplicate, and deterministically order consensus errors."""
    unique: dict[tuple[int, str, str], dict[str, Any]] = {}
    for record_list in records:
        if not isinstance(record_list, list):
            continue
        for record in record_list:
            if not isinstance(record, dict):
                continue
            key = (
                int(record.get("rank", -1)),
                str(record.get("type", "unknown")),
                str(record.get("message", "")),
            )
            unique[key] = {"rank": key[0], "type": key[1], "message": key[2]}
    return [unique[key] for key in sorted(unique)]


def _patch_megatron_fsdp_050_tp_dtensor_reshape_with_consensus(
    *,
    tp_group: Any,
    dp_group: Any,
    package_version: str | None = None,
    param_and_grad_buffer: ModuleType | None = None,
    megatron_fsdp_module: ModuleType | None = None,
) -> dict[str, str | bool]:
    """Apply and verify all compatibility fixes with TP-then-DP Cartesian error consensus.

    Every rank participates in both object collectives, even after a local
    validation failure. The TP stage propagates errors across each tensor
    parallel row; the DP/DP_CP stage then propagates those rows across the
    complete FSDP×TP slice. Thus no peer can enter TP parameter collectives
    while another peer exits because its installed dependency differs.
    """
    if not dist.is_initialized():
        raise RuntimeError("Megatron-FSDP TP compatibility consensus requires initialized distributed state")
    local_status: dict[str, str | bool] | None = None
    local_errors: list[dict[str, Any]] = []
    try:
        _patch_megatron_fsdp_050_tp_dtensor_reshape(
            package_version=package_version,
            param_and_grad_buffer=param_and_grad_buffer,
            megatron_fsdp_module=megatron_fsdp_module,
        )
        local_status = _megatron_fsdp_050_tp_dtensor_patch_status(
            package_version=package_version,
            param_and_grad_buffer=param_and_grad_buffer,
            megatron_fsdp_module=megatron_fsdp_module,
        )
    except Exception as error:
        local_errors.append(
            {
                "rank": dist.get_rank(),
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    tp_records: list[Any] = [None] * dist.get_world_size(tp_group)
    dist.all_gather_object(tp_records, local_errors, group=tp_group)
    tp_errors = _normalized_patch_errors(tp_records)

    dp_records: list[Any] = [None] * dist.get_world_size(dp_group)
    dist.all_gather_object(dp_records, tp_errors, group=dp_group)
    errors = _normalized_patch_errors(dp_records)
    if errors:
        detail = "; ".join(f"rank={record['rank']} {record['type']}: {record['message']}" for record in errors)
        raise RuntimeError("Megatron-FSDP 0.5.0 compatibility-set consensus failed: " + detail)
    if local_status is None:
        raise RuntimeError("Megatron-FSDP compatibility-set consensus lost its local success status")
    return local_status


__all__ = [
    "_megatron_fsdp_050_tp_dtensor_patch_status",
    "_patch_megatron_fsdp_050_tp_dtensor_reshape",
    "_patch_megatron_fsdp_050_tp_dtensor_reshape_with_consensus",
]
