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

"""Shared fp32-compute helpers for GatedDeltaNet (GDN) linear-attention layers.

GDN layers carry intrinsically-fp32 bare parameters (``A_log`` and ``dt_bias``)
that feed the decay gate ``g = -exp(A_log) * softplus(a + dt_bias)``. Under FSDP2
mixed precision with fp32 master weights, the bulk of a model computes in bf16
(``param_dtype=bf16``) while these parameters must stay in fp32 -- ``A_log`` is
exponentiated, so bf16 rounding becomes a proportional error on the decay rate
that the recurrence compounds across the sequence.

``fully_shard_by_dtype`` groups parameters into FSDP units by *module subtree*, so
a bare fp32 parameter sitting next to bf16 ones cannot earn its own fp32 unit. To
make it shardable, ``isolate_fp32_params`` first forces the intrinsic GDN params
to fp32, then moves those bare params into a ``_fp32_params`` submodule
(``Fp32GateParamHolder``) whose ``forward`` computes the gate. Because the gate is
computed *inside* the holder's forward, FSDP2's unshard/reshard lifecycle and
post-backward gradient reduce-scatter work naturally.

This module is shared by the dense Qwen3.5 path (``qwen3_5_moe.cp_linear_attn``),
the native Qwen3.5-MoE path, and the Qwen3-Next path.
"""

from __future__ import annotations

import re

import torch

HOLDER_NAME = "_fp32_params"
_GETATTR_PATCHED_FLAG = "_fp32_getattr_patched"

# Intrinsically-fp32 GatedDeltaNet bare params routed through the holder.
FP32_GDN_PARAM_NAMES = ("A_log", "dt_bias")

_FP32_HOLDER_KEY_RE = re.compile(r"(\.linear_attn)\._fp32_params\.")


def strip_fp32_holder_key(key: str) -> str:
    """Rewrite ``...linear_attn._fp32_params.X`` -> ``...linear_attn.X``.

    Used by state-dict adapters so saved checkpoints hide the ``_fp32_params``
    wrapping and stay directly HF-loadable.
    """
    return _FP32_HOLDER_KEY_RE.sub(r"\1.", key)


def route_fp32_holder_key(key: str, param_names: tuple[str, ...] = FP32_GDN_PARAM_NAMES) -> str:
    """Rewrite a bare ``...linear_attn.X`` GDN param key into the ``_fp32_params`` holder.

    Inverse of :func:`strip_fp32_holder_key` for the param names in ``param_names``.
    No-op when the key is already routed, is not under ``linear_attn``, or is not a
    tracked fp32 GDN param.
    """
    if not key.endswith(param_names):
        return key
    if "._fp32_params." in key:
        return key
    if ".linear_attn." not in key:
        return key
    head, tail = key.rsplit(".linear_attn.", 1)
    return f"{head}.linear_attn._fp32_params.{tail}"


def is_gated_delta_net_fp32_param_key(key: str, param_names: tuple[str, ...] = FP32_GDN_PARAM_NAMES) -> bool:
    """Return whether ``key`` names an intrinsically-fp32 GDN parameter."""
    return key.endswith(param_names) and ".linear_attn." in key


def upcast_gated_delta_net_fp32_state_tensor(
    key: str, tensor: object, param_names: tuple[str, ...] = FP32_GDN_PARAM_NAMES
) -> object:
    """Cast loaded GDN fp32-param tensors to fp32 while leaving other state untouched.

    Construction-time upcasting is not enough for checkpoint and HF load paths that
    replace or carry tensor values from disk. This helper preserves the fp32 GDN
    contract at adapter boundaries before tensors enter the live model state dict.
    """
    if not is_gated_delta_net_fp32_param_key(key, param_names):
        return tensor
    if getattr(tensor, "dtype", None) == torch.float32:
        return tensor
    is_floating_point = getattr(tensor, "is_floating_point", None)
    if callable(is_floating_point) and is_floating_point():
        return tensor.to(dtype=torch.float32)
    return tensor


class Fp32GateParamHolder(torch.nn.Module):
    """Holds fp32 GDN params (``A_log``, optionally ``dt_bias``) in their own module.

    ``forward`` computes the gate that the GatedDeltaNet would otherwise compute
    inline. Doing it inside this module's forward keeps FSDP's unshard/reshard
    lifecycle natural -- the params are unsharded for the computation and the
    holder's forward output participates in autograd so gradients are reduced.

    ``dt_bias`` may live either here (when it is available and was moved in by
    ``isolate_fp32_params``) or remain a bare param on the parent module in legacy
    / partial-holder cases. The forward prefers the holder-owned ``dt_bias`` so
    the value used in compute is the unsharded fp32 parameter; otherwise it falls
    back to the ``dt_bias`` passed in by the caller.
    """

    def forward(self, a: torch.Tensor, dt_bias: torch.Tensor | None = None) -> torch.Tensor:
        import torch.nn.functional as F

        own_dt_bias = self._parameters.get("dt_bias")
        bias = own_dt_bias if own_dt_bias is not None else dt_bias
        return -self.A_log.float().exp() * F.softplus(a.float() + bias)


def make_fp32_getattr(orig_getattr):
    """Create a ``__getattr__`` that resolves fp32 params from ``_fp32_params``.

    Allows ``self.A_log`` / ``self.dt_bias`` to resolve from the holder submodule
    so code outside the holder forward (state_dict, checkpointing, the inline
    fallback gate) can still access the parameter by name.
    """

    def _getattr_with_fp32(self, name):
        modules = self.__dict__.get("_modules", {})
        fp32_holder = modules.get(HOLDER_NAME)
        if fp32_holder is not None and name in fp32_holder._parameters:
            return fp32_holder._parameters[name]
        return orig_getattr(self, name)

    return _getattr_with_fp32


def force_fp32_gated_delta_net_params(
    module: torch.nn.Module, param_names: tuple[str, ...] = FP32_GDN_PARAM_NAMES
) -> bool:
    """Force intrinsically-fp32 GDN params to fp32 before holder isolation.

    ``from_config`` and custom construction can create ``A_log`` / ``dt_bias`` under
    a bf16 default dtype. Upcasting here records the architecture contract on the
    live parameters themselves; optimizer fp32 master weights alone cannot recover
    precision after the forward parameter has already been rounded to bf16.

    Returns ``True`` when at least one parameter was converted.
    """
    forced_any = False

    for owner in (module, module._modules.get(HOLDER_NAME)):
        if owner is None:
            continue
        for pname in param_names:
            param = owner._parameters.get(pname)
            if param is not None and param.dtype != torch.float32 and param.is_floating_point():
                param.data = param.data.to(dtype=torch.float32)
                forced_any = True

    return forced_any


def isolate_fp32_params(module: torch.nn.Module) -> Fp32GateParamHolder | None:
    """Move ``module``'s fp32 bare params into a ``_fp32_params`` holder submodule.

    For FSDP mixed-dtype compatibility: bare fp32 params (``A_log``, ``dt_bias``)
    are moved into a ``Fp32GateParamHolder`` so ``fully_shard_by_dtype`` can wrap
    them in their own fp32 FSDP unit. The parent class's ``__getattr__`` is patched
    so ``module.A_log`` still resolves to the moved parameter.

    Idempotent: if a holder already exists the existing one is returned after
    re-asserting fp32 dtype on holder-owned GDN params. Returns the holder, or
    ``None`` when the module has no fp32 bare params.
    """
    force_fp32_gated_delta_net_params(module)

    existing = module._modules.get(HOLDER_NAME)
    if existing is not None:
        return existing

    holder: Fp32GateParamHolder | None = None
    for pname in list(module._parameters.keys()):
        param = module._parameters[pname]
        if param is not None and param.dtype == torch.float32:
            if holder is None:
                holder = Fp32GateParamHolder()
            setattr(holder, pname, param)
            del module._parameters[pname]

    if holder is None:
        return None

    module.add_module(HOLDER_NAME, holder)

    cls = type(module)
    if not getattr(cls, _GETATTR_PATCHED_FLAG, False):
        cls.__getattr__ = make_fp32_getattr(cls.__getattr__)
        setattr(cls, _GETATTR_PATCHED_FLAG, True)

    return holder


def mark_keep_in_fp32_modules_strict(model: torch.nn.Module) -> None:
    """Declare ``_fp32_params`` on ``model._keep_in_fp32_modules_strict``.

    ``fully_shard_by_dtype`` reads this to keep the holder in fp32 compute even
    under fp32 master weights (bf16 compute for the bulk).
    """
    existing = tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ())
    if HOLDER_NAME not in existing:
        model._keep_in_fp32_modules_strict = existing + (HOLDER_NAME,)


def isolate_gated_delta_net_fp32_params(model: torch.nn.Module) -> bool:
    """Isolate fp32 GatedDeltaNet params across ``model`` for fp32 FSDP compute.

    Walks the model, and for every GatedDeltaNet linear-attention module
    (identified by a ``linear_attn`` qualified module name plus a bare ``A_log``
    parameter, or an already-created ``_fp32_params`` holder) moves its fp32 bare
    params into a ``_fp32_params`` holder. When at least one holder exists,
    declares ``_fp32_params`` on ``model._keep_in_fp32_modules_strict`` so the
    custom-MoE FSDP path shards those params in their own fp32 unit.

    Idempotent and safe to call on models without GatedDeltaNet layers (no-op).
    The ``linear_attn`` name guard is intentional: other architectures such as
    Mamba also have ``A_log`` parameters, but they do not call the fp32 holder's
    ``forward`` and therefore must not be wrapped by this GDN-specific helper.
    Returns ``True`` if any fp32 holder is present after the walk.
    """
    named_modules_fn = getattr(model, "named_modules", None)
    if not callable(named_modules_fn):
        return False

    isolated_any = False
    # Materialize the module list first: isolate_fp32_params adds a ``_fp32_params``
    # child, mutating the module tree mid-walk.
    for name, module in list(named_modules_fn()):
        if isinstance(module, Fp32GateParamHolder):
            continue
        if name != "linear_attn" and not name.endswith(".linear_attn"):
            continue
        params = getattr(module, "_parameters", None)
        mods = getattr(module, "_modules", None)
        if params is None or mods is None:
            continue
        if "A_log" in params or mods.get(HOLDER_NAME) is not None:
            holder = isolate_fp32_params(module)
            if holder is not None:
                isolated_any = True
    if isolated_any:
        mark_keep_in_fp32_modules_strict(model)
    return isolated_any
