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
"""
Transformer Engine compatibility patches.

Runtime monkey-patches applied directly to TE classes in memory so they
take effect immediately in the current process.

Call `apply_te_patches()` early in the process, before TE optimizers are used.
"""

from __future__ import annotations

import logging

import torch

_logger = logging.getLogger(__name__)

_TE_PATCHES_APPLIED = False


def _apply_fused_adam_quantized_tensor_patch() -> None:
    """Patch FusedAdam._initialize_state to handle QuantizedTensor params.

    TE's FusedAdam uses ``torch.zeros(param.shape, ...)`` /
    ``torch.empty(param.shape, ...)`` in ``_initialize_state``, which fails for
    QuantizedTensor parameters because their ``.shape`` does not carry the
    correct metadata for allocation.  The fix dequantizes the param first and
    uses ``torch.zeros_like`` / ``torch.empty_like`` instead.

    The fix was merged upstream in TE 2.12 via
    https://github.com/NVIDIA/TransformerEngine/pull/2535.
    """
    from nemo_automodel.shared.import_utils import is_te_min_version

    if is_te_min_version("2.12"):
        _logger.debug("TE >= 2.12 detected; FusedAdam QuantizedTensor patch not needed.")
        return

    try:
        import inspect

        from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
    except ImportError as e:
        _logger.debug(f"Skipping FusedAdam QuantizedTensor patch (import failed): {e}")
        return

    # Skip if upstream already contains the full fix from
    # https://github.com/NVIDIA/TransformerEngine/pull/2535
    _src = inspect.getsource(FusedAdam._initialize_state)
    _upstream_fix_lines = (
        "param.dequantize() if isinstance(param, QuantizedTensor)",
        "torch.zeros_like(param_for_empty",
        "torch.empty_like(param_for_empty",
    )
    if all(line in _src for line in _upstream_fix_lines):
        _logger.debug("FusedAdam._initialize_state already contains upstream QuantizedTensor fix, skipping patch.")
        return

    def _patched_initialize_state(self, param, state_name, zero_buffer, store_param_remainders=False):
        """Initialize one of the optimizer states according to `state_name`.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            zero_buffer (bool): Whether to initialize the optimizer state with zeros.
            store_param_remainders (bool): Store only trailing remainder bits.
        """
        dtype = self.name_to_dtype_map[state_name]
        # Handle QuantizedTensor by dequantizing first
        param_for_empty = param.dequantize() if isinstance(param, QuantizedTensor) else param
        if store_param_remainders:
            data = torch.zeros_like(param_for_empty, dtype=torch.int16)
        else:
            data = torch.empty_like(param_for_empty, dtype=dtype)
        if zero_buffer:
            data.zero_()

        if dtype == torch.uint8:
            import transformer_engine_torch as tex
            from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

            quantizer = Float8Quantizer(
                scale=torch.ones([1], dtype=torch.float32, device=param.device),
                amax=torch.zeros([1], dtype=torch.float32, device=param.device),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
            self.state[param][state_name] = quantizer.make_empty(param.shape)
            self.state[param][state_name].quantize_(data.float())
        else:
            self.state[param][state_name] = data

        # Create scale if necessary.
        if dtype != torch.float32:
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param][state_name] = torch.ones([1], dtype=torch.float32, device=param.device)

    FusedAdam._initialize_state = _patched_initialize_state
    _logger.info("Applied FusedAdam QuantizedTensor monkey-patch.")


def _apply_fused_adam_empty_shard_patch() -> None:
    """Keep zero-element FSDP shards out of TE's native multi-tensor kernel.

    TE's multi_tensor_apply metadata writer can overrun its final slot when a
    zero-element tensor occupies that slot (TE issue #3202).  Filter complete
    parameter/state columns at the Python/native boundary so state_dict keys,
    parameter groups, and the update for every nonempty shard stay intact.
    """
    from nemo_automodel.shared.import_utils import safe_import, safe_import_te

    have_te, _ = safe_import_te()
    available, fused_adam = (
        safe_import("transformer_engine.pytorch.optimizers.fused_adam") if have_te else (False, None)
    )
    if not available:
        _logger.debug("Skipping FusedAdam empty-shard patch: Transformer Engine is unavailable")
        return

    original = fused_adam.multi_tensor_applier
    if getattr(original, "_nemo_empty_shard_patch", False) is True:
        return

    reported_empty_shards = False

    def _apply_without_empty_shards(op, noop_flag_buffer, tensor_lists, *args):
        """Forward aligned nonempty shard columns to TE's existing applier.

        Args:
            op: Native multi-tensor optimizer kernel.
            noop_flag_buffer: Device integer tensor of shape [1], passed unchanged.
            tensor_lists: Role-major lists [role][parameter]. Each entry is a tensor
                of arbitrary shape or a DTensor whose local shard has that shape.
                Roles start with gradients and parameters, followed by optimizer
                state; corresponding columns describe the same local parameter.
                Tensor storage is preserved and may be updated in place by op.
            *args: Native kernel arguments, forwarded unchanged.

        Returns:
            The original applier's result, or None when every local shard is empty.
        """
        nonlocal reported_empty_shards
        if not tensor_lists or not tensor_lists[0]:
            return original(op, noop_flag_buffer, tensor_lists, *args)

        width = len(tensor_lists[0])
        if any(len(tensors) != width for tensors in tensor_lists):
            raise RuntimeError("FusedAdam multi-tensor lists have different lengths")

        def local_numel(tensor):
            """Return the element count of a tensor's local shard.

            Args:
                tensor: Tensor of arbitrary shape, or DTensor with an arbitrary
                    global shape and possibly empty local shard.

            Returns:
                Number of local elements, independent of the global DTensor shape.
            """
            # TE's original applier unwraps DTensor immediately before the
            # native call.  Check the same local shard here.
            return tensor._local_tensor.numel() if hasattr(tensor, "_local_tensor") else tensor.numel()

        nonempty = [index for index, grad in enumerate(tensor_lists[0]) if local_numel(grad)]
        if len(nonempty) == width:
            return original(op, noop_flag_buffer, tensor_lists, *args)

        # Each column is one parameter, gradient, and optimizer state tuple.
        # A zero gradient with a nonempty parameter would indicate an unrelated
        # error; do not silently omit its update.
        for index in set(range(width)) - set(nonempty):
            if local_numel(tensor_lists[1][index]):
                raise RuntimeError("FusedAdam has an empty gradient for a nonempty parameter shard")

        if not reported_empty_shards:
            _logger.info("FusedAdam filtered %d empty local shards from native update", width - len(nonempty))
            reported_empty_shards = True
        if not nonempty:
            return None
        filtered = [[tensors[index] for index in nonempty] for tensors in tensor_lists]
        return original(op, noop_flag_buffer, filtered, *args)

    _apply_without_empty_shards._nemo_empty_shard_patch = True
    fused_adam.multi_tensor_applier = _apply_without_empty_shards
    _logger.info("Applied FusedAdam empty-shard monkey-patch.")


class _FusedAdamGroupPlan:
    """The multi-tensor lists of one parameter group, in the order ``FusedAdam.step`` builds them."""

    __slots__ = (
        "params",
        "n_params",
        "skipped",
        "f16_params",
        "f16_m",
        "f16_v",
        "f16_master",
        "f32_params",
        "f32_m",
        "f32_v",
        "has_fp16",
        "has_bf16",
    )

    def __init__(self, params: list) -> None:
        self.params = params
        self.n_params = len(params)
        self.skipped: list = []
        self.f16_params: list = []
        self.f16_m: list = []
        self.f16_v: list = []
        self.f16_master: list = []
        self.f32_params: list = []
        self.f32_m: list = []
        self.f32_v: list = []
        self.has_fp16 = False
        self.has_bf16 = False

    def still_valid(self, group: dict) -> bool:
        """True when the group's parameters and gradient presence match the plan."""
        params = group["params"]
        if params is not self.params or len(params) != self.n_params:
            return False
        for param in self.f16_params:
            if param.grad is None:
                return False
        for param in self.f32_params:
            if param.grad is None:
                return False
        for param in self.skipped:
            if param.grad is not None:
                return False
        return True


def _apply_fused_adam_cached_lists_patch() -> None:
    """Reuse FusedAdam's multi-tensor lists across steps instead of rebuilding them every step.

    ``FusedAdam.step`` walks every parameter on every step — three DTensor ``.data`` dispatches,
    the optimizer-state lookups, the dtype branches and the DTensor-parity asserts — and only
    then launches the fused kernels. Under FSDP2 every parameter is a DTensor and a large MoE
    model has on the order of a thousand sharded parameter tensors per rank, so this is tens of
    milliseconds of Python per step during which the GPU is idle (DeepSeek-V4.1 Flash on 16
    GB200 nodes: ``optimizer`` timer 85 ms, Adam kernels 27 ms). Parameters, optimizer states
    and master weights are the same tensor objects on every step; only the gradient objects
    change (FSDP2 installs a fresh sharded gradient after each reduce-scatter). The patch builds
    the per-group plan once — the same lists in the same order as the original loop — and on
    later steps only re-reads ``param.grad``, so the kernels receive identical inputs.

    Anything the plan does not cover falls back to the original ``step`` for that call:
    capturable mode, decoupled gradients, a grad scaler or closure, FP8 / block-scaled
    parameters, optimizer states that need an unscale pass (fp16 or non-fused bf16 moments),
    uninitialised states (the first step), or a gradient that appears or disappears. The plan is
    dropped on ``load_state_dict`` and ``add_param_group``.
    """

    from nemo_automodel.shared.import_utils import safe_import, safe_import_te

    have_te, _ = safe_import_te()
    available, fused_adam = (
        safe_import("transformer_engine.pytorch.optimizers.fused_adam") if have_te else (False, None)
    )
    if not available:
        _logger.debug("Skipping FusedAdam cached-lists patch: Transformer Engine is unavailable")
        return

    fused_adam_cls = fused_adam.FusedAdam
    if getattr(fused_adam_cls.step, "_nemo_cached_lists_patch", False) is True:
        return

    from torch.distributed.tensor import DTensor

    original_step = fused_adam_cls.step
    original_load_state_dict = fused_adam_cls.load_state_dict
    original_add_param_group = fused_adam_cls.add_param_group
    plain_tensor_types = (torch.Tensor, torch.nn.Parameter)

    def local_tensor(tensor):
        """The tensor the native kernel receives: the local shard of a DTensor, else the tensor itself."""
        return tensor._local_tensor if isinstance(tensor, DTensor) else tensor

    def state_supported(optimizer, name: str) -> bool:
        """States the original loop hands to the kernel unchanged (no unscale pass before or after)."""
        dtype = optimizer.name_to_dtype_map[name]
        if dtype == torch.float32:
            return True
        return dtype == torch.bfloat16 and bool(optimizer.fuse_unscale) and name in ("exp_avg", "exp_avg_sq")

    def build_plan(optimizer, group: dict):
        """Replay the original per-parameter loop once; None when any parameter needs the original path."""
        plan = _FusedAdamGroupPlan(group["params"])
        for param in group["params"]:
            state = optimizer.state[param]
            if len(state) == 0:
                return None  # the original step initialises the states on its first call
            if param.grad is None:
                plan.skipped.append(param)
                continue
            local_param = local_tensor(param)
            if type(local_param) not in plain_tensor_types:
                return None  # Float8Tensor / QuantizedTensor parameters keep their dedicated paths
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                return None
            if not (state_supported(optimizer, "exp_avg") and state_supported(optimizer, "exp_avg_sq")):
                return None
            moment = local_tensor(state["exp_avg"])
            velocity = local_tensor(state["exp_avg_sq"])
            if param.dtype in (torch.float16, torch.bfloat16):
                if optimizer.master_weights:
                    if "master_param" not in state or not state_supported(optimizer, "master_param"):
                        return None
                    plan.f16_master.append(local_tensor(state["master_param"]).data)
                plan.has_fp16 = plan.has_fp16 or param.dtype == torch.float16
                plan.has_bf16 = plan.has_bf16 or param.dtype == torch.bfloat16
                plan.f16_params.append(param)
                plan.f16_m.append(moment)
                plan.f16_v.append(velocity)
            elif param.dtype == torch.float32:
                plan.f32_params.append(param)
                plan.f32_m.append(moment)
                plan.f32_v.append(velocity)
            else:
                return None
        if plan.has_fp16 and plan.has_bf16:
            return None  # the original step raises for this mix
        return plan

    def grads_of(params: list) -> list:
        return [local_tensor(param.grad).data for param in params]

    def datas_of(params: list) -> list:
        return [local_tensor(param).data for param in params]

    def cached_step(self, closure=None, grad_scaler=None):
        """``FusedAdam.step`` with the per-group tensor lists reused across steps."""
        if (
            closure is not None
            or grad_scaler is not None
            or self.capturable
            or getattr(self, "use_decoupled_grad", False)
        ):
            return original_step(self, closure, grad_scaler)

        plans = self.__dict__.get("_nemo_fused_adam_plans")
        if plans is None or len(plans) != len(self.param_groups):
            plans = [None] * len(self.param_groups)
            self.__dict__["_nemo_fused_adam_plans"] = plans
        # Decide for the whole optimizer before touching any group: the original step updates
        # every group, so a fallback must happen before the first kernel of this call.
        for index, group in enumerate(self.param_groups):
            if len(group["params"]) == 0:
                continue
            plan = plans[index]
            if plan is None or not plan.still_valid(group):
                plan = build_plan(self, group)
                plans[index] = plan
                if plan is None:
                    return original_step(self, closure, grad_scaler)

        applier = fused_adam.multi_tensor_applier
        for index, group in enumerate(self.param_groups):
            if len(group["params"]) == 0:
                continue
            plan = plans[index]
            group["step"] = group["step"] + 1 if "step" in group else 1
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]
            kernel_args = (
                group["lr"],
                beta1,
                beta2,
                group["eps"],
                group["step"],
                self.adam_w_mode,
                bias_correction,
                group["weight_decay"],
            )
            if plan.f16_params:
                tensor_lists = [grads_of(plan.f16_params), datas_of(plan.f16_params), plan.f16_m, plan.f16_v]
                adam_func = self.multi_tensor_adam
                if self.master_weights:
                    tensor_lists.append(plan.f16_master)
                    if self.store_param_remainders and plan.has_bf16 and not plan.has_fp16:
                        adam_func = self.multi_tensor_adam_param_remainder
                applier(adam_func, self._dummy_overflow_buf, tensor_lists, *kernel_args)
            if plan.f32_params:
                tensor_lists = [grads_of(plan.f32_params), datas_of(plan.f32_params), plan.f32_m, plan.f32_v]
                applier(self.multi_tensor_adam, self._dummy_overflow_buf, tensor_lists, *kernel_args)
        return None

    def load_state_dict(self, state_dict):
        self.__dict__.pop("_nemo_fused_adam_plans", None)
        return original_load_state_dict(self, state_dict)

    def add_param_group(self, param_group):
        self.__dict__.pop("_nemo_fused_adam_plans", None)
        return original_add_param_group(self, param_group)

    cached_step._nemo_cached_lists_patch = True
    cached_step._nemo_original_step = original_step
    fused_adam_cls.step = cached_step
    fused_adam_cls.load_state_dict = load_state_dict
    fused_adam_cls.add_param_group = add_param_group
    _logger.info("Applied FusedAdam cached-lists monkey-patch.")


def apply_te_patches() -> None:
    """Apply all Transformer Engine runtime patches.

    This function is idempotent and safe to call multiple times.
    """
    global _TE_PATCHES_APPLIED
    if _TE_PATCHES_APPLIED:
        return

    _apply_fused_adam_quantized_tensor_patch()
    _apply_fused_adam_empty_shard_patch()
    _apply_fused_adam_cached_lists_patch()

    _TE_PATCHES_APPLIED = True
