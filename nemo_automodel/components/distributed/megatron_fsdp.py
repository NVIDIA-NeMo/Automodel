# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron_fsdp import MixedPrecisionPolicy
from megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_optimizer,
)
from megatron_fsdp.experimental.module import FsdpModule
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.distributed.parallelizer import (
    _derive_megatron_fsdp_unit_modules,
    import_classes_from_paths,
)

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.config import DistributedConfig

logger = logging.getLogger(__name__)


class MegatronFSDPManager:
    """Shard a model with Megatron Core MFSDP v2.

    MFSDP v2 initially supports only data-parallel ZeRO-3 in AutoModel. Selected
    transformer blocks are sharded bottom-up before the root module, and every
    data-parallel buffer uses a Flat placement.

    Args:
        config: Megatron FSDP strategy configuration.
        device_mesh: Device mesh whose ``dp`` axis is used for sharding.
    """

    def __init__(self, config: MegatronFSDPConfig, device_mesh: DeviceMesh) -> None:
        self.config = config
        self.device_mesh = device_mesh
        self.megatron_fsdp_unit_modules = config.megatron_fsdp_unit_modules
        self.activation_checkpointing = config.activation_checkpointing
        self._validate_config(config)
        self.mp_policy = self._mixed_precision_policy(config)

    @staticmethod
    def _mixed_precision_policy(config: MegatronFSDPConfig):
        """Translate supported legacy precision fields to MFSDP v2."""
        return MixedPrecisionPolicy(
            main_params_dtype=torch.float32 if config.preserve_fp32_weights else torch.bfloat16,
            main_grads_dtype=torch.float32 if config.grad_reduce_in_fp32 else None,
            grad_comm_dtype=None,
        )

    @staticmethod
    def _validate_config(config: MegatronFSDPConfig) -> None:
        """Reject legacy settings that MFSDP v2 does not implement."""
        if config.zero_dp_strategy != 3:
            raise ValueError(f"megatron_fsdp with MFSDP v2 requires zero_dp_strategy=3; got {config.zero_dp_strategy}.")

        unsupported_nondefaults = {
            "init_fsdp_with_meta_device": (config.init_fsdp_with_meta_device, False),
            "overlap_grad_reduce": (config.overlap_grad_reduce, True),
            "overlap_param_gather": (config.overlap_param_gather, True),
            "check_for_nan_in_grad": (config.check_for_nan_in_grad, True),
            "report_nan_in_param_grad": (config.report_nan_in_param_grad, False),
            "average_in_collective": (config.average_in_collective, False),
            "disable_bucketing": (config.disable_bucketing, False),
            "calculate_per_token_loss": (config.calculate_per_token_loss, False),
            "keep_fp8_transpose_cache": (config.keep_fp8_transpose_cache, False),
            "nccl_ub": (config.nccl_ub, False),
            "fsdp_double_buffer": (config.fsdp_double_buffer, False),
        }
        changed = [
            f"{name}={value!r}" for name, (value, default) in unsupported_nondefaults.items() if value != default
        ]
        if changed:
            raise ValueError(
                "megatron_fsdp with MFSDP v2 does not support these non-default legacy options: " + ", ".join(changed)
            )

    def _dp_mesh(self) -> DeviceMesh:
        """Return the one-dimensional data-parallel mesh."""
        for axis, feature in (("tp", "tensor parallelism"), ("cp", "context parallelism")):
            if self.device_mesh[axis].size() > 1:
                raise ValueError(f"megatron_fsdp with MFSDP v2 does not support {feature}")
        dp_mesh = self.device_mesh["dp"]
        if dp_mesh.ndim != 1:
            raise ValueError(f"megatron_fsdp requires a one-dimensional DP mesh; got ndim={dp_mesh.ndim}.")
        return dp_mesh

    @staticmethod
    def _flat_placements():
        """Build ZeRO-3 placements for the data-parallel mesh."""
        return Placements(
            dp_axes=[0],
            parameter=[Flat()],
            gradient=[Flat()],
            optimizer=[Flat()],
        )

    def _fp32_policy(self):
        """Keep explicitly protected FP32 parameters lossless in main storage."""
        return MixedPrecisionPolicy(
            main_params_dtype=torch.float32,
            main_grads_dtype=self.mp_policy.main_grads_dtype,
            grad_comm_dtype=self.mp_policy.grad_comm_dtype,
        )

    @staticmethod
    def _protected_fp32_modules(model: nn.Module) -> list[nn.Module]:
        """Find modules covered by the model's FP32-storage contract."""
        protected_names = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
        if not protected_names:
            return []
        return [
            module
            for name, module in model.named_modules()
            if module is not model
            and any(name == protected or name.endswith(f".{protected}") for protected in protected_names)
        ]

    @staticmethod
    def _remap_optimizer_parameters(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        parameter_names_by_id: dict[int, str],
    ) -> None:
        """Retarget pre-built optimizer groups to MFSDP's sharded parameters.

        Args:
            model: Model whose parameters now contain sharded DTensors.
            optimizer: Optimizer whose parameter groups may still reference the
                pre-shard parameters.
            parameter_names_by_id: Mapping from each pre-shard parameter identity
                to its model parameter name.
        """
        sharded_parameters = dict(model.named_parameters(remove_duplicate=False))
        for group in optimizer.param_groups:
            remapped_parameters = []
            for parameter in group["params"]:
                parameter_name = parameter_names_by_id.get(id(parameter))
                if parameter_name is None:
                    remapped_parameters.append(parameter)
                    continue
                sharded_parameter = sharded_parameters[parameter_name]
                if parameter in optimizer.state:
                    optimizer.state[sharded_parameter] = optimizer.state.pop(parameter)
                remapped_parameters.append(sharded_parameter)
            group["params"] = remapped_parameters

    @staticmethod
    def _register_external_output_embedding_hooks(model: nn.Module) -> None:
        """Unshard root-owned output weights when callers reuse the LM head.

        Some recipes apply the output embedding outside ``model.forward`` (for
        example, to score MTP hidden states). The root has already resharded by
        then, so the head would otherwise receive a regular activation and a
        DTensor weight. These hooks unshard only the root parameter groups for
        that standalone call. Calls made during the root forward are already
        unsharded and remain untouched.
        """
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        if not callable(get_output_embeddings):
            return
        output_embeddings = get_output_embeddings()
        if output_embeddings is None or isinstance(output_embeddings, FsdpModule):
            return

        required_methods = (
            "_lazy_init_context",
            "_unshard_parameter_groups",
            "_reshard_parameter_groups",
        )
        if not all(callable(getattr(model, method, None)) for method in required_methods):
            return

        unsharded_for_call: list[bool] = []
        external_state = {"held": False, "pending_backwards": 0}

        def pre_forward(_module, _args) -> None:
            model._lazy_init_context()
            if external_state["held"]:
                external_state["pending_backwards"] += 1
                unsharded_for_call.append(True)
                return
            if model._unshard_event is not None:
                unsharded_for_call.append(False)
                return
            model._unshard_parameter_groups()
            assert model._unshard_event is not None
            model.context.current_stream().wait_event(model._unshard_event)
            external_state["held"] = True
            external_state["pending_backwards"] = 1
            unsharded_for_call.append(True)

        def release_after_backward(grad):
            external_state["pending_backwards"] -= 1
            if external_state["pending_backwards"] == 0:
                model._reshard_parameter_groups()
                external_state["held"] = False
            return grad

        def post_forward(_module, args, output) -> None:
            if not unsharded_for_call.pop():
                return
            backward_input = next(
                (arg for arg in args if isinstance(arg, torch.Tensor) and arg.requires_grad),
                None,
            )
            if torch.is_grad_enabled() and output is not None and backward_input is not None:
                backward_input.register_hook(release_after_backward)
            else:
                release_after_backward(None)

        output_embeddings.register_forward_pre_hook(pre_forward)
        output_embeddings.register_forward_hook(post_forward, always_call=True)

    @staticmethod
    def _register_structured_output_hooks(model: nn.Module) -> None:
        """Bridge MFSDP root hooks for models returning structured outputs.

        PyTorch module backward hooks only support a Tensor or tuple of Tensors.
        HF-style ModelOutput objects therefore skip MFSDP's root pre-backward
        hook. Keep root-owned parameters materialized after such a forward,
        enter the MFSDP backward lifecycle from the first output gradient, and
        run its fallback completion at the end of autograd.
        """

        def output_tensors(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from output_tensors(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from output_tensors(item)

        def post_forward(_module, _args, output) -> None:
            if isinstance(output, torch.Tensor) or (
                isinstance(output, tuple) and all(isinstance(item, torch.Tensor) for item in output)
            ):
                return

            tensors = [tensor for tensor in output_tensors(output) if tensor.requires_grad]
            if not tensors:
                return

            model._unshard_parameter_groups()
            assert model._unshard_event is not None
            model.context.current_stream().wait_event(model._unshard_event)
            backward_started = False

            def begin_backward(grad):
                nonlocal backward_started
                if backward_started:
                    return grad
                backward_started = True
                original_reshard = model._reshard_parameter_groups
                original_post_backward = model.post_backward
                post_backward_ran = False

                def defer_reshard() -> None:
                    return None

                def tracked_post_backward() -> None:
                    nonlocal post_backward_ran
                    if post_backward_ran:
                        return
                    post_backward_ran = True
                    original_post_backward()

                model._reshard_parameter_groups = defer_reshard
                model.post_backward = tracked_post_backward
                model.pre_backward()

                def finish_backward() -> None:
                    if not post_backward_ran:
                        model.post_backward()
                    model._reshard_parameter_groups = original_reshard
                    model.post_backward = original_post_backward
                    if model._unshard_event is not None:
                        original_reshard()

                torch.autograd.Variable._execution_engine.queue_callback(finish_backward)
                return grad

            for tensor in tensors:
                tensor.register_hook(begin_backward)

        model.register_forward_hook(post_forward)

    @staticmethod
    def _defer_reshard_until_module_backward(
        module: nn.Module,
        backward_state: dict | None = None,
    ) -> None:
        """Keep mixed frozen/trainable units live through all backward branches.

        PEFT can produce one branch for adapter gradients and another frozen
        base-weight branch for activation gradients. MFSDP's parameter-complete
        callback may observe all trainable gradients before the frozen branch
        consumes its weight. Require both parameter completion and the module
        full-backward boundary before releasing storage.
        """
        if not isinstance(module, FsdpModule):
            return
        groups = module.parameter_groups
        if not any(group.requires_grad for group in groups) or not any(not group.requires_grad for group in groups):
            return
        if backward_state is None:
            backward_state = {"retained_module": None, "reshard": None}

        original_post_backward = module.post_backward
        original_reshard = module._reshard_parameter_groups
        state = {
            "grads_finished": False,
            "module_finished": False,
            "inputs_require_grad": False,
            "completion_queued": False,
        }

        def reset_state(_module, args) -> None:
            state["grads_finished"] = False
            state["module_finished"] = False
            state["inputs_require_grad"] = any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args)
            state["completion_queued"] = False

        def maybe_reshard() -> None:
            if state["grads_finished"] and state["module_finished"] and module._unshard_event is not None:
                if state["inputs_require_grad"]:
                    original_reshard()
                else:
                    backward_state["retained_module"] = module
                    backward_state["reshard"] = original_reshard

        def post_backward() -> None:
            if state["grads_finished"]:
                return
            module._reshard_parameter_groups = lambda: None
            try:
                original_post_backward()
            finally:
                module._reshard_parameter_groups = original_reshard
            state["grads_finished"] = True
            maybe_reshard()

        def finish_module_backward(_module, _grad_input, _grad_output) -> None:
            if state["inputs_require_grad"]:
                state["module_finished"] = True
                maybe_reshard()
            elif not state["completion_queued"]:
                state["module_finished"] = True
                maybe_reshard()
                state["completion_queued"] = True

                def finish_at_autograd_completion() -> None:
                    if backward_state["retained_module"] is module:
                        backward_state["reshard"]()
                        backward_state["retained_module"] = None
                        backward_state["reshard"] = None

                torch.autograd.Variable._execution_engine.queue_callback(finish_at_autograd_completion)

        module.post_backward = post_backward
        module.register_forward_pre_hook(reset_state)
        module.register_full_backward_hook(finish_module_backward)

    @staticmethod
    def _disable_backward_prefetch(
        module: nn.Module,
        backward_state: dict | None = None,
    ) -> None:
        """Avoid overlapping two full parameter units during backward.

        Large PEFT units may need to retain frozen weights until their backward
        finishes. Prefetching the next full unit can exceed 48 GB-class device
        capacity, so the initial DP-only adapter favors bounded memory here.
        """
        if not isinstance(module, FsdpModule):
            return
        if backward_state is None:
            backward_state = {"retained_module": None, "reshard": None}
        original_pre_backward = module.pre_backward

        def pre_backward() -> None:
            retained_module = backward_state["retained_module"]
            if retained_module is not None and retained_module is not module:
                backward_state["reshard"]()
                backward_state["retained_module"] = None
                backward_state["reshard"] = None
            order = module.context.backward_order
            original_next_item = order.next_item
            order.next_item = lambda _item: None
            try:
                original_pre_backward()
            finally:
                order.next_item = original_next_item

        module.pre_backward = pre_backward

    def parallelize(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[nn.Module, torch.optim.Optimizer | None]:
        """Shard selected model blocks and the root module in place.

        Args:
            model: BF16 CPU-materialized or meta-initialized module to shard onto the DP mesh.
            optimizer: Optional optimizer to adapt after model sharding.

        Returns:
            The original model and optimizer objects after in-place adaptation.
        """
        if dist.get_world_size() == 1:
            logger.info("World size is 1, skipping Megatron FSDP sharding.")
            model.to(device=torch.device(self.device_mesh.device_type))
            if self.activation_checkpointing:
                checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
                if checkpointing_enable is None:
                    raise ValueError("Model does not support activation checkpointing.")
                checkpointing_enable()
            return model, optimizer

        parameter_names_by_id = (
            {id(parameter): name for name, parameter in model.named_parameters()} if optimizer is not None else {}
        )
        dp_mesh = self._dp_mesh()
        placements = self._flat_placements()
        if self.megatron_fsdp_unit_modules:
            unit_types = import_classes_from_paths(self.megatron_fsdp_unit_modules)
        else:
            unit_types = _derive_megatron_fsdp_unit_modules(model)

        selected_modules = [
            module for module in model.modules() if module is not model and isinstance(module, tuple(unit_types))
        ]
        if not selected_modules:
            raise ValueError("No modules matched distributed.megatron_fsdp_unit_modules.")
        backward_state = {"retained_module": None, "reshard": None}
        for module in reversed(self._protected_fp32_modules(model)):
            fully_shard(
                module,
                mesh=dp_mesh,
                placements=placements,
                mixed_precision_policy=self._fp32_policy(),
            )
            self._defer_reshard_until_module_backward(module, backward_state)
            self._disable_backward_prefetch(module, backward_state)
        for module in reversed(selected_modules):
            fully_shard(
                module,
                mesh=dp_mesh,
                placements=placements,
                mixed_precision_policy=self.mp_policy,
            )
            self._defer_reshard_until_module_backward(module, backward_state)
            self._disable_backward_prefetch(module, backward_state)
        fully_shard(
            model,
            mesh=dp_mesh,
            placements=placements,
            mixed_precision_policy=self.mp_policy,
        )
        self._disable_backward_prefetch(model, backward_state)
        self._register_external_output_embedding_hooks(model)
        self._register_structured_output_hooks(model)

        if self.activation_checkpointing:
            checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
            if checkpointing_enable is None:
                raise ValueError("Model does not support activation checkpointing.")
            checkpointing_enable()

        if optimizer is not None:
            self._remap_optimizer_parameters(model, optimizer, parameter_names_by_id)
            fully_shard_optimizer(optimizer)
        return model, optimizer

    @staticmethod
    def sync_model_weights(model: nn.Module) -> None:
        """Refresh MFSDP compute weights after initialization or checkpoint loading.

        Args:
            model: Fully sharded model whose main weights were updated.
        """
        for module in model.modules():
            if isinstance(module, FsdpModule):
                for parameter_group in module.parameter_groups:
                    parameter_group.sync_model_weight_from_main_weight()


def maybe_shard_optimizer(
    model_part: nn.Module,
    optimizer: torch.optim.Optimizer,
    distributed_config: "DistributedConfig | None",
    *,
    allow: bool = True,
) -> torch.optim.Optimizer:
    """Adapt an optimizer when the model uses the Megatron FSDP strategy.

    Args:
        model_part: Sharded model part associated with the optimizer.
        optimizer: Optimizer to adapt in place.
        distributed_config: Active distributed strategy configuration.
        allow: Whether this optimizer supports MFSDP adaptation.

    Returns:
        The original optimizer, adapted when MFSDP is active.
    """
    if isinstance(distributed_config, MegatronFSDPConfig) and dist.get_world_size() > 1:
        if not allow:
            raise ValueError("This optimizer does not support fully_shard_optimizer.")
        if not isinstance(model_part, FsdpModule):
            raise ValueError("The optimizer's model must be sharded with MFSDP v2 before adaptation.")
        fully_shard_optimizer(optimizer)
    return optimizer
