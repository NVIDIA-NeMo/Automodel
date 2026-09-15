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

import importlib
import inspect
import logging
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from types import FunctionType
from typing import Any, Dict, Generator, List, Sequence, Tuple, Union

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.activation_checkpointing import (
    SELECTIVE_AC_WRAPPER_FLAG,
    apply_full_layer_checkpointing_to_layers,
    apply_selective_checkpointing_to_layers,
    apply_submodule_checkpointing,
    detect_kv_sharing_and_maybe_disable_cache,
    is_selective_activation_checkpointing,
    query_activation_checkpointing_spec,
    unwrap_checkpoint_wrapper,
)
from nemo_automodel.components.distributed.config import (
    ActivationCheckpointingScope,
    normalize_activation_checkpointing_scope,
)
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.optimized_tp_plans import VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec, query_parallel_spec
from nemo_automodel.components.distributed.parallel_styles import translate_to_lora
from nemo_automodel.shared.import_utils import UnavailableMeta, safe_import_from
from nemo_automodel.shared.multimodal_fsdp import (
    MULTIMODAL_TOWER_NAMES,
    FrozenMultimodalSharding,
    ignored_params_for_root,
    is_multimodal_module_name,
    iter_multimodal_modules,
    module_is_fully_frozen,
    module_parameters,
    normalize_frozen_multimodal_sharding,
)
from nemo_automodel.shared.tied_weights import ensure_tied_lm_head
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_accumulated_grad_guard as _patch_fsdp_accumulated_grad_guard,
)

_MEGATRON_FSDP_050_REQUIRED_MSG = (
    "megatron_fsdp.MixedPrecisionPolicy could not be imported: NeMo Automodel requires megatron-fsdp==0.5.0"
)

HAVE_MEGATRON_FSDP = False
logging.getLogger("megatron_fsdp").setLevel(logging.WARNING)
try:
    from megatron_fsdp import fully_shard as megatron_fsdp_fully_shard
    from megatron_fsdp import fully_shard_model as megatron_fsdp_fully_shard_model

    # megatron-fsdp==0.5.0, the only supported release, always exports
    # MixedPrecisionPolicy. safe_import_from keeps module import safe on any
    # other install; constructing the returned placeholder then raises
    # _MEGATRON_FSDP_050_REQUIRED_MSG instead of silently degrading.
    _, MegatronFSDPMixedPrecisionPolicy = safe_import_from(
        "megatron_fsdp", "MixedPrecisionPolicy", msg=_MEGATRON_FSDP_050_REQUIRED_MSG
    )

    HAVE_MEGATRON_FSDP = True
except (ImportError, FileNotFoundError, OSError):
    # megatron_fsdp itself is unavailable; every use is already guarded by
    # HAVE_MEGATRON_FSDP, and this placeholder fails loudly like the one above.
    MegatronFSDPMixedPrecisionPolicy = UnavailableMeta(
        "MixedPrecisionPolicy", (), {"_msg": _MEGATRON_FSDP_050_REQUIRED_MSG}
    )

# Import as module so tests can patch nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype
import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# One-time flag: megatron-fsdp 0.5.0 removed the legacy buffer-level NaN check,
# so a truthy check_for_nan_in_grad is dropped by _megatron_fsdp_compat_kwargs
# and only warned about once per process.
_megatron_fsdp_nan_check_noop_warned = False


def apply_selective_activation_checkpointing(
    model: nn.Module,
    *,
    enable_compile: bool = False,
    activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
) -> None:
    """Apply selective activation checkpointing to ``model`` end to end.

    Standalone entry point (detects KV-sharing, disables ``use_cache``, and
    wraps transformer blocks) for paths where the FSDP2 parallelize flow is
    skipped -- notably single-GPU training.

    Args:
        model: The model to checkpoint.
        enable_compile: Whether per-layer ``torch.compile`` will be applied.
        activation_checkpointing_scope: Which extracted layer groups to wrap.
    """
    layer_groups = _extract_model_layer_groups(model)
    layers, _ = _filter_layer_groups_for_activation_checkpointing(layer_groups, activation_checkpointing_scope)
    if not layers:
        logger.warning("No transformer layers found; skipping selective activation checkpointing.")
        return
    has_kv_sharing = detect_kv_sharing_and_maybe_disable_cache(model)
    apply_selective_checkpointing_to_layers(model, layers, has_kv_sharing, enable_compile=enable_compile)


class ParallelizationStrategy(ABC):
    """Abstract base class for model parallelization strategies."""

    @abstractmethod
    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reshard_after_forward: bool | None = None,
        activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        """Apply parallelization strategy to the model."""
        pass


def _fully_shard_untied_input_output_embeddings(
    model: nn.Module,
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: OffloadPolicy | None,
    input_reshard_after_forward: bool,
    fully_shard_fn: Callable[..., nn.Module],
) -> None:
    """Give large trainable untied embedding tables independent FSDP buffers.

    The generic dense path otherwise leaves both tables in the root FSDP unit.
    With fp32 gradient reduction, that unit allocates one contiguous
    reduce-scatter input containing both gradients. Keeping the two trainable
    leaf modules in separate FSDP units bounds that allocation by the larger
    table instead of their sum. Tied weights stay in one unit to preserve
    aliasing, and frozen tables stay in the root because they have no gradient
    communication buffer to split.

    Args:
        model: Model whose input and output embedding modules may be sharded.
        mesh: Device mesh that owns the FSDP shards.
        mp_policy: Mixed-precision policy used by the surrounding FSDP units.
        offload_policy: Optional offload policy used by the surrounding FSDP
            units.
        input_reshard_after_forward: Whether the input embedding unit reshards
            its parameters after forward.
        fully_shard_fn: FSDP sharding callable, injectable for unit tests.
    """
    weights_are_tied = ensure_tied_lm_head(model)

    def _resolve(getter_name: str) -> nn.Module | None:
        getter = getattr(model, getter_name, None)
        if not callable(getter):
            return None
        try:
            module = getter()
        except (AttributeError, NotImplementedError):
            return None
        return module if isinstance(module, nn.Module) else None

    input_embeddings = _resolve("get_input_embeddings")
    output_embeddings = _resolve("get_output_embeddings")
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    weights_are_physically_tied = input_embeddings is not None and (
        input_embeddings is output_embeddings or (input_weight is not None and input_weight is output_weight)
    )
    if weights_are_tied or weights_are_physically_tied:
        logger.info("Keeping tied input/output embeddings in the root FSDP unit")
        return

    seen: set[int] = set()
    for role, module, module_reshard_after_forward in (
        ("input embedding", input_embeddings, input_reshard_after_forward),
        # The output projection is the last compute unit. Keep it gathered until
        # backward, matching the old root-owned behavior and allowing
        # FusedLinearCrossEntropy to consume its mixed-precision compute weight
        # outside the module's forward.
        ("output embedding", output_embeddings, False),
    ):
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        if not any(param.requires_grad for param in module.parameters()):
            continue
        fully_shard_fn(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=module_reshard_after_forward,
            offload_policy=offload_policy,
        )
        logger.info("Sharded %s as an independent FSDP unit", role)


class DefaultParallelizationStrategy(ParallelizationStrategy):
    """Default parallelization strategy used by most models."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        enable_async_tensor_parallel: bool = False,
        enable_compile: bool = False,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: bool | None = None,
        activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        fully_shard_fn=None,
    ) -> nn.Module:
        """Apply the default parallelization flow."""
        frozen_multimodal_sharding = normalize_frozen_multimodal_sharding(frozen_multimodal_sharding)
        tp_mesh = device_mesh[tp_mesh_name]
        if fully_shard_fn is None:
            fully_shard_fn = fully_shard

        # Set FSDP sharding mesh to context parallel mesh if CP > 1, else default to the data parallel mesh.
        # if dp_replicate_size > 1, use HSDP, else use FSDP
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        pp_enabled = "pp" in dp_mesh.mesh_dim_names and dp_mesh["pp"].size() > 1
        if pp_enabled and reshard_after_forward is True:
            logger.warning(
                "reshard_after_forward=True overrides the pipeline-parallel default of keeping layer weights "
                "gathered across microbatches. This may increase per-microbatch all-gathers and reduce throughput."
            )

        # Extract layers from the model for parallelization.
        layer_groups = _extract_model_layer_groups(model)
        spec = query_parallel_spec(model)

        # TP sharding with enhanced plan generation
        if tp_mesh.size() > 1:
            # async-TP (_micro_pipeline_tp) overlaps ReduceScatter with compute.
            # Without SP, row-parallel layers emit AllReduce (not ReduceScatter),
            # so there is nothing for the micro-pipeline to overlap — force SP on.
            if enable_async_tensor_parallel and not sequence_parallel:
                raise ValueError("enable_async_tensor_parallel=True requires sequence_parallel=True")

            # Generate or use tensor parallel plan
            model_parallel_plan = {
                k: translate_to_lora(v)
                for k, v in _get_parallel_plan(
                    model,
                    sequence_parallel,
                    tp_shard_plan,
                    tp_size=tp_mesh.size(),
                ).items()
            }

            # Head counts must divide the TP size only when the plan splits attention across heads;
            # a plan that shards MLPs alone (hybrid Mamba stacks, diffusion transformers) or keeps
            # attention replicated places no such constraint.
            if _attention_is_head_sharded(model_parallel_plan):
                validate_tp_mesh(model, tp_mesh)

            # Apply tensor parallelism
            if model_parallel_plan:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*could not be resolved.*",
                        category=UserWarning,
                    )
                    parallelize_module(model, tp_mesh, model_parallel_plan)
                # TP styles replace module weights with DTensors independently.
                # Restore an architectural embedding/head alias before FSDP
                # records ownership; re-tying after FSDP can leave two roots
                # that disagree about the shared parameter.
                ensure_tied_lm_head(model)
                if _attention_is_head_sharded(model_parallel_plan):
                    _update_attention_head_counts_for_tp(model, tp_mesh.size(), layer_groups)

            if enable_async_tensor_parallel:
                torch._inductor.config._micro_pipeline_tp = True
                logger.info("Async tensor parallel enabled — ensure torch.compile is also enabled")
                # Enable symmetric memory for the TP group so Inductor's
                # fused_all_gather_matmul and fused_matmul_reduce_scatter kernels
                # can fire (both are gated on is_symm_mem_enabled_for_group).
                if tp_mesh.size() > 1:
                    try:
                        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

                        tp_group_name = tp_mesh._dim_group_names[0]
                        enable_symm_mem_for_group(tp_group_name)
                        logger.info(f"Symmetric memory enabled for TP group '{tp_group_name}'")
                    except Exception as e:
                        logger.warning(f"Could not enable symmetric memory for TP group: {e}")

        # Apply activation checkpointing to transformer blocks if requested
        if activation_checkpointing:
            _has_kv_sharing = detect_kv_sharing_and_maybe_disable_cache(model)
            ac_layers, ac_scopes = self.select_activation_checkpointing_layers(
                model, layer_groups, activation_checkpointing_scope
            )

            if is_selective_activation_checkpointing(activation_checkpointing):
                apply_selective_checkpointing_to_layers(
                    model,
                    ac_layers,
                    _has_kv_sharing,
                    enable_compile=enable_compile,
                )
            elif query_activation_checkpointing_spec(model).granularity == "layer":
                apply_full_layer_checkpointing_to_layers(model, ac_layers)
                logger.info(
                    "Using the model's declared whole-layer activation checkpointing; skipping submodule wrappers."
                )
            elif enable_compile:
                # NO_REENTRANT is required for compile: REENTRANT's first forward runs under
                # no_grad, causing AOT autograd to trace a forward-only graph that drops LoRA
                # (and other trainable) weight gradients.  Wrapping must happen BEFORE FSDP2
                # sharding so the module structure is stable when fully_shard() indexes params.
                for layer in ac_layers:
                    for attr in ("self_attn", "attention", "attn", "linear_attn", "mlp", "feed_forward", "ffn"):
                        m = getattr(layer, attr, None)
                        if m is not None:
                            setattr(layer, attr, checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT))
            else:
                if _should_use_hf_native_gradient_checkpointing(
                    model,
                    layer_groups,
                    ac_scopes,
                    enable_compile=enable_compile,
                ) and (not _has_kv_sharing or _kv_sharing_survives_checkpoint_replay(model)):
                    # Work around a PyTorch FSDP2 bug that skips mixed-precision input casts during
                    # checkpoint recomputation. Remove when the minimum PyTorch version is 2.13.
                    apply_full_layer_checkpointing_to_layers(model, ac_layers)
                else:
                    apply_submodule_checkpointing(ac_layers, _has_kv_sharing)

        if reapply_trainability is not None:
            reapply_trainability(model)

        # Evaluate frozen-module ownership only after TP/AC transformations and
        # trainability rebinding so FSDP sees the final module hierarchy.
        frozen_multimodal_modules = [
            name for name, module in iter_multimodal_modules(model) if module_is_fully_frozen(module)
        ]
        if frozen_multimodal_sharding == "per_layer" and frozen_multimodal_modules:
            logger.warning(
                "distributed.multimodal.frozen_sharding='per_layer' selected for %s. Every rank in the FSDP "
                "group must execute or skip these modules the same number of times and in the same order on every "
                "microbatch; rank-asymmetric modality execution can hang or desynchronize FSDP collectives.",
                ", ".join(frozen_multimodal_modules),
            )

        # Set up mixed precision policy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        # Install this only when NeMo actually enters FSDP2 sharding.
        _patch_fsdp_accumulated_grad_guard()

        ignored_multimodal_params: set[nn.Parameter] = set()

        # Find transformer layers and apply parallelisms
        apply_fsdp2_sharding_recursively(
            model,
            dp_mesh,
            mp_policy,
            offload_policy,
            enable_fsdp2_prefetch,
            fsdp2_backward_prefetch_depth,
            fsdp2_forward_prefetch_depth,
            reshard_after_forward,
            fully_shard_fn=fully_shard_fn,
            frozen_multimodal_sharding=frozen_multimodal_sharding,
            ignored_multimodal_params=ignored_multimodal_params,
            shard_by_dtype=spec.shard_by_dtype,
            fp32_compute_module_names=tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ()),
        )

        input_embedding_reshard_after_forward = (
            reshard_after_forward if reshard_after_forward is not None else not pp_enabled
        )
        _fully_shard_untied_input_output_embeddings(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            input_reshard_after_forward=input_embedding_reshard_after_forward,
            fully_shard_fn=fully_shard_fn,
        )

        # Apply FSDP to the root model
        # Do not reshard after forward for root model because its parameters
        # will be used in backward immediately
        root_ignored_params = ignored_params_for_root(model, ignored_multimodal_params)
        root_kwargs = {
            "mesh": dp_mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": False,
            "offload_policy": offload_policy,
        }
        if root_ignored_params is not None:
            root_kwargs["ignored_params"] = root_ignored_params
        model = fully_shard_fn(model, **root_kwargs)

        cp_enabled = "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1
        if cp_enabled:
            configured_units = parallelizer_utils.configure_fsdp_unused_param_reduction(model)
            logger.info(
                "Enabled unused-parameter reduce-scatter on %d FSDP units for context parallelism",
                configured_units,
            )

        return model

    def select_activation_checkpointing_layers(
        self,
        model: nn.Module,
        layer_groups: Dict[str, List[nn.Module]],
        activation_checkpointing_scope: ActivationCheckpointingScope | None,
    ) -> Tuple[List[nn.Module], Tuple[str, ...]]:
        """Choose the layers activation checkpointing wraps: the scope's trainable layer groups.

        A strategy whose stack mixes block kinds narrows this selection instead of re-implementing
        the flow (Nemotron-H checkpoints its MLP and Mamba blocks but not its attention blocks).

        Args:
            model: The model being parallelized.
            layer_groups: Its transformer blocks by role (see :func:`get_model_layer_groups`).
            activation_checkpointing_scope: The requested scope (``"all"``, roles or ``"multimodal"``).

        Returns:
            The selected layers and the normalized scope tuple.
        """
        del model
        return _filter_layer_groups_for_activation_checkpointing(layer_groups, activation_checkpointing_scope)


def _patch_dtensor_spec_hash_for_symint() -> None:
    """Fix a crash when torch.compile + DTensor are used together.

    Problem: torch.compile traces with symbolic shapes (SymInt). DTensorSpec hashes
    its shape to cache sharding decisions, but SymInt is not hashable -> crash.

    Fix: if hashing the shape fails, fall back to hashing only (mesh, placements).
    Cache hits are slightly reduced but correctness is unaffected.
    """
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    if getattr(DTensorSpec, "_symint_hash_patched", False):
        return

    _original_hash_impl = DTensorSpec._hash_impl

    def _hash_impl_symint_safe(self) -> int:
        try:
            return _original_hash_impl(self)
        except TypeError:
            return hash((self.mesh, self.placements))

    DTensorSpec._hash_impl = _hash_impl_symint_safe
    DTensorSpec._symint_hash_patched = True


def _apply_per_layer_compile(model: nn.Module) -> None:
    """Compile each decoder layer in-place after FSDP2 sharding.

    Compiles at decoder-layer granularity (not sub-module) so that AOT autograd traces
    the joint fwd+bwd graph under the training loop's enable_grad context.  Sub-module
    compile (e.g. on mlp alone) would be traced during activation checkpointing's first
    forward pass which runs under no_grad, producing a forward-only graph that drops
    LoRA and other trainable-parameter gradients.

    Prerequisite: NO_REENTRANT checkpoint_wrapper must already be applied to self_attn
    and mlp before FSDP2 sharding (done in DefaultParallelizationStrategy).  This
    function only handles the compile step.

    Whole-block selective-AC wrappers (tagged with ``SELECTIVE_AC_WRAPPER_FLAG``)
    are compiled OUTER -- the wrapper itself is compiled so the selective policy
    is traced and the partitioner honors its recompute tags. Other layer-level
    CheckpointWrappers (e.g. the PP path) are unwrapped and the decoder layer is
    compiled directly.

    nn.Module.compile() is used instead of torch.compile() to compile in-place without
    introducing an _orig_mod wrapper, which would add a key prefix and break checkpoint
    loading.

    _patch_dtensor_spec_hash_for_symint() is called to allow torch.compile with dynamic
    shapes to coexist with DTensor's lru_cache-based sharding propagation.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    _patch_dtensor_spec_hash_for_symint()

    compiled_count = 0
    compiled_modules: set[int] = set()

    def _compile_target(layer: nn.Module) -> nn.Module:
        # Whole-block selective-AC wrappers must be compiled OUTER so the SAC
        # policy is traced and the partitioner honors its recompute tags.
        # Other CheckpointWrappers (e.g. PP full-layer wrap with sub-module AC
        # inside) are unwrapped so the decoder layer is compiled directly.
        if isinstance(layer, CheckpointWrapper):
            if getattr(layer, SELECTIVE_AC_WRAPPER_FLAG, False):
                return layer
            return layer._checkpoint_wrapped_module
        return layer

    def _compile_module_list(module_list: nn.ModuleList | nn.ModuleDict) -> None:
        nonlocal compiled_count
        # PP converts model.model.layers from nn.ModuleList to nn.ModuleDict (str keys).
        # enumerate(nn.ModuleDict) yields string keys, not modules -- use .items() instead.
        items = module_list.items() if isinstance(module_list, nn.ModuleDict) else enumerate(module_list)
        for _, layer in items:
            actual_layer = _compile_target(layer)
            module_id = id(actual_layer)
            if module_id in compiled_modules:
                continue
            actual_layer.compile()
            compiled_modules.add(module_id)
            compiled_count += 1

    module_lists = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        module_lists.append(model.model.layers)
    if hasattr(model, "layers"):
        module_lists.append(model.layers)
    for attr_name in ("transformer_blocks", "single_transformer_blocks"):
        module_list = getattr(model, attr_name, None)
        if isinstance(module_list, (nn.ModuleList, nn.ModuleDict)):
            module_lists.append(module_list)

    if module_lists:
        for module_list in module_lists:
            _compile_module_list(module_list)
    else:
        logger.warning("_apply_per_layer_compile: using heuristic layer extraction")
        for layer in _extract_model_layers(model):
            actual_layer = _compile_target(layer)
            module_id = id(actual_layer)
            if module_id in compiled_modules:
                continue
            actual_layer.compile()
            compiled_modules.add(module_id)
            compiled_count += 1

    logger.info("Per-layer torch.compile applied to %d decoder layers", compiled_count)


def apply_fsdp2_sharding_recursively(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None = None,
    enable_fsdp2_prefetch: bool = True,
    fsdp2_backward_prefetch_depth: int = 2,
    fsdp2_forward_prefetch_depth: int = 1,
    reshard_after_forward: bool | None = None,
    fully_shard_fn=None,
    frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
    ignored_multimodal_params: set[nn.Parameter] | None = None,
    shard_by_dtype: bool = False,
    fp32_compute_module_names: Tuple[str, ...] = (),
) -> None:
    """
    Recursively apply FSDP2 sharding to modules, with optimizations for ModuleList.

    This utility function traverses a model hierarchy and applies FSDP2 sharding
    to each module. For ModuleList instances (commonly used for transformer layers),
    it applies an optimization where the last layer doesn't reshard after forward
    since FSDP2 will prefetch it immediately.

    Handles both single-level and nested ModuleList/ModuleDict structures. If a
    ModuleList contains other ModuleLists, it will recurse into them instead of trying
    to wrap them (since ModuleList doesn't have a forward method).

    Args:
        module (nn.Module): The module to apply FSDP sharding to.
        mesh (DeviceMesh): The device mesh for FSDP sharding.
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy for FSDP.
        offload_policy (Optional[OffloadPolicy]): CPU offload policy for FSDP.
            Defaults to None.
        enable_fsdp2_prefetch (bool): Enable explicit forward/backward prefetch chains.
        fsdp2_backward_prefetch_depth (int): Backward prefetch depth.
        fsdp2_forward_prefetch_depth (int): Forward prefetch depth.
        reshard_after_forward (Optional[bool]): Optional override for each layer's
            ``fully_shard`` reshard behavior.
        frozen_multimodal_sharding: Whether fully frozen multimodal modules are
            owned by the root FSDP unit, sharded per layer, or replicated.
        ignored_multimodal_params: Accumulator for replicated frozen multimodal
            parameters that must be ignored by ancestor FSDP roots.
        shard_by_dtype: Shard each layer with ``parallelizer_utils.fully_shard_by_dtype`` so
            its fp32-pinned parameters get their own fp32 compute unit (``ParallelSpec.shard_by_dtype``).
        fp32_compute_module_names: Parameter-name substrings that must compute in fp32 under
            ``shard_by_dtype``; the model's ``_keep_in_fp32_modules_strict``.
    Note:
        This function modifies the module in-place by replacing modules with their
        FSDP2-subclassed versions.
    """
    frozen_multimodal_sharding = normalize_frozen_multimodal_sharding(frozen_multimodal_sharding)
    if fully_shard_fn is None:
        fully_shard_fn = fully_shard

    pp_enabled = "pp" in mesh.mesh_dim_names and mesh["pp"].size() > 1

    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        # After pipeline splitting, functional.py replaces nn.ModuleList with nn.ModuleDict
        # (keyed by string layer indices). Normalise both to a list of (key, child) pairs.
        if isinstance(module, nn.ModuleDict):
            all_items = list(module.items())
            _is_container = lambda c: isinstance(c, (nn.ModuleList, nn.ModuleDict))
        else:
            all_items = [(i, module[i]) for i in range(len(module))]
            _is_container = lambda c: isinstance(c, nn.ModuleList)

        flat_layer_items = [(k, c) for k, c in all_items if not _is_container(c)]
        nested_items = [(k, c) for k, c in all_items if _is_container(c)]
        nested_lists = nested_items  # kept for len() checks below

        # Recurse into any nested ModuleLists first (unchanged behavior).
        for layer_id, child_module in nested_lists:
            apply_fsdp2_sharding_recursively(
                child_module,
                mesh,
                mp_policy,
                offload_policy,
                enable_fsdp2_prefetch,
                fsdp2_backward_prefetch_depth,
                fsdp2_forward_prefetch_depth,
                reshard_after_forward,
                fully_shard_fn=fully_shard_fn,
                frozen_multimodal_sharding=frozen_multimodal_sharding,
                ignored_multimodal_params=ignored_multimodal_params,
                shard_by_dtype=shard_by_dtype,
                fp32_compute_module_names=fp32_compute_module_names,
            )

        for enum_id, (layer_key, child_module) in enumerate(flat_layer_items):
            # With PP: keep weights gathered across microbatches (no per-microbatch all-gather).
            # Without PP: reshard all but last layer to enable forward+backward weight prefetching.
            if reshard_after_forward is not None:
                layer_reshard_after_forward = reshard_after_forward
            elif pp_enabled:
                layer_reshard_after_forward = False
            else:
                layer_reshard_after_forward = enum_id < len(flat_layer_items) - 1
            if shard_by_dtype:
                parallelizer_utils.fully_shard_by_dtype(
                    child_module,
                    mesh,
                    mp_policy,
                    offload_policy,
                    fp32_compute_module_names=fp32_compute_module_names,
                    reshard_after_forward=layer_reshard_after_forward,
                    fully_shard_fn=fully_shard_fn,
                )
            else:
                fully_shard_fn(
                    child_module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=layer_reshard_after_forward,
                    offload_policy=offload_policy,
                )
            module[layer_key] = child_module

        # Set up explicit forward/backward prefetch chains when layers are being resharded.
        # With PP or an explicit no-reshard override, weights are always gathered -- no prefetch needed.
        if reshard_after_forward is False:
            should_prefetch = False
        else:
            should_prefetch = not pp_enabled and enable_fsdp2_prefetch
        if should_prefetch:
            # Only layers that became FSDP units take part: ``fully_shard_by_dtype`` leaves a layer whose
            # parameters split into three or more compute dtypes unwrapped and shards its subtrees instead.
            fsdp_units = [c for _, c in flat_layer_items if hasattr(c, "set_modules_to_forward_prefetch")]
            if fsdp2_forward_prefetch_depth > 0:
                for i in range(len(fsdp_units) - 1):
                    targets = [
                        fsdp_units[i + j] for j in range(1, fsdp2_forward_prefetch_depth + 1) if i + j < len(fsdp_units)
                    ]
                    if targets:
                        fsdp_units[i].set_modules_to_forward_prefetch(targets)
            for i in range(1, len(fsdp_units)):
                targets = []
                for d in range(1, fsdp2_backward_prefetch_depth + 1):
                    if i - d >= 0:
                        targets.append(fsdp_units[i - d])
                if targets:
                    fsdp_units[i].set_modules_to_backward_prefetch(targets)
    else:
        for name, sub_module in module.named_children():
            if is_multimodal_module_name(name) and module_is_fully_frozen(sub_module):
                if frozen_multimodal_sharding in ("root", "replicate"):
                    logger.info(
                        "Keeping frozen multimodal module %s at FSDP policy %s",
                        name,
                        frozen_multimodal_sharding,
                    )
                    if frozen_multimodal_sharding == "replicate" and ignored_multimodal_params is not None:
                        ignored_multimodal_params.update(module_parameters(sub_module))
                    continue
            apply_fsdp2_sharding_recursively(
                sub_module,
                mesh,
                mp_policy,
                offload_policy,
                enable_fsdp2_prefetch,
                fsdp2_backward_prefetch_depth,
                fsdp2_forward_prefetch_depth,
                reshard_after_forward,
                fully_shard_fn=fully_shard_fn,
                frozen_multimodal_sharding=frozen_multimodal_sharding,
                ignored_multimodal_params=ignored_multimodal_params,
                shard_by_dtype=shard_by_dtype,
                fp32_compute_module_names=fp32_compute_module_names,
            )


def has_hf_tp_plan(model: nn.Module) -> bool:
    """Whether :func:`get_hf_tp_shard_plan` has a HuggingFace ``_tp_plan`` to translate (class or instance)."""
    return getattr(type(model), "_tp_plan", None) is not None or getattr(model, "_tp_plan", None) is not None


def _input_embedding_fqn(model: nn.Module) -> str | None:
    """FQN of the module ``model.get_input_embeddings()`` returns, or ``None`` when the model has no such API."""
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        return None
    try:
        embedding = get_input_embeddings()
    except Exception:
        return None
    return next((name for name, module in model.named_modules() if module is embedding), None)


def _plan_pattern_resolves(model: nn.Module, pattern: str) -> bool:
    """Whether a ``parallelize_module`` key (``*`` = one path segment) matches at least one module of ``model``."""
    regex = re.compile("^" + re.escape(pattern).replace(r"\*", r"[^.]+") + "$")
    return any(regex.match(name) for name, _ in model.named_modules())


def get_hf_tp_shard_plan(model):
    """Translate the HuggingFace ``_tp_plan`` transformers assembled on ``model`` into ``ParallelStyle`` objects.

    transformers copies the class plan onto the instance and merges the config's ``base_model_tp_plan``
    and every child's plan under the child's attribute name (``DistributedMixin.init_parallel_plans``),
    so the keys are already fully qualified and no model-specific root is needed. On top of that:

    - the input embedding (``model.get_input_embeddings()``) is row-sharded when the plan omits it;
    - ``lm_head`` keeps a vocab-sharded output instead of HF's replicated one (a speed-up);
    - HF's MoE-only styles are skipped (their weights stay replicated, see below);
    - keys that match no module are reported, since ``parallelize_module`` would silently ignore them.

    Taken and modified from: https://github.com/NVIDIA/NeMo/blob/6c6169db01bcca73ae8ad3ac35242fadbb9a78ba/nemo/lightning/pytorch/strategies/utils.py#L532

    Args:
        model: A Hugging Face model instance

    Returns:
        dict: A dictionary mapping model component paths to their parallelization strategies

    Raises:
        AssertionError: If no TP plan is found
    """
    model_cls = type(model)
    hf_tp_plan = {}
    if getattr(model_cls, "_tp_plan", None) is not None:
        assert isinstance(model_cls._tp_plan, dict), f"model_cls._tp_plan is not a dict: {model_cls._tp_plan}"
        hf_tp_plan.update(model_cls._tp_plan)
    if getattr(model, "_tp_plan", None) is not None:
        hf_tp_plan.update(model._tp_plan)

    assert len(hf_tp_plan) > 0, (
        f"Hugging Face tp plan is not supported for {model_cls}, please set dtensor_cfg.tensor_parallel_size to 1 or provide a custom_parallel_plan. "
        "The usage example of custom_parallel_plan can refer to `docs/design-docs/fsdp2-parallel-plan.md`."
    )

    # HF plans rarely include the input embedding; shard it by vocabulary rows.
    embedding_fqn = _input_embedding_fqn(model)
    if embedding_fqn is not None and embedding_fqn not in hf_tp_plan:
        hf_tp_plan[embedding_fqn] = "rowwise_rep"

    # Build translated plan, skipping HF's MoE-related styles.
    #
    # HuggingFace transformers v5 introduced these styles for MoE models, but they do NOT
    # implement true expert parallelism (where each rank stores only a subset of experts).
    # Instead, HF's approach:
    # - local_colwise/local_rowwise: Store expert weights as local tensors (NOT sharded).
    #   Despite the names, these do NOT perform tensor parallelism on the experts.
    #   Each rank stores ALL expert weights (full shape), which is memory inefficient.
    # - ep_router: Modifies routing so each rank only computes with a subset of experts.
    #   This distributes compute but not memory.
    # - gather: All-reduces expert outputs across ranks.
    # - packed_colwise/packed_rowwise: 3-D packed expert weights (Llama 4).
    #
    # Since these styles result in replicated expert weights (not sharded), and we don't
    # support HF's routing modification approach, we skip them entirely. The experts will
    # be replicated across all ranks and computed redundantly, which is correct but not
    # memory/compute efficient for large MoE models.
    _hf_moe_styles = {"ep_router", "local_colwise", "local_rowwise", "gather", "packed_colwise", "packed_rowwise"}
    translated_plan = {}
    for k, v in hf_tp_plan.items():
        if isinstance(v, str) and (v.startswith("ep_") or v in _hf_moe_styles):
            continue
        # speed up the tp plan for lm_head
        if (k == "lm_head" or k == "language_model.lm_head") and v == "colwise_rep":
            translated_plan[k] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
        else:
            style = translate_to_torch_parallel_style(v)
            # Translator returns None for styles that should be skipped (e.g.
            # "replicated_with_grad_allreduce" under FSDP where leaving the
            # param un-wrapped is equivalent).
            if style is None:
                continue
            translated_plan[k] = style

    unresolved = [k for k in translated_plan if not _plan_pattern_resolves(model, k)]
    if unresolved:
        logger.warning(
            "HF tp plan entries for %s match no module and will shard nothing: %s", model_cls.__name__, unresolved
        )
    logger.info(f"Hugging Face tp plan: {translated_plan}")
    return translated_plan


def import_class_from_path(name: str) -> Any:
    """Import a class from a string path (e.g. 'torch.optim.AdamW').

    Args:
        full_path: Full path to class including module path and class name

    Returns:
        The imported class object
    """
    module_name, cls_name = name.rsplit(".", 1)
    cls_instance = getattr(importlib.import_module(module_name), cls_name)
    return cls_instance


def import_classes_from_paths(class_paths: List[str]):
    """
    Helper function to import classes from string paths.

    Args:
        class_paths (List[str]): The list of string paths to the classes.

    Returns:
        List of imported classes.
    """
    classes = []
    for path in class_paths:
        try:
            cls = import_class_from_path(path)
            classes.append(cls)
        except Exception as e:
            print(f"Warning: Could not import class from path '{path}': {e}")
    return classes


@lru_cache
def translate_to_torch_parallel_style(style: str):
    """
    Translates string descriptions to parallelism plans.

    In model configurations, we use a neutral type (string) to specify parallel
    styles, here we translate them into torch.distributed tensor-parallel
    types.
    """
    assert isinstance(style, str), f"parallel style type should be str, but got {type(style)}"

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "colwise_gather_output":
        # HF maps this to ColwiseParallel(gather_output=True); gathering the output
        # is the same as replicating it, so this matches "colwise_rep" above.
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "embedding_rowwise":
        # transformers v5 style for vocabulary-sharded embeddings: rows are sharded and the
        # lookup result is all-reduced, which is the "rowwise_rep" embedding above.
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    elif style == "replicated_with_grad_allreduce":
        # transformers v5 style for norm weights (q_norm, k_norm, etc.) that are
        # replicated across TP ranks but need gradient all-reduce. Under FSDP+TP,
        # leaving the param un-wrapped (no TP style) is equivalent: FSDP handles
        # grad sync on its DP/DP_shard mesh, and since the param is replicated on
        # the TP mesh, no TP-level collective is needed in forward.
        return None
    else:
        raise ValueError(f"Unknown parallel style: {style}")


def _attention_is_head_sharded(model_parallel_plan: dict) -> bool:
    """Return True when the TP plan column-wise shards any QKV attention projection.

    When Q/K/V projections use ``ColwiseParallel`` with sharded output (the
    default), each TP rank holds ``num_heads / tp_size`` heads and the model
    config / layer attributes must be updated accordingly.

    Plans that keep attention replicated (e.g. Phi-3 with ``RowwiseParallel``
    on fused QKV and ``Replicate`` output) should *not* trigger a head-count
    update.
    """
    attn_proj_suffixes = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.qkv_proj")
    for key, style in model_parallel_plan.items():
        if not any(key.endswith(s) for s in attn_proj_suffixes):
            continue
        if isinstance(style, ColwiseParallel):
            out = getattr(style, "output_layouts", None)
            if out is None:
                return True
            if isinstance(out, (list, tuple)):
                if any(isinstance(p, Shard) for p in out):
                    return True
            elif isinstance(out, Shard):
                return True
    return False


def _update_attention_head_counts_for_tp(
    model: nn.Module, tp_size: int, layer_groups: Dict[str, List[nn.Module]] | None = None
) -> None:
    """
    After TP sharding, the Q/K/V outputs are split across ranks (each rank has
    num_heads/tp_size heads). Update the config and each attention layer's
    num_heads / num_key_value_heads so the forward uses the local head count
    instead of the global one (avoids shape mismatches in .view()).

    The attention layers are the model's ``language`` layer group (``layer_groups`` when the
    caller already resolved it, :func:`get_model_layer_groups` otherwise).
    """
    if tp_size <= 1:
        return
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "num_attention_heads"):
        return
    if layer_groups is None:
        layer_groups = get_model_layer_groups(model)
    layers = layer_groups.get("language", [])
    if not layers:
        return
    # Preserve the true head_dim before dividing num_attention_heads.
    # RoPE utilities derive head_dim via getattr(config, "head_dim",
    # config.hidden_size // config.num_attention_heads).  Without an
    # explicit head_dim, the division would compute a wrong (too large)
    # head_dim after we halve num_attention_heads for TP.
    if not hasattr(config, "head_dim") or config.head_dim is None:
        config.head_dim = config.hidden_size // config.num_attention_heads
    local_num_attention_heads = config.num_attention_heads // tp_size
    local_num_key_value_heads = None
    if hasattr(config, "num_key_value_heads") and config.num_key_value_heads is not None:
        local_num_key_value_heads = config.num_key_value_heads // tp_size

    for layer in layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "num_heads"):
                attn.num_heads = local_num_attention_heads
            if hasattr(attn, "num_key_value_heads"):
                # Use config's value if set, else derive from local num_heads and num_key_value_groups (e.g. DeciLM)
                if local_num_key_value_heads is not None:
                    attn.num_key_value_heads = local_num_key_value_heads
                elif hasattr(attn, "num_key_value_groups"):
                    attn.num_key_value_heads = local_num_attention_heads // attn.num_key_value_groups
                else:
                    attn.num_key_value_heads = local_num_attention_heads


def validate_tp_mesh(model, tp_mesh):
    """
    Validate that attention heads and key value heads are divisible by TP size
    """
    if tp_mesh.size() == 1:
        return  # if tp_mesh.size() == 1, we don't need to validate

    config = getattr(model, "config", None)
    # Composite (VLM) configs keep the attention head counts on their text config; transformers
    # exposes it uniformly, and non-composite configs return themselves.
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        config = get_text_config()
    num_attention_heads = getattr(config, "num_attention_heads", None) or 0
    num_key_value_heads = getattr(config, "num_key_value_heads", None) or 0

    # TP sharding with enhanced plan generation
    # Validate that attention heads are divisible by TP size
    assert num_key_value_heads % tp_mesh.size() == 0, (
        f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"
    )
    assert num_attention_heads % tp_mesh.size() == 0, (
        f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
    )


_DEFAULT_STRATEGY = DefaultParallelizationStrategy()


def get_parallelization_strategy(model: nn.Module) -> ParallelizationStrategy:
    """Get the appropriate parallelization strategy for the given model."""
    strategy = query_parallel_spec(model).strategy
    return _DEFAULT_STRATEGY if strategy is None else strategy


def _find_largest_module_list(model: nn.Module) -> Union[nn.ModuleList, nn.ModuleDict] | None:
    """
    Heuristic function to find the largest layer container in a model.

    This function recursively traverses the model to find all nn.ModuleList and
    pipeline-split nn.ModuleDict instances and returns the one with the most
    modules. This is useful as a fallback when the model architecture is unknown,
    since transformer layers are typically organized in ModuleLists. Pipeline
    splitting converts ModuleLists to ModuleDicts keyed by original layer index.

    Args:
        model (nn.Module): The model to search through.

    Returns:
        Optional[Union[nn.ModuleList, nn.ModuleDict]]: The largest layer container found, or None.
    """
    largest_module_list: Union[nn.ModuleList, nn.ModuleDict] | None = None
    largest_size = 0

    def _is_pp_layer_module_dict(module: nn.ModuleDict) -> bool:
        # functional.py converts split ModuleLists to ModuleDicts with stringified
        # numeric indices. Avoid treating arbitrary named ModuleDicts (for example
        # adapter registries) as transformer layer containers in the heuristic path.
        return all(key.isdigit() for key in module.keys())

    def _recursive_search(module: nn.Module, path: str = ""):
        nonlocal largest_module_list, largest_size

        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.ModuleList) or (
                isinstance(child, nn.ModuleDict) and _is_pp_layer_module_dict(child)
            ):
                current_size = len(child)
                if current_size > largest_size:
                    largest_size = current_size
                    largest_module_list = child
                    logger.debug(f"Found {type(child).__name__} at {current_path} with {current_size} modules")

            # Continue recursive search
            _recursive_search(child, current_path)

    _recursive_search(model)

    if largest_module_list is not None:
        logger.info(f"Largest layer container found with {largest_size} modules")
    else:
        logger.warning("No ModuleList or ModuleDict found in the model")

    return largest_module_list


def _reduce_attrs(model: nn.Module, fqns: Sequence[str]) -> List[nn.Module]:
    ans = []
    for fqn in fqns:
        parts = fqn.split(".")
        obj = model
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            ans.append(obj)
    return ans


def _extend_layers(layers: List[nn.Module], modules: Sequence[nn.Module]) -> None:
    for m in modules:
        if isinstance(m, nn.ModuleList):
            layers.extend(m)
        elif isinstance(m, nn.ModuleDict):
            layers.extend(m.values())
        else:
            layers.append(m)


def _no_split_module_names(model: nn.Module) -> set[str]:
    """Class names of the transformer blocks ``model`` declares in ``_no_split_modules``.

    HF ``PreTrainedModel``, the native NeMo models and ``diffusers`` ``ModelMixin`` all declare the
    attribute as a list of class *names* (``["LlamaDecoderLayer"]``); matching names rather than
    classes makes a native port and its transformers twin (distinct classes sharing a name) resolve
    alike. Empty when the model declares nothing.
    """
    return set(getattr(model, "_no_split_modules", None) or ())


def _is_no_split_block(module: nn.Module, no_split_names: set[str]) -> bool:
    """Whether ``module`` is one of the declared blocks, seen through the wrappers sharding adds.

    ``checkpoint_wrapper`` nests the block and FSDP2 swaps its class for an ``FSDP<Name>`` subclass,
    so the declared class is looked up on the MRO of the unwrapped module.
    """
    return any(cls.__name__ in no_split_names for cls in type(unwrap_checkpoint_wrapper(module)).__mro__)


def _derive_layer_groups(model: nn.Module) -> Dict[str, List[nn.Module]]:
    """Layer groups from what the model declares: its no-split blocks, its decoder API and its tower names.

    A layer container is a ``ModuleList`` (or the ``ModuleDict`` pipeline splitting leaves behind) whose
    children are all ``_no_split_modules`` blocks. Its role is the multimodal tower it lives under
    (``vision`` / ``audio``, by the tower attribute names ``nemo_automodel.shared.multimodal_fsdp``
    defines); otherwise ``language`` when the model exposes a decoder (``get_decoder``, i.e. it is a
    language model) and ``backbone`` when it does not (a diffusion transformer). Empty when the model
    declares no blocks or none of them sit in a container.
    """
    no_split_names = _no_split_module_names(model)
    if not no_split_names:
        return {}
    is_language_model = callable(getattr(model, "get_decoder", None))
    layer_groups: Dict[str, List[nn.Module]] = {}
    for fqn, module in model.named_modules():
        if not isinstance(module, (nn.ModuleList, nn.ModuleDict)) or len(module) == 0:
            continue
        children = list(module.values()) if isinstance(module, nn.ModuleDict) else list(module)
        if not all(_is_no_split_block(child, no_split_names) for child in children):
            continue
        tower = next((part for part in fqn.split(".") if part in MULTIMODAL_TOWER_NAMES), None)
        if tower is not None:
            role = "audio" if "audio" in tower else "vision"
        else:
            role = "language" if is_language_model else "backbone"
        layer_groups.setdefault(role, []).extend(children)
    return layer_groups


def _extract_model_layer_groups(model: nn.Module) -> Dict[str, List[nn.Module]]:
    """Extract transformer layers grouped by model role.

    Resolution order: the ``layer_groups`` the model's :class:`ParallelSpec` declares; else the groups
    derived from the model's own declarations (:func:`_derive_layer_groups`); else the llama-style
    ``model.layers`` container; else the largest ``ModuleList`` in the tree (role ``"unknown"``). A
    wrapper that declares none of this resolves through the model its ``base_model_prefix`` names.
    """
    model_cls = type(model)
    layer_group_specs = query_parallel_spec(model).layer_groups
    if layer_group_specs is None and not _no_split_module_names(model):
        # A wrapper that declares nothing of its own defers to the model it holds, named by the transformers
        # convention ``base_model_prefix`` (the retrieval bi-/cross-encoders keep their encoder at ``model``).
        base_model = getattr(model, getattr(model, "base_model_prefix", ""), None)
        if isinstance(base_model, nn.Module) and base_model is not model:
            return _extract_model_layer_groups(base_model)

    layer_groups: Dict[str, List[nn.Module]] = {}
    if layer_group_specs is not None:
        for group_name, fqns in layer_group_specs.items():
            layers: List[nn.Module] = []
            # Candidate FQNs are alternative locations of the same container
            # across transformers versions; take the first that resolves so
            # deprecated alias properties (e.g. top-level `visual` on
            # standardized 4.x Qwen2-VL aliasing `model.visual`) cannot
            # double-count layers.
            for fqn in fqns:
                _extend_layers(layers, _reduce_attrs(model, [fqn]))
                if layers:
                    break
            if layers:
                layer_groups[group_name] = layers
        if not layer_groups:
            logger.warning(
                "Layer-group spec for %s resolved no modules: none of the expected FQNs %s exist in the "
                "model tree (likely transformers version drift). Activation checkpointing and layer-based "
                "sharding will skip this model until the spec is updated.",
                model_cls.__name__,
                {group_name: list(fqns) for group_name, fqns in layer_group_specs.items()},
            )
    elif derived_layer_groups := _derive_layer_groups(model):
        layer_groups = derived_layer_groups
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Default case for all other models (assumed to be a causal LM).
        layer_groups["language"] = (
            list(model.model.layers.values())
            if isinstance(model.model.layers, nn.ModuleDict)
            else list(model.model.layers)
        )
    elif hasattr(model, "layers"):
        layer_groups["language"] = (
            list(model.layers.values()) if isinstance(model.layers, nn.ModuleDict) else list(model.layers)
        )
    else:
        # Use heuristic to find the largest layer container in the model.
        logger.warning(f"Unknown model type: {model_cls}. Using heuristic to find transformer layers.")
        largest_module_list = _find_largest_module_list(model)
        if largest_module_list is None:
            # If no layer container is found, still raise an exception.
            print(model)
            raise ValueError(
                f"Unknown model type: {model_cls} and no ModuleList or ModuleDict found in model structure"
            )

        layer_groups["unknown"] = (
            list(largest_module_list.values())
            if isinstance(largest_module_list, nn.ModuleDict)
            else list(largest_module_list)
        )
        logger.info(f"Successfully extracted {len(largest_module_list)} layers using heuristic")

    layers = [layer for group_layers in layer_groups.values() for layer in group_layers]
    assert all(isinstance(m, nn.Module) for m in layers), "layers should be nn.Module instances"
    return layer_groups


def get_model_layer_groups(model: nn.Module) -> Dict[str, List[nn.Module]]:
    """Return transformer layers grouped by model role (``language``, ``vision``, ``audio``, ``backbone``).

    The one accessor for a model's transformer blocks: activation checkpointing, the TP head-count
    update, the FSDP2/DDP managers, the MoE parallelizer and pipeline splitting all read this mapping,
    which comes from the model's ``ParallelSpec.layer_groups`` declaration or, absent one, from the
    blocks the model declares in ``_no_split_modules`` (see :func:`_extract_model_layer_groups`).

    Args:
        model: Root model to extract grouped layers from.

    Returns:
        Mapping from group name to the list of transformer blocks in that group.
    """
    return _extract_model_layer_groups(model)


def _extract_model_layers(model: nn.Module) -> List[nn.Module]:
    """
    Extract layers from different model architectures for parallelization.

    This compatibility wrapper flattens grouped language/vision/audio layers.
    New activation-checkpointing code should use ``_extract_model_layer_groups``
    so scope decisions can be explicit.
    """
    layer_groups = _extract_model_layer_groups(model)
    return [layer for group_layers in layer_groups.values() for layer in group_layers]


def _dedupe_layers(layers: Sequence[nn.Module]) -> List[nn.Module]:
    deduped: List[nn.Module] = []
    seen: set[int] = set()
    for layer in layers:
        layer_id = id(layer)
        if layer_id in seen:
            continue
        seen.add(layer_id)
        deduped.append(layer)
    return deduped


def _has_trainable_parameters(module: nn.Module) -> bool:
    return any(param.requires_grad for param in module.parameters(recurse=True))


def _filter_layer_groups_for_activation_checkpointing(
    layer_groups: Dict[str, List[nn.Module]],
    activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
) -> Tuple[List[nn.Module], Tuple[str, ...]]:
    """Select trainable activation-checkpointed layers from grouped model layers."""
    scopes = normalize_activation_checkpointing_scope(activation_checkpointing_scope)
    all_layers = [layer for group_layers in layer_groups.values() for layer in group_layers]

    selected = []
    if scopes == ("all",):
        selected = all_layers
    else:
        for scope in scopes:
            if scope == "multimodal":
                selected.extend(layer_groups.get("vision", []))
                selected.extend(layer_groups.get("audio", []))
            else:
                selected.extend(layer_groups.get(scope, []))

    selected = _dedupe_layers(selected)
    skipped_frozen = [layer for layer in selected if not _has_trainable_parameters(layer)]
    if skipped_frozen:
        selected = [layer for layer in selected if _has_trainable_parameters(layer)]
    group_counts = {name: len(layers) for name, layers in layer_groups.items()}
    selected_counts = {
        name: sum(1 for layer in layers if any(layer is selected_layer for selected_layer in selected))
        for name, layers in layer_groups.items()
    }
    logger.info(
        "Activation checkpointing scope %s selected %d/%d trainable layers; groups=%s selected_groups=%s "
        "skipped_frozen=%d",
        scopes,
        len(selected),
        len(all_layers),
        group_counts,
        selected_counts,
        len(skipped_frozen),
    )
    if all_layers and not selected:
        logger.warning("Activation checkpointing scope %s selected no layers.", scopes)
    return selected, scopes


def _should_use_hf_native_gradient_checkpointing(
    model: nn.Module,
    layer_groups: Dict[str, List[nn.Module]],
    scopes: Tuple[str, ...],
    *,
    enable_compile: bool = False,
) -> bool:
    """Return whether HF-native gradient checkpointing can preserve AutoModel's AC scope."""
    if enable_compile or scopes != ("all",):
        return False
    if set(layer_groups) != {"language"}:
        return False
    language_layers = layer_groups.get("language", [])
    if not language_layers:
        return False
    if any(not _has_trainable_parameters(layer) for layer in language_layers):
        return False
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer as _HFGradLayer
    except ImportError:
        return False
    return (
        language_layers[0].__class__.__module__.startswith("transformers.")
        and isinstance(language_layers[0], _HFGradLayer)
        and getattr(model, "supports_gradient_checkpointing", False)
        and hasattr(model, "gradient_checkpointing_enable")
    )


def _kv_sharing_survives_checkpoint_replay(model: nn.Module) -> bool:
    """Return whether whole-block activation checkpointing is safe for a KV-shared model.

    ``checkpoint_wrapper`` replays a whole decoder block during backward with the
    arguments the forward saw. Unlike HF's ``GradientCheckpointingLayer.__call__``
    it cannot drop ``past_key_values`` from that replay, so every layer that
    writes to a cache writes to it a second time. A KV-shared model then needs
    both halves to hold:

    * the layers that populate the cache -- the *non*-shared ones, which are what
      call ``Cache.update()`` -- must not accumulate on the replay, or the
      recomputed K/V stops matching the forward;
    * the shared layers must still read the K/V their source layer produced.

    Neither holds for a model backed by an accumulating ``Cache``: the second
    ``Cache.update()`` grows the entry and backward dies with a
    ``CheckpointError`` about changed tensor metadata (observed on native HF
    ``Gemma3nForCausalLM`` with ``use_cache=True``). KV-shared models therefore
    stay on ``apply_submodule_checkpointing``, which leaves attention unwrapped,
    by default.

    A model that satisfies both halves opts in by setting the class attribute
    ``kv_sharing_survives_checkpoint_replay = True``. Gemma4 E2B/E4B qualify: a
    pass-through holder stands in for the cache, and the shared layers read a
    separate store that the replay does not disturb (see
    ``gemma4_moe/model.py``).

    Args:
        model: The model about to be checkpointed.

    Returns:
        Whether the model declares its KV sharing safe under whole-block replay.
    """
    return bool(getattr(model, "kv_sharing_survives_checkpoint_replay", False))


def _uses_custom_moe_modules(model: nn.Module) -> bool:
    """Return whether ``model`` contains Automodel's grouped custom-MoE layer."""
    iter_modules = getattr(model, "modules", None)
    if not callable(iter_modules):
        return False
    try:
        from nemo_automodel.components.moe.layers import MoE
    except ImportError:
        return False
    return any(isinstance(module, MoE) for module in iter_modules())


def _get_parallel_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
    tp_size: int = 1,
) -> Dict[str, ParallelStyle]:
    """
    Select the tensor-parallel plan for the given model.

    Priority order:
    1) If ``tp_shard_plan`` is provided as a dict or import path, use it.
    2) If the model's ``ParallelSpec`` declares a ``tp_plan``, use it (with its sequence-parallel overlay).
    3) Otherwise, prefer the model's HF-native ``_tp_plan`` (``ParallelSpec.from_hf_model``).
    4) Otherwise, fall back to the default base plan.

    When ``tp_size > 1`` and the model falls through to path 4 *and* the
    model class was loaded from a custom-code source (HF's
    ``trust_remote_code=True`` path, where the dynamic class lives under
    ``transformers_modules.*``), this raises ``ValueError`` instead of
    returning the default base plan. On recent PyTorch the default plan's
    placements do not populate ``shard_order`` and trip an internal assert in
    ``torch.distributed.tensor._redistribute`` on the first weight
    redistribute, which surfaces to the user as an opaque PyTorch internal
    error. Custom-code architectures are the only known-broken case (see
    https://github.com/NVIDIA-NeMo/Automodel/issues/2243); known HF
    architectures that happen to fall through (e.g. Mixtral) are left on the
    default plan with a warning, since they have been working in practice.

    When the model *did* define a ``_tp_plan`` but ``get_hf_tp_shard_plan``
    raised while translating it (e.g. styles nemo does not recognize), the
    translator's error message is folded into the ``ValueError`` as a
    diagnostic so the user can tell whether to add a ``_tp_plan`` from
    scratch or fix the styles in the one they already have.
    """
    model_parallel_plan = None
    model_cls = type(model)
    spec = query_parallel_spec(model)

    if isinstance(tp_shard_plan, dict):
        model_parallel_plan = tp_shard_plan
        col_w = max(55, max(map(len, tp_shard_plan.keys()), default=0))
        plan_lines = "\n".join(f"  {k:<{col_w}} {v}" for k, v in tp_shard_plan.items())
        logger.info(f"Using parallel plan (dictionary):\n{plan_lines}")
    elif tp_shard_plan is not None:
        try:
            plan_obj = import_class_from_path(tp_shard_plan)
            if isinstance(plan_obj, FunctionType):
                model_parallel_plan = plan_obj()
            else:
                model_parallel_plan = plan_obj
            assert isinstance(model_parallel_plan, dict), (
                f"Parallel plan must be a dictionary, got {type(model_parallel_plan)}"
            )
            logger.info(f"Using provided parallel plan (from path). {tp_shard_plan}")
        except Exception as e:
            raise ValueError(
                f"Custom parallel plan '{tp_shard_plan}' is not valid. "
                f"Please ensure it is one of the following:\n"
                "1. A dictionary mapping module names to parallel styles\n"
                "2. A path to a dictionary\n"
                "3. A path to a function that returns a dictionary\n"
                f"Error: {e}"
            )

    elif (declared_plan := spec.resolved_tp_plan(sequence_parallel)) is not None:
        if sequence_parallel and spec.sequence_parallel_plan is None:
            logger.warning(
                "%s declares no sequence-parallel plan; sequence_parallel=True is ignored for its tensor-parallel plan.",
                model_cls.__name__,
            )
        model_parallel_plan = declared_plan
        logger.info(f"Using optimized parallel plan for {model_cls.__name__}.")

    else:
        # Try HF's per-model _tp_plan first — it correctly handles multimodal
        # architectures like Mistral3ForConditionalGeneration whose text layers
        # live under model.language_model.layers.* and would be missed by the
        # hardcoded llama-style wildcards below.
        hf_plan = None
        hf_plan_error: Exception | None = None
        try:
            hf_spec = ParallelSpec.from_hf_model(model)
        except Exception as e:
            hf_plan_error = e
            logger.info(f"HF tp plan not available ({e}). Falling back to default base plan.")
        else:
            hf_plan = hf_spec.resolved_tp_plan(sequence_parallel) if hf_spec is not None else None

        if hf_plan:
            model_parallel_plan = hf_plan
            logger.info(f"Using HF-native tp plan for {model_cls.__name__}.")
        else:
            # HF places dynamic classes loaded via ``trust_remote_code=True`` under the
            # ``transformers_modules.*`` namespace. Those are the only archs known to
            # actually crash inside ``_redistribute`` with the default base plan, so we
            # only fail-fast for them. See https://github.com/NVIDIA-NeMo/Automodel/issues/2243.
            is_remote_code = (model_cls.__module__ or "").startswith("transformers_modules.")
            if tp_size > 1 and is_remote_code:
                # If the model author *did* define `_tp_plan` but it was unusable
                # (e.g. styles nemo does not recognize), surface that diagnostic so the
                # user knows whether to (a) add a missing `_tp_plan` from scratch or
                # (b) fix the styles in the one they already have.
                diag = f" Note: {hf_plan_error}." if hf_plan_error is not None else ""
                raise ValueError(
                    f"No tensor-parallel plan is registered for the custom-code architecture "
                    f"'{model_cls.__name__}' (loaded via trust_remote_code=True), and no usable "
                    f"HuggingFace `_tp_plan` was found.{diag} The default base plan cannot be used "
                    f"at tp_size={tp_size}: it produces DTensor placements without `shard_order` "
                    "metadata, which trips an internal assert in "
                    "`torch.distributed.tensor._redistribute` on the first weight redistribute. "
                    "Register a working plan in one of the following ways:\n"
                    "  1. Declare `parallel_spec = ParallelSpec(tp_plan=...)` on the model class, or, for a "
                    f"transformers / trust_remote_code class, on a class named '{model_cls.__name__}' in "
                    "`components/models/<model_type>/parallelization.py`.\n"
                    "  2. Define a `_tp_plan` on the model class with styles nemo recognizes "
                    "(e.g. `colwise`, `rowwise`, `colwise_rep`, `rowwise_rep`).\n"
                    "  3. Pass `tp_shard_plan` (dict or import path) when constructing the parallelizer.\n"
                    "Alternatively, run with tp_size=1."
                )
            if tp_size > 1:
                logger.warning(
                    "No usable tensor-parallel plan is registered for '%s'. Falling back to the "
                    "default base plan at tp_size=%d. If you hit an internal assert in "
                    "`torch.distributed.tensor._redistribute` on `shard_order is not None`, "
                    "register a plan via `ParallelSpec.tp_plan`, `_tp_plan`, or `tp_shard_plan`.",
                    model_cls.__name__,
                    tp_size,
                )
            base_model_tp_plan = {
                "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
                "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Combined QKV projection
                "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate and up projection
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(),
                "lm_head": ColwiseParallel(output_layouts=Replicate()),
            }
            if sequence_parallel:
                base_model_sp_plan = {
                    "model.embed_tokens": VocabParallelEmbedding(
                        input_layouts=Replicate(),
                        output_layouts=Shard(1),
                        use_local_output=False,
                    ),
                    "model.norm": SequenceParallel(),
                    "model.layers.*.input_layernorm": SequenceParallel(),
                    "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
                    "model.layers.*.post_attention_layernorm": SequenceParallel(),
                    "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
                    "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
                }
                base_model_tp_plan.update(base_model_sp_plan)
            model_parallel_plan = base_model_tp_plan
            logger.info("Using default base TP plan. Compatible with huggingface llama3-style models.")

    for fqn in spec.sharded_output_only:
        style = model_parallel_plan.get(fqn)
        if style is None:
            continue
        output_layouts = getattr(style, "output_layouts", ())
        if not isinstance(output_layouts, (tuple, list)):
            output_layouts = (output_layouts,)
        if not any(isinstance(layout, Shard) for layout in output_layouts):
            model_parallel_plan.pop(fqn)
            logger.info(
                "Dropped the %s entry of the plan for %s: it requires a sharded output.", fqn, model_cls.__name__
            )

    # EP=1 uses this generic FSDP2 path rather than the dedicated MoE
    # parallelizer. Apply the same routed-expert ownership validation here so
    # an explicit/wildcard TP plan cannot silently shard expert or router
    # modules merely because expert parallelism is disabled.
    if tp_size > 1 and _uses_custom_moe_modules(model):
        from nemo_automodel.components.moe.tp_plan_validation import _validate_moe_tp_plan

        model_parallel_plan = _validate_moe_tp_plan(model_parallel_plan, model=model)

    return model_parallel_plan


# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def fsdp2_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
    dp_replicate_mesh_name: str = "dp_replicate",
    dp_shard_cp_mesh_name: str = "dp_shard_cp",
    tp_mesh_name: str = "tp",
    enable_async_tensor_parallel: bool = False,
    enable_compile: bool = False,
    enable_fsdp2_prefetch: bool = True,
    fsdp2_backward_prefetch_depth: int = 2,
    fsdp2_forward_prefetch_depth: int = 1,
    reshard_after_forward: bool | None = None,
    activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
    frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
    reapply_trainability: Callable[[nn.Module], None] | None = None,
) -> nn.Module:
    """
    Apply parallelisms and activation checkpointing to the model.

    Enhanced version that uses a strategy pattern for different model parallelization approaches:
    - Automatic strategy selection based on model type
    - Polymorphic parallelization strategies for different model families
    - Custom parallel plan support (dict or string path)
    - Sequence parallel support
    - Activation checkpointing for linear layers
    - Model validation (attention heads divisible by TP size)
    - Better fallback logic

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh for distributed training.
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy for model parallelism.
        offload_policy (Optional[OffloadPolicy]): The offload policy for FSDP.
        sequence_parallel (bool): Whether to use sequence parallelism. Defaults to False.
        activation_checkpointing (bool): Whether to use activation checkpointing. Defaults to False.
        tp_shard_plan (Optional[Union[Dict[str, ParallelStyle], str]]):
            Custom tensor parallel plan for the model. Can be:
            - A dictionary mapping module names to parallel styles
            - A string path to a dictionary or function that returns a dictionary
            If provided, this takes precedence over automatic plan generation.
        dp_replicate_mesh_name (str): Key name for the data parallel replicate mesh in device_mesh.
            Used when data parallel replicate is enabled. Defaults to "dp_replicate".
        dp_shard_cp_mesh_name (str): Key name for the data parallel shard + context parallel mesh in device_mesh.
            Used when data parallel shard is enabled. Defaults to "dp_shard_cp".
        tp_mesh_name (str): Key name for the tensor parallel mesh in device_mesh.
            Defaults to "tp".
        frozen_multimodal_sharding: Whether fully frozen multimodal modules are
            owned by the root FSDP unit (``"root"``), sharded normally
            (``"per_layer"``), or excluded from FSDP and copied on every rank
            (``"replicate"``).
        reapply_trainability: Optional callback that re-resolves parameter
            trainability after strategy-specific model surgery and immediately
            before FSDP construction.

    Returns:
        The parallelized model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # Get the appropriate parallelization strategy for this model
    strategy = get_parallelization_strategy(model)

    # Delegate to the strategy
    return strategy.parallelize(
        model=model,
        device_mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        sequence_parallel=sequence_parallel,
        activation_checkpointing=activation_checkpointing,
        tp_shard_plan=tp_shard_plan,
        dp_replicate_mesh_name=dp_replicate_mesh_name,
        dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
        tp_mesh_name=tp_mesh_name,
        enable_async_tensor_parallel=enable_async_tensor_parallel,
        enable_compile=enable_compile,
        enable_fsdp2_prefetch=enable_fsdp2_prefetch,
        fsdp2_backward_prefetch_depth=fsdp2_backward_prefetch_depth,
        fsdp2_forward_prefetch_depth=fsdp2_forward_prefetch_depth,
        reshard_after_forward=reshard_after_forward,
        activation_checkpointing_scope=activation_checkpointing_scope,
        frozen_multimodal_sharding=frozen_multimodal_sharding,
        reapply_trainability=reapply_trainability,
    )


def _megatron_fsdp_compat_kwargs(
    shard_fn,
    *,
    grad_reduce_in_fp32: bool,
    preserve_fp32_weights: bool,
    check_for_nan_in_grad: bool,
    report_nan_in_param_grad: bool,
) -> Dict[str, Any]:
    """Translate the config precision controls to the Megatron-FSDP 0.5.0 API.

    megatron-fsdp==0.5.0, the only supported release, expresses precision
    through a ``MixedPrecisionPolicy`` plus a more expensive per-parameter NaN
    reporter. The reporter stays a separate opt-in rather than being silently
    enabled from the legacy buffer-check setting; because 0.5.0 has no
    buffer-level NaN check at all, a truthy ``check_for_nan_in_grad`` is
    dropped with a one-time warning that points at ``report_nan_in_param_grad``
    as the opt-in replacement. Any other ``fully_shard`` signature — older or
    newer releases alike — fails loudly instead of guessing a translation.
    """
    try:
        parameters = inspect.signature(shard_fn).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cannot determine the installed Megatron-FSDP fully_shard API") from exc

    required_names = {"mixed_precision_policy", "report_nan_in_param_grad"}
    if not required_names.issubset(parameters):
        raise RuntimeError(
            "unsupported Megatron-FSDP fully_shard API: NeMo Automodel requires megatron-fsdp==0.5.0, "
            f"whose signature has the arguments {sorted(required_names)!r}; got {sorted(parameters)!r}"
        )

    global _megatron_fsdp_nan_check_noop_warned
    if check_for_nan_in_grad and not _megatron_fsdp_nan_check_noop_warned:
        _megatron_fsdp_nan_check_noop_warned = True
        logger.warning(
            "check_for_nan_in_grad=True is a no-op with megatron-fsdp==0.5.0, which removed the "
            "legacy buffer-level NaN check: gradient NaN checking is now DISABLED. Set "
            "report_nan_in_param_grad=True to restore per-parameter gradient NaN checking."
        )
    return {
        "mixed_precision_policy": MegatronFSDPMixedPrecisionPolicy(
            main_params_dtype=torch.float32 if preserve_fp32_weights else None,
            main_grads_dtype=torch.float32 if grad_reduce_in_fp32 else None,
            # In megatron-fsdp 0.5.0, None makes communication use the main
            # gradient dtype, matching the legacy grad_reduce_in_fp32 flag.
            grad_comm_dtype=None,
        ),
        "report_nan_in_param_grad": report_nan_in_param_grad,
    }


def _derive_megatron_fsdp_unit_modules(model: nn.Module) -> list[type[nn.Module]]:
    """Derive the MegatronFSDP wrap classes from a model's ``_no_split_modules``.

    Used when a config does not specify ``megatron_fsdp_unit_modules``. HF
    ``PreTrainedModel`` and the NeMo custom models both define ``_no_split_modules``
    as a list of block class *names* (for example ``["LlamaDecoderLayer"]``).
    Walking ``model.modules()`` and matching ``type(module).__name__`` against those
    names resolves the actual instantiated classes, so the result is correct for
    both the HF backend and the NeMo-custom backend (which use distinct classes that
    share the same name). For VLM/MoE models whose top-level ``_no_split_modules``
    lists several block classes (for example vision and language towers), every
    matching class found anywhere in the module tree is collected.

    Args:
        model: The (already TP-parallelized) model to be wrapped by MegatronFSDP.

    Returns:
        The de-duplicated list of submodule classes to wrap as MegatronFSDP units,
        in module-traversal order.

    Raises:
        ValueError: If the model does not expose a non-empty ``_no_split_modules``,
            or if none of those names match an instantiated submodule. Raised with
            an actionable message instead of letting MegatronFSDP later fail with
            ``ZeroDivisionError`` (``total_fsdp_module=0``) when zero modules are wrapped.
    """
    no_split_names = _no_split_module_names(model)
    if not no_split_names:
        raise ValueError(
            "distributed.megatron_fsdp_unit_modules was not provided and the model does not define a "
            "non-empty '_no_split_modules' to derive them from. Set distributed.megatron_fsdp_unit_modules "
            "explicitly to the transformer block class path(s) to wrap as MegatronFSDP units."
        )
    derived: list[type[nn.Module]] = []
    seen: set[type[nn.Module]] = set()
    for submodule in model.modules():
        cls = type(submodule)
        if cls.__name__ in no_split_names and cls not in seen:
            seen.add(cls)
            derived.append(cls)
    if not derived:
        raise ValueError(
            "distributed.megatron_fsdp_unit_modules was not provided and none of the model's "
            f"_no_split_modules {sorted(no_split_names)} matched an instantiated submodule; cannot derive "
            "MegatronFSDP unit modules. Set distributed.megatron_fsdp_unit_modules explicitly."
        )
    logger.info(
        "Auto-derived MegatronFSDP unit modules from _no_split_modules: %s",
        [cls.__name__ for cls in derived],
    )
    return derived


def megatron_fsdp_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    optimizer=None,
    megatron_fsdp_unit_modules: List[str] | None = None,
    tp_shard_plan: Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]] | None = None,
    zero_dp_strategy: int = 3,
    init_fsdp_with_meta_device: bool = False,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = False,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    check_for_nan_in_grad: bool = True,
    report_nan_in_param_grad: bool = False,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    dp_shard_dim: str = "dp",
    tp_dim: str = "tp",
    reapply_trainability: Callable[[nn.Module], None] | None = None,
):
    """
    Apply tensor/data parallelism (MegatronFSDP) and optional activation-checkpointing to the model.

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh describing the physical devices
            used for distributed training.
        megatron_fsdp_unit_modules (Optional[List[str]]): Class paths of the sub-modules that
            should become individual MegatronFSDP units. When None or empty, the wrap classes
            are auto-derived from the model's ``_no_split_modules`` (see
            :func:`_derive_megatron_fsdp_unit_modules`).
        tp_shard_plan (Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]]):
            A tensor-parallel sharding plan.
            Keys are module names; values specify the parallel style to apply
            (e.g., RowwiseParallel, ColwiseParallel, SequenceParallel).
        zero_dp_strategy (int): The zero-DP strategy to use.
        init_fsdp_with_meta_device (bool): If True, construct the model on a
            meta device first and materialize weights lazily to reduce memory
            fragmentation.
        grad_reduce_in_fp32 (bool): Reduce gradients in FP32 irrespective of the
            parameter precision to improve numerical stability.
        preserve_fp32_weights (bool): Keep a master FP32 copy of weights when
            training in reduced precision (e.g., FP16/BF16).
        overlap_grad_reduce (bool): If True, overlap gradient reduction with
            backward computation.
        overlap_param_gather (bool): If True, overlap parameter gathering with
            forward computation.
        check_for_nan_in_grad (bool): Legacy buffer-level gradient NaN check.
            BREAKING CHANGE on megatron-fsdp 0.5.0: this flag is a no-op,
            preserved only for config compatibility. 0.5.0 removed the
            buffer-level NaN check entirely, so gradient NaN checking is now OFF
            regardless of this value; a truthy value is dropped with a one-time
            warning per process. Enable ``report_nan_in_param_grad`` to restore
            gradient NaN checking.
        report_nan_in_param_grad (bool): Whether Megatron-FSDP should perform
            its precise per-parameter gradient NaN check. This is the 0.5.0
            replacement for ``check_for_nan_in_grad`` and is disabled by default
            because it can significantly reduce training throughput.
        average_in_collective (bool): Perform gradient averaging inside the
            collective operation instead of dividing afterward.
        disable_bucketing (bool): Disable gradient bucketing; gradients are
            reduced immediately as they are produced.
        calculate_per_token_loss (bool): Compute loss normalized by the number of
            tokens instead of the number of sequences.
        keep_fp8_transpose_cache (bool): Retain the FP8
            transpose cache when using a custom MegatronFSDP wrapper.
        nccl_ub (bool): Enable NCCL user-buffer API (experimental) for reduced
            latency on some networks.
        fsdp_double_buffer (bool): Enable double buffering of parameters to
            overlap communication and computation in MegatronFSDP.
        dp_shard_dim (str): Key name for the data parallel mesh in device_mesh.
            Defaults to "dp".
        tp_dim (str): Key name for the tensor parallel mesh in device_mesh.
            Defaults to "tp".
        reapply_trainability: Optional callback that re-resolves parameter
            trainability after tensor-parallel surgery and immediately before
            Megatron-FSDP construction.

    NOTE: The passed-in model should preferably reside on the meta device.
    Otherwise, ensure the model fits into available GPU or CPU memory.

    NOTE: The user must ensure that the provided tp_shard_plan is compatible
    with the model architecture.
    """
    assert HAVE_MEGATRON_FSDP, (
        "MegatronFSDP is not installed, please visit \
        https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src for \
        more information"
    )

    # DP_CP ranks are sharded by FSDP.
    dp_mesh = device_mesh[dp_shard_dim]
    tp_mesh = device_mesh[tp_dim]

    if dp_mesh.size() > 1:
        # TODO(boxiangw): remove this once HSDP is supported.
        assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    # TP sharding.
    if tp_mesh.size() > 1:
        parallelize_module(model, tp_mesh, tp_shard_plan)

    if reapply_trainability is not None:
        reapply_trainability(model)

    # MegatronFSDP requires a sharded DP dimension to create its param/grad buffers.
    # In practice, configurations like world_size=2,tp=2 -> dp=1 frequently hit
    # DTensor metadata assertions inside megatron_fsdp. In that case, we still
    # support training by applying TP-only and skipping the MegatronFSDP wrapper.
    if dp_mesh.size() == 1:
        logger.warning(
            "MegatronFSDP DP shard group size is 1; skipping MegatronFSDP wrapping and returning the "
            "TP-parallelized model. To enable MegatronFSDP sharding, use dp_size>1 (e.g., tp_size=1 "
            "for world_size=2)."
        )
        # `parallelize_module` only moves/shards modules covered by the TP plan.
        # Ensure the remaining (non-sharded) parameters/buffers are on the local device.
        if getattr(device_mesh, "device_type", None) == "cuda" and torch.cuda.is_available():
            try:
                model = model.to(torch.device("cuda", torch.cuda.current_device()))
            except Exception:
                # Best-effort fallback (e.g., if current_device isn't set).
                model = model.to("cuda")
        return model, optimizer

    # Resolve the MegatronFSDP unit (wrap) modules (only needed on the wrapping path).
    # When the config specifies them, import the class paths as-is. Otherwise derive
    # them from the model's `_no_split_modules` so the real instantiated block classes
    # are wrapped regardless of backend (HF or NeMo-custom); a mismatched hard-coded
    # class path would otherwise wrap zero modules and MegatronFSDP would raise a
    # ZeroDivisionError (total_fsdp_module=0).
    if megatron_fsdp_unit_modules:
        megatron_fsdp_unit_modules = import_classes_from_paths(megatron_fsdp_unit_modules)
    else:
        megatron_fsdp_unit_modules = _derive_megatron_fsdp_unit_modules(model)

    # Wrap model with MegatronFSDP.
    # When an optimizer is provided, use the combined fully_shard which handles
    # both model wrapping and optimizer sharding in one step.
    # When optimizer is None (e.g., during model creation before optimizer
    # instantiation), use fully_shard_model to wrap only the model and prepare
    # distributed parameters so the optimizer can be sharded later via
    # fully_shard_optimizer.
    fsdp_kwargs = dict(
        fsdp_unit_modules=megatron_fsdp_unit_modules,
        device_mesh=device_mesh,
        dp_shard_dim=dp_shard_dim,
        tp_dim=tp_dim,
        zero_dp_strategy=zero_dp_strategy,
        init_model_with_meta_device=init_fsdp_with_meta_device,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        average_in_collective=average_in_collective,
        disable_bucketing=disable_bucketing,
        calculate_per_token_loss=calculate_per_token_loss,
        keep_fp8_transpose_cache=keep_fp8_transpose_cache,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
    )
    if optimizer is not None:
        fsdp_kwargs.update(
            _megatron_fsdp_compat_kwargs(
                megatron_fsdp_fully_shard,
                grad_reduce_in_fp32=grad_reduce_in_fp32,
                preserve_fp32_weights=preserve_fp32_weights,
                check_for_nan_in_grad=check_for_nan_in_grad,
                report_nan_in_param_grad=report_nan_in_param_grad,
            )
        )
        model, optimizer = megatron_fsdp_fully_shard(module=model, optimizer=optimizer, **fsdp_kwargs)
    else:
        fsdp_kwargs.update(
            _megatron_fsdp_compat_kwargs(
                megatron_fsdp_fully_shard_model,
                grad_reduce_in_fp32=grad_reduce_in_fp32,
                preserve_fp32_weights=preserve_fp32_weights,
                check_for_nan_in_grad=check_for_nan_in_grad,
                report_nan_in_param_grad=report_nan_in_param_grad,
            )
        )
        model = megatron_fsdp_fully_shard_model(module=model, **fsdp_kwargs)
        model._replace_param_with_distributed_if_needed()

    return model, optimizer


@contextmanager
def unshard_fsdp2_model(model: nn.Module) -> Generator[None, None, None]:
    """Explicitly unshard and then reshard the FSDP2 modules. Useful for logprob inference."""
    try:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
