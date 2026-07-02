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
import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union

import torch
import transformers
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
from torch.distributed.tensor.placement_types import Shard
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForConditionalGeneration,
)

try:
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration,
    )
except (ImportError, ModuleNotFoundError):

    class Gemma4ForConditionalGeneration:  # type: ignore[no-redef]
        """Placeholder when the installed transformers build has no Gemma4."""

        pass


from nemo_automodel.components.distributed.activation_checkpointing import (
    SELECTIVE_AC_WRAPPER_FLAG,
    apply_selective_checkpointing_to_layers,
    apply_submodule_checkpointing,
    detect_kv_sharing_and_maybe_disable_cache,
    is_selective_activation_checkpointing,
)
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh


def _is_transformers_v5_or_higher() -> bool:
    """Check if transformers version is 5.x or higher."""
    version = transformers.__version__
    major_version = int(version.split(".")[0])
    return major_version >= 5


from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
)
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoForConditionalGeneration,
)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionForConditionalGeneration,
)
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
)
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration

from nemo_automodel.components.distributed.parallel_styles import translate_to_lora
from nemo_automodel.components.distributed.tp_plan import (
    get_hf_tp_shard_plan,  # noqa: F401
    get_tp_plan,
    translate_to_torch_parallel_style,  # noqa: F401
)

# TODO(boxiangw): Change to MegatronFSDP once it got published
HAVE_MEGATRON_FSDP = False
logging.getLogger("megatron_fsdp").setLevel(logging.WARNING)
try:
    from megatron_fsdp import fully_shard as megatron_fsdp_fully_shard
    from megatron_fsdp import fully_shard_model as megatron_fsdp_fully_shard_model

    HAVE_MEGATRON_FSDP = True
except (ImportError, FileNotFoundError, OSError):
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _patch_fsdp_accumulated_grad_guard() -> None:
    """Guard FSDP2 post-backward against params that were never unsharded.

    This is needed for text-only configs that still instantiate and shard a
    full VLM model, e.g.
    ``examples/llm_finetune/mistral/ministral3_3b_squad.yaml`` when
    ``ci.checkpoint_robustness.distributed.tp_size: 2`` reruns the recipe.
    Ministral3 FP8 is loaded through ``Mistral3FP8VLMForConditionalGeneration``
    so the vision tower remains separately FSDP-sharded, but SQuAD batches do
    not execute that tower. Those FSDP params never create PyTorch's lazy
    ``_unsharded_param`` field, and the fp32 grad-reduce post-backward helper
    dereferences it unconditionally. If the field is absent, there is no
    unsharded grad to upcast, so returning early preserves the no-grad case.
    The wrapper still calls PyTorch first and only handles the exact
    ``AttributeError`` from the missing lazy field.
    Permalinks:
    - Trigger YAML: https://github.com/NVIDIA-NeMo/Automodel/blob/0990cb2c047496bae50e2035dac7b8c509316076/examples/llm_finetune/mistral/ministral3_3b_squad.yaml#L114-L128
    - Mistral3 layer extraction: https://github.com/NVIDIA-NeMo/Automodel/blob/0990cb2c047496bae50e2035dac7b8c509316076/nemo_automodel/components/distributed/parallelizer.py#L1522-L1530
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
        return

    orig = FSDPParam.to_accumulated_grad_if_needed
    if getattr(orig, "_nemo_automodel_guarded", False):
        return

    def guarded(self: Any) -> Any:
        try:
            return orig(self)
        except AttributeError as exc:
            if "_unsharded_param" not in str(exc) or hasattr(self, "_unsharded_param"):
                raise
            return None

    setattr(guarded, "_nemo_automodel_guarded", True)
    FSDPParam.to_accumulated_grad_if_needed = guarded


def apply_selective_activation_checkpointing(model: nn.Module, *, enable_compile: bool = False) -> None:
    """Apply selective activation checkpointing to ``model`` end to end.

    Standalone entry point (detects KV-sharing, disables ``use_cache``, and
    wraps transformer blocks) for paths where the FSDP2 parallelize flow is
    skipped -- notably single-GPU training.

    Args:
        model: The model to checkpoint.
        enable_compile: Whether per-layer ``torch.compile`` will be applied.
    """
    layers = _extract_model_layers(model)
    if not layers:
        logger.warning("No transformer layers found; skipping selective activation checkpointing.")
        return
    has_kv_sharing = detect_kv_sharing_and_maybe_disable_cache(model)
    apply_selective_checkpointing_to_layers(model, layers, has_kv_sharing, enable_compile=enable_compile)


_BAGEL_FULL_LAYER_CHECKPOINT_MODULE_LISTS = (
    "model.language_model.model.layers",
    "model.vit_model.vision_model.encoder.layers",
)


def _get_module_by_fqn(module: nn.Module, fqn: str) -> Optional[nn.Module]:
    obj = module
    for part in fqn.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_checkpoint_wrapped(module: nn.Module) -> bool:
    return hasattr(module, "_checkpoint_wrapped_module")


def _apply_bagel_full_layer_activation_checkpointing(model: nn.Module) -> bool:
    """Apply native BAGEL-style activation checkpointing to whole logical layers."""
    if type(model).__name__ != "BagelForUnifiedMultimodal":
        return False

    wrapped_count = 0
    for fqn in _BAGEL_FULL_LAYER_CHECKPOINT_MODULE_LISTS:
        container = _get_module_by_fqn(model, fqn)
        if container is None:
            logger.warning("BAGEL activation checkpointing skipped missing module list %s", fqn)
            continue
        if not isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            logger.warning(
                "BAGEL activation checkpointing expected %s to be a module list, got %s",
                fqn,
                type(container),
            )
            continue

        items = container.items() if isinstance(container, nn.ModuleDict) else enumerate(container)
        for key, layer in list(items):
            if _is_checkpoint_wrapped(layer):
                continue
            container[key] = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            wrapped_count += 1

    logger.info("Applied BAGEL full-layer activation checkpointing to %d layers", wrapped_count)
    return wrapped_count > 0


class ParallelizationStrategy(ABC):
    """Abstract base class for model parallelization strategies."""

    @abstractmethod
    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reshard_after_forward: Optional[bool] = None,
        **kwargs,
    ) -> nn.Module:
        """Apply parallelization strategy to the model."""
        pass


class DefaultParallelizationStrategy(ParallelizationStrategy):
    """Default parallelization strategy used by most models."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        enable_async_tensor_parallel: bool = False,
        enable_compile: bool = False,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: Optional[bool] = None,
    ) -> nn.Module:
        """Apply the default parallelization flow."""
        tp_mesh = device_mesh[tp_mesh_name]

        # Set FSDP sharding mesh to context parallel mesh if CP > 1, else default to the data parallel mesh.
        # if dp_replicate_size > 1, use HSDP, else use FSDP
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        pp_enabled = "pp" in dp_mesh.mesh_dim_names and dp_mesh["pp"].size() > 1
        if pp_enabled and reshard_after_forward is True:
            logger.warning(
                "reshard_after_forward=True overrides the pipeline-parallel default of keeping layer weights "
                "gathered across microbatches. This may increase per-microbatch all-gathers and reduce throughput."
            )

        # Extract layers from the model for parallelization
        layers = _extract_model_layers(model)

        # TP sharding with enhanced plan generation
        if tp_mesh.size() > 1:
            # async-TP (_micro_pipeline_tp) overlaps ReduceScatter with compute.
            # Without SP, row-parallel layers emit AllReduce (not ReduceScatter),
            # so there is nothing for the micro-pipeline to overlap — force SP on.
            if enable_async_tensor_parallel and not sequence_parallel:
                raise ValueError("enable_async_tensor_parallel=True requires sequence_parallel=True")

            # Validate that attention heads are divisible by TP size
            validate_tp_mesh(model, tp_mesh)

            # Generate or use tensor parallel plan
            model_parallel_plan = {
                k: translate_to_lora(v)
                for k, v in get_tp_plan(
                    model,
                    sequence_parallel=sequence_parallel,
                    tp_shard_plan=tp_shard_plan,
                    tp_size=tp_mesh.size(),
                ).items()
            }

            # Apply tensor parallelism
            if model_parallel_plan:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*could not be resolved.*",
                        category=UserWarning,
                    )
                    parallelize_module(model, tp_mesh, model_parallel_plan)
                if _attention_is_head_sharded(model_parallel_plan):
                    _update_attention_head_counts_for_tp(model, tp_mesh.size())

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

            if is_selective_activation_checkpointing(activation_checkpointing):
                apply_selective_checkpointing_to_layers(model, layers, _has_kv_sharing, enable_compile=enable_compile)
            elif _apply_bagel_full_layer_activation_checkpointing(model):
                logger.info("Using BAGEL full-layer activation checkpointing; skipping submodule checkpoint wrappers.")
            elif enable_compile:
                # NO_REENTRANT is required for compile: REENTRANT's first forward runs under
                # no_grad, causing AOT autograd to trace a forward-only graph that drops LoRA
                # (and other trainable) weight gradients.  Wrapping must happen BEFORE FSDP2
                # sharding so the module structure is stable when fully_shard() indexes params.
                for layer in layers:
                    for attr in ("self_attn", "mlp"):
                        m = getattr(layer, attr, None)
                        if m is not None:
                            setattr(layer, attr, checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT))
            else:
                # Preserve original behavior when compile is disabled.
                # For HF models on transformers >= 5.3.0, GradientCheckpointingLayer applies
                # AC at full-layer granularity via __call__ -- fewer CheckpointWrapper objects
                # and no memory-fragmentation overhead from sub-module wrapping.
                _use_hf_native_grad_ckpt = False
                try:
                    from transformers.modeling_layers import GradientCheckpointingLayer as _HFGradLayer

                    _use_hf_native_grad_ckpt = (
                        bool(layers)
                        and layers[0].__class__.__module__.startswith("transformers.")
                        and isinstance(layers[0], _HFGradLayer)
                        and getattr(model, "supports_gradient_checkpointing", False)
                        and hasattr(model, "gradient_checkpointing_enable")
                    )
                except ImportError:
                    pass

                if _use_hf_native_grad_ckpt:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
                else:
                    apply_submodule_checkpointing(layers, _has_kv_sharing)

        # Set up mixed precision policy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        # Install this only when NeMo actually enters FSDP2 sharding.
        _patch_fsdp_accumulated_grad_guard()

        # Find transformer layers and apply parallelisms
        self._shard_modules_recursively(
            model,
            dp_mesh,
            mp_policy,
            offload_policy,
            enable_fsdp2_prefetch,
            fsdp2_backward_prefetch_depth,
            fsdp2_forward_prefetch_depth,
            reshard_after_forward,
        )

        # Apply FSDP to the root model
        # Do not reshard after forward for root model because its parameters
        # will be used in backward immediately
        return self._shard_module(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
            offload_policy=offload_policy,
        )

    def _shard_module(
        self,
        module: nn.Module,
        *,
        mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy],
        offload_policy: Optional[OffloadPolicy],
        reshard_after_forward: Optional[bool],
    ) -> nn.Module:
        """Wrap one FSDP unit.

        Model-owned strategies override this hook when a model needs a custom
        FSDP unit layout or precision policy. The generic strategy never takes
        a model-specific sharding callback.
        """
        return fully_shard(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )

    def _shard_modules_recursively(
        self,
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy],
        offload_policy: Optional[OffloadPolicy] = None,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: Optional[bool] = None,
    ) -> None:
        """Recursively wrap FSDP units, optimizing decoder-layer containers."""
        pp_enabled = "pp" in mesh.mesh_dim_names and mesh["pp"].size() > 1

        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            # After pipeline splitting, functional.py replaces nn.ModuleList with nn.ModuleDict
            # (keyed by string layer indices). Normalise both to a list of (key, child) pairs.
            if isinstance(module, nn.ModuleDict):
                all_items = list(module.items())
                is_container = lambda child: isinstance(child, (nn.ModuleList, nn.ModuleDict))
            else:
                all_items = [(index, module[index]) for index in range(len(module))]
                is_container = lambda child: isinstance(child, nn.ModuleList)

            flat_layer_items = [(key, child) for key, child in all_items if not is_container(child)]
            nested_items = [(key, child) for key, child in all_items if is_container(child)]

            # Recurse into nested layer containers before wrapping their leaves.
            for _, child_module in nested_items:
                self._shard_modules_recursively(
                    child_module,
                    mesh,
                    mp_policy,
                    offload_policy,
                    enable_fsdp2_prefetch,
                    fsdp2_backward_prefetch_depth,
                    fsdp2_forward_prefetch_depth,
                    reshard_after_forward,
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
                self._shard_module(
                    child_module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=layer_reshard_after_forward,
                    offload_policy=offload_policy,
                )
                module[layer_key] = child_module

            # Set up explicit forward/backward prefetch chains when layers are being resharded.
            # With PP or an explicit no-reshard override, weights are always gathered -- no prefetch needed.
            should_prefetch = reshard_after_forward is not False and not pp_enabled and enable_fsdp2_prefetch
            if should_prefetch:
                fsdp_units = [child for _, child in flat_layer_items if not is_container(child)]
                if fsdp2_forward_prefetch_depth > 0:
                    for index in range(len(fsdp_units) - 1):
                        targets = [
                            fsdp_units[index + depth]
                            for depth in range(1, fsdp2_forward_prefetch_depth + 1)
                            if index + depth < len(fsdp_units)
                        ]
                        if targets and hasattr(fsdp_units[index], "set_modules_to_forward_prefetch"):
                            fsdp_units[index].set_modules_to_forward_prefetch(targets)
                for index in range(1, len(fsdp_units)):
                    targets = []
                    for depth in range(1, fsdp2_backward_prefetch_depth + 1):
                        if index - depth >= 0:
                            targets.append(fsdp_units[index - depth])
                    if targets and hasattr(fsdp_units[index], "set_modules_to_backward_prefetch"):
                        fsdp_units[index].set_modules_to_backward_prefetch(targets)
            return

        for name, sub_module in module.named_children():
            # A frozen audio tower never runs in the forward on image/text-only
            # data (in gemma E4B and E2B models), so wrapping its layers as their own FSDP units leaves those
            # units never all-gathered. Under gradient accumulation FSDP's
            # deferred post-backward then dereferences their (never-created)
            # ``_unsharded_param`` and raises ``AttributeError``. Skip it so its
            # params stay with the always-run root FSDP unit (which is still
            # sharded, and whose frozen params have ``grad is None`` so the
            # accumulate path is a no-op). Mirrors the audio_tower guard in
            # ``components/moe/parallelizer.py``.
            if name == "audio_tower" and _subtree_all_frozen(sub_module):
                continue
            self._shard_modules_recursively(
                sub_module,
                mesh,
                mp_policy,
                offload_policy,
                enable_fsdp2_prefetch,
                fsdp2_backward_prefetch_depth,
                fsdp2_forward_prefetch_depth,
                reshard_after_forward,
            )


# The registry is a stable extension point for out-of-tree strategies. Built-in
# model and adapter strategies use the lazy per-instance protocol below instead
# of mutating global state during import.
PARALLELIZATION_STRATEGIES: Dict[str, ParallelizationStrategy] = {}

_DEFAULT_STRATEGY = DefaultParallelizationStrategy()


def get_parallelization_strategy(model: nn.Module) -> ParallelizationStrategy:
    """Return a model-owned, adapter-owned, or registered strategy.

    A native model may expose ``_nemo_parallelization_strategy_factory``. The
    factory is called only when FSDP2 parallelization is requested, so model
    imports do not pull in a distributed strategy sidecar. Adapters for
    upstream model classes may instead attach a ready strategy instance as
    ``_nemo_parallelization_strategy`` immediately before parallelization.
    """
    strategy = getattr(model, "_nemo_parallelization_strategy", None)
    if strategy is not None:
        if not isinstance(strategy, ParallelizationStrategy):
            raise TypeError("_nemo_parallelization_strategy must be a ParallelizationStrategy")
        return strategy

    strategy_factory = getattr(model, "_nemo_parallelization_strategy_factory", None)
    if strategy_factory is not None:
        strategy = strategy_factory()
        if not isinstance(strategy, ParallelizationStrategy):
            raise TypeError("_nemo_parallelization_strategy_factory must return a ParallelizationStrategy")
        return strategy

    return PARALLELIZATION_STRATEGIES.get(type(model).__name__, _DEFAULT_STRATEGY)


def register_parallel_strategy(arg=None, *, name: Optional[str] = None):
    """Register an out-of-tree strategy under a model class name.

    Args:
        name: Model class name resolved by get_parallelization_strategy.

    Returns:
        A decorator that registers a ParallelizationStrategy subclass.
    """

    def _register(cls):
        if not isinstance(cls, type) or not issubclass(cls, ParallelizationStrategy):
            raise ValueError(f"Expected ParallelizationStrategy subclass, got {cls!r}")
        if name is None:
            raise ValueError("name is required")
        if name in PARALLELIZATION_STRATEGIES:
            raise ValueError(f"name {name!r} is already registered")
        PARALLELIZATION_STRATEGIES[name] = cls()
        return cls

    if arg is not None:
        raise ValueError("register_parallel_strategy must be called with name=<model class name>")
    if name is None:
        raise ValueError("name is required")
    return _register


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


def _subtree_all_frozen(module: nn.Module) -> bool:
    """Return True if ``module`` owns parameters and none of them require grad.

    Used to skip FSDP-wrapping a frozen submodule that never runs in the forward
    (e.g. the audio tower on image/text-only data); see
    ``apply_fsdp2_sharding_recursively``.
    """
    params = list(module.parameters())
    return len(params) > 0 and not any(p.requires_grad for p in params)


def apply_fsdp2_sharding_recursively(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy],
    offload_policy: Optional[OffloadPolicy] = None,
    enable_fsdp2_prefetch: bool = True,
    fsdp2_backward_prefetch_depth: int = 2,
    fsdp2_forward_prefetch_depth: int = 1,
    reshard_after_forward: Optional[bool] = None,
) -> None:
    """Apply the default strategy's FSDP2 traversal in place.

    Model-specific traversal and wrapping belong in their own
    :class:`ParallelizationStrategy`; this compatibility helper deliberately
    exposes no callback for replacing ``fully_shard``.
    """
    DefaultParallelizationStrategy()._shard_modules_recursively(
        module,
        mesh,
        mp_policy,
        offload_policy,
        enable_fsdp2_prefetch,
        fsdp2_backward_prefetch_depth,
        fsdp2_forward_prefetch_depth,
        reshard_after_forward,
    )


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


def _update_attention_head_counts_for_tp(model: nn.Module, tp_size: int) -> None:
    """
    After TP sharding, the Q/K/V outputs are split across ranks (each rank has
    num_heads/tp_size heads). Update the config and each attention layer's
    num_heads / num_key_value_heads so the forward uses the local head count
    instead of the global one (avoids shape mismatches in .view()).
    """
    if tp_size <= 1:
        return
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "num_attention_heads"):
        return
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None and hasattr(model, "language_model"):
        inner = model.language_model
        layers = getattr(inner, "layers", None)
    if layers is None:
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

    # PP converts ModuleList → ModuleDict; iterating a ModuleDict yields keys, not modules.
    layer_iter = layers.values() if isinstance(layers, nn.ModuleDict) else layers
    for layer in layer_iter:
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


def validate_tp_mesh_for_nemotron_nas(model, tp_size):
    """Validate that a Nemotron-NAS model can be tensor-parallel sharded."""
    num_attention_heads = model.config.num_attention_heads
    assert num_attention_heads % tp_size == 0, "num_attention_heads in config does not match the TP size"

    assert len(model.config.block_configs) >= model.config.num_hidden_layers, (
        "num_hidden_layers in config does not match the number of block configs"
    )

    for i in range(model.config.num_hidden_layers):
        # Valid layer
        if model.config.block_configs[i].attention.replace_with_linear:
            print(f"By pass checking for linear layer in layer {i}")
            # TODO: Check if the linear layer could support TP.
        else:
            if model.config.block_configs[i].attention.n_heads_in_group is not None:
                num_key_value_heads = num_attention_heads // model.config.block_configs[i].attention.n_heads_in_group
                assert num_key_value_heads % tp_size == 0, (
                    f"layer {i}: num_key_value_heads in config does not match the TP size"
                )
            else:
                assert model.config.block_configs[i].attention.no_op == True


def validate_tp_mesh(model, tp_mesh):
    """
    Validate that attention heads and key value heads are divisible by TP size
    """
    if tp_mesh.size() == 1:
        return  # if tp_mesh.size() == 1, we don't need to validate

    model_cls = type(model)

    # There are cases like DeciLMForCausalLM is defined in transformers_modules
    # which hardly has predefined path to import. Guard access to config/architectures.
    model_arch = None
    if hasattr(model, "config") and hasattr(model.config, "architectures") and model.config.architectures:
        try:
            model_arch = model.config.architectures[0]
        except Exception:
            model_arch = None

    if model_cls in [
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    ]:
        # VL models have the language model at model.language_model
        num_attention_heads = model.language_model.config.num_attention_heads
        num_key_value_heads = model.language_model.config.num_key_value_heads

    elif model_cls == SmolVLMForConditionalGeneration:
        num_attention_heads = model.model.text_model.config.num_attention_heads
        num_key_value_heads = model.model.text_model.config.num_key_value_heads

    elif model_cls in [
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextVideoForConditionalGeneration,
        LlavaOnevisionForConditionalGeneration,
    ]:
        num_attention_heads = model.language_model.config.num_attention_heads
        num_key_value_heads = model.language_model.config.num_key_value_heads

    elif model_cls == Mistral3ForConditionalGeneration:
        num_attention_heads = model.model.language_model.config.num_attention_heads
        num_key_value_heads = model.model.language_model.config.num_key_value_heads

    elif model_cls == Llama4ForConditionalGeneration:
        num_attention_heads = model.language_model.model.config.num_attention_heads
        num_key_value_heads = model.language_model.model.config.num_key_value_heads

    elif model_cls in [Gemma3ForConditionalGeneration, Gemma4ForConditionalGeneration]:
        num_attention_heads = model.config.text_config.num_attention_heads
        num_key_value_heads = model.config.text_config.num_key_value_heads
    elif model_arch == "DeciLMForCausalLM" and getattr(model.config, "model_type", None) == "nemotron-nas":
        validate_tp_mesh_for_nemotron_nas(model, tp_mesh.size())

        # SKip following code and return.
        return
    elif hasattr(model, "config"):
        num_attention_heads = getattr(model.config, "num_attention_heads", 0)
        num_key_value_heads = getattr(model.config, "num_key_value_heads", 0)
    else:
        num_attention_heads = 0
        num_key_value_heads = 0

    # TP sharding with enhanced plan generation
    # Validate that attention heads are divisible by TP size
    assert num_key_value_heads % tp_mesh.size() == 0, (
        f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"
    )
    assert num_attention_heads % tp_mesh.size() == 0, (
        f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
    )


def _find_largest_module_list(model: nn.Module) -> Optional[Union[nn.ModuleList, nn.ModuleDict]]:
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
    largest_module_list: Optional[Union[nn.ModuleList, nn.ModuleDict]] = None
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


def _extract_model_layers(model: nn.Module) -> List[nn.Module]:
    """
    Extract layers from different model architectures for parallelization.

    This function handles various model types including vision-language models,
    causal language models, and multimodal models. It collects both language
    model layers and vision model layers where applicable.

    Args:
        model (nn.Module): The model to extract layers from.

    Returns:
        List[nn.Module]: A list of all layers that should be parallelized.
    """

    def _reduce_attrs(model, fqns: List[str]) -> List[nn.Module]:
        if isinstance(fqns, str):
            fqns = [fqns]
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

    # Gemma3 layer paths depend on transformers version
    _gemma3_layers = (
        ["model.layers", "model.vision_tower.vision_model.encoder.layers"]
        if _is_transformers_v5_or_higher()
        else ["language_model.layers", "vision_tower.vision_model.encoder.layers"]
    )
    VLM_MODEL_CLS_TO_LAYERS = {
        Gemma3ForConditionalGeneration: _gemma3_layers,
        Qwen2_5_VLForConditionalGeneration: ["language_model.layers", "visual.blocks"],
        Qwen2VLForConditionalGeneration: ["language_model.layers", "visual.blocks"],
        # Note: `model.` is not a mistake here, it's the full fqn
        SmolVLMForConditionalGeneration: ["model.text_model.layers", "model.vision_model.encoder.layers"],
        LlavaForConditionalGeneration: ["model.language_model.layers", "vision_tower.vision_model.encoder.layers"],
        LlavaNextForConditionalGeneration: ["model.language_model.layers", "vision_tower.vision_model.encoder.layers"],
        LlavaNextVideoForConditionalGeneration: [
            "model.language_model.layers",
            "vision_tower.vision_model.encoder.layers",
        ],
        LlavaOnevisionForConditionalGeneration: [
            "model.language_model.layers",
            "vision_tower.vision_model.encoder.layers",
        ],
        Mistral3ForConditionalGeneration: ["model.language_model.layers", "model.vision_tower.transformer.layers"],
        # FP8 VLM subclass (own FP8 dequant on top of HF's Mistral3). String-keyed
        # because NeMo Auto wraps the class via HFCheckpointingMixin into a new
        # type with the same __name__ but distinct identity, so direct class
        # comparison misses; the elif `model_cls.__name__ in MAP` check catches it.
        "Mistral3FP8VLMForConditionalGeneration": [
            "model.language_model.layers",
            "model.vision_tower.transformer.layers",
        ],
        Llama4ForConditionalGeneration: ["language_model.model.layers", "vision_model.model.layers"],
        # String-keyed to avoid eagerly importing transformers.models.qwen3_5 at
        # module load (which would defeat test monkeypatches that stub the
        # module before first import).
        "Qwen3_5ForConditionalGeneration": ["model.language_model.layers", "model.visual.blocks"],
        Gemma4ForConditionalGeneration: ["model.language_model.layers"],
        # String fallback in case of class identity mismatch across imports
        "Gemma4ForConditionalGeneration": ["model.language_model.layers"],
        # BAGEL (text-to-image + understanding). String-keyed to avoid an
        # import cycle: parallelizer is core distributed code, the BAGEL
        # model lives under components/models/bagel/. Lists both the Qwen2
        # decoder ModuleList and the SigLIP encoder ModuleList so each
        # member becomes its own FSDP unit (matching upstream BAGEL's
        # transformer_auto_wrap_policy class set; without the SigLIP
        # entry, Stage 2 OOMs on 8x80GB because the SigLIP layers sit in
        # the root FSDP unit's all-gather peak).
        "BagelForUnifiedMultimodal": [
            "model.language_model.model.layers",
            "model.vit_model.vision_model.encoder.layers",
        ],
    }
    LLM_MODEL_CLS_TO_LAYERS = {
        "NemotronHForCausalLM": ["backbone.layers", "model.layers"],
        GPT2LMHeadModel: ["transformer.h"],
    }

    MODEL_CLS_TO_LAYERS = VLM_MODEL_CLS_TO_LAYERS | LLM_MODEL_CLS_TO_LAYERS

    def _extend_layers(layers, modules):
        for m in modules:
            if isinstance(m, nn.ModuleList):
                layers.extend(m)
            elif isinstance(m, nn.ModuleDict):
                layers.extend(m.values())
            else:
                layers.append(m)

    model_cls = type(model)
    layers: List[nn.Module] = []
    if model_cls in MODEL_CLS_TO_LAYERS:
        _extend_layers(layers, _reduce_attrs(model, MODEL_CLS_TO_LAYERS[model_cls]))
    elif model_cls.__name__ in MODEL_CLS_TO_LAYERS:
        _extend_layers(layers, _reduce_attrs(model, MODEL_CLS_TO_LAYERS[model_cls.__name__]))
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Default case for all other models (assumed to be a causal LM)
        if isinstance(model.model.layers, nn.ModuleDict):
            layers.extend(model.model.layers.values())
        else:
            layers.extend(model.model.layers)
    elif hasattr(model, "layers"):
        layers.extend(model.layers)
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

        if isinstance(largest_module_list, nn.ModuleDict):
            layers.extend(largest_module_list.values())
        else:
            layers.extend(largest_module_list)
        logger.info(f"Successfully extracted {len(largest_module_list)} layers using heuristic")

    assert all(isinstance(m, nn.Module) for m in layers), "layers shoudl be nn.Module instances"
    return layers


# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def fsdp2_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional[OffloadPolicy] = None,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
    dp_replicate_mesh_name: str = "dp_replicate",
    dp_shard_cp_mesh_name: str = "dp_shard_cp",
    tp_mesh_name: str = "tp",
    enable_async_tensor_parallel: bool = False,
    enable_compile: bool = False,
    enable_fsdp2_prefetch: bool = True,
    fsdp2_backward_prefetch_depth: int = 2,
    fsdp2_forward_prefetch_depth: int = 1,
    reshard_after_forward: Optional[bool] = None,
):
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
    )


def megatron_fsdp_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    optimizer=None,
    megatron_fsdp_unit_modules: Optional[List[str]] = None,
    tp_shard_plan: Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]] = None,
    zero_dp_strategy: int = 3,
    init_fsdp_with_meta_device: bool = False,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = False,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    check_for_nan_in_grad: bool = True,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    dp_shard_dim: str = "dp",
    tp_dim: str = "tp",
):
    """
    Apply tensor/data parallelism (MegatronFSDP) and optional activation-checkpointing to the model.

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh describing the physical devices
            used for distributed training.
        megatron_fsdp_unit_modules (Optional[List[str]]): Names of sub-modules that should
            become individual MegatronFSDP units. If None, the full model is wrapped as
            a single unit.
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
        check_for_nan_in_grad (bool): Whether to check gradients for NaNs/Infs
            before applying the optimizer step.
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

    # Import MegatronFSDP unit modules specified by the user.
    megatron_fsdp_unit_modules = import_classes_from_paths(megatron_fsdp_unit_modules)

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
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        preserve_fp32_weights=preserve_fp32_weights,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        check_for_nan_in_grad=check_for_nan_in_grad,
        average_in_collective=average_in_collective,
        disable_bucketing=disable_bucketing,
        calculate_per_token_loss=calculate_per_token_loss,
        keep_fp8_transpose_cache=keep_fp8_transpose_cache,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
    )
    if optimizer is not None:
        model, optimizer = megatron_fsdp_fully_shard(module=model, optimizer=optimizer, **fsdp_kwargs)
    else:
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
