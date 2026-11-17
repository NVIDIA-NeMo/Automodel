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

"""Model-owned distributed parallelization for ERNIE 5.5.

ERNIE 5.5 is distributed with FSDP2 and expert parallelism only. The sidecar rejects every other topology before
the shared dispatcher runs, and with it a model whose routed experts exchange tokens through HybridEP unless the
model and its FSDP policies compute in BF16. It keeps the shared expert-parallel flow, and adds two things while
that flow builds the FSDP units:

* Every HCA compressor receives the process group of the mesh over which its parameters are sharded. A row shorter
  than one compression window does not use the compressor. With the group, the ranks that share the compressor
  parameters agree, for every microbatch, on whether the compressor takes part in backward, so they issue the same
  FSDP collectives in the same order.
* The hyper-head becomes an FP32 unit of its own before the root unit, which computes in the bulk dtype, claims its
  parameters.

No other trainable parameter of the model is used by some ranks and not by others, so the sidecar does not enable
FSDP's reduction of unused parameters.

The sidecar also holds the two rules that LoRA needs. Before the shared dispatcher runs, it rejects an adapter that
the model does not support: on a module without LoRA support, of another kind than plain LoRA, or in another dtype
than the weight it adapts (``_reject_unsupported_lora_adapters``). After the shared flow has sharded the routed
experts, it records the expert-parallel size on them: a PEFT checkpoint save reads that attribute to decide whether
every rank collects the adapter tensors itself.
"""

import logging
from typing import Required, TypedDict, Unpack

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy

from nemo_automodel.components.distributed import FSDP2Config, MeshContext, ModelParallelizer
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5ModelPlan
from nemo_automodel.components.models.ernie5_5.hca import Ernie5_5HCAAttention
from nemo_automodel.components.models.ernie5_5.layers import Ernie5_5HyperHead
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)

# Parameters of ``Ernie5_5HyperHead``, named relative to that module. All of them compute in FP32.
_HYPER_HEAD_FP32_NAMES = ("hc_fn", "hc_base", "hc_scale")
# Compute dtypes that the model implements; an FSDP policy may also leave the storage dtype in place (None).
_COMPUTE_DTYPES = (None, torch.float32, torch.bfloat16)
# Plain linear projections that may carry a LoRA adapter, named relative to the attention module of an HCA layer,
# to the attention module of a DSA layer, and to a dense FFN or a shared expert. The routed experts of a MoE layer
# are the fourth supported target. A LoRA wrapper leaves each of them numerically unchanged at initialization.
_HCA_LORA_TARGETS = ("wq_a", "wq_b", "wkv", "wo_b", "gate_proj")
_DSA_LORA_TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "gate_proj", "o_proj")
_FFN_LORA_TARGETS = ("gate_proj", "up_proj", "down_proj")
# Adapter tensors of the two supported wrappers, named relative to the module that owns them: the two factors of a
# LoRA linear layer and the four grouped factors of the routed experts. These are the names that the checkpoint
# adapter saves and restores. A DoRA wrapper registers ``lora_magnitude`` as well; it is not supported.
_LINEAR_LORA_TENSORS = ("lora_A.weight", "lora_B.weight")
_EXPERT_LORA_TENSORS = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")


class _FullyShardKwargs(TypedDict, total=False):
    """Keyword arguments with which the shared flows call ``_fully_shard_module``."""

    mesh: Required[DeviceMesh]
    mp_policy: MixedPrecisionPolicy | None
    offload_policy: OffloadPolicy | None
    reshard_after_forward: bool | int | None
    ignored_params: set[nn.Parameter] | None


def _axis_names(mesh: DeviceMesh) -> str:
    """Return the axis names of a mesh as text, for example ``dp_replicate, dp_shard``."""
    return ", ".join(str(getattr(name, "value", name)) for name in mesh.mesh_dim_names or ())


def _is_pipeline_stage(model: nn.Module) -> bool:
    """Return whether pipeline-stage surgery removed modules or decoder layers from an ERNIE 5.5 model.

    Args:
        model: The ``Ernie5_5_MoeForCausalLM`` handed to the sidecar.

    Returns:
        True when the embedding, the hyper-head, the final norm, the output head or MultiMax is missing, or the
        decoder holds another number of layers than its configuration declares.
    """
    inner = model.model
    modules = (inner.embed_tokens, inner.hyper_head, inner.norm, inner.lm_head, inner.multimax)
    return any(module is None for module in modules) or len(inner.layers) != model.config.num_hidden_layers


def _lora_tensors(model: nn.Module) -> dict[str, dict[str, nn.Parameter]]:
    """Return the LoRA adapter tensors of a model, grouped by the module that owns them.

    An adapter tensor is a parameter with a name component that starts with ``lora_``, as the LoRA wrappers of
    linear layers (``lora_A``, ``lora_B``, and ``lora_magnitude`` with DoRA) and of grouped experts
    (``lora_gate_and_up_A`` and three more) register them. The module that owns it is named by the components
    before that one.

    Args:
        model: Model whose parameters are inspected.

    Returns:
        For every owning module, in the order of ``named_parameters``, its adapter tensors of any shape keyed by
        their name relative to the module, for example ``lora_A.weight``.
    """
    owners: dict[str, dict[str, nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        for position, part in enumerate(parts):
            if part.startswith("lora_"):
                owners.setdefault(".".join(parts[:position]), {})[".".join(parts[position:])] = parameter
                break
    return owners


def _supported_lora_modules(model: nn.Module) -> set[str]:
    """Return the names of the modules of an ERNIE 5.5 model that support a LoRA adapter.

    Args:
        model: The ``Ernie5_5_MoeForCausalLM`` handed to the sidecar, with all its decoder layers.

    Returns:
        For every decoder layer, the plain linear projections that its attention module owns directly (not those
        of its indexer or compressor), and the three projections of its dense FFN or the routed experts and the
        three projections of the shared expert of its MoE.
    """
    names: set[str] = set()
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        attention = _DSA_LORA_TARGETS if layer.is_dsa else _HCA_LORA_TARGETS
        names.update(f"{prefix}.attention.{target}" for target in attention)
        if layer.is_moe:
            names.add(f"{prefix}.moe.experts")
            names.update(f"{prefix}.moe.shared_experts.{target}" for target in _FFN_LORA_TARGETS)
        else:
            names.update(f"{prefix}.mlp.{target}" for target in _FFN_LORA_TARGETS)
    return names


def _lora_rejection_reason(name: str) -> str:
    """Say why a module of an ERNIE 5.5 model does not support a LoRA adapter.

    Args:
        name: Name of a module outside the supported targets.

    Returns:
        The reason, as the middle part of the error message.
    """
    if ".attention.indexer." in name:
        return "the DSA indexer only selects keys and is not trained"
    if name.endswith(".attention.wo_a"):
        return "wo_a is a grouped projection, and a LoRA wrapper would replace its grouped forward"
    if name.endswith((".moe.fc1_latent_proj", ".moe.fc2_latent_proj")):
        return (
            "the latent projections of the MoE cast their FP32 weight in a forward of their own, which a LoRA "
            "wrapper would replace"
        )
    return "LoRA has not been validated on this module"


def _reject_unsupported_lora_adapters(model: nn.Module) -> None:
    """Reject every LoRA adapter that an ERNIE 5.5 model does not support.

    The module is checked first. The supported targets are the plain linear projections that an attention module
    owns directly, the routed experts, and the projections of the dense FFN and of the shared expert. The LoRA code
    also wraps other linear layers when a pattern covers them, and its default pattern does. Three of them are
    known to fail later: the FP32 latent projections of the MoE while the shared flow shards the model, and the
    grouped projection ``wo_a`` of an HCA layer and the projections of a DSA indexer in the first training
    forward. Every other module, such as the FP32 projections of the HCA compressor or the output head, has not
    been validated with an adapter. A pattern that names a parameter that is not a module of its own (the key and
    value projections of a DSA layer, the router) or a module that is neither a linear layer nor the routed
    experts adds no adapter and is not seen here.

    On a supported module the adapter tensors are checked. Only the two factors of a LoRA linear layer and the four
    grouped factors of the routed experts are admitted: these are the tensors that the checkpoint adapter saves
    and restores. DoRA adds a magnitude vector to a linear layer and is rejected. A model that the framework
    builds with its checkpoint adapter attached is rejected for that vector even earlier, when the adapter plans
    the checkpoint keys of the unsharded model, and with the routed experts among the targets the LoRA installer
    rejects DoRA before that.

    Every admitted factor must have the dtype of the weight it adapts. With ``peft.lora_dtype`` set to another
    dtype and an FSDP policy without a compute dtype, the shared flow cannot group the parameters of a decoder
    block by dtype and fails while it shards the model, with a message about the frozen weights.

    The factors also differ from the weight when ``peft.lora_dtype`` is unset, the model dtype is FP32 and the FSDP
    policy computes in BF16, which is the default of ``FSDP2Config``: before it calls the sidecar, the framework
    stores the frozen weights of a PEFT model in the compute dtype, and the trainable factors keep FP32. That
    layout has not been validated with this model; the message names the policy as well.

    Args:
        model: The ``Ernie5_5_MoeForCausalLM`` handed to the sidecar, with or without LoRA adapters.

    Raises:
        ValueError: If a module outside the supported targets owns a LoRA adapter tensor; if a supported module
            owns an adapter tensor other than the LoRA factors, such as the magnitude vector of DoRA; or if a
            factor has another dtype than the weight it adapts.
    """
    adapters = _lora_tensors(model)
    supported = _supported_lora_modules(model)
    unsupported = [name for name in adapters if name not in supported]
    if unsupported:
        first = unsupported[0]
        count = f" (the first of {len(unsupported)} such modules)" if len(unsupported) > 1 else ""
        hca, dsa, ffn = (", ".join(targets) for targets in (_HCA_LORA_TARGETS, _DSA_LORA_TARGETS, _FFN_LORA_TARGETS))
        raise ValueError(
            f"ERNIE 5.5 does not support LoRA on {first}{count}: {_lora_rejection_reason(first)}. Set "
            "peft.target_modules to patterns that select supported modules only. Under model.layers.<index> these "
            f"are attention.<name> with <name> in ({hca}) for an HCA layer and in ({dsa}) for a DSA layer, "
            f"moe.experts, and moe.shared_experts.<name> and mlp.<name> with <name> in ({ffn})"
        )
    for owner, tensors in adapters.items():
        experts = owner.endswith(".moe.experts")
        factors = _EXPERT_LORA_TENSORS if experts else _LINEAR_LORA_TENSORS
        others = [name for name in tensors if name not in factors]
        if "lora_magnitude" in others:
            raise ValueError(
                f"ERNIE 5.5 does not support DoRA: {owner} carries the magnitude vector lora_magnitude. Disable "
                "peft.use_dora"
            )
        if others:
            raise ValueError(
                f"ERNIE 5.5 does not support the adapter tensor {owner}.{others[0]}: only the two factors of a "
                "LoRA linear layer and the four grouped factors of the routed experts are supported"
            )
        module = model.get_submodule(owner)
        weight = module.gate_and_up_projs if experts else module.weight
        for name, tensor in tensors.items():
            if tensor.dtype != weight.dtype:
                raise ValueError(
                    f"ERNIE 5.5 requires LoRA factors in the dtype of the weight they adapt: {owner}.{name} is "
                    f"{tensor.dtype} and {owner} is {weight.dtype}. Leave peft.lora_dtype unset. If it is unset, the "
                    "frozen weights were stored in the FSDP compute dtype: set distributed.mp_policy.param_dtype to "
                    "null or to the model dtype"
                )


class Ernie5_5ModelParallelizer(ModelParallelizer):
    """Distribute ERNIE 5.5 with FSDP2 and expert parallelism, and reject every other topology.

    The supported topology is ``distributed.strategy: fsdp2`` with ``ep_size`` greater than one, one data-parallel
    sharding dimension over all ranks of the job (``tp_size``, ``cp_size``, ``pp_size`` and ``dp_replicate_size``
    equal to one), FSDP compute in FP32 or BF16, and no activation checkpointing. A model built with the HybridEP
    dispatcher (``model.backend.dispatcher: hybridep``) is admitted for BF16 compute only: a BF16 model whose FSDP
    policies have no compute dtype or BF16. The model was validated with HybridEP in BF16 only, and the HybridEP
    combine fails a C++ assertion, which aborts the process, for activations that are not 16 bits wide.

    A model with a DSA layer also restricts the number of ranks: it must divide
    ``config.hybrid_mla_num_attention_heads``. A checkpoint stores the key and value projections of a DSA layer as
    one matrix with the rows of each attention head together, and the checkpoint adapter loads and saves that
    matrix only when every rank holds the rows of whole heads. With 64 heads the supported jobs have 2, 4, 8, 16,
    32 or 64 ranks.

    A model with LoRA adapters is admitted when every adapter sits on a supported target (a plain linear
    projection that an attention module owns directly, the routed experts, or a projection of the dense FFN or of
    the shared expert), consists of the LoRA factors only, and has the dtype of the weight it adapts. DoRA is not
    supported.
    """

    _customizes_moe_fsdp = True

    def _prepare_model(self, model: nn.Module, mesh_context: MeshContext) -> None:
        """Reject every unsupported topology and LoRA adapter before an executor changes the model.

        The shared dispatcher has not run when this hook is called, so a rejected model keeps its parameters, its
        modules and its synchronization groups as they were handed to the hook. The hook runs for every strategy,
        including DDP and Megatron-FSDP. Under pipeline parallelism the framework has already copied the model
        and cut the copies into stages when it calls the hook, once per stage; the hook rejects the stage.

        The framework installs LoRA adapters on the unsharded model before it calls the hook, and loads the base
        weights after sharding, so an unsupported adapter is rejected before the model is sharded and before a
        weight is read.

        One condition is not decided here. The mesh over which the FSDP units are sharded is derived inside the
        shared expert-parallel flow, so ``_fully_shard_module`` checks its number of dimensions as a
        post-condition.

        Args:
            model: The ``Ernie5_5_MoeForCausalLM`` to distribute, with or without LoRA adapters.
            mesh_context: Meshes, parallelism sizes, strategy, precision policies and activation-checkpointing
                mode of the run.

        Raises:
            ValueError: If the strategy is not FSDP2; if the tensor-, context- or pipeline-parallel size or the
                replicate size is not one; if ``model`` is a pipeline stage; if expert parallelism is off; if
                activation checkpointing is on; if an FSDP compute dtype is neither FP32 nor BF16; if the model
                was built with the HybridEP dispatcher and its dtype or an FSDP compute dtype is FP32; if the
                data-parallel mesh does not cover all ranks of the job; if the model has a DSA layer and the
                number of ranks does not divide its number of DSA attention heads; or if a LoRA adapter sits on a
                module that does not support one, holds a tensor other than the LoRA factors, or has another
                dtype than the weight it adapts.
        """
        strategy = mesh_context.strategy_config
        if not isinstance(strategy, FSDP2Config):
            raise ValueError(
                f"ERNIE 5.5 requires distributed.strategy fsdp2 with expert parallelism, got {type(strategy).__name__}"
            )
        sizes = {
            "tp_size": mesh_context.tp_size,
            "cp_size": mesh_context.cp_size,
            "pp_size": mesh_context.pp_size,
            "dp_replicate_size": mesh_context.dp_replicate_size or 1,
        }
        unsupported = {name: size for name, size in sizes.items() if size != 1}
        if unsupported:
            raise ValueError(
                "ERNIE 5.5 supports FSDP2 with expert parallelism only; set these distributed sizes to 1: "
                f"{unsupported}"
            )
        if _is_pipeline_stage(model):
            raise ValueError(
                "ERNIE 5.5 does not support pipeline parallelism: the model handed to the parallelizer is a "
                "pipeline stage that lacks modules or decoder layers of the full model"
            )
        ep_size = mesh_context.ep_size
        if ep_size <= 1:
            raise ValueError(
                "ERNIE 5.5 requires expert parallelism: set distributed.ep_size to a value greater than 1, got "
                f"{ep_size}"
            )
        if mesh_context.activation_checkpointing:
            raise ValueError(
                "ERNIE 5.5 does not support activation checkpointing: set distributed.activation_checkpointing to "
                f"false, got {mesh_context.activation_checkpointing!r}"
            )
        moe = mesh_context.moe_parallel_config
        policies = {
            "distributed.mp_policy": strategy.mp_policy,
            "distributed.moe.mp_policy": None if moe is None else moe.mp_policy,
        }
        for name, policy in policies.items():
            if policy is not None and policy.param_dtype not in _COMPUTE_DTYPES:
                raise ValueError(
                    f"ERNIE 5.5 computes in float32 or bfloat16 only: set {name}.param_dtype to one of them, got "
                    f"{policy.param_dtype}"
                )
        if model.backend.dispatcher == "hybridep":
            # The routed experts hand the latent activations to HybridEP in the compute dtype of their decoder
            # block: the model dtype, or the compute dtype of an FSDP policy when it has one. An FP32 model
            # under a BF16 policy would compute in BF16 as well; it is rejected because it was not validated.
            model_dtype = dtype_from_str(model.config.dtype, torch.bfloat16)
            if model_dtype != torch.bfloat16:
                raise ValueError(
                    "ERNIE 5.5 with model.backend.dispatcher hybridep is validated with model.dtype bfloat16 "
                    "only: set model.dtype to bfloat16, or set model.backend.experts and "
                    f"model.backend.dispatcher to torch; got model.dtype {model_dtype}"
                )
            for name, policy in policies.items():
                if policy is not None and policy.param_dtype not in (None, torch.bfloat16):
                    raise ValueError(
                        "ERNIE 5.5 with model.backend.dispatcher hybridep computes in bfloat16 only: set "
                        f"{name}.param_dtype to null or bfloat16, or set model.backend.experts and "
                        f"model.backend.dispatcher to torch; got {policy.param_dtype}"
                    )
        device_mesh = mesh_context.device_mesh
        if device_mesh is None:
            raise ValueError("ERNIE 5.5 requires a data-parallel device mesh, got mesh_context.device_mesh None")
        ranks, world = device_mesh.size(), dist.get_world_size()
        if ranks != world:
            raise ValueError(
                f"ERNIE 5.5 requires a data-parallel mesh over all {world} ranks of the job, got a mesh of {ranks} "
                "ranks"
            )
        config = model.config
        if any(layer.attention == "dsa" for layer in Ernie5_5ModelPlan.from_config(config).decoder_layers):
            # The condition of the checkpoint adapter for the key and value projections of a DSA layer, which
            # are sharded over all ranks.
            heads = config.hybrid_mla_num_attention_heads
            if heads % ranks != 0:
                raise ValueError(
                    "ERNIE 5.5 with a DSA layer requires a number of ranks that divides "
                    f"hybrid_mla_num_attention_heads={heads}, so that every rank holds whole attention heads when "
                    f"a checkpoint is loaded or saved; got a job of {ranks} ranks"
                )
        _reject_unsupported_lora_adapters(model)

    def _apply_moe(self, model: nn.Module, mesh_context: MeshContext) -> nn.Module:
        """Run the shared expert-parallel flow, verify it, and record the expert-parallel size on the experts.

        For this model, which has neither pipeline stages nor quantized modules, a PEFT checkpoint save lets every
        rank collect the adapter tensors itself only when a module carries an ``ep_size`` attribute greater than
        one; otherwise only rank 0 receives them and the other ranks hand an empty state to the checkpoint
        adapter. Expert modules with a token dispatcher set the attribute when the shared flow initializes the
        dispatcher. The torch experts of this model have no dispatcher, so the sidecar sets it once the shared
        flow has sharded them and its post-condition holds. A save without PEFT does not read the attribute.

        Args:
            model: The ``Ernie5_5_MoeForCausalLM`` to distribute, admitted by ``_prepare_model``.
            mesh_context: Meshes, parallelism sizes and strategy of the run.

        Returns:
            The model, with its parameters sharded by the shared expert-parallel flow and ``ep_size`` set on the
            routed experts of every MoE layer.

        Raises:
            RuntimeError: If the shared flow returned without making the hyper-head an FSDP unit.
        """
        model = super()._apply_moe(model, mesh_context)
        heads = [module for module in model.modules() if isinstance(module, Ernie5_5HyperHead)]
        if not heads or not all(isinstance(head, FSDPModule) for head in heads):
            raise RuntimeError(
                "The expert-parallel flow did not shard ERNIE 5.5 through its sidecar: the hyper-head is not an "
                "FSDP unit, so its FP32 boundary and the HCA synchronization groups are missing"
            )
        for layer in model.model.layers:
            if layer.is_moe:
                layer.moe.experts.ep_size = mesh_context.ep_size
        return model

    def _fully_shard_module(self, module: nn.Module, **kwargs: Unpack[_FullyShardKwargs]) -> nn.Module:
        """Shard ``module`` after giving ERNIE-specific units inside it their boundary and group.

        The shared flow calls this hook for each decoder block, for the modules inside a block that compute in
        FP32, for the embedding, the output head and the roots. Whichever call contains an HCA compressor sets the
        compressor's synchronization group from the mesh of that call; setting the same group again is harmless.
        The call that contains the hyper-head shards it first as an FP32 unit.

        The check of the mesh is a post-condition of the shared flow, not an admission check: when it fails, the
        flow has already distributed the expert weights. A mesh context built by ``MeshContext.build`` that
        ``_prepare_model`` admitted yields a one-dimensional mesh.

        Args:
            module: Module that becomes an FSDP unit, with parameters of any shape.
            **kwargs: Arguments of ``fully_shard``. ``mesh`` is the one-dimensional data-parallel mesh over which
                the parameters of ``module`` are sharded on their leading axis. ``ignored_params`` holds
                parameters of any shape that another unit already owns; it is passed to ``fully_shard``
                unchanged, as are ``mp_policy``, ``offload_policy`` and ``reshard_after_forward``.

        Returns:
            ``module`` as an FSDP unit.

        Raises:
            ValueError: If ``mesh`` has more than one dimension.
        """
        mesh = kwargs["mesh"]
        if mesh.ndim != 1:
            raise ValueError(
                "ERNIE 5.5 shards its FSDP units over a one-dimensional data-parallel mesh, got a mesh with the "
                f"dimensions ({_axis_names(mesh)})"
            )
        group = mesh.get_group()
        compressors = 0
        head_sharded = False
        for submodule in module.modules():
            if isinstance(submodule, Ernie5_5HCAAttention):
                submodule.compressor._set_hca_param_sync_group(group)
                compressors += 1
            elif isinstance(submodule, Ernie5_5HyperHead) and not isinstance(submodule, FSDPModule):
                fully_shard_by_dtype(
                    submodule,
                    mesh,
                    kwargs.get("mp_policy"),
                    kwargs.get("offload_policy"),
                    fp32_compute_module_names=_HYPER_HEAD_FP32_NAMES,
                    reshard_after_forward=kwargs.get("reshard_after_forward"),
                )
                head_sharded = True
        if head_sharded:
            logger.info(
                "ERNIE 5.5 FSDP on the %d-rank mesh (%s): hyper-head sharded as an FP32 unit, synchronization group "
                "set on %d HCA compressors",
                mesh.size(),
                _axis_names(mesh),
                compressors,
            )
        if isinstance(module, Ernie5_5HyperHead):
            return module
        return super()._fully_shard_module(module, **kwargs)


PARALLELIZER = Ernie5_5ModelParallelizer()

__all__ = ["PARALLELIZER"]
