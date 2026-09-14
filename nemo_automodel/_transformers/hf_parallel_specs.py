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

"""Parallelization contracts for architectures Automodel does not implement.

Stock ``transformers`` classes and ``trust_remote_code`` checkpoints cannot declare a
``parallel_spec`` themselves, so this bridge owns their contracts and
:func:`parallel_spec_for` binds one onto the wrapper class ``_get_mixin_wrapped_class``
creates. Automodel-owned models declare ``parallel_spec`` on their own class instead.
"""

from __future__ import annotations

import dataclasses
from typing import cast

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import (
    RotaryEmbedParallel,
    SequenceParallelAllGatherActivation,
    VocabParallelEmbedding,
)
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy
from nemo_automodel.components.models.llama.parallelization import llama_tp_plan


def _gemma3_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a Gemma3ForCausalLM model across data and tensor parallel dimensions."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

    if isinstance(model, Gemma3ForConditionalGeneration):
        model_prefix = "model.language_model"
    else:
        model_prefix = "model"

    base_model_tp_plan: dict[str, ParallelStyle] = {
        f"{model_prefix}.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        f"{model_prefix}.layers.*.self_attn.q_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.k_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.v_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(),
        f"{model_prefix}.layers.*.mlp.up_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.gate_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    base_model_sp_plan = {
        f"{model_prefix}.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        f"{model_prefix}.rotary_emb": RotaryEmbedParallel(use_local_output=True),
        f"{model_prefix}.rotary_emb_local": RotaryEmbedParallel(use_local_output=True),
        f"{model_prefix}.layers.*.input_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        f"{model_prefix}.layers.*.post_attention_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.pre_feedforward_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        f"{model_prefix}.layers.*.post_feedforward_layernorm": SequenceParallel(),
        f"{model_prefix}.norm": SequenceParallel(),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _phi_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a PhiForCausalLM (Phi-2) model across tensor parallel dimensions.

    Phi-2 uses ``self_attn.dense`` instead of ``self_attn.o_proj`` and
    ``mlp.fc1``/``mlp.fc2`` instead of ``mlp.gate_proj``/``mlp.up_proj``/``mlp.down_proj``.
    """
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.dense": RowwiseParallel(),
        "model.layers.*.mlp.fc1": ColwiseParallel(),
        "model.layers.*.mlp.fc2": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        base_model_sp_plan: dict[str, ParallelStyle] = {
            "model.embed_tokens": VocabParallelEmbedding(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "model.final_layernorm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.dense": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            "model.layers.*.mlp.fc2": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
        }
        base_model_tp_plan.update(base_model_sp_plan)

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _phi3_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        # Fused Attention can not be sharded
        "model.layers.*.self_attn.qkv_proj": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "model.layers.*.self_attn.o_proj": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        # Shard MLP layers
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
        "model.layers.*.mlp.down_proj": RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=Replicate(),
        ),
        "lm_head": ColwiseParallel(
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
    }

    return cast(
        dict[str, ParallelStyle],
        base_model_tp_plan,
    )


def _falcon_h1_tp_plan(
    model,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelize Falcon-H1 (hybrid Transformer + Mamba2 SSM).

    Every Falcon-H1 decoder layer runs an attention branch (``self_attn``) and a
    Mamba2 branch (``mamba``) in parallel, followed by an MLP (``feed_forward``).
    Only the attention and MLP linears are tensor-parallel sharded; the Mamba2
    mixer stays replicated because its SSM scan / causal conv1d are not
    TP-shardable with stock kernels (same approach as Qwen3.5's GatedDeltaNet
    linear-attention branch).

    A dedicated plan is required because HuggingFace ships only
    ``_tp_plan = {"lm_head": "colwise_gather_output"}`` for FalconH1, and its MLP
    is named ``feed_forward`` (not ``mlp``). The generic llama-style fallback plan
    therefore matches neither the HF plan (the ``colwise_gather_output`` style is
    rejected) nor the MLP module names, leaving the dominant ``feed_forward``
    weights replicated across TP ranks — which OOMs large variants such as
    Falcon-H1-34B even under LoRA.

    ``sequence_parallel`` is accepted for signature compatibility but ignored: the
    parallel Mamba2 branch emits non-sequence-parallel activations that cannot be
    combined with sequence-parallel attention outputs.
    """
    return cast(
        dict[str, ParallelStyle],
        {
            "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.feed_forward.gate_proj": ColwiseParallel(),
            "model.layers.*.feed_forward.up_proj": ColwiseParallel(),
            "model.layers.*.feed_forward.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        },
    )


def _qwen_classification_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    from nemo_automodel.components.models.qwen2.parallelization import qwen_tp_plan

    plan = qwen_tp_plan(model, sequence_parallel)
    assert not hasattr(model, "lm_head"), "Expected model not to have lm_head"
    del plan["lm_head"]
    assert hasattr(model, "score"), "Expected model to have score"
    # `Qwen3ForSequenceClassification` pools over the *sequence* dimension in Python.
    # Ensure the classifier logits are replicated (full num_labels) for correct pooling/loss.
    plan["score"] = ColwiseParallel(output_layouts=Replicate())
    return plan


def _mistral3_vlm_tp_plan(
    model,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """TP plan for Mistral3ForConditionalGeneration (and subclasses like
    Mistral3FP8VLMForConditionalGeneration). The Ministral3 text backbone
    lives at ``model.language_model.{embed_tokens, layers.*}``; vision_tower
    and multi_modal_projector stay replicated across TP ranks.
    """
    model_prefix = "model.language_model"
    base_model_tp_plan: dict[str, ParallelStyle] = {
        f"{model_prefix}.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        f"{model_prefix}.layers.*.self_attn.q_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.k_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.v_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(),
        f"{model_prefix}.layers.*.mlp.up_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.gate_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }
    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _nemotron_labs_diffusion_tp_plan(
    model,  # NemotronLabsDiffusionModel — loaded via trust_remote_code, not importable.
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """TP plan for ``NemotronLabsDiffusionModel`` (Nemotron-Labs-Diffusion).

    Same shape as the Ministral3 plan but the model uses
    ``encoder.*`` (not ``model.*``) and the output head is ``diffusion_head``
    (not ``lm_head``).
    """
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "encoder.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "encoder.layers.*.self_attn.q_proj": ColwiseParallel(),
        "encoder.layers.*.self_attn.k_proj": ColwiseParallel(),
        "encoder.layers.*.self_attn.v_proj": ColwiseParallel(),
        "encoder.layers.*.self_attn.o_proj": RowwiseParallel(),
        "encoder.layers.*.mlp.up_proj": ColwiseParallel(),
        "encoder.layers.*.mlp.gate_proj": ColwiseParallel(),
        "encoder.layers.*.mlp.down_proj": RowwiseParallel(),
        "diffusion_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    base_model_sp_plan = {
        "encoder.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "encoder.norm": SequenceParallel(),
        "encoder.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "encoder.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "encoder.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "encoder.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "diffusion_head": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
    }

    if sequence_parallel:
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def get_decilm_nemotron_tp_plan(
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Return a TP plan for remote-code DeciLM Nemotron-NAS checkpoints.

    DeciLM/Nemotron-NAS is close to Llama structurally, but its remote-code forward
    path performs model-level rotary embedding setup and per-layer block-config
    dispatch. In practice, the generic base-style plan is a safer match than the
    Llama-optimized named plan for this architecture.
    """
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Replicate()),
    }

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

    if sequence_parallel:
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _decilm_nemotron_tp_plan(
    model,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    if getattr(getattr(model, "config", None), "model_type", None) != "nemotron-nas":
        raise ValueError("DeciLM TP plan is only registered for Nemotron-NAS checkpoints")
    return get_decilm_nemotron_tp_plan(sequence_parallel=sequence_parallel)


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


def _nemotron_flash_adjust_tp_plan(model: nn.Module, plan: dict[str, ParallelStyle]) -> dict[str, ParallelStyle]:
    """Keep ``lm_head`` sharding consistent with Nemotron-Flash's normalized logits.

    Its forward computes ``logits / self.lm_head.weight.norm(p=2, dim=1)``, so the logits and
    the weight norm must either both be plain tensors or both use the same vocab-sharded
    DTensor layout. A replicated-output ColwiseParallel plan mixes a plain logits tensor with a
    sharded weight norm, so that entry is dropped; a vocab-sharded output keeps both operands
    aligned and is required under FSDP+TP.
    """
    for k in ("lm_head", "language_model.lm_head"):
        style = plan.get(k)
        output_layouts = getattr(style, "output_layouts", ())
        if not isinstance(output_layouts, (tuple, list)):
            output_layouts = (output_layouts,)
        if any(isinstance(layout, Shard) for layout in output_layouts):
            continue
        plan.pop(k, None)
    return plan


# Layer containers list every known location across transformers releases; the first
# candidate that resolves wins, so no version gating is needed. Canonical paths come first:
# standardized 4.x releases keep deprecated top-level alias properties (``language_model``,
# ``vision_tower``, ``visual``) that also resolve, and first-match-wins must not pick the
# alias. Shapes verified by meta-instantiating each class on transformers 4.51.3, 4.57.1,
# 5.8.1 and 5.12.1.
_GEMMA3_LAYERS = {
    "language": ("model.language_model.layers", "language_model.model.layers"),
    "vision": (
        "model.vision_tower.vision_model.encoder.layers",
        "model.vision_tower.encoder.layers",
        "vision_tower.vision_model.encoder.layers",
    ),
}
_QWEN2_VL_LAYERS = {
    "language": ("model.language_model.layers", "model.layers"),
    "vision": ("model.visual.blocks", "visual.blocks"),
}
# Llava family: same tree history as Gemma3 (CLIP instead of SigLIP tower).
_LLAVA_SPEC = ParallelSpec(
    layer_groups=_GEMMA3_LAYERS,
    text_config_path="language_model.config",
    hf_tp_plan_prefix=("model.language_model",),
)
_QWEN2_VL_SPEC = ParallelSpec(
    layer_groups=_QWEN2_VL_LAYERS,
    text_config_path="language_model.config",
    hf_tp_plan_prefix=("model.language_model",),
)
# Mistral3 VLM (Pixtral + Ministral3); the Automodel FP8 subclass reuses this spec.
MISTRAL3_VLM_PARALLEL_SPEC = ParallelSpec(
    tp_plan=_mistral3_vlm_tp_plan,
    layer_groups={
        "language": ("model.language_model.layers",),
        "vision": (
            "model.vision_tower.encoder.layers",
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.transformer.layers",
        ),
    },
    text_config_path="model.language_model.config",
    hf_tp_plan_prefix=("model.language_model",),
)

HF_PARALLEL_SPECS: dict[str, ParallelSpec] = {
    # transformers dense causal LMs
    "PhiForCausalLM": ParallelSpec(tp_plan=_phi_tp_plan),
    "Phi3ForCausalLM": ParallelSpec(tp_plan=_phi3_tp_plan),
    "Gemma3ForCausalLM": ParallelSpec(tp_plan=_gemma3_tp_plan),
    "Qwen3ForSequenceClassification": ParallelSpec(tp_plan=_qwen_classification_tp_plan),
    # Falcon-H1 (hybrid Transformer + Mamba2): HF ships only a minimal _tp_plan and names its
    # MLP ``feed_forward``, so the generic fallback leaves it replicated and OOMs the 34B variant.
    "FalconH1ForCausalLM": ParallelSpec(tp_plan=_falcon_h1_tp_plan),
    "GPT2LMHeadModel": ParallelSpec(layer_groups={"language": ("transformer.h",)}),
    # transformers VLMs
    "Gemma3ForConditionalGeneration": ParallelSpec(
        tp_plan=_gemma3_tp_plan,
        layer_groups=_GEMMA3_LAYERS,
        text_config_path="config.text_config",
        # Pre-standardization releases hang the text tower off a top-level ``language_model``.
        hf_tp_plan_prefix=("model", "language_model"),
    ),
    "Qwen2VLForConditionalGeneration": _QWEN2_VL_SPEC,
    "Qwen2_5_VLForConditionalGeneration": _QWEN2_VL_SPEC,
    "SmolVLMForConditionalGeneration": ParallelSpec(
        layer_groups={"language": ("model.text_model.layers",), "vision": ("model.vision_model.encoder.layers",)},
        text_config_path="model.text_model.config",
    ),
    "LlavaForConditionalGeneration": _LLAVA_SPEC,
    "LlavaNextForConditionalGeneration": _LLAVA_SPEC,
    "LlavaNextVideoForConditionalGeneration": _LLAVA_SPEC,
    "LlavaOnevisionForConditionalGeneration": _LLAVA_SPEC,
    "Llama4ForConditionalGeneration": ParallelSpec(
        layer_groups={"language": ("language_model.model.layers",), "vision": ("vision_model.model.layers",)},
        text_config_path="language_model.model.config",
        hf_tp_plan_prefix=("language_model.model",),
    ),
    "Mistral3ForConditionalGeneration": MISTRAL3_VLM_PARALLEL_SPEC,
    # trust_remote_code architectures (matched by class name; their module path carries a snapshot hash)
    "NemotronFlashForCausalLM": ParallelSpec(tp_plan=llama_tp_plan, adjust_tp_plan=_nemotron_flash_adjust_tp_plan),
    "DeciLMForCausalLM": ParallelSpec(tp_plan=_decilm_nemotron_tp_plan, validate_tp=validate_tp_mesh_for_nemotron_nas),
    "NemotronLabsDiffusionModel": ParallelSpec(tp_plan=_nemotron_labs_diffusion_tp_plan),
}


def parallel_spec_for(model_cls: type) -> ParallelSpec | None:
    """Contract for a class Automodel does not own, or ``None`` when the generic defaults apply.

    Walks the MRO so subclasses (``HFCheckpointingMixin`` wrappers, FP8 variants, remote-code
    ports) inherit their base architecture's contract. Per class, an explicit
    :data:`HF_PARALLEL_SPECS` entry wins; otherwise the native Automodel implementation
    registered under the same architecture name is authoritative for its transformers twin.
    """
    from nemo_automodel._transformers.registry import ModelRegistry

    for cls in model_cls.__mro__:
        spec = HF_PARALLEL_SPECS.get(cls.__name__)
        if spec is None and ModelRegistry.has_custom_model(cls.__name__):
            spec = getattr(ModelRegistry.get_model_cls_from_model_arch(cls.__name__), "parallel_spec", None)
        if spec is not None:
            return spec
    return None


def register_parallel_strategy(arg=None, *, name: str | None = None):
    """Decorator binding an out-of-tree ``ParallelizationStrategy`` to a class name.

    For classes you own, declare ``parallel_spec`` on the class instead; this is for
    third-party classes that reach the HF bridge, and must run before the model is built.

    Supports:
    - @register_parallel_strategy(name="CustomModelName")
    """

    def _register(cls):
        # The decorator receives a class, not an instance.
        assert isinstance(cls, type) and issubclass(cls, ParallelizationStrategy), (
            f"cls must be a subclass of ParallelizationStrategy, but got {type(cls)} {cls}"
        )
        assert name is not None, "name is required"
        spec = HF_PARALLEL_SPECS.get(name, ParallelSpec())
        assert spec.strategy is None, f"name {name} already registered"
        HF_PARALLEL_SPECS[name] = dataclasses.replace(spec, strategy=cls())
        return cls

    if name is None:
        raise ValueError("name is required")
    # If used with parentheses (possibly with arguments)
    return _register
