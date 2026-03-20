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

"""Model-specific parallel plans for tensor parallelism.

This module contains optimized tensor parallel plans for different model architectures
including LLaMA, Qwen, Gemma3, and Ministral3 models.
"""

from typing import Callable, Dict, Union, cast

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

# Import model classes for type checking and parallel plan mapping
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3ForSequenceClassification

from nemo_automodel.components.distributed.parallel_styles import (
    FusedColwiseParallel,  # noqa: F401
    RotaryEmbedParallel,
    SequenceParallelAllGatherActivation,
    VocabParallelEmbedding,
)
from nemo_automodel.components.models.llama.model import LlamaForCausalLM as CustomLlamaForCausalLM
from nemo_automodel.components.models.mistral3.model import Ministral3ForCausalLM
from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM as CustomQwen2ForCausalLM


def _parallelize_gemma3(
    model: Union[Gemma3ForCausalLM, Gemma3ForConditionalGeneration],
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a Gemma3ForCausalLM model across data and tensor parallel dimensions."""
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


def get_llama_nemotron_super_tp_plan(
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Return the tensor parallel plan for Llama / Llama-3.3-Nemotron Super.

    Same topology as Llama-3.3-Nemotron (e.g. nvidia/Llama-3_3-Nemotron-Super-49B-v1_5):
    fused QKV, fused gate+up, VocabParallelEmbedding, Row/ColwiseParallel for attention and MLP.

    Use this plan explicitly by passing it as tp_shard_plan (dict) or by name
    ``llama_nemotron_super_tp_plan`` when calling fsdp2_strategy_parallelize / _get_parallel_plan.
    """
    return _parallelize_llama(None, sequence_parallel)  # type: ignore[arg-type]


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


def _parallelize_llama(
    model: LlamaForCausalLM | None,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a LlamaForCausalLM model across data and tensor parallel dimensions."""
    # Compute per-section sizes for the fused QKV projection (GQA-aware).
    # For GQA, Q has more heads than K/V so the sections are unequal;
    # FusedColwiseParallel shards each section independently.
    # model may be None when called from get_llama_nemotron_super_tp_plan.
    if model is not None and hasattr(model, "config"):
        head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
        q_size = model.config.num_attention_heads * head_dim
        kv_size = model.config.num_key_value_heads * head_dim
    else:
        head_dim = q_size = kv_size = None

    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": FusedColwiseParallel(
            section_sizes=(q_size, kv_size, kv_size),
        )
        if q_size is not None
        else FusedColwiseParallel(num_sections=3),
        "model.layers.*.mlp.gate_up_proj": FusedColwiseParallel(num_sections=2),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    base_model_sp_plan = {
        "model.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "model.norm": SequenceParallel(),
        "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _parallelize_ministral3(
    model: Ministral3ForCausalLM,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a Ministral3ForCausalLM model across data and tensor parallel dimensions."""
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    base_model_sp_plan = {
        "model.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "model.norm": SequenceParallel(),
        "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _parallelize_qwen(
    model: Union[Qwen2ForCausalLM, Qwen3ForCausalLM],
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a Qwen2/Qwen3 causal LM across data and tensor parallel dimensions."""

    if sequence_parallel:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
            "model.embed_tokens": VocabParallelEmbedding(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                # Keep DTensor outputs so HF modeling code (e.g. cache_position) can
                # observe the *global* sequence length via DTensor.shape.
                use_local_output=False,
            ),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            # Rowwise projections reduce-scatter back to sequence-sharded activations.
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            # NOTE: Qwen3 has `q_norm`/`k_norm` inside attention. These operate on the
            # head-sharded outputs of q_proj/k_proj. Do NOT wrap them with SequenceParallel,
            # which would incorrectly tag head-sharded activations as sequence-sharded.
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        }

    else:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
            "model.embed_tokens": VocabParallelEmbedding(
                input_layouts=Replicate(),
            ),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
        }

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _parallelize_qwen_classification(
    model: Union[Qwen3ForSequenceClassification],
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    plan = _parallelize_qwen(model, sequence_parallel)
    assert not hasattr(model, "lm_head"), "Expected model not to have lm_head"
    del plan["lm_head"]
    assert hasattr(model, "score"), "Expected model to have score"
    # `Qwen3ForSequenceClassification` pools over the *sequence* dimension in Python.
    # Ensure the classifier logits are replicated (full num_labels) for correct pooling/loss.
    plan["score"] = ColwiseParallel(output_layouts=Replicate())
    return plan


# Phi3: fused attention cannot be sharded; shard MLP as in HF guidance
def _parallelize_phi3(
    model: Phi3ForCausalLM,
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


# Named TP plan for use with tp_shard_plan="llama_nemotron_super_tp_plan" in parallelizer
LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME = "llama_nemotron_super_tp_plan"


def _get_class_qualname(cls: type) -> str:
    """Return the fully qualified name of a class as ``module.qualname``.

    Used as a stable dict key for PARALLELIZE_FUNCTIONS instead of the class
    object itself.

    When NeMo-RL uses automodel, ``force_hf=True`` is auto-set for models
    (e.g. ``LlamaForCausalLM``) whose adapter does not implement
    ``convert_single_tensor_to_hf``. This causes ``_get_mixin_wrapped_class``
    in ``model_init.py`` to create a new class via ``type(...)`` that wraps
    the original with ``HFCheckpointingMixin``. The wrapper copies
    ``__module__`` and ``__qualname__`` from the original but is a **different
    Python object**, so ``type(model) in PARALLELIZE_FUNCTIONS`` (identity
    check) returns ``False`` and the default plan is used instead of the
    optimized one.

    String comparison on ``module.qualname`` survives this wrapping and
    correctly identifies the model class.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


# Keyed by qualified class name — see _get_class_qualname for why.
PARALLELIZE_FUNCTIONS: Dict[str, Callable[..., Dict[str, ParallelStyle]]] = {
    _get_class_qualname(Qwen2ForCausalLM): _parallelize_qwen,
    _get_class_qualname(Qwen3ForCausalLM): _parallelize_qwen,
    _get_class_qualname(Qwen3ForSequenceClassification): _parallelize_qwen_classification,
    _get_class_qualname(LlamaForCausalLM): _parallelize_llama,
    _get_class_qualname(Ministral3ForCausalLM): _parallelize_ministral3,
    # gemma-3-1b-it uses Gemma3ForCausalLM since it is a text-only model
    _get_class_qualname(Gemma3ForCausalLM): _parallelize_gemma3,
    # The larger gemma models use Gemma3ForConditionalGeneration, which are for text-image input
    _get_class_qualname(Gemma3ForConditionalGeneration): _parallelize_gemma3,
    _get_class_qualname(Phi3ForCausalLM): _parallelize_phi3,
    _get_class_qualname(CustomLlamaForCausalLM): _parallelize_llama,
    _get_class_qualname(CustomQwen2ForCausalLM): _parallelize_qwen,
}
