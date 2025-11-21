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
including LLaMA, Qwen, and Gemma3 models.
"""

from typing import Callable, Dict, Union, cast

import torch
from torch.distributed.tensor import DTensor
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

from nemo_automodel.components.models.llama.model import LlamaForCausalLM as CustomLlamaForCausalLM
from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM as CustomQwen2ForCausalLM


class SequenceParallelAllGatherActivation(SequenceParallel):
    """SequenceParallel that all-gathers activations for sequence parallelism."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        """Prepare outputs by redistributing sharded DTensors to replicated placement."""
        # If output is a DTensor with Shard placement, redistribute to Replicate
        if isinstance(outputs, DTensor):
            if any(isinstance(p, Shard) for p in outputs.placements):
                # Redistribute to replicated placement (performs all-gather)
                outputs = outputs.redistribute(device_mesh=device_mesh, placements=[Replicate()])
        else:
            raise ValueError(f"Expected output to be a DTensor, but got {type(outputs)}")

        # Call the parent's prepare_output_fn to handle use_local_output
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


class RotaryEmbedParallel(SequenceParallel):
    """Custom SequenceParallel class for Qwen2 / Gemma3 rotary embeddings because the input is a tuple."""

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        new_inputs = list(inputs)

        if not isinstance(inputs[0], DTensor):
            """Guard the metadata for Sequence Parallel here"""
            try:
                new_inputs[0] = DTensor.from_local(
                    local_tensor=inputs[0],
                    device_mesh=device_mesh,
                    placements=sequence_sharding,
                    run_check=True,
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to shard tensor for sequence parallelism. Local Shape is ({inputs[0].shape}) "
                    f"at rank {torch.distributed.get_rank()}. Different TP ranks must have the same shape. "
                    f"Original error: {str(e)}"
                ) from e

        if not isinstance(inputs[1], DTensor):
            new_inputs[1] = DTensor.from_local(
                local_tensor=inputs[1],
                device_mesh=device_mesh,
                placements=(Replicate(),),
                run_check=False,
            )

        return type(inputs)(new_inputs)

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return type(outputs)([o.to_local() if use_local_output else o for o in outputs])


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
        f"{model_prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
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
        f"{model_prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        f"{model_prefix}.rotary_emb": RotaryEmbedParallel(use_local_output=True),
        f"{model_prefix}.rotary_emb_local": RotaryEmbedParallel(use_local_output=True),
        f"{model_prefix}.layers.*.input_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        f"{model_prefix}.layers.*.post_attention_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.pre_feedforward_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        f"{model_prefix}.layers.*.post_feedforward_layernorm": SequenceParallel(),
        f"{model_prefix}.norm": SequenceParallel(),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _parallelize_llama(
    model: LlamaForCausalLM,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a LlamaForCausalLM model across data and tensor parallel dimensions."""
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Combined QKV projection
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate and up projection
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    base_model_sp_plan = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        "model.norm": SequenceParallel(),
        "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
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
    """Parallelizes a Qwen2ForCausalLM model across data and tensor parallel dimensions."""

    class Qwen3QKNorm(SequenceParallel):
        @staticmethod
        def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
            input_tensor = inputs[0]

            if isinstance(input_tensor, DTensor):
                assert input_tensor.placements == (Shard(dim=2),)
                return input_tensor
            elif isinstance(input_tensor, torch.Tensor):
                # assume the input passed in already sharded on the sequence dim and create the DTensor
                return DTensor.from_local(input_tensor, device_mesh, sequence_sharding, run_check=False)
            else:
                raise ValueError(f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}")

    if sequence_parallel:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.self_attn.q_norm": Qwen3QKNorm(),
            "model.layers.*.self_attn.k_norm": Qwen3QKNorm(),
            "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        }

    else:
        base_model_tp_plan = {
            "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
            "model.embed_tokens": RowwiseParallel(
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
    plan["score"] = ColwiseParallel()
    return plan


# Phi3: fused attention cannot be sharded; shard MLP as in HF guidance
def _parallelize_phi3(
    model: Phi3ForCausalLM,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": RowwiseParallel(
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


# Create the model-specific parallel plan mapping
PARALLELIZE_FUNCTIONS: Dict[type, Callable[..., Dict[str, ParallelStyle]]] = {
    Qwen2ForCausalLM: _parallelize_qwen,
    Qwen3ForCausalLM: _parallelize_qwen,
    Qwen3ForSequenceClassification: _parallelize_qwen_classification,
    LlamaForCausalLM: _parallelize_llama,
    # gemma-3-1b-it uses Gemma3ForCausalLM since it is a text-only model
    Gemma3ForCausalLM: _parallelize_gemma3,
    # The larger gemma models use Gemma3ForConditionalGeneration, which are for text-image input
    Gemma3ForConditionalGeneration: _parallelize_gemma3,
    Phi3ForCausalLM: _parallelize_phi3,
    CustomLlamaForCausalLM: _parallelize_llama,
    CustomQwen2ForCausalLM: _parallelize_qwen,
}
