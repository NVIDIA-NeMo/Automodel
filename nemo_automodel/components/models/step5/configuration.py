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

"""Checkpoint configuration for Step-5 Preview."""

from dataclasses import dataclass
from typing import TypeAlias

import torch
from transformers import PretrainedConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p7.configuration_step3p7 import (
    Step3p7TextConfig,
    StepRoboticsVisionEncoderConfig,
)

# JSON config values plus the torch dtype accepted by PretrainedConfig.
ConfigValue: TypeAlias = bool | int | float | str | torch.dtype | None | list["ConfigValue"] | dict[str, "ConfigValue"]


class Step5SparseConfig(PretrainedConfig):
    """CSA region selection geometry and normalization semantics."""

    model_type = "step5_sparse"

    def __init__(
        self,
        *,
        enabled: bool = True,
        proxy_dim: int = 256,
        sparse_indexer_rope_dim: int = 32,
        sparse_indexer_use_rope: bool = True,
        sparse_indexer_num_heads: int = 16,
        sparse_indexer_num_k_heads: int = 1,
        sparse_indexer_q_norm_type: str = "rmsnorm",
        sparse_indexer_k_norm_type: str = "layernorm",
        sparse_indexer_csa_z_norm_type: str = "none",
        sparse_indexer_softmax_variant: str = "ssmax",
        sparse_indexer_ssmax_s_granularity: str = "q_head",
        num_provider_groups: int = 4,
        topk: int = 512,
        region_block_size: int = 8,
        compression_method: str = "csa_block_compress",
        attention_impl: str = "sparse_gqa",
        apply_to_layer_types: list[str] | None = None,
        **kwargs: ConfigValue,
    ) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.proxy_dim = proxy_dim
        self.sparse_indexer_rope_dim = sparse_indexer_rope_dim
        self.sparse_indexer_use_rope = sparse_indexer_use_rope
        self.sparse_indexer_num_heads = sparse_indexer_num_heads
        self.sparse_indexer_num_k_heads = sparse_indexer_num_k_heads
        self.sparse_indexer_q_norm_type = sparse_indexer_q_norm_type
        self.sparse_indexer_k_norm_type = sparse_indexer_k_norm_type
        self.sparse_indexer_csa_z_norm_type = sparse_indexer_csa_z_norm_type
        self.sparse_indexer_softmax_variant = sparse_indexer_softmax_variant
        self.sparse_indexer_ssmax_s_granularity = sparse_indexer_ssmax_s_granularity
        self.num_provider_groups = num_provider_groups
        self.topk = topk
        self.region_block_size = region_block_size
        self.compression_method = compression_method
        self.attention_impl = attention_impl
        self.apply_to_layer_types = ["full_attention"] if apply_to_layer_types is None else list(apply_to_layer_types)
        semantics = (
            sparse_indexer_num_k_heads,
            sparse_indexer_q_norm_type,
            sparse_indexer_k_norm_type,
            sparse_indexer_csa_z_norm_type,
            sparse_indexer_softmax_variant,
            sparse_indexer_ssmax_s_granularity,
            compression_method,
            attention_impl,
        )
        if enabled and semantics != (
            1,
            "rmsnorm",
            "layernorm",
            "none",
            "ssmax",
            "q_head",
            "csa_block_compress",
            "sparse_gqa",
        ):
            raise ValueError(f"Unsupported Step5 sparse semantics: {semantics}")
        if min(proxy_dim, sparse_indexer_num_heads, num_provider_groups, topk, region_block_size) < 1:
            raise ValueError("Step5 sparse dimensions, topk and region size must be positive")
        if sparse_indexer_num_heads % num_provider_groups:
            raise ValueError("Indexer heads must be divisible by provider groups")
        if sparse_indexer_rope_dim % 2 or not 0 <= sparse_indexer_rope_dim <= proxy_dim:
            raise ValueError("Indexer rotary dimension must be even and no larger than proxy_dim")


class Step5Config(Step3p7TextConfig):
    """Step-5 language backbone with per-layer RoPE and CSA attention."""

    model_type = "step5"
    base_config_key = "text_config"
    architectures = ["Step5ForCausalLM"]

    def __init__(
        self,
        *,
        sparse_config: Step5SparseConfig | dict[str, ConfigValue] | None = None,
        quantization_config: dict[str, ConfigValue] | None = None,
        moe_layer_list: list[int] | None = None,
        fp32_residual_connection: bool = True,
        use_moe: bool = True,
        norm_dtype: str = "float32",
        att_impl_type: str = "GQA",
        num_sliding_attention_heads: int | None = None,
        head_dim_full_attention: int | None = None,
        head_dim_sliding_attention: int | None = None,
        num_attention_groups_full_attention: int | None = None,
        num_attention_groups_sliding_attention: int | None = None,
        **kwargs: ConfigValue,
    ) -> None:
        defaults: dict[str, ConfigValue] = {
            "num_hidden_layers": 92,
            "intermediate_size": 13824,
            "num_attention_groups": 4,
            "head_dim": 192,
            "vocab_size": 128896,
            "moe_num_experts": 352,
            "moe_intermediate_size": 1536,
            "share_expert_dim": 1536,
            "moe_router_activation": "sigmoid",
            "moe_router_scaling_factor": 3.0,
            "use_moe_router_bias": True,
            "need_fp32_gate": True,
            "use_head_wise_attn_gate": True,
            "sliding_window": 512,
            "max_position_embeddings": 1048576,
            "num_nextn_predict_layers": 3,
            "tie_word_embeddings": False,
            "architectures": ["Step5ForCausalLM"],
        }
        defaults.update(kwargs)
        n = int(defaults["num_hidden_layers"])
        kinds = ["full_attention" if i % 4 == 3 else "sliding_attention" for i in range(n)]
        defaults.setdefault("layer_types", kinds)
        defaults.setdefault("rope_theta", [1e7 if k == "full_attention" else 1e4 for k in kinds])
        defaults.setdefault("partial_rotary_factors", [1 / 3 if k == "full_attention" else 1.0 for k in kinds])
        defaults.setdefault("swiglu_limits", [7.0] * n)
        defaults.setdefault("swiglu_limits_shared", [7.0] * n)
        defaults.setdefault("yarn_only_types", ["full_attention"])
        if moe_layer_list is None:
            enum = defaults.get("moe_layers_enum")
            moe_layer_list = (
                [int(x) for x in enum.split(",")]
                if isinstance(enum, str)
                else (list(enum) if isinstance(enum, (list, tuple)) else list(range(3, 91)))
            )
        defaults["moe_layers_enum"] = list(moe_layer_list)
        super().__init__(**defaults)
        self.quantization_config = quantization_config
        self.moe_layer_list = list(moe_layer_list)
        self.fp32_residual_connection = fp32_residual_connection
        self.use_moe = use_moe
        self.sparse_config = (
            sparse_config
            if isinstance(sparse_config, Step5SparseConfig)
            else Step5SparseConfig(**({} if sparse_config is None else sparse_config))
        )
        self.norm_dtype, self.att_impl_type = norm_dtype, att_impl_type
        if norm_dtype != "float32" or att_impl_type != "GQA":
            raise ValueError("Step5 Preview requires float32 normalization and GQA attention")
        geometry = {
            "num_sliding_attention_heads": (num_sliding_attention_heads, self.num_attention_heads),
            "head_dim_full_attention": (head_dim_full_attention, self.head_dim),
            "head_dim_sliding_attention": (head_dim_sliding_attention, self.head_dim),
            "num_attention_groups_full_attention": (num_attention_groups_full_attention, self.num_attention_groups),
            "num_attention_groups_sliding_attention": (
                num_attention_groups_sliding_attention,
                self.num_attention_groups,
            ),
        }
        for name, (configured, expected) in geometry.items():
            if configured is not None and configured != expected:
                raise ValueError(f"Step5 requires {name}={expected}, got {configured}")
            setattr(self, name, expected)
        if self.attention_other_setting is not None:
            for name, expected in (
                ("num_attention_heads", self.num_attention_heads),
                ("num_attention_groups", self.num_attention_groups),
                ("head_dim", self.head_dim),
                ("true_head_dim", self.head_dim),
                ("attention_type", "sliding_attention"),
            ):
                if self.attention_other_setting.get(name, expected) != expected:
                    raise ValueError(f"Unsupported Step5 attention_other_setting: {name}")
        if self.attention_dropout != 0:
            raise ValueError("Step5 Preview attention does not use dropout")
        if self.use_rope_layers is not None and not all(self.use_rope_layers):
            raise ValueError("Step5 Preview requires rotary embeddings on every layer")
        if any(kind not in ("sliding_attention", "full_attention") for kind in self.layer_types):
            raise ValueError("Unsupported Step5 layer_types")
        self.num_key_value_heads = self.num_attention_groups
        self.num_experts = self.moe_num_experts
        if self.num_attention_heads % self.num_attention_groups:
            raise ValueError("Query heads must be divisible by KV groups")
        if self.sparse_config.enabled and self.sparse_config.num_provider_groups != self.num_attention_groups:
            raise ValueError("Sparse provider groups must equal KV groups")
        if self.rope_scaling is not None and self.rope_scaling.get("rope_type", "default") not in ("llama3", "default"):
            raise ValueError("Step5 supports only llama3 rotary scaling")
        if not fp32_residual_connection:
            raise ValueError("Step5 Preview requires fp32_residual_connection=True")
        if self.moe_router_activation != "sigmoid":
            raise ValueError("Step5 Preview uses sigmoid MoE routing")


class Step5VisionConfig(StepRoboticsVisionEncoderConfig):
    """Perception configuration with explicit rotary semantics."""

    def __init__(
        self,
        *,
        rope_theta: float = 10000.0,
        rope_theta_rescale_factor: float = 1.0,
        **kwargs: ConfigValue,
    ) -> None:
        super().__init__(**kwargs)
        self.rope_theta = rope_theta
        self.rope_theta_rescale_factor = rope_theta_rescale_factor
        if self.width % self.heads or (self.width // self.heads) % 4:
            raise ValueError("Step5 vision requires a head dimension divisible by four")
        if rope_theta <= 0 or rope_theta_rescale_factor <= 0:
            raise ValueError("Step5 vision rotary scales must be positive")


class Step5VLConfig(PretrainedConfig):
    """Step-5 perception encoder and language model configuration."""

    model_type = "step5_vl"
    sub_configs = {"text_config": Step5Config, "vision_config": Step5VisionConfig}

    def __init__(
        self,
        text_config: Step5Config | dict[str, ConfigValue] | None = None,
        vision_config: Step5VisionConfig | dict[str, ConfigValue] | None = None,
        *,
        image_token_id: int = 128001,
        projector_bias: bool = False,
        understand_projector_stride: int = 2,
        **kwargs: ConfigValue,
    ) -> None:
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("architectures", ["Step5ForConditionalGeneration"])
        super().__init__(**kwargs)
        self.text_config = text_config if isinstance(text_config, Step5Config) else Step5Config(**(text_config or {}))
        self.vision_config = (
            vision_config
            if isinstance(vision_config, Step5VisionConfig)
            else (Step5VisionConfig(**(vision_config or {})))
        )
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        self.max_position_embeddings = self.text_config.max_position_embeddings
        self.image_token_id = image_token_id
        self.projector_bias = projector_bias
        self.understand_projector_stride = understand_projector_stride
        if understand_projector_stride != 2:
            raise ValueError("Step5 vision uses two stride-2 downsamplers")


@dataclass(kw_only=True)
class Step5BackendConfig(BackendConfig):
    """Model-owned execution options, disabled by default.

    Attributes:
        csa_forward_full_blocks: Skip custom masks on proven fully selected
            forward blocks for single-document, unpadded B1 CP1 inputs on SM90
            with head dimension 192. Backward block lists stay unchanged.
            Requires attn="fa4".
        csa_backward_full_blocks: Skip custom masks on proven complete backward
            blocks under the same single-document guard. May change FP32
            dK/dV accumulation order. Requires attn="fa4".
        swa_varlen: Use native FA4 window attention for valid single-batch
            SM90 D192 query prefixes. Padding keeps its custom-mask path.
            Requires enabled sparse metadata; communication is unchanged.
    """

    csa_forward_full_blocks: bool = False
    csa_backward_full_blocks: bool = False
    swa_varlen: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.csa_forward_full_blocks, bool):
            raise TypeError("csa_forward_full_blocks must be a bool")
        if self.csa_forward_full_blocks and self.attn != "fa4":
            raise ValueError("csa_forward_full_blocks requires attn=fa4")
        if not isinstance(self.csa_backward_full_blocks, bool):
            raise TypeError("csa_backward_full_blocks must be a bool")
        if self.csa_backward_full_blocks and self.attn != "fa4":
            raise ValueError("csa_backward_full_blocks requires attn=fa4")
        if not isinstance(self.swa_varlen, bool):
            raise TypeError("swa_varlen must be a bool")
