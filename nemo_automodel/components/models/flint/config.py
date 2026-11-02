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

"""Local configuration for Flint's hybrid-attention MoE architecture."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters

if TYPE_CHECKING:
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.flint.model import FlintForCausalLM


@strict
class FlintConfig(PreTrainedConfig):
    """Architecture settings matching the source Flint checkpoint.

    The local class removes dependence on checkpoint remote code. Full layers
    have no RoPE; sliding layers rotate all head channels with NeoX RoPE. The
    window counts the current token. ``router_aux_loss_coef`` is preserved as
    checkpoint metadata; the native training implementation requires an explicit
    zero value until the original auxiliary objective is specified.

    ``attention_compute_dtype='fp16'`` is an explicit TE-only QKV/kernel-output
    boundary within otherwise FP32 projection compute. It does not change the
    bulk parameter dtype or enable autocast. Small-gradient training requires an
    appropriate loss-scaling policy; the default None retains native precision.
    """

    model_type = "flint"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 150144
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 49
    schedule_num_hidden_layers: int | None = None
    residual_dtype: str | None = None
    attention_compute_dtype: str | None = None
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = False
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict[str, object] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    use_sliding_window: bool = True
    sliding_window: int | None = 513
    layer_types: list[str] | None = None
    no_rope_layer_indices: list[int] | None = None
    softmax_type: str = "learnable"
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    num_experts_per_tok: int = 6
    num_experts: int = 256
    norm_topk_prob: bool = True
    router_score_function: str = "sigmoid"
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.0
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def validate_layer_type(self) -> None:
        """Validate the preserved source schedule independently of active prefix depth."""
        depth = self.num_hidden_layers if self.schedule_num_hidden_layers is None else self.schedule_num_hidden_layers
        if self.layer_types is not None and (
            len(self.layer_types) != depth
            or any(kind not in ("full_attention", "sliding_attention") for kind in self.layer_types)
        ):
            raise ValueError("Flint layer_types must contain one valid entry per source schedule layer")

    def __post_init__(self, **kwargs: object) -> None:
        """Resolve schedules and upstream RoPE fields, then validate the contract."""
        if self.schedule_num_hidden_layers is None:
            self.schedule_num_hidden_layers = self.num_hidden_layers
        schedule_depth = self.schedule_num_hidden_layers
        if not 0 < self.num_hidden_layers <= schedule_depth:
            raise ValueError("Flint active depth must be positive and no greater than schedule_num_hidden_layers")
        if self.residual_dtype not in (None, "fp32", "bf16"):
            raise ValueError("Flint residual_dtype must be None, 'fp32', or 'bf16'")
        if self.attention_compute_dtype not in (None, "fp16"):
            raise ValueError("Flint attention_compute_dtype must be None or explicit 'fp16' for TE")
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i < 2 or (i - 1) % 4 == 0 or i == schedule_depth - 1 else "sliding_attention"
                for i in range(schedule_depth)
            ]
        full_layers = [i for i, kind in enumerate(self.layer_types) if kind == "full_attention"]
        if self.no_rope_layer_indices is None:
            self.no_rope_layer_indices = full_layers
        if self.mlp_only_layers is None:
            self.mlp_only_layers = list(range(min(2, schedule_depth)))
        if self.architectures is None:
            self.architectures = ["FlintForCausalLM"]
        if self.rope_parameters is None:
            kwargs.setdefault("rope_theta", 10000.0)
        super().__post_init__(**kwargs)
        positive = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.num_experts,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Flint dimensions and layer/expert counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("Flint requires Q heads divisible by KV heads and an even explicit head_dim")
        if len(self.layer_types) != schedule_depth or any(
            kind not in ("full_attention", "sliding_attention") for kind in self.layer_types
        ):
            raise ValueError(
                "Flint layer_types must name one full_attention/sliding_attention entry per source schedule layer"
            )
        if sorted(self.no_rope_layer_indices) != full_layers:
            raise ValueError("Flint no_rope_layer_indices must identify exactly the full-attention layers")
        if len(set(self.mlp_only_layers)) != len(self.mlp_only_layers) or any(
            i < 0 or i >= schedule_depth for i in self.mlp_only_layers
        ):
            raise ValueError("Flint mlp_only_layers must contain unique valid layer indices")
        if "sliding_attention" in self.layer_types and (
            not self.use_sliding_window or self.sliding_window is None or self.sliding_window <= 0
        ):
            raise ValueError("Flint sliding layers require a positive sliding_window including the current token")
        active_dense_layers = sum(index < self.num_hidden_layers for index in self.mlp_only_layers)
        if active_dense_layers < self.num_hidden_layers and not 2 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("Flint native MoE currently requires 2 <= num_experts_per_tok <= num_experts")
        if self.decoder_sparse_step != 1 or self.hidden_act != "silu":
            raise ValueError("Flint supports decoder_sparse_step=1 and hidden_act='silu'")
        if self.attention_bias or self.attention_dropout != 0 or self.softmax_type != "learnable":
            raise ValueError("Flint requires bias-free attention, zero attention dropout and learnable softmax sinks")
        if self.router_score_function != "sigmoid" or not self.norm_topk_prob or self.output_router_logits:
            raise ValueError("Flint requires normalized sigmoid routing without output_router_logits")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("Flint rms_norm_eps must be positive and finite")
        if not math.isfinite(self.initializer_range) or self.initializer_range <= 0:
            raise ValueError("Flint initializer_range must be positive and finite")
        if self.rope_parameters is None or self.rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("Flint supports only default NeoX RoPE")
        theta = self.rope_parameters["rope_theta"]
        if not isinstance(theta, (float, int)) or not math.isfinite(theta) or theta <= 0:
            raise ValueError("Flint rope_theta must be positive and finite")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("Flint pad_token_id must be within the vocabulary")

    def build(self, *, backend: BackendConfig | None = None) -> FlintForCausalLM:
        """Construct a native model without changing these declarative settings.

        Args:
            backend: Runtime kernel choices; defaults to the verified eager/torch path.

        Returns:
            An initialized native causal language model.
        """
        from nemo_automodel.components.models.flint.model import FlintForCausalLM

        return FlintForCausalLM(self, backend=backend)
