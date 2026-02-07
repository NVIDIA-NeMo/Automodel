# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""DeepSeek V3.2 Model.

Contains DeepseekV32Block, DeepseekV32Model, and DeepseekV32ForCausalLM.
These classes subclass from DeepSeek V3, with the main difference being
the use of DeepseekV32MLA (with Indexer) instead of the standard MLA.
"""

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.deepseek_v3.model import (
    Block,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
)
from nemo_automodel.components.models.deepseek_v3.rope_utils import precompute_freqs_cis
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.layers import DeepseekV32MLA
from nemo_automodel.components.models.deepseek_v32.state_dict_adapter import DeepSeekV32StateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class DeepseekV32Block(Block):
    """Transformer block for DeepSeek V3.2.

    Subclasses V3 Block, using DeepseekV32MLA (with Indexer) instead of the standard MLA.
    """

    def __init__(
        self,
        layer_idx: int,
        config: DeepseekV32Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        # Call grandparent __init__ to skip Block's __init__ which creates MLA
        nn.Module.__init__(self)

        # Use V3.2 MLA with Indexer
        self.self_attn = DeepseekV32MLA(config, backend)

        # Import here to avoid circular imports
        from nemo_automodel.components.models.common import initialize_rms_norm_module
        from nemo_automodel.components.moe.layers import MLP, MoE

        if layer_idx < config.first_k_dense_replace:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear)
        else:
            self.mlp = MoE(moe_config, backend)

        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx


class DeepseekV32Model(DeepseekV3Model):
    """DeepSeek V3.2 Model.

    Subclasses V3 Model, using DeepseekV32Block instead of Block.
    """

    def __init__(
        self,
        config: DeepseekV32Config,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
    ):
        # Call grandparent __init__ to skip DeepseekV3Model's __init__
        nn.Module.__init__(self)

        self.backend = backend
        self.config = config
        self.moe_config = moe_config or MoEConfig(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=0.001,
            score_func="sigmoid",
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0,
            norm_topk_prob=config.norm_topk_prob,
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
        )
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            # Use V3.2 Block instead of V3 Block
            self.layers[str(layer_id)] = DeepseekV32Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)

        self.max_seq_len = config.max_position_embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.qk_rope_head_dim,
                self.max_seq_len,
                config.rope_parameters["rope_theta"] if hasattr(config, "rope_parameters") else config.rope_theta,
                config.rope_parameters if hasattr(config, "rope_parameters") else config.rope_scaling,
            ),
            persistent=False,
        )


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    """DeepSeek V3.2 for Causal Language Modeling.

    Subclasses V3 ForCausalLM, using DeepseekV32Model and DeepSeekV32StateDictAdapter.
    """

    @classmethod
    def from_config(
        cls,
        config: DeepseekV32Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = DeepseekV32Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: DeepseekV32Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        # Call grandparent __init__ to skip DeepseekV3ForCausalLM's __init__
        nn.Module.__init__(self)

        from nemo_automodel.components.models.common import initialize_linear_module

        self.config = config
        self.backend = backend or BackendConfig()
        # Use V3.2 Model instead of V3 Model
        self.model = DeepseekV32Model(config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if self.backend.enable_hf_state_dict_adapter:
            # Use V3.2 adapter instead of V3 adapter
            self.state_dict_adapter = DeepSeekV32StateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
            )


ModelClass = DeepseekV32ForCausalLM
