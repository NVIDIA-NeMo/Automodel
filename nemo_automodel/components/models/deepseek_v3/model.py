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

from typing import Any

import torch
import torch.nn as nn
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.deepseek_v3.layers import MLA
from nemo_automodel.components.models.deepseek_v3.rope_utils import freqs_cis_from_position_ids, precompute_freqs_cis
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Block(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        config: DeepseekV3Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        self.self_attn = MLA(config, backend)
        if layer_idx < config.first_k_dense_replace:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear)
        else:
            self.mlp = MoE(moe_config, backend)
        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            padding_mask (torch.Tensor): Boolean tensor indicating padding positions.

        Returns:
            torch.Tensor: Output tensor after block computation.
            torch.Tensor | None: Auxiliary loss for load balancing (if applicable).
        """
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        
        # DEBUG: Log attention output for layer 1 (MoE layer)
        if self.layer_idx == 1 and not hasattr(self, '_attn_debug_logged'):
            self._attn_debug_logged = True
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                h = attn_out.float()
                print(f"[NEMO_LAYER1_DEBUG] === Layer 1 Attention Debug ===", flush=True)
                print(f"[NEMO_LAYER1_DEBUG] attn_out.shape={attn_out.shape}", flush=True)
                print(f"[NEMO_LAYER1_DEBUG] attn_out[0,0,:32]={h[0,0,:32].tolist()}", flush=True)
                print(f"[NEMO_LAYER1_DEBUG] attn_out stats: mean={h.mean().item():.8f}, std={h.std().item():.8f}, min={h.min().item():.6f}, max={h.max().item():.6f}", flush=True)
                print(f"[NEMO_LAYER1_DEBUG] attn_out[0,0] full vector stats: mean={h[0,0].mean().item():.8f}, std={h[0,0].std().item():.8f}", flush=True)
        
        x = x + attn_out
        
        # # DEBUG: Log post-attention hidden states for layer 1
        # if self.layer_idx == 1 and not hasattr(self, '_post_attn_debug_logged'):
        #     self._post_attn_debug_logged = True
        #     import torch.distributed as dist
        #     rank = dist.get_rank() if dist.is_initialized() else 0
        #     if rank == 0:
        #         h = x.float()
        #         print(f"[NEMO_LAYER1_DEBUG] x (after residual).shape={x.shape}", flush=True)
        #         print(f"[NEMO_LAYER1_DEBUG] x (after residual)[0,0,:32]={h[0,0,:32].tolist()}", flush=True)
        #         print(f"[NEMO_LAYER1_DEBUG] x (after residual) stats: mean={h.mean().item():.8f}, std={h.std().item():.8f}", flush=True)
        
        mlp_input = self.post_attention_layernorm(x)
        
        # # DEBUG: Log MoE input for layer 1
        # if self.layer_idx == 1 and not hasattr(self, '_mlp_input_debug_logged'):
        #     self._mlp_input_debug_logged = True
        #     import torch.distributed as dist
        #     rank = dist.get_rank() if dist.is_initialized() else 0
        #     if rank == 0:
        #         h = mlp_input.float()
        #         print(f"[NEMO_LAYER1_DEBUG] mlp_input (post_attn_ln).shape={mlp_input.shape}", flush=True)
        #         print(f"[NEMO_LAYER1_DEBUG] mlp_input (post_attn_ln)[0,0,:32]={h[0,0,:32].tolist()}", flush=True)
        #         print(f"[NEMO_LAYER1_DEBUG] mlp_input stats: mean={h.mean().item():.8f}, std={h.std().item():.8f}, min={h.min().item():.6f}, max={h.max().item():.6f}", flush=True)
        #         print(f"[NEMO_LAYER1_DEBUG] mlp_input[0,0] full vector stats: mean={h[0,0].mean().item():.8f}, std={h[0,0].std().item():.8f}", flush=True)

        mlp_out = self._mlp(
            x=mlp_input,
            padding_mask=padding_mask,
        )
        x = x + mlp_out

        return x

    def _mlp(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.mlp, MLP):
            return self.mlp(x)
        else:
            assert isinstance(self.mlp, MoE)
            return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class DeepseekV3Model(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
    ):
        super().__init__()
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
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)

        self.max_seq_len = config.max_position_embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.qk_rope_head_dim,
                self.max_seq_len,
                config.rope_theta,
                config.rope_scaling,
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)


        with torch.no_grad():
            freqs_cis = freqs_cis_from_position_ids(
                position_ids,
                self.freqs_cis,
                qkv_format=attn_kwargs.get("qkv_format", "bshd"),
                for_fused_rope=self.backend.rope_fusion,
                cp_size=attn_kwargs.get("cp_size", 1),
            )

        h = inputs_embeds

        # # ===== DEBUG: Per-layer logging =====
        # import os
        # import time
        # _DEBUG_ACTIVATIONS = os.environ.get("DEBUG_ACTIVATIONS", "0") == "1"
        # _DEBUG_DIR = "/lustre/fsw/portfolios/coreai/users/huiyingl/kimi/Automodel/debug"
        # _rank = 0
        # _should_log = False
        # _is_shape_inference = False
        # if _DEBUG_ACTIVATIONS:
        #     import torch.distributed as dist
        #     _rank = dist.get_rank() if dist.is_initialized() else 0
        #     _layer_keys = list(self.layers.keys())
        #     # Always log what this rank has
        #     print(f"[FINETUNE][rank={_rank}] DeepseekV3Model.forward() called, layers on this rank: {_layer_keys}", flush=True)
        #     # Detect shape inference: input is all zeros
        #     _emb_max = inputs_embeds.max().item()
        #     _emb_min = inputs_embeds.min().item()
        #     if _emb_max == 0 and _emb_min == 0:
        #         _is_shape_inference = True
        #         print(f"[FINETUNE][rank={_rank}] Skipping LLM layer debug - shape inference (all-zero inputs_embeds)", flush=True)
        #     else:
        #         print(f"[FINETUNE][rank={_rank}] inputs_embeds stats: min={_emb_min:.6f}, max={_emb_max:.6f} (REAL DATA)", flush=True)
        #     # Log on all ranks that have layers (PP distributes layers across ranks)
        #     _should_log = len(self.layers) > 0 and not _is_shape_inference
        # # ===== END DEBUG SETUP =====

        # Apply the transformer layers.
        for layer_key, layer in self.layers.items():
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )
        #     # ===== DEBUG: Log layer output =====
        #     if _DEBUG_ACTIVATIONS and _should_log:
        #         h_float = h.detach().float()
        #         print(f"[FINETUNE][rank={_rank}] After LLM layer {layer_key}: shape={tuple(h.shape)}, dtype={h.dtype}, "
        #               f"mean={h_float.mean().item():.6f}, min={h_float.min().item():.6f}, "
        #               f"max={h_float.max().item():.6f}", flush=True)
        #         # Save with rank to avoid overwrites in PP
        #         _path = os.path.join(_DEBUG_DIR, f"finetune_hidden_states_layer{layer_key}_rank{_rank}.pt")
        #         if not os.path.exists(_path):
        #             torch.save(h.detach().cpu(), _path)
        #             print(f"[FINETUNE][rank={_rank}] Saved hidden_states_layer{layer_key} to: {_path}", flush=True)
        #     # ===== END DEBUG =====

        # # ===== DEBUG: Log completion (no exit - PP schedule must complete) =====
        # if _DEBUG_ACTIVATIONS and _should_log:
        #     print(f"[FINETUNE][rank={_rank}] LLM layers on this rank complete. Files saved to {_DEBUG_DIR}/", flush=True)
        # # ===== END DEBUG =====

        h = self.norm(h) if self.norm else h
        return h

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for _, block in self.layers.named_children():
                if isinstance(block.mlp, MoE):
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            self.freqs_cis = precompute_freqs_cis(
                self.config.qk_rope_head_dim,
                self.max_seq_len,
                self.config.rope_theta,
                self.config.rope_scaling,
            )
            self.freqs_cis = self.freqs_cis.to(buffer_device)
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class DeepseekV3ForCausalLM(nn.Module, MoEFSDPSyncMixin):
    @classmethod
    def from_config(
        cls,
        config: DeepseekV3Config,
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
        config = DeepseekV3Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: DeepseekV3Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = DeepseekV3Model(config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = DeepSeekV3StateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        logits = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        logits = self.lm_head(logits) if self.lm_head else logits
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            logits = logits.unsqueeze(0)
        return logits

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for _, block in self.model.layers.named_children():
                if isinstance(block.mlp, MoE):
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )

        self.to(dtype)
        with buffer_device:
            self.model.freqs_cis = precompute_freqs_cis(
                self.config.qk_rope_head_dim,
                self.model.max_seq_len,
                self.config.rope_theta,
                self.config.rope_scaling,
            )


ModelClass = DeepseekV3ForCausalLM
