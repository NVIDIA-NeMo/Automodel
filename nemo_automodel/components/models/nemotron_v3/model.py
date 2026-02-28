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

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import (
    BackendConfig,
    HFCheckpointingMixin,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class NemotronV3Model(nn.Module):
    """NemotronV3 base model (without LM head).

    This is a hybrid architecture with Mamba2, Attention, MLP, and MoE layers.
    """

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        *,
        moe_config: MoEConfig | None = None,
    ):
        """Initialize NemotronV3Model.

        Args:
            config: NemotronH config with model parameters
            backend: Backend configuration for MoE and other components
            moe_config: MoE configuration (optional, will create default if None)
        """
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.moe_config = moe_config or MoEConfig(
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=1,  # NemotronV3 has 1 shared expert
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=False,  # Router weights are trained but not using bias updates
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,  # No aux loss for NemotronV3
            score_func="sigmoid",  # NemotronV3 uses sigmoid scoring
            route_scale=config.routed_scaling_factor,
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,  # For shared expert
            moe_inter_dim=config.moe_intermediate_size,  # For routed experts
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=config.mlp_bias,
            expert_activation="relu2",  # NemotronV3 uses ReLU² activation
            dtype=config.torch_dtype,
            shared_expert_gate=False,
            shared_expert_inter_dim=config.moe_shared_expert_intermediate_size,
            shared_expert_activation="relu2",  # Use ReLU² for shared experts
            force_e_score_correction_bias=True,  # NemotronV3 checkpoint has this buffer
        )

        # Embeddings
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        # Transformer layers (hybrid: mamba, attention, mlp, moe)
        self.layers = nn.ModuleDict()
        for idx in range(config.num_hidden_layers):
            self.layers[str(idx)] = NemotronV3Block(
                config, layer_idx=idx, moe_config=self.moe_config, backend=self.backend
            )

        # Final norm
        self.norm = initialize_rms_norm_module(
            self.backend.rms_norm,
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        causal_mask_mapping: dict[str, torch.Tensor] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len] (optional)
            attention_mask: 2D padding mask [batch_size, seq_len] (1=real, 0=padding)
            causal_mask_mapping: Dict with precomputed 4D causal masks for attention layers
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size] (optional)
            past_key_values: Optional NemotronHybridCache for incremental decoding.
            cache_position: Token position indices for cache updates.
            **kwargs: Additional arguments (ignored)

        Returns:
            Hidden states tensor [batch_size, seq_len, hidden_size]
        """
        # Get embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids must be provided if inputs_embeds is not provided")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # TODO: attention mask currently does not work. A default causal mask is applied.

        # Get 4D causal mask for attention layers (from precomputed masks)
        causal_mask = causal_mask_mapping.get("full_attention") if causal_mask_mapping is not None else None

        # Apply transformer layers
        for layer in self.layers.values():
            # Pass appropriate mask based on layer type
            if layer.block_type == "attention":
                # Attention layers use 4D causal mask
                mask = causal_mask
            elif layer.block_type == "mamba":
                # Mamba layers use 2D padding mask during prefill, None during decode
                mask = None if (past_key_values is not None and past_key_values.has_previous_state) else attention_mask
            else:
                # MLP/MoE layers don't use mask
                mask = None

            hidden_states = layer(
                hidden_states,
                attention_mask=mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights according to NemotronV3 spec.

        Args:
            buffer_device: Device to use for buffer initialization
        """
        # Embedding weights: normal initialization
        with buffer_device:
            nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.initializer_range)
            self.norm.reset_parameters()

        # Initialize all layers via delegation
        for block in self.layers.values():
            block.init_weights(buffer_device=buffer_device)


class NemotronHForCausalLM(HFCheckpointingMixin, GenerationMixin, nn.Module, MoEFSDPSyncMixin):
    """NemotronV3 model with language modeling head.

    Supports ``.generate()`` from ``transformers.generation.GenerationMixin`` with O(1)
    per-step KV caching for attention layers and recurrent state caching for Mamba2 layers.
    """

    # Prevent GenerationMixin from creating a DynamicCache: the hybrid Mamba2/Attention
    # architecture uses its own NemotronHybridCache.
    _is_stateful: bool = True
    main_input_name: str = "input_ids"

    @classmethod
    def from_config(
        cls,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Create model from config.

        Args:
            config: NemotronH config
            backend: Backend configuration
            **kwargs: Additional arguments

        Returns:
            NemotronHForCausalLM instance
        """
        return cls(config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """Load pretrained model.

        Args:
            pretrained_model_name_or_path: Path or name of pretrained model
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            NemotronHForCausalLM instance
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Initialize NemotronV3ForCausalLM.

        Args:
            config: NemotronH config
            backend: Backend configuration
            **kwargs: Additional arguments
        """
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        # Base model
        self.model = NemotronV3Model(config, backend=self.backend)

        # LM head
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )

        # Create state_dict_adapter if enabled (needed to convert HF checkpoints)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = NemotronV3StateDictAdapter(
                config=config,
                moe_config=self.model.moe_config,
                backend=self.backend,
                dtype=dtype,
            )

        # Required by GenerationMixin.generate()
        self.generation_config = GenerationConfig()

    @property
    def device(self) -> torch.device:
        """Return the device of the first model parameter (required by GenerationMixin)."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the first model parameter (used by cache construction)."""
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask_mapping: Optional[dict[str, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len] (optional)
            attention_mask: 2D padding mask [batch_size, seq_len]
            causal_mask_mapping: Dict with precomputed 4D causal masks
            inputs_embeds: Pre-computed input embeddings (optional)
            labels: Token IDs for loss computation [batch_size, seq_len] (optional)
            past_key_values: Optional NemotronHybridCache for incremental decoding.
            use_cache: Whether to return past_key_values for subsequent steps.
            cache_position: Token position indices for cache updates.
            position_ids: Unused – accepted for API compatibility with GenerationMixin.
            logits_to_keep: If > 0, only compute logits for the last ``logits_to_keep``
                token positions (avoids materialising the full logit matrix during generation).
            output_hidden_states: Whether to return hidden states
            return_dict: Accepted for API compatibility (always returns CausalLMOutputWithPast)
            **kwargs: Additional arguments forwarded to the base model.

        Returns:
            :class:`~transformers.modeling_outputs.CausalLMOutputWithPast` with
            ``logits`` (float32, ``[batch_size, seq_len, vocab_size]``), optional
            ``loss``, ``past_key_values``, and ``hidden_states``.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        # Forward through base model
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            causal_mask_mapping=causal_mask_mapping,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        # Mark cache as having state after the first forward pass (prefill done)
        if past_key_values is not None:
            past_key_values.has_previous_state = True

        # Optionally restrict logit computation to the last few positions.
        # When logits_to_keep == 0 we compute all positions (training default).
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=None,
        )

    @staticmethod
    def _make_causal_mask(
        query_len: int,
        kv_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a 4D SDPA-compatible causal mask.

        Prefill (query_len == kv_len): standard lower-triangular causal mask.
        Decode (query_len == 1): all-zeros row allowing attention to all cached positions.
        """
        if query_len == 1:
            # Decode: attend to all positions
            return torch.zeros(batch_size, 1, 1, kv_len, dtype=dtype, device=device)
        # Prefill: lower-triangular causal mask
        mask = torch.zeros(batch_size, 1, query_len, kv_len, dtype=dtype, device=device)
        mask.masked_fill_(
            torch.triu(torch.ones(query_len, kv_len, device=device), diagonal=1).bool(),
            float("-inf"),
        )
        return mask

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> dict:
        """Prepare model inputs for each generation step.

        On the first call (prefill), creates a :class:`NemotronHybridCache` and
        forwards the full prompt.  On subsequent calls (decode), only the newly
        generated token is forwarded.

        Args:
            input_ids: Accumulated token ids [batch_size, current_seq_len].
            attention_mask: Padding mask [batch_size, current_seq_len].
            inputs_embeds: Pre-computed embeddings for the first step (optional).
            past_key_values: NemotronHybridCache from the previous step (None on first call).
            cache_position: Token position indices.
            use_cache: Whether to use caching (default True).
            **kwargs: Remaining model kwargs.

        Returns:
            Dict of keyword arguments to pass to :meth:`forward`.
        """
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache

        batch_size = input_ids.shape[0]

        # Create cache on first call
        if past_key_values is None:
            past_key_values = NemotronHybridCache(self.config, batch_size, self.dtype, self.device)
            # First call: cache_position covers the full prompt
            if cache_position is None:
                cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

        # After prefill, send only the new token
        if past_key_values.has_previous_state:
            input_ids = input_ids[:, -1:]
            if cache_position is None:
                kv_len = past_key_values.get_seq_length()
                cache_position = torch.tensor([kv_len], device=input_ids.device)

        # On the first step, prefer inputs_embeds when available
        if inputs_embeds is not None and not past_key_values.has_previous_state:
            model_inputs = {"input_ids": None, "inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        # Build causal mask for attention layers
        query_len = (
            input_ids.shape[1] if model_inputs["inputs_embeds"] is None else model_inputs["inputs_embeds"].shape[1]
        )
        kv_len = past_key_values.get_seq_length() + query_len
        causal_mask = self._make_causal_mask(query_len, kv_len, batch_size, self.dtype, self.device)

        model_inputs["causal_mask_mapping"] = {"full_attention": causal_mask}
        model_inputs["past_key_values"] = past_key_values
        model_inputs["cache_position"] = cache_position
        model_inputs["use_cache"] = use_cache
        model_inputs["attention_mask"] = attention_mask
        return model_inputs

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize model weights.

        Args:
            buffer_device: Device to use for buffer initialization
            dtype: Target dtype for model weights
        """
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.initialize_weights(buffer_device=buffer_device)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)

        self.to(dtype)


ModelClass = NemotronHForCausalLM
