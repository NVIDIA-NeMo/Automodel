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

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoConfig, AutoModel, GenerationMixin, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.donut.modeling_donut_swin import DonutSwinModelOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from transformers.models.mbart.modeling_mbart import (
    BaseModelOutputWithPastAndCrossAttentions,
    MBartConfig,
    MBartDecoderLayer,
    MBartPreTrainedModel,
    MBartScaledWordEmbedding,
)

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

# -----------------------------------------------------------------------------
# NemotronParse configuration
# -----------------------------------------------------------------------------


class NemotronParseTextConfig(PretrainedConfig):
    """Configuration class for NemotronParse text decoder (mBART-based)."""

    model_type = "nemotron_parse_text"

    def __init__(
        self,
        vocab_size: int = 250027,
        d_model: int = 1024,
        encoder_layers: int = 12,
        decoder_layers: int = 12,
        encoder_attention_heads: int = 16,
        decoder_attention_heads: int = 16,
        decoder_ffn_dim: int = 4096,
        encoder_ffn_dim: int = 4096,
        activation_function: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        init_std: float = 0.02,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        scale_embedding: bool = False,
        use_cache: bool = True,
        num_labels: int = 3,
        forced_eos_token_id: int = 2,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        decoder_start_token_id: int = 2,
        add_cross_attention: bool = True,
        is_decoder: bool = True,
        max_sequence_length: int = 9000,
        **kwargs,
    ):
        # Populate special token ids on the config so downstream components
        # (e.g., mBART decoder layers) can rely on them.
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            add_cross_attention=add_cross_attention,
            is_decoder=is_decoder,
            use_cache=use_cache,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.classifier_dropout = classifier_dropout
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache
        self.num_labels = num_labels
        self.add_cross_attention = add_cross_attention
        self.is_decoder = is_decoder
        self.hidden_size = self.d_model
        self.num_attention_heads = self.encoder_attention_heads
        self.max_sequence_length = max_sequence_length


class NemotronParseEncoderConfig(PretrainedConfig):
    """Configuration class for NemotronParse vision encoder (RADIO-based)."""

    model_type = "nemotron_parse_encoder"

    def __init__(
        self,
        patch_size: int = 16,
        max_resolution: int = 2048,
        preferred_resolution: List[int] = None,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_resolution = max_resolution
        self.preferred_resolution = preferred_resolution or [768, 768]
        self.torch_dtype = torch_dtype
        # Store any additional args from the original RADIO config
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class NemotronParseConfig(PretrainedConfig):
    """Configuration class for NemotronParse model."""

    model_type = "nemotron_parse"
    is_composition = True

    def __init__(
        self,
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
        tie_word_embeddings: bool = False,
        decoder_start_token_id: int = 2,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        bos_token_id: int = 0,
        image_size: List[int] = None,
        is_encoder_decoder: bool = True,
        max_sequence_length: int = 9000,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            decoder_start_token_id=decoder_start_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

        if decoder is None:
            decoder = {}
        if encoder is None:
            encoder = {}

        if encoder:
            radio_model_path = encoder.get("_name_or_path", "nvidia/C-RADIOv2-H")
            self.encoder = AutoConfig.from_pretrained(radio_model_path, trust_remote_code=True)
            # Update with any overrides from encoder dict
            for key, value in encoder.items():
                if hasattr(self.encoder, key):
                    setattr(self.encoder, key, value)
        else:
            self.encoder = PretrainedConfig()

        decoder["max_sequence_length"] = max_sequence_length
        self.decoder = NemotronParseTextConfig(**decoder)
        self.image_size = image_size or [2048, 1648]
        self.vocab_size = self.decoder.vocab_size
        self.is_encoder_decoder = is_encoder_decoder
        self.max_sequence_length = max_sequence_length

    def to_dict(self):
        output = super().to_dict()
        output["encoder"] = self.encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.model_type
        output["is_encoder_decoder"] = self.is_encoder_decoder
        return output


# -----------------------------------------------------------------------------
# NemotronParse modeling
# -----------------------------------------------------------------------------


class NemotronParseDecoder(MBartPreTrainedModel):
    """Transformer decoder consisting of *config.decoder_layers* layers."""

    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = MBartScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.config = config

        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.config._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self.config._attn_implementation == "sdpa" and not output_attentions:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            else:
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        hidden_states = self.layernorm_embedding(inputs_embeds)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,  # past_key_values
                    output_attentions,
                    False,  # use_cache
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=None,
                    output_attentions=output_attentions,
                    use_cache=False,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, None, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class RadioWithNeck(nn.Module):
    """Vision encoder using RADIO model with custom neck."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create RADIO encoder from config (which is now actual RADIOConfig)
        self.model_encoder = AutoModel.from_config(config, trust_remote_code=True)

        # Neck components (dtype conversion done later via .to())
        last_hidden_state = 1024
        self.conv1 = nn.Conv1d(1280, last_hidden_state, 1)
        self.layer_norm1 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)
        self.conv2 = nn.Conv2d(
            last_hidden_state, last_hidden_state, kernel_size=(1, 4), stride=(1, 4), padding=0, bias=False
        )
        self.layer_norm2 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)
        self.sum_proj = nn.Linear(3840, last_hidden_state)
        self.layer_norm3 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=False, **kwargs):
        radio_output = self.model_encoder(pixel_values)
        summary, feature = radio_output

        output = self.conv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.layer_norm1(output)

        patch_size = self.config.patch_size
        output = rearrange(
            output,
            "b (h w) d -> b d h w",
            h=pixel_values.shape[-2] // patch_size,
            w=pixel_values.shape[-1] // patch_size,
        )
        output = self.conv2(output)
        output = rearrange(output, "b d h w -> b (h w) d")
        output = self.layer_norm2(output)
        summary = self.layer_norm3(self.sum_proj(summary))
        output = torch.cat((output, summary.unsqueeze(1)), dim=1)

        return DonutSwinModelOutput(last_hidden_state=output)


class NemotronParsePreTrainedModel(PreTrainedModel):
    """Abstract class to handle weights initialization."""

    config_class = NemotronParseConfig
    base_model_prefix = "vision_encoder_decoder"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RadioWithNeck", "MBartDecoder"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.decoder.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.decoder.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class NemotronParseForConditionalGeneration(HFCheckpointingMixin, NemotronParsePreTrainedModel, GenerationMixin):
    """NemotronParse model for conditional generation tasks."""

    def __init__(self, config: NemotronParseConfig, loss_fn=None, **kwargs):
        super().__init__(config)
        self.loss_fn = loss_fn

        self.encoder = RadioWithNeck(config.encoder)
        self.encoder.main_input_name = "pixel_values"
        self.encoder = self.encoder.to(torch.bfloat16)

        self.decoder = NemotronParseDecoder(config.decoder)
        self.decoder = self.decoder.to(torch.bfloat16)

        self.lm_head = nn.Linear(config.decoder.d_model, config.decoder.vocab_size, bias=False, dtype=torch.bfloat16)

        num_extra_heads = getattr(config, "num_extra_heads", 0)
        self.decoder.extra_heads = nn.ModuleList(
            [
                nn.Linear(config.decoder.d_model, config.decoder.d_model, dtype=torch.bfloat16)
                for _ in range(num_extra_heads)
            ]
        )
        self.decoder.extra_proj = nn.ModuleList(
            [
                nn.Linear(config.decoder.d_model, config.decoder.d_model, dtype=torch.bfloat16)
                for _ in range(num_extra_heads)
            ]
        )

        self.class_token_indx_start = getattr(config, "class_token_start_idx", 50000)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs_encoder = {k: v for k, v in kwargs.items() if not k.startswith("decoder_")}
        kwargs_decoder = {k[len("decoder_") :]: v for k, v in kwargs.items() if k.startswith("decoder_")}

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")
            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        output_hidden_states = True

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        logits = self.lm_head(decoder_outputs.last_hidden_state)

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=None,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=getattr(encoder_outputs, "hidden_states", None),
            encoder_attentions=getattr(encoder_outputs, "attentions", None),
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.decoder._reorder_cache(past_key_values, beam_idx)


AutoConfig.register("nemotron_parse", NemotronParseConfig)

ModelClass = NemotronParseForConditionalGeneration
