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

import logging
import traceback
import types
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants for identifying text/language modules in multimodal models
TEXT_MODULE_ATTRS = ("language_model", "text_model", "text_decoder")
MULTIMODAL_SUFFIXES = (
    "vision_tower",
    "visual",
    "image_encoder",
    "vision_encoder",
    "audio_tower",
    "audio_encoder",
    "audio_model",
    "mm_projector",
    "multi_modal_projector",
    "multimodal_projector",
    "vision_projector",
    "audio_projector",
)


def get_text_module(model: nn.Module) -> nn.Module:
    """Return the nested text/LLM module if present, else the model itself."""
    if model is None:
        return model
    for attr_name in TEXT_MODULE_ATTRS:
        if hasattr(model, attr_name):
            nested = getattr(model, attr_name)
            if nested is not None:
                return nested
    return model


def create_pipeline_forward_inner(model_class_name: str = "AutoModel") -> Callable:
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def pipeline_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        causal_mask_mapping: Optional[dict] = None,
        **kwargs,
    ) -> Union[torch.Tensor, BaseModelOutputWithPast]:
        # Embeddings handling
        if inputs_embeds is None:
            if hasattr(self, "embed_tokens") and self.embed_tokens is not None:
                if input_ids is None:
                    raise ValueError("You must provide either input_ids or inputs_embeds")
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                if (
                    input_ids is not None
                    and isinstance(input_ids, torch.Tensor)
                    and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    inputs_embeds = input_ids
                else:
                    raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")

        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Attention mask handling (compilation-friendly):
        # causal_mask_mapping should be precomputed in data pipeline via default_collater
        # If not provided, model will fail - this enforces clean separation
        if causal_mask_mapping is None:
            # If causal_mask_mapping is missing, fall back to on-the-fly computation.
            # This is not recommended for compilation, as it introduces runtime overhead.
            #
            # DEBUG: log why we're in the fallback path.  This should NOT happen after
            # the collate_fn fix — if you see this log, causal_mask_mapping was dropped
            # somewhere between collate and this forward call.
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(
                f"[DEBUG rank={rank}] causal_mask_mapping not provided; computing it here. "
                f"inputs_embeds.shape={inputs_embeds.shape if inputs_embeds is not None else None}  "
                f"attention_mask={attention_mask.shape if isinstance(attention_mask, torch.Tensor) else attention_mask}  "
                f"_attn_implementation={getattr(getattr(self, 'config', None), '_attn_implementation', 'N/A')}\n"
                f"Call stack:\n{''.join(traceback.format_stack(limit=12))}",
                flush=True,
            )
            if not isinstance((causal_mask_mapping := attention_mask), dict):
                # Skip create_causal_mask entirely: for standard (non-packed) causal
                # training, passing None lets FA2 use its native is_causal=True fast
                # path with zero mask-materialization overhead.
                causal_mask_mapping = {"full_attention": None}
                if hasattr(self, "has_sliding_layers") and self.has_sliding_layers:
                    from transformers.masking_utils import create_sliding_window_causal_mask
                    mask_kwargs = {
                        "config": self.config,
                        "input_embeds": inputs_embeds,
                        "attention_mask": attention_mask,
                        "cache_position": cache_position,
                        "past_key_values": None,
                        "position_ids": None,
                    }
                    # causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            # DEBUG: log that causal_mask_mapping was received correctly (log only first call to avoid spam)
            if not getattr(pipeline_forward, "_mask_logged", False):
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                fa_mask = causal_mask_mapping.get("full_attention")
                print(
                    f"[DEBUG rank={rank}] causal_mask_mapping received OK: keys={list(causal_mask_mapping.keys())}  "
                    f"full_attention type={type(fa_mask)}  "
                    f"value={fa_mask.shape if isinstance(fa_mask, torch.Tensor) else fa_mask}",
                    flush=True,
                )
                pipeline_forward._mask_logged = True

        hidden_states = inputs_embeds

        # Rotary embeddings precomputation (shared across layers)
        position_embeddings = None
        rotary_emb = get_text_module(self).rotary_emb
        if rotary_emb is not None:
            position_embeddings = rotary_emb(hidden_states, position_ids)

        if hasattr(self, "layers") and self.layers is not None:
            # Works for dict-like or list-like containers
            layer_iter = self.layers.values() if hasattr(self.layers, "values") else self.layers
            _layer_dbg_done = getattr(pipeline_forward, "_layer_mask_logged", False)
            for decoder_layer in layer_iter:
                layer_attention_mask = causal_mask_mapping.get("full_attention")
                if hasattr(decoder_layer, "attention_type"):
                    layer_attention_mask = causal_mask_mapping.get(
                        getattr(decoder_layer, "attention_type"), causal_mask_mapping.get("full_attention")
                    )

                # DEBUG: confirm mask is None (full causal) at every decoder layer call.
                if not _layer_dbg_done:
                    import torch.distributed as dist
                    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                    print(
                        f"[DEBUG layer mask rank={rank}] layer={type(decoder_layer).__name__}  "
                        f"attention_mask={type(layer_attention_mask).__name__}  "
                        f"value={layer_attention_mask.shape if isinstance(layer_attention_mask, torch.Tensor) else layer_attention_mask}  "
                        f"<-- should be None for full-causal FA2 (no mask materialization)",
                        flush=True,
                    )
                    pipeline_forward._layer_mask_logged = True
                    _layer_dbg_done = True

                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        if hasattr(self, "norm") and self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if model_class_name == "PipelineStage":
            return hidden_states
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

    return pipeline_forward


def create_pipeline_forward_causal_lm() -> Callable:
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def pipeline_forward_causal_lm(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if hasattr(self, "model") and self.model is not None:
            # DEBUG: log what kwargs (specifically causal_mask_mapping) reach the causal-LM wrapper
            if not getattr(pipeline_forward_causal_lm, "_clm_logged", False):
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                cmm = kwargs.get("causal_mask_mapping", "NOT_IN_KWARGS")
                print(
                    f"[DEBUG rank={rank}] pipeline_forward_causal_lm: "
                    f"input_ids.shape={input_ids.shape if isinstance(input_ids, torch.Tensor) else type(input_ids).__name__}  "
                    f"causal_mask_mapping in kwargs={'causal_mask_mapping' in kwargs}  "
                    f"value={({k: (type(v).__name__, v.shape if isinstance(v, torch.Tensor) else v) for k, v in cmm.items()} if isinstance(cmm, dict) else cmm)}",
                    flush=True,
                )
                pipeline_forward_causal_lm._clm_logged = True

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )
            if isinstance(outputs, BaseModelOutputWithPast):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs
                outputs = None
        else:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            elif input_ids is not None and input_ids.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                hidden_states = input_ids
            else:
                raise ValueError("Expected hidden states as input for pipeline stage without inner model")
            outputs = None

        if hasattr(self, "lm_head") and self.lm_head is not None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            return logits
        else:
            return hidden_states

    return pipeline_forward_causal_lm


def _patch_is_packed_sequence_for_pp_training() -> None:
    """Eliminate CPU-GPU sync in flash attention during PP training.

    transformers._is_packed_sequence() returns a GPU bool scalar when batch_size==1,
    which causes Python's `if` to call aten::is_nonzero — a CPU-GPU sync — once per
    attention layer per microbatch forward pass.  With PP+gradient-checkpointing this
    fires ~2560 times per iteration, costing ~36 s/iter on 70B.

    For standard (non-packed) PP training sequences are never packed, so returning the
    Python False immediately is both correct and avoids the sync.  Do NOT apply this
    patch when using packed-sequence training (multiple sequences concatenated into one
    tensor with position_ids that reset to 0 mid-sequence).
    """
    try:
        import transformers.modeling_flash_attention_utils as _fa_utils

        if getattr(_fa_utils, "_is_packed_sequence_patched_for_pp", False):
            return  # already patched

        def _is_packed_sequence_no_sync(position_ids, batch_size):
            # Non-packed training: position_ids is always a simple arange — never packed.
            return False

        _fa_utils._is_packed_sequence = _is_packed_sequence_no_sync

        # Patch _flash_attention_forward to add one-time debug confirming:
        #   1) is_causal=True (full causal path — no recompile)
        #   2) attention_mask=None (no mask materialization)
        #   3) which branch is taken (standard causal else-branch expected)
        #
        # Must patch BOTH the source module AND the already-imported name in modeling_llama,
        # because modeling_llama imports _flash_attention_forward by reference at import time.
        # FSDP2 wrapping prevents class-level method patches from firing, so this is the
        # only reliable interception point.
        try:
            _orig_fa_fwd = _fa_utils._flash_attention_forward

            _fa_call_count = [0]

            def _flash_attention_forward_debug(*args, _orig=_orig_fa_fwd, _count=_fa_call_count, **kwargs):
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

                attn_mask = kwargs.get("attention_mask", args[3] if len(args) > 3 else "N/A")
                is_causal = kwargs.get("is_causal", args[5] if len(args) > 5 else "N/A")
                position_ids = kwargs.get("position_ids", args[7] if len(args) > 7 else None)
                cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
                cu_seq_lens_k = kwargs.get("cu_seq_lens_k")

                has_mask = attn_mask is not None and attn_mask != "N/A"
                has_varlen = cu_seq_lens_q is not None and cu_seq_lens_k is not None
                has_packed = _fa_utils._is_packed_sequence(position_ids, batch_size=args[0].size(0)) if position_ids is not None else False

                if has_mask:
                    branch = "BRANCH1-UNPAD (slow: _upad_input + flash_varlen_fn + pad_fn)"
                elif has_varlen or has_packed:
                    branch = "BRANCH2-VARLEN (packed/varlen: flash_varlen_fn)"
                else:
                    branch = "BRANCH3-FAST (optimal: flash_fn with is_causal)"

                _count[0] += 1
                if _count[0] <= 5 or _count[0] % 1000 == 0:
                    print(
                        f"[FA2 rank={rank} call#{_count[0]}] {branch}  "
                        f"attn_mask={'None' if not has_mask else type(attn_mask).__name__ + str(attn_mask.shape)}  "
                        f"is_causal={is_causal}  "
                        f"position_ids={'None' if position_ids is None else type(position_ids).__name__ + str(position_ids.shape)}  "
                        f"is_packed={has_packed}  cu_seq_lens={has_varlen}",
                        flush=True,
                    )
                return _orig(*args, **kwargs)

            # Patch source module
            _fa_utils._flash_attention_forward = _flash_attention_forward_debug

            # Patch every module that imported _flash_attention_forward by reference
            # at import time, so the wrapper fires regardless of the code path.
            _patch_targets = [
                "transformers.models.llama.modeling_llama",
                "transformers.integrations.flash_attention",
            ]
            for _mod_name in _patch_targets:
                try:
                    import importlib
                    _mod = importlib.import_module(_mod_name)
                    if hasattr(_mod, "_flash_attention_forward"):
                        _mod._flash_attention_forward = _flash_attention_forward_debug
                except Exception as _e2:
                    print(f"[PP] {_mod_name} FA2 patch failed: {_e2}", flush=True)

        except Exception as _e:
            print(f"[PP] _flash_attention_forward debug patch failed: {_e}", flush=True)
        _fa_utils._is_packed_sequence_patched_for_pp = True
        print(
            "[PP] Patched transformers._is_packed_sequence to avoid CPU-GPU sync during training.",
            flush=True,
        )
    except (ImportError, AttributeError):
        pass


def patch_hf_model_for_pp(model, patch_inner_model: bool = True, patch_causal_lm_model: bool = True) -> None:
    """Patch a HF model/module to produce pipeline-compatible forward.

    - If model has .model (e.g., LlamaForCausalLM), patch inner and outer.
    - Else, patch the module itself.
    """
    _patch_is_packed_sequence_for_pp_training()

    if hasattr(model, "model"):
        if patch_inner_model and getattr(model, "model", None) is not None:
            model.model.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), model.model)

        if patch_causal_lm_model:
            model.forward = types.MethodType(create_pipeline_forward_causal_lm(), model)
    else:
        if patch_inner_model:
            model.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), model)


def init_hf_model_buffers(model: torch.nn.Module, device: torch.device) -> None:
    if hasattr(getattr(model, "model", model), "rotary_emb"):
        rotary_owner = getattr(model, "model", model)
        if hasattr(rotary_owner.rotary_emb, "rope_init_fn"):
            inv_freq, _ = rotary_owner.rotary_emb.rope_init_fn(rotary_owner.rotary_emb.config, device)
            rotary_owner.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)


def validate_hf_model_for_pipeline_support(model: torch.nn.Module) -> None:
    """Validate if a model is compatible with torch.distributed.pipelining."""
    model_name = getattr(getattr(model, "config", object()), "pretrained_model_name_or_path", "Unknown")
    config = getattr(model, "config", None)

    issues: list[str] = []

    if config is not None:
        if getattr(config, "tie_word_embeddings", False):
            issues.append(
                "tie_word_embeddings=True is not supported for pipelining. Use separate input/output embeddings."
            )
        if getattr(config, "is_encoder_decoder", False):
            issues.append("Encoder-Decoder models with cross-attention are not supported yet for pipeline parallelism.")

    if issues:
        error_msg = f"Model '{model_name}' is not compatible with pipeline parallelism:\n\n"
        for i, issue in enumerate(issues, 1):
            error_msg += f"{i}. {issue}\n"
        raise ValueError(error_msg)
