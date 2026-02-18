# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Optional

from transformers import AutoConfig


def _should_load_before_shard(
    *,
    autopipeline: Optional[object],
    tp_size: int,
    ep_size: int,
    pretrained_model_name_or_path: str,
    load_base_model: bool,
    peft_config: Optional[object],
) -> bool:
    """Decide whether to load the checkpoint before FSDP/TP/EP sharding.

    Load-before-shard is only safe when running single-GPU (no PP, TP, or EP),
    a checkpoint actually needs loading, and no PEFT adapter is involved.
    With any model parallelism the post-shard load path must be used to avoid
    NCCL collective mismatches or key/device inconsistencies.
    """
    no_pp = autopipeline is None
    no_tp = tp_size <= 1
    no_ep = ep_size <= 1
    need_checkpoint_load = bool(pretrained_model_name_or_path and load_base_model)
    return no_pp and no_tp and no_ep and need_checkpoint_load and (peft_config is None)


def sliding_window_overwrite(model_name: str) -> dict[str, Any]:
    """Returns configuration overrides to handle sliding window settings based on model rules.

    Args:
        model_name: The HuggingFace model name or path to load configuration from

    Returns:
        dict: Dictionary with overwrite values, or empty dict if no overwrites needed
    """
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    overwrite_dict = {}

    # Override sliding_window setting to address a HF mismatch relevant to use_sliding_window
    # TODO(@zhiyul): remove this once the bug is fixed https://github.com/huggingface/transformers/issues/38002
    if hasattr(hf_config, "use_sliding_window") and hf_config.use_sliding_window == False:
        assert hasattr(hf_config, "sliding_window")
        overwrite_dict = {
            "sliding_window": None,
        }
        print(f"use_sliding_window=False in config - overriding sliding_window parameter to None: {overwrite_dict}")

    return overwrite_dict


def apply_qwen3_omni_config_patch():
    """Fix Qwen3OmniMoeTalkerCodePredictorConfig accessing use_sliding_window."""
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig

    if not hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window"):
        Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = False


def apply_cache_compatibility_patches():
    """Apply compatibility patches for transformers cache utilities."""
    # Alias cache API for models expecting get_usable_length
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length") and hasattr(DynamicCache, "get_seq_length"):
        DynamicCache.get_usable_length = DynamicCache.get_seq_length

    # ---------------------------------------------------------------------
    # DTensor/TP compatibility patches
    # ---------------------------------------------------------------------
    # HF Qwen3 slices `hidden_states[:, slice(0, None), :]` when logits_to_keep=0.
    # Under DTensor this can dispatch to `aten.alias`, which lacks a sharding strategy
    # on some torch nightly builds used in CI.
    #
    # Patch: skip the no-op slice and call lm_head(hidden_states) directly.
    try:  # pragma: no cover
        import functools

        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        if not getattr(Qwen3ForCausalLM.forward, "__nemo_dtensor_logits_to_keep_patched__", False):
            from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: WPS433

            _orig_forward = Qwen3ForCausalLM.forward

            @functools.wraps(_orig_forward)
            def _patched_forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                cache_position=None,
                logits_to_keep=0,
                **kwargs,
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

                hidden_states = outputs.last_hidden_state
                if isinstance(logits_to_keep, int) and logits_to_keep == 0:
                    logits = self.lm_head(hidden_states)
                else:
                    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                    logits = self.lm_head(hidden_states[:, slice_indices, :])

                loss = None
                if labels is not None:
                    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

            _patched_forward.__nemo_dtensor_logits_to_keep_patched__ = True  # type: ignore[attr-defined]
            Qwen3ForCausalLM.forward = _patched_forward  # type: ignore[method-assign]
    except Exception:
        # Best-effort patch; ignore if transformers/qwen3 is unavailable.
        pass
