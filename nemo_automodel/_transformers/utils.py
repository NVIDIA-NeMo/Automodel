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

import logging
from typing import Any, Optional

import torch
from transformers import AutoConfig

logger = logging.getLogger(__name__)


def _should_load_before_shard(
    *,
    autopipeline: Optional[object],
    tp_size: int,
    ep_size: int,
    dp_shard_size: int = 1,
    pretrained_model_name_or_path: str,
    load_base_model: bool,
    peft_config: Optional[object],
) -> bool:
    """Decide whether to load the checkpoint before FSDP/TP/EP sharding.

    Load-before-shard is only safe when running single-GPU (no PP, TP, EP, or
    DP sharding) and a checkpoint actually needs loading.
    With any model parallelism the post-shard load path must be used to avoid
    NCCL collective mismatches or key/device inconsistencies.

    PEFT models skip this path and use the post-shard load so that base and
    adapter weights load in the same way as multi-GPU.
    """
    no_pp = autopipeline is None
    no_tp = tp_size <= 1
    no_ep = ep_size <= 1
    no_dp_shard = dp_shard_size <= 1
    no_peft = peft_config is None
    need_checkpoint_load = bool(pretrained_model_name_or_path and load_base_model)
    result = no_pp and no_tp and no_ep and no_dp_shard and no_peft and need_checkpoint_load
    logger.debug(
        "[_should_load_before_shard] no_pp={} no_tp={} no_ep={} no_dp_shard={} no_peft={} need_load={} -> {}".format(
            no_pp, no_tp, no_ep, no_dp_shard, no_peft, need_checkpoint_load, result
        )
    )
    return result


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


def _patch_bytes_to_unicode():
    """Re-export bytes_to_unicode on transformers.models.gpt2.tokenization_gpt2.

    In transformers v5 this helper was removed from the GPT-2 tokenizer module,
    but some custom tokenizers shipped with model weights (e.g. Kimi) still
    import it from there via ``trust_remote_code``.  Monkey-patching it back
    avoids an ImportError without modifying the transformers package.
    """
    import importlib

    gpt2_tok = importlib.import_module("transformers.models.gpt2.tokenization_gpt2")
    if hasattr(gpt2_tok, "bytes_to_unicode"):
        return

    from functools import lru_cache

    @lru_cache()
    def bytes_to_unicode():
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    gpt2_tok.bytes_to_unicode = bytes_to_unicode


def _patch_special_tokens_pattern():
    """Default ``special_tokens_pattern`` to ``"none"`` for PreTrainedTokenizer.

    Transformers v5 introduced ``special_tokens_pattern`` (default ``"cls_sep"``)
    which makes ``build_inputs_with_special_tokens`` prepend ``cls_token_id`` and
    append ``sep_token_id``.  Custom tokenizers (e.g. TikToken-based Kimi) that
    lack CLS/SEP tokens end up with ``None`` IDs in the sequence, crashing
    ``pad()``.
    """
    from transformers.tokenization_python import PreTrainedTokenizer

    _orig_init = PreTrainedTokenizer.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("special_tokens_pattern", "none")
        return _orig_init(self, *args, **kwargs)

    if not getattr(PreTrainedTokenizer.__init__, "_nemo_stp_patched", False):
        PreTrainedTokenizer.__init__ = _patched_init
        PreTrainedTokenizer.__init__._nemo_stp_patched = True  # type: ignore[attr-defined]


def apply_cache_compatibility_patches():
    """Apply compatibility patches for transformers cache utilities.

    Patches applied here fix API removals/changes between transformers versions
    so that both native and remote-code models can load and run.
    """
    _patch_bytes_to_unicode()
    _patch_special_tokens_pattern()

    import transformers.cache_utils as cache_utils

    # SlidingWindowCache was removed in transformers v5.x
    if not hasattr(cache_utils, "SlidingWindowCache"):
        cache_utils.SlidingWindowCache = cache_utils.StaticCache

    # Cache.get_usable_length was removed in transformers v5.x
    if not hasattr(cache_utils.Cache, "get_usable_length"):

        def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            max_length = self.get_max_cache_shape()
            if max_length is not None and isinstance(max_length, dict):
                max_length = max_length.get(layer_idx)
            if max_length is not None and self.get_seq_length(layer_idx) + new_seq_length > max_length:
                return max_length - new_seq_length
            return self.get_seq_length(layer_idx)

        cache_utils.Cache.get_usable_length = _get_usable_length

    # Alias on DynamicCache as well
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length") and hasattr(DynamicCache, "get_seq_length"):
        DynamicCache.get_usable_length = DynamicCache.get_seq_length

    # DynamicCache.to_legacy_cache was removed in transformers v5.x
    if not hasattr(DynamicCache, "to_legacy_cache"):

        def _to_legacy_cache(self):
            legacy_cache = ()
            for layer_idx in range(len(self)):
                legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
            return legacy_cache

        DynamicCache.to_legacy_cache = _to_legacy_cache

    # _tied_weights_keys changed from list to dict in transformers v5.x.
    # Patch post_init to auto-convert list -> dict for remote-code models.
    import transformers.modeling_utils as mu

    if not getattr(mu.PreTrainedModel.post_init, "_nemo_tied_keys_patched", False):
        _orig_post_init = mu.PreTrainedModel.post_init

        def _find_embedding_source(model):
            """Resolve the weight name of the input embedding layer.

            Prefer get_input_embeddings() (explicit HF contract), fall back
            to scanning for the first nn.Embedding in the module tree.
            """
            embed = model.get_input_embeddings()
            if embed is not None:
                for name, module in model.named_modules():
                    if module is embed:
                        return f"{name}.weight"
            # Fallback: first nn.Embedding (custom models that don't
            # override get_input_embeddings).
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    return f"{name}.weight"
            return None

        def _patched_post_init(self):
            tied = getattr(self, "_tied_weights_keys", None)
            # if tied is list -> model is pre 5.x -> we will tie the weights after _model_init.
            # between post_init and returned value of _model_init, there's code we don't control or can test for regressions,
            # thus seems safer to tie weights after _model_init.
            if isinstance(tied, list):
                source = _find_embedding_source(self)
                if source is None:
                    raise ValueError("Could not find the source of the embedding layer")
                tied_dict = {k: source for k in tied}
                self._nemo_tied_weights_keys = tied_dict
                # Keep the v5 dict form on the model so that any downstream HF
                # code path (e.g. vanilla ``AutoModelForCausalLM.from_pretrained``
                # used by the checkpoint-robustness test) ties the weights via
                # HF's own ``tie_weights`` and does not leave the tied sibling
                # (``lm_head.weight``) zero-initialised — which would cause
                # NaN logits for tied-embedding remote-code models like
                # Nemotron-Flash-1B whose forward does
                # ``logits / lm_head.weight.norm()``.
                self._tied_weights_keys = tied_dict
            # call orig post init
            _orig_post_init(self)

        mu.PreTrainedModel.post_init = _patched_post_init
        mu.PreTrainedModel.post_init._nemo_tied_keys_patched = True  # type: ignore[attr-defined]

    _patch_phi4mm_processor()
    _patch_peft_prepare_inputs()

    from nemo_automodel._transformers.kernel_patches import _patch_legacy_flash_attn_flag

    _patch_legacy_flash_attn_flag()


def _patch_phi4mm_processor():
    """Patch AutoProcessor.from_pretrained to fall back to the remote
    Phi4MMProcessor when the native Phi4MultimodalProcessor fails
    (hub processor_config.json points to native class but the tokenizer
    lacks image_token/audio_token attributes the native processor expects).
    """
    import transformers.processing_utils as pu

    if getattr(pu.ProcessorMixin.__dict__.get("from_pretrained"), "_nemo_phi4mm_patched", False):
        return
    _orig = pu.ProcessorMixin.from_pretrained.__func__

    @classmethod  # type: ignore[misc]
    def _patched(cls, pretrained_model_name_or_path, *args, **kwargs):
        try:
            return _orig(cls, pretrained_model_name_or_path, *args, **kwargs)
        except AttributeError as e:
            if "image_token" not in str(e) and "audio_token" not in str(e):
                raise
            import json

            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            kwargs.pop("trust_remote_code", None)
            repo = pretrained_model_name_or_path

            ProcessorCls = get_class_from_dynamic_module("processing_phi4mm.Phi4MMProcessor", repo)
            ImageProcCls = get_class_from_dynamic_module("processing_phi4mm.Phi4MMImageProcessor", repo)
            AudioProcCls = get_class_from_dynamic_module("processing_phi4mm.Phi4MMAudioFeatureExtractor", repo)

            pp_path = hf_hub_download(repo, "preprocessor_config.json")
            with open(pp_path) as f:
                pp_cfg = json.load(f)

            return ProcessorCls(
                ImageProcCls(dynamic_hd=pp_cfg.get("dynamic_hd", 36)),
                AudioProcCls(
                    audio_compression_rate=pp_cfg.get("audio_compression_rate", 8),
                    audio_downsample_rate=pp_cfg.get("audio_downsample_rate", 1),
                    audio_feat_stride=pp_cfg.get("audio_feat_stride", 1),
                ),
                AutoTokenizer.from_pretrained(repo, trust_remote_code=True),
            )

    _patched._nemo_phi4mm_patched = True  # type: ignore[attr-defined]
    pu.ProcessorMixin.from_pretrained = _patched


def _patch_peft_prepare_inputs():
    """Patch PeftModelForCausalLM.__init__ to handle models whose inner
    backbone lacks prepare_inputs_for_generation (e.g. Phi4MM applies PEFT
    to the inner Phi4MMModel, not the outer ForCausalLM).
    """
    try:
        import peft.peft_model as pm

        if getattr(pm.PeftModelForCausalLM.__init__, "_nemo_peft_patched", False):
            return
        _orig = pm.PeftModelForCausalLM.__init__

        def _patched(self, model, peft_config, adapter_name="default", **kwargs):
            try:
                _orig(self, model, peft_config, adapter_name=adapter_name, **kwargs)
            except AttributeError as e:
                if "prepare_inputs_for_generation" not in str(e):
                    raise
                model.prepare_inputs_for_generation = lambda *a, **kw: {}
                _orig(self, model, peft_config, adapter_name=adapter_name, **kwargs)

        _patched._nemo_peft_patched = True  # type: ignore[attr-defined]
        pm.PeftModelForCausalLM.__init__ = _patched
    except ImportError:
        pass

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
