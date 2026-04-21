# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Patch ``NemotronFlashForCausalLM.forward`` lm_head norm-divide to cope with
DTensor / plain-tensor mixing under FSDP2 + PEFT.

The remote-code model file at
``transformers_modules/.../modeling_nemotron_flash.py`` divides the logits by
``self.lm_head.weight.norm(p=2, dim=1)`` at the tail of its forward. With
FSDP2, ``self.lm_head.weight`` is a ``DTensor`` outside of the lm_head's
unshard window. With PEFT/LoRA wrapping the interior ``*_proj`` linears, the
``hidden_states`` / ``logits`` can emerge as plain tensors, so the divide
operation mixes plain ``torch.Tensor`` with ``DTensor`` and raises

    RuntimeError: aten.div.Tensor got mixed torch.Tensor and DTensor, need to
    convert all torch.Tensor to DTensor before calling distributed operators!

The SFT (non-PEFT) sibling does not hit this because its forward pipeline
preserves DTensor all the way through, so the divide is DTensor / DTensor.

We can't edit the remote-code model (it's cached from HF Hub), so we
monkey-patch the bound ``forward`` method post-materialization to unwrap
``self.lm_head.weight`` via ``.to_local()`` before calling ``.norm()``.
"""

from __future__ import annotations

import logging
import types

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _find_nemotron_flash_causallm_class(module: nn.Module):
    """Walk ``type(module).__mro__`` and return the original
    ``NemotronFlashForCausalLM`` class shipped by the remote-code module,
    or ``None`` if the module isn't a Nemotron-Flash causal LM.

    ``fully_shard`` dynamically subclasses the module class and prefixes its
    ``__name__`` with ``"FSDP"`` (e.g. ``FSDPNemotronFlashForCausalLM``); walk
    the MRO so both the pre- and post-FSDP types are recognised.
    """
    for cls in type(module).__mro__:
        name = getattr(cls, "__name__", "")
        mod = getattr(cls, "__module__", "") or ""
        if name == "NemotronFlashForCausalLM" and "transformers_modules" in mod:
            return cls
    return None


def _is_nemotron_flash_causallm(module: nn.Module) -> bool:
    return _find_nemotron_flash_causallm_class(module) is not None


def _patched_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    calc_logits_for_entire_prompt=True,
    fla_past_key_values=None,
    mamba_inference_params=None,
):
    """Mirror the original ``NemotronFlashForCausalLM.forward`` body but use a
    local-unwrapped ``lm_head.weight`` for the post-logits normalization.
    """
    from torch.distributed.tensor import DTensor
    from torch.nn import CrossEntropyLoss

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        fla_past_key_values=fla_past_key_values,
        mamba_inference_params=mamba_inference_params,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if calc_logits_for_entire_prompt:
        logits = self.lm_head(hidden_states)
    else:
        logits = self.lm_head(hidden_states[..., -1:, :])

    # Fetch the weight and materialize the full tensor if it's a DTensor.
    # ``logits`` has vocab_size in its last dim (full, not sharded), because the
    # underlying ``lm_head`` temporarily all-gathers its weight during its own
    # forward. After the call returns, ``self.lm_head.weight`` may be back to
    # DTensor (post-reshard) — we need the replicated full tensor to match
    # ``logits``' shape when dividing.
    lm_head_weight = self.lm_head.weight
    if isinstance(lm_head_weight, DTensor):
        lm_head_weight_full = lm_head_weight.full_tensor()
    else:
        lm_head_weight_full = lm_head_weight

    norm = lm_head_weight_full.norm(p=2, dim=1)

    # If ``logits`` is a DTensor (SFT path), keep the divide on DTensor by
    # leaving the weight as DTensor. Otherwise divide with a plain tensor.
    if isinstance(logits, DTensor) and isinstance(lm_head_weight, DTensor):
        norm = lm_head_weight.norm(p=2, dim=1)

    logits = logits / norm
    logits = logits.float()

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # Mirror the upstream return type from modeling_nemotron_flash.
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast

    return MoeCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
        hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
    )


def _is_nemotron_flash_config(cfg) -> bool:
    if cfg is None:
        return False
    if getattr(cfg, "model_type", None) == "nemotron_flash":
        return True
    architectures = getattr(cfg, "architectures", None) or ()
    if "NemotronFlashForCausalLM" in architectures:
        return True
    name_or_path = getattr(cfg, "name_or_path", "") or ""
    return "nemotron-flash" in name_or_path.lower()


def should_fix_lm_head_norm(model_parts) -> bool:
    for mp in model_parts:
        if isinstance(mp, nn.Module):
            if _is_nemotron_flash_causallm(mp):
                return True
            if _is_nemotron_flash_config(getattr(mp, "config", None)):
                return True
            for _, module in mp.named_modules():
                if _is_nemotron_flash_causallm(module):
                    return True
                if _is_nemotron_flash_config(getattr(module, "config", None)):
                    return True
        elif _is_nemotron_flash_config(getattr(mp, "config", None)):
            return True
    return False


def _patch_class_forward(cls) -> bool:
    """Patch the class's ``forward`` attribute in-place. Idempotent."""
    if getattr(cls.forward, "_nemo_lm_head_norm_patched", False):
        return False
    _patched_forward._nemo_lm_head_norm_patched = True  # type: ignore[attr-defined]
    cls.forward = _patched_forward
    logger.info(
        "[fix_lm_head_norm] patched class forward on %s.%s",
        cls.__module__,
        cls.__name__,
    )
    return True


def _find_nemotron_flash_classes_in_sys_modules():
    """Scan ``sys.modules`` for the remote-code ``NemotronFlashForCausalLM`` class.

    Remote-code modules live under ``transformers_modules.<repo>.<hash>.modeling_nemotron_flash``
    after transformers dynamic module loading.
    """
    import sys

    classes = []
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("transformers_modules."):
            continue
        if "modeling_nemotron_flash" not in mod_name and mod is not None:
            continue
        cls = getattr(mod, "NemotronFlashForCausalLM", None) if mod is not None else None
        if cls is not None:
            classes.append(cls)
    return classes


def fix_lm_head_norm(model_parts) -> int:
    """Monkey-patch ``NemotronFlashForCausalLM.forward`` to DTensor-safe version.

    Prefers patching at the class level (via MRO + ``sys.modules`` scan) so that
    all current and future instances — including vanilla-HF loaded models in
    the checkpoint robustness Phase 4 — use the DTensor-safe forward. Also
    patches per-instance as a fallback (FSDP2's dynamic subclass shadows the
    base ``forward`` on the instance).
    """
    fixed = 0
    patched_classes = set()

    # Pass 1: scan sys.modules for the remote-code class(es) and patch them at
    # class level. This covers ALL instances (training FSDP-wrapped + Phase 4
    # vanilla HF) without having to find the exact module on each one.
    for cls in _find_nemotron_flash_classes_in_sys_modules():
        if cls not in patched_classes and _patch_class_forward(cls):
            patched_classes.add(cls)
            fixed += 1

    # Pass 2: for any module parts that are the Nemotron-Flash causal LM
    # (NOT arbitrary submodules sharing the same config), patch class-level
    # via MRO (in case the sys.modules scan missed) and bind per-instance
    # as a belt-and-suspenders safeguard against FSDP2 dynamic-subclass
    # shadowing of the base ``forward``.
    for mp in model_parts:
        if not isinstance(mp, nn.Module):
            continue
        candidates = []
        if _is_nemotron_flash_causallm(mp):
            candidates.append(mp)
        for _, module in mp.named_modules():
            if _is_nemotron_flash_causallm(module):
                candidates.append(module)
        for module in candidates:
            cls = _find_nemotron_flash_causallm_class(module)
            if cls is not None and cls not in patched_classes:
                if _patch_class_forward(cls):
                    patched_classes.add(cls)
                    fixed += 1
            if not getattr(module.forward, "_nemo_lm_head_norm_patched", False):
                module.forward = types.MethodType(_patched_forward, module)
                try:
                    module.forward.__func__._nemo_lm_head_norm_patched = True  # type: ignore[attr-defined]
                except AttributeError:
                    pass
                logger.info(
                    "[fix_lm_head_norm] patched instance forward on %s",
                    type(module).__name__,
                )
                fixed += 1
    return fixed


__all__ = ["fix_lm_head_norm", "should_fix_lm_head_norm"]
