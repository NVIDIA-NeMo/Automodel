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

"""Target-model wrapper for minimal EAGLE-3 training."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from nemo_automodel.components.datasets.llm.packed_sequence import build_block_causal_additive_mask
from nemo_automodel.components.speculative.eagle.backend import Eagle3TargetBackend


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a batched sequence tensor left and zero-fill the tail.

    This matches the reference EAGLE-3 target preparation used by SpecForge:
    sequence-aligned tensors are shifted with ``padding(..., left=False)``.
    See SpecForge ``eagle3_target_model.py`` around the target preparation
    logic referenced by the user.
    """
    tail = torch.zeros_like(tensor[:, :1])
    return torch.cat((tensor[:, 1:], tail), dim=1)


def default_eagle3_aux_layer_ids(num_layers: int) -> list[int]:
    """Return the EAGLE-3 default 3-layer (low / mid / high) aux capture recipe.

    The downstream draft model's ``fc`` projection is sized for exactly
    ``num_aux_hidden_states`` layers (default 3) of concatenated target hidden
    states. Silently deduplicating collisions on shallow targets would yield
    fewer than 3 captured tensors and crash later inside the draft ``fc`` with a
    confusing shape-mismatch error -- raise here instead so the caller picks 3
    distinct in-bounds ids that match the draft config. Shared by every target
    backend (co-located, remote, SGLang) so they all default identically.
    """
    candidates = [1, num_layers // 2 - 1, num_layers - 4]
    if any(c < 0 or c >= num_layers for c in candidates) or len(set(candidates)) != 3:
        raise ValueError(
            f"Target model has num_hidden_layers={num_layers}, which is too shallow "
            f"for the default EAGLE-3 aux recipe {candidates}. Pass aux_layer_ids "
            f"explicitly (must be 3 distinct in-bounds layer indices, matching the "
            f"draft model's num_aux_hidden_states)."
        )
    return candidates


def validate_eagle3_aux_layer_ids(aux_layer_ids: Sequence[int], num_layers: int) -> list[int]:
    """Validate an aux-layer selection against a target of ``num_layers`` depth.

    Shared by every target backend so an explicit ``aux_layer_ids`` is checked
    identically whether the target runs co-located, remote, or under SGLang.
    """
    aux_layer_ids = list(aux_layer_ids)
    if len(aux_layer_ids) != 3:
        raise ValueError(
            f"EAGLE-3 expects exactly 3 aux_layer_ids, but got {len(aux_layer_ids)}: "
            f"{aux_layer_ids}. This must match the draft model's num_aux_hidden_states."
        )
    if len(set(aux_layer_ids)) != len(aux_layer_ids):
        raise ValueError(
            f"EAGLE-3 aux_layer_ids must be distinct, but got {aux_layer_ids}. "
            "Duplicate ids would collapse the captured aux hidden states."
        )
    for layer_id in aux_layer_ids:
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(f"aux layer id {layer_id} is out of bounds for model with {num_layers} layers")
    return aux_layer_ids


def _to_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialise a (possibly tensor-parallel) tensor as a plain local tensor.

    With a tensor-parallel target the lm_head is column-parallel, so its logits
    come back as a vocab-sharded ``DTensor``. The draft consumes plain tensors,
    so gather the full tensor before handing it on. A no-op for an already-plain
    (unsharded or pure-FSDP-replicated) tensor.
    """
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


@dataclass
class Eagle3TargetBatch:
    """Target-model supervision for one draft-training batch.

    Carries exactly one supervision encoding (validated in ``__post_init__``),
    both consumed directly by ``Eagle3TrainerModule.forward``:

    - ``logits`` -- the target's full-vocab logits; the draft-vocab projection
      happens trainer-side. Used by the co-located backend, where the tensor
      never leaves the GPU.
    - ``target_probs`` + ``position_mask`` -- the already-projected draft-vocab
      distribution, so a backend that computes it itself (e.g. a remote server)
      only transfers draft-vocab-sized tensors.
    """

    aux_hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    logits: torch.Tensor | None = None
    target_probs: torch.Tensor | None = None
    position_mask: torch.Tensor | None = None
    # Packing metadata (None unless packing is enabled), unshifted slot frame:
    # per-document position_ids / seq_lens (block-causal mask) and doc_remaining
    # (gates cross-document TTT supervision).
    position_ids: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    doc_remaining: torch.Tensor | None = None

    def __post_init__(self) -> None:
        has_logits = self.logits is not None
        has_precomputed = self.target_probs is not None and self.position_mask is not None
        if has_logits == has_precomputed:
            raise ValueError(
                "Eagle3TargetBatch requires exactly one supervision source: either "
                "`logits` (full-vocab, projected trainer-side) or both `target_probs` "
                "and `position_mask` (precomputed over the draft vocab)."
            )

    def to_trainer_inputs(self) -> dict[str, torch.Tensor]:
        """Return kwargs for ``Eagle3TrainerModule.forward``, dispatching on
        whichever supervision encoding this batch carries."""
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "loss_mask": self.loss_mask,
            "aux_hidden_states": self.aux_hidden_states,
        }
        if self.logits is not None:
            inputs["target_logits"] = self.logits
        else:
            inputs["target_probs"] = self.target_probs
            inputs["position_mask"] = self.position_mask
        if self.seq_lens is not None:
            inputs["position_ids"] = self.position_ids
            inputs["seq_lens"] = self.seq_lens
            inputs["doc_remaining"] = self.doc_remaining
        return inputs


class HFEagle3TargetModel(Eagle3TargetBackend):
    """Co-located backend that captures three auxiliary hidden states from a causal LM."""

    def __init__(self, model: nn.Module, aux_layer_ids: Sequence[int] | None = None, cp_mesh=None):
        self.model = model.eval()
        candidate_ids = list(aux_layer_ids) if aux_layer_ids is not None else self._default_aux_layer_ids()
        self.aux_layer_ids = self._validate_aux_layer_ids(candidate_ids)
        # Context parallelism shards the (frozen) target forward along the sequence
        # dim; generate_batch gathers the aux/logits back to the full sequence so
        # the draft stays CP-unaware. cp_mesh is the "cp" device submesh (or None).
        self.cp_mesh = cp_mesh
        self._cp_size = cp_mesh.size() if cp_mesh is not None else 1
        if self._cp_size > 1:
            from nemo_automodel.components.distributed.context_parallel.utils import attach_context_parallel_hooks
            from nemo_automodel.components.speculative.target_cp import attach_cp_kv_gather_hooks

            attach_context_parallel_hooks(self.model)
            n_self_attn = sum(1 for name, _ in self.model.named_modules() if name.endswith("self_attn"))
            if n_self_attn == 0:
                raise ValueError(
                    "Context parallelism requires the target's attention modules to be named "
                    "'*self_attn' (so the causal mask is fixed under sequence sharding), but none "
                    "were found on the target model."
                )
            # torch's context_parallel ring does not fire for a plain HF forward (q/k/v
            # reach SDPA as local tensors), so each rank would attend only to its own
            # shard. Since the target is frozen, all-gather K/V and attend the local Q
            # against the full sequence.
            attach_cp_kv_gather_hooks(self.model, cp_mesh)

    def _check_captured(self, captured: dict[int, torch.Tensor]) -> None:
        if len(captured) != len(self.aux_layer_ids):
            raise RuntimeError(
                f"Expected {len(self.aux_layer_ids)} captured aux layers but got {len(captured)}: {sorted(captured)}"
            )

    def _num_hidden_layers(self) -> int:
        """Return the target's decoder depth.

        Multimodal targets (e.g. ``Gemma4ForConditionalGeneration``) carry no
        top-level ``num_hidden_layers``; the text backbone's depth lives on
        ``config.text_config``. Text-only targets (and the lightweight config
        stubs used in tests) expose ``num_hidden_layers`` directly.
        """
        config = self.model.config
        text_config = getattr(config, "text_config", None)
        if text_config is not None and getattr(text_config, "num_hidden_layers", None) is not None:
            return text_config.num_hidden_layers
        return config.num_hidden_layers

    def _default_aux_layer_ids(self) -> list[int]:
        return default_eagle3_aux_layer_ids(self._num_hidden_layers())

    def _validate_aux_layer_ids(self, aux_layer_ids: Sequence[int]) -> list[int]:
        """Validate aux-layer selection before any forward hooks are registered."""
        return validate_eagle3_aux_layer_ids(aux_layer_ids, self._num_hidden_layers())

    def _get_transformer_layers(self) -> list[nn.Module]:
        """Return decoder layers as an ordered list indexable by integer.

        Supports both the HuggingFace layouts (where ``layers`` is a
        ``ModuleList``) and AutoModel's custom-impl layouts (where
        ``layers`` is a ``ModuleDict`` keyed by ``str(i)``). Returning a
        plain list normalizes the access pattern for downstream
        ``register_forward_hook`` calls.
        """
        # Common HF causal-LM layouts:
        #   model.model.layers                     (Llama, Qwen, Mistral, Gemma, Phi, ...)
        #   model.model.language_model.layers      (multimodal wrappers: Gemma4ForConditionalGeneration)
        #   model.language_model.layers            (some VLM text backbones nested one level up)
        #   model.layers                           (some VLM text backbones exposed directly)
        #   model.transformer.h                    (GPT2 / Falcon-style)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            container = self.model.model.layers
        elif (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "language_model")
            and hasattr(self.model.model.language_model, "layers")
        ):
            container = self.model.model.language_model.layers
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            container = self.model.language_model.layers
        elif hasattr(self.model, "layers"):
            container = self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            container = self.model.transformer.h
        else:
            raise ValueError("Unsupported model structure for EAGLE-3 aux-layer capture")
        if isinstance(container, nn.ModuleDict):
            # AutoModel custom impls use ModuleDict keyed by ``str(i)``.
            return [container[str(i)] for i in range(len(container))]
        return list(container)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the target model input embeddings."""
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> Eagle3TargetBatch:
        """Run the target model and capture aux hidden states plus logits.

        With ``seq_lens`` (packing), the target runs with a ``[B, 1, T, T]``
        block-causal mask and per-document ``position_ids`` so its outputs respect
        document boundaries; the packing metadata is forwarded unshifted to the
        trainer. ``seq_lens=None`` keeps the original 2D-mask path.
        """
        layers = self._get_transformer_layers()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(layer_id: int):
            def _hook(_module, _inputs, outputs):
                captured[layer_id] = outputs[0] if isinstance(outputs, tuple) else outputs

            return _hook

        for layer_id in self.aux_layer_ids:
            if layer_id < 0 or layer_id >= len(layers):
                raise ValueError(f"aux layer id {layer_id} is out of bounds for model with {len(layers)} layers")
            handles.append(layers[layer_id].register_forward_hook(_make_hook(layer_id)))

        # AutoModel's custom causal LMs only declare ``input_ids``,
        # ``attention_mask``, ``position_ids``, ``padding_mask`` and a
        # ``**attn_kwargs`` catch-all; the HF flags below mean nothing to
        # them and are dropped to keep the call site honest.
        forward_params = inspect.signature(self.model.forward).parameters
        extra_kwargs = {
            name: False for name in ("output_hidden_states", "output_attentions", "use_cache") if name in forward_params
        }

        # Packing isolates documents per attention backend; the mask strategy
        # differs because FlashAttention has no 4D-mask code path:
        #   * SDPA / eager consume the [B, 1, T, T] block-causal additive mask.
        #   * FlashAttention infers per-document cu_seqlens from the reset points
        #     in a per-document ``position_ids`` and is passed ``attention_mask=None``.
        #     Feeding FA the 4D additive mask instead drives its unpad gather out
        #     of bounds: the mask flattens to B*T*T entries and the gather indexes
        #     a B*T-row tensor with them. transformers only packs from position_ids
        #     at batch size 1 (see ``_is_packed_sequence``).
        target_attention_mask = attention_mask
        if seq_lens is not None:
            if position_ids is None or "position_ids" not in forward_params:
                raise ValueError(
                    "EAGLE-3 sequence packing requires per-document position_ids, but none were "
                    "provided or the target model's forward does not accept a `position_ids` argument."
                )
            extra_kwargs["position_ids"] = position_ids
            attn_impl = getattr(self.model.config, "_attn_implementation", None) or ""
            if "flash" in attn_impl:
                if input_ids.shape[0] != 1:
                    raise ValueError(
                        "EAGLE-3 sequence packing with a FlashAttention target only supports "
                        f"micro_batch_size=1 (got {input_ids.shape[0]}). FlashAttention infers "
                        "document boundaries from per-document position_ids, which transformers "
                        "packs only at batch size 1. Set micro_batch_size=1 or load the target "
                        "with attn_implementation='sdpa'."
                    )
                # attention_mask=None + per-document position_ids -> FA varlen packing.
                target_attention_mask = None
            else:
                param_dtype = next(self.model.parameters()).dtype
                target_attention_mask = build_block_causal_additive_mask(
                    seq_lens,
                    seq_length=input_ids.shape[1],
                    dtype=param_dtype,
                    device=input_ids.device,
                )

        try:
            if self._cp_size > 1:
                # Context parallelism: shard the sequence, run the target as ring
                # attention, then gather aux/logits back to the full sequence.
                # Packing (which needs the 4D block-causal mask) is gated off
                # upstream, so attention_mask is None and the self_attn hooks force
                # is_causal.
                from nemo_automodel.components.speculative.target_cp import run_target_cp_forward_and_gather

                def _collect(outputs):
                    raw_logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    self._check_captured(captured)
                    return [captured[layer_id] for layer_id in self.aux_layer_ids] + [raw_logits]

                cp_extra = {k: v for k, v in extra_kwargs.items() if k != "position_ids"}
                _outputs, gathered = run_target_cp_forward_and_gather(
                    self.cp_mesh, self.model, input_ids, cp_extra, _collect, position_ids=position_ids
                )
                aux_hidden_states = torch.cat(gathered[:-1], dim=-1)
                target_logits = gathered[-1]
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=target_attention_mask,
                    **extra_kwargs,
                )
                self._check_captured(captured)
                aux_hidden_states = torch.cat([captured[layer_id] for layer_id in self.aux_layer_ids], dim=-1)
                # HF causal LM outputs wrap logits in a dataclass; AutoModel's
                # custom causal LM returns the logits tensor directly.
                target_logits = outputs.logits if hasattr(outputs, "logits") else outputs
        finally:
            for handle in handles:
                handle.remove()
        # A tensor-parallel target returns vocab-sharded (DTensor) logits from its
        # column-parallel lm_head; gather to a full tensor before the draft-vocab
        # projection and shift. No-op without TP. (TP is gated off with CP for now,
        # so the CP branch above never produces a TP-sharded logits tensor.)
        target_logits = _to_full_tensor(target_logits)
        shifted_logits = _shift_left_with_zero(target_logits)
        shifted_input_ids = _shift_left_with_zero(input_ids)
        shifted_loss_mask = _shift_left_with_zero(loss_mask)
        return Eagle3TargetBatch(
            aux_hidden_states=aux_hidden_states,
            logits=shifted_logits,
            input_ids=shifted_input_ids,
            attention_mask=attention_mask,
            loss_mask=shifted_loss_mask,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )
