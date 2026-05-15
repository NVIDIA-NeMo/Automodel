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

"""Composite model for joint fine-tuning of a Gemma 4 base + its drafter.

The composite orchestrates a forward pass that:

1. Runs the base ``Gemma4ForConditionalGeneration`` with
   ``return_shared_kv_states=True`` and ``output_hidden_states=True``.
2. Builds the drafter's ``inputs_embeds`` by concatenating the (already
   ``sqrt(H_b)``-scaled) base token embeddings with the base's final hidden
   state along the feature axis.
3. Runs the drafter ``Gemma4AssistantForCausalLM`` with the captured
   ``shared_kv_states`` and the concatenated embeddings.
4. Returns a :class:`Gemma4JointOutput` that exposes both base logits and a
   per-step list of drafter logits so the training recipe can compute
   ``L = L_base + drafter_loss_weight * sum_k L_drafter_k``.

Both sub-models are trainable. Gradients from the drafter loss flow back into
the base through:
- the "store" KV layers (last non-shared layer of each ``layer_type``) via
  ``shared_kv_states``;
- the base's input embedding (consumed by the drafter's first projection);
- the base's final hidden state.

This is the EAGLE-2 / Medusa-2 style co-training pattern: the drafter stays
aligned with a base that is itself moving.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

if TYPE_CHECKING:
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

logger = logging.getLogger(__name__)


@dataclass
class Gemma4JointOutput:
    """Output of :class:`Gemma4WithDrafter`.

    Attributes:
        logits: Base model logits ``[B, S, V]``.
        drafter_logits: Per-step list of drafter logits, each ``[B, S, V]``.
            For the default single-step recurrent cell this list has length 1.
        drafter_loss_weight: ``lambda`` multiplier the recipe applies to the
            drafter loss when summing it with the base loss.
        hidden_states: Optional list of base hidden states (mirrors HF).
        loss: Placeholder, populated by the recipe if needed.
    """

    logits: torch.Tensor
    drafter_logits: list[torch.Tensor] = field(default_factory=list)
    drafter_loss_weight: float = 1.0
    hidden_states: Optional[tuple] = None
    loss: Optional[torch.Tensor] = None


class Gemma4WithDrafter(nn.Module, HFCheckpointingMixin):
    """Composite model that wraps a Gemma 4 base + its released drafter.

    Both sub-modules are loaded via NeMo's ``NeMoAutoModel*`` paths so they
    receive the standard distributed infrastructure (FSDP2 sharding, freeze
    config, checkpoint loading, kernel patches, ...) independently. The
    composite is a thin :class:`nn.Module` that owns both and exposes a joint
    forward and a ``save_pretrained`` that writes the pair as two HF-format
    sub-directories (``base/`` and ``drafter/``).

    Args:
        base: Loaded base model (typically a ``Gemma4ForConditionalGeneration``
            instance returned by ``NeMoAutoModelForImageTextToText.from_pretrained``).
        drafter: Loaded drafter (a ``Gemma4DrafterForCausalLM`` instance
            returned by ``NeMoAutoModelForCausalLM.from_pretrained``).
        drafter_loss_weight: Multiplier ``lambda`` applied to the drafter loss
            in the recipe.
        drafter_num_steps: Number of recurrent drafter steps to run. Default 1
            matches the released checkpoint, which is a one-step recurrent cell.
    """

    supports_gradient_checkpointing = True

    def __init__(
        self,
        base: nn.Module,
        drafter: nn.Module,
        *,
        drafter_loss_weight: float = 1.0,
        drafter_num_steps: int = 1,
        freeze_base_for_drafter: bool = False,
        share_embedding_with_base: bool = False,
        base_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.base = base
        self.drafter = drafter
        self.drafter_loss_weight = float(drafter_loss_weight)
        self.drafter_num_steps = int(drafter_num_steps)
        self.freeze_base_for_drafter = bool(freeze_base_for_drafter)
        self.share_embedding_with_base = bool(share_embedding_with_base)
        self.base_activation_checkpointing = bool(base_activation_checkpointing)

        if self.drafter_num_steps != 1:
            raise NotImplementedError(
                "Gemma4WithDrafter currently supports drafter_num_steps=1 (matches the "
                f"released drafter's one-step recurrent cell). Got {self.drafter_num_steps}."
            )

        # Backbone hidden size used to build the drafter's pre-projection input.
        # The drafter's pre_projection layer expects ``2 * backbone_hidden_size``
        # features (concatenation of base embed and base final hidden state).
        base_text_config = self._get_base_text_config(base)
        drafter_config = getattr(drafter, "config", None)
        if drafter_config is not None and hasattr(drafter_config, "backbone_hidden_size"):
            assert drafter_config.backbone_hidden_size == base_text_config.hidden_size, (
                f"drafter.config.backbone_hidden_size ({drafter_config.backbone_hidden_size}) "
                f"must match base text_config.hidden_size ({base_text_config.hidden_size})"
            )

        if self.share_embedding_with_base:
            # One-shot init alignment: copy the base's input-embedding weight into
            # the drafter's ``embed_tokens``. The drafter's ``lm_head`` is tied to
            # its own ``embed_tokens`` (per HF ``_tied_weights_keys``), so the
            # ``lm_head`` row weights start aligned with the base's embedding too.
            # The two embeddings then evolve as independent parameters during
            # training (each accumulates its own gradient). True shared-storage
            # tying is intentionally avoided -- it conflicts with FSDP2's
            # per-module ownership model.
            base_embed = base.get_input_embeddings()
            drafter_embed = drafter.get_input_embeddings()
            if base_embed.weight.shape != drafter_embed.weight.shape:
                raise ValueError(
                    f"share_embedding_with_base=True but base embed_tokens shape "
                    f"{tuple(base_embed.weight.shape)} != drafter embed_tokens shape "
                    f"{tuple(drafter_embed.weight.shape)}. The two models must share a vocabulary."
                )
            with torch.no_grad():
                drafter_embed.weight.copy_(base_embed.weight)
            logger.info(
                "Gemma4WithDrafter: copied base embed_tokens into drafter at init (share_embedding_with_base=True)."
            )

        if self.freeze_base_for_drafter:
            for p in self.base.parameters():
                p.requires_grad_(False)
            logger.info("Gemma4WithDrafter: froze all base parameters (freeze_base_for_drafter=True).")

        if self.base_activation_checkpointing:
            enable_fn = getattr(self.base, "gradient_checkpointing_enable", None)
            if enable_fn is None:
                raise RuntimeError(
                    "base_activation_checkpointing=True but the base model does not expose "
                    "`gradient_checkpointing_enable`. Pass a HF-style model."
                )
            enable_fn()
            logger.info("Gemma4WithDrafter: enabled gradient checkpointing on the base.")

    @staticmethod
    def _get_base_text_config(base: nn.Module):
        cfg = getattr(base, "config", None)
        if cfg is None:
            raise ValueError("base model has no `config` attribute")
        return cfg.text_config if hasattr(cfg, "text_config") else cfg

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        base_path: Optional[str] = None,
        drafter_path: Optional[str] = None,
        *,
        pretrained_model_name_or_path: Optional[str] = None,
        drafter_loss_weight: float = 1.0,
        drafter_num_steps: int = 1,
        freeze_base_for_drafter: bool = False,
        share_embedding_with_base: bool = False,
        base_activation_checkpointing: bool = False,
        torch_dtype: Any = None,
        attn_implementation: Optional[str] = None,
        use_liger_kernel: Optional[bool] = None,
        use_sdpa_patching: Optional[bool] = None,
        text_config: Optional[dict] = None,
        peft_config: Any = None,
        device_mesh: Any = None,
        moe_mesh: Any = None,
        distributed_config: Any = None,
        pipeline_config: Any = None,
        freeze_config: Any = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Gemma4WithDrafter":
        """Build the composite by loading base and drafter via the NeMoAuto paths.

        Args:
            base_path: HF repo id or local path of the Gemma 4 base model.
            drafter_path: HF repo id or local path of the released drafter.
            pretrained_model_name_or_path: Alias for ``base_path``. Kept so that
                YAML configs can set ``pretrained_model_name_or_path`` and have
                the recipe's processor / checkpoint-config helpers (which read
                this key from the model config) keep working.
            drafter_loss_weight: ``lambda`` multiplier on the drafter loss.
            drafter_num_steps: Number of recurrent drafter steps. Currently
                must be 1.
            freeze_base_for_drafter: If True, freeze all base parameters so
                only the drafter is trained (drafter-only sub-case). Default
                False (joint training).
            share_embedding_with_base: If True, copy the base's input
                embedding into the drafter's ``embed_tokens`` once at init.
                The drafter's ``lm_head`` is tied to its own ``embed_tokens``
                so the row weights start aligned with the base too. The two
                embeddings then evolve as independent parameters during
                training.
            base_activation_checkpointing: If True, enable HF gradient
                checkpointing on the base to reduce activation memory.
                Important for the 4B + drafter + long-context setting.
            torch_dtype: dtype to use for both sub-models. Must be
                ``torch.bfloat16`` -- the drafter is bf16-only.
            attn_implementation: Forwarded to both sub-loads.
            use_liger_kernel: Forwarded to both sub-loads.
            use_sdpa_patching: Forwarded to both sub-loads.
            text_config: Optional overrides forwarded to the base load.
            peft_config: PEFT config (currently expected to be ``None`` --
                joint drafter PEFT is out of scope for the initial recipe).
            device_mesh: Distributed device mesh shared by base and drafter.
            moe_mesh: MoE mesh shared by base and drafter (drafter is dense).
            distributed_config: FSDP2 / Megatron-FSDP / DDP config object.
            pipeline_config: Must be ``None`` -- pipeline parallelism is not
                supported when the drafter is attached.
            freeze_config: Forwarded to the base only (the drafter is trained
                end-to-end). Customize the drafter's freezing with explicit
                ``requires_grad_`` calls on the returned composite if needed.
            cache_dir: HuggingFace cache directory.
            **kwargs: Additional kwargs forwarded to both sub-loads.

        Returns:
            An instantiated :class:`Gemma4WithDrafter`.
        """
        if base_path is None:
            base_path = pretrained_model_name_or_path
        if base_path is None:
            raise ValueError(
                "Gemma4WithDrafter.from_pretrained requires `base_path` "
                "(or `pretrained_model_name_or_path` as an alias)."
            )
        if drafter_path is None:
            raise ValueError("Gemma4WithDrafter.from_pretrained requires `drafter_path`.")

        if pipeline_config is not None:
            raise ValueError(
                "Pipeline parallelism is not supported with Gemma4WithDrafter "
                "(the KV-sharing path between base and drafter is not pipeline-safe). "
                "Set `pp_size: 1` in the distributed config."
            )
        if peft_config is not None:
            raise NotImplementedError(
                "PEFT (LoRA/QLoRA) for joint base + drafter fine-tuning is not "
                "supported yet. Run full SFT or open an issue."
            )
        if device_mesh is not None and "cp" in getattr(device_mesh, "mesh_dim_names", ()):
            if device_mesh["cp"].size() > 1:
                raise ValueError(
                    "Context parallelism is not supported with Gemma4WithDrafter "
                    "(the drafter's shared_kv_states path is not CP-safe). "
                    "Set `cp_size: 1` in the distributed config."
                )
        if torch_dtype is not None and torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Gemma4WithDrafter requires torch_dtype=torch.bfloat16 (the drafter is bf16-only). Got {torch_dtype}."
            )

        # Imported here to avoid a circular import at module load time.
        from nemo_automodel._transformers.auto_model import (
            NeMoAutoModelForCausalLM,
            NeMoAutoModelForImageTextToText,
        )

        base_kwargs = dict(kwargs)
        if torch_dtype is not None:
            base_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            base_kwargs["attn_implementation"] = attn_implementation
        if use_liger_kernel is not None:
            base_kwargs["use_liger_kernel"] = use_liger_kernel
        if use_sdpa_patching is not None:
            base_kwargs["use_sdpa_patching"] = use_sdpa_patching
        if text_config is not None:
            base_kwargs["text_config"] = text_config
        if cache_dir is not None:
            base_kwargs["cache_dir"] = cache_dir

        logger.info("Gemma4WithDrafter: loading base from %s", base_path)
        base = NeMoAutoModelForImageTextToText.from_pretrained(
            base_path,
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
            distributed_config=distributed_config,
            pipeline_config=None,
            freeze_config=freeze_config,
            **base_kwargs,
        )

        # Drafter is a text-only causal LM. Reuse the same mesh / dist config so
        # both sub-modules end up on the same FSDP2 axes. Strip multimodal /
        # text_config knobs that don't apply.
        drafter_kwargs = dict(kwargs)
        if torch_dtype is not None:
            drafter_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            drafter_kwargs["attn_implementation"] = attn_implementation
        if use_liger_kernel is not None:
            drafter_kwargs["use_liger_kernel"] = use_liger_kernel
        if use_sdpa_patching is not None:
            drafter_kwargs["use_sdpa_patching"] = use_sdpa_patching
        if cache_dir is not None:
            drafter_kwargs["cache_dir"] = cache_dir

        logger.info("Gemma4WithDrafter: loading drafter from %s", drafter_path)
        drafter = NeMoAutoModelForCausalLM.from_pretrained(
            drafter_path,
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
            distributed_config=distributed_config,
            pipeline_config=None,
            freeze_config=None,
            **drafter_kwargs,
        )

        return cls(
            base,
            drafter,
            drafter_loss_weight=drafter_loss_weight,
            drafter_num_steps=drafter_num_steps,
            freeze_base_for_drafter=freeze_base_for_drafter,
            share_embedding_with_base=share_embedding_with_base,
            base_activation_checkpointing=base_activation_checkpointing,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Gemma4JointOutput:
        """Joint forward: base first, then drafter consuming the base's outputs.

        Any extra kwargs (``pixel_values``, ``mm_token_type_ids``,
        ``pixel_values_videos``, ``input_features``, ...) are passed straight
        through to the base. Multimodal kwargs are *not* forwarded to the
        drafter (the drafter is text-only).
        """
        # Drop ``labels`` so the base doesn't compute its own internal loss;
        # the recipe handles loss computation against the returned logits.
        kwargs.pop("labels", None)
        # The recipe's MaskedCrossEntropy path doesn't use ``logits_to_keep``;
        # pop it just in case to keep the composite forward stable.
        kwargs.pop("logits_to_keep", None)

        # Note: do NOT force ``use_cache=False`` here. For Gemma 4 the
        # presence of a ``DynamicCache`` changes how sliding-window attention
        # masks are constructed (``masking_utils._preprocess_mask_arguments``
        # pulls ``is_sliding`` / ``get_mask_sizes`` from the cache). Without
        # the cache the SDPA mask-skip optimization can collapse sliding
        # layers into plain causal attention, which silently inflates the
        # initial training loss. The YAML sets ``text_config.use_cache: true``
        # so the cache gets allocated per forward and discarded; small
        # overhead, correct math.
        base_out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_shared_kv_states=True,
            **kwargs,
        )

        logits_base = base_out.logits
        hidden_states = getattr(base_out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Base model did not return `hidden_states`. Ensure `output_hidden_states=True` "
                "reaches the inner text model (most Gemma 4 base configs require "
                "`text_config.output_hidden_states=True`)."
            )
        h_final = hidden_states[-1]

        shared_kv = getattr(base_out, "shared_kv_states", None)
        if shared_kv is None:
            raise RuntimeError(
                "Base model did not return `shared_kv_states`. Ensure transformers TOT "
                "(>=5.8.0.dev) is installed and the base model is a Gemma 4 family "
                "checkpoint with KV-sharing enabled."
            )

        # The Gemma 4 input embedding already multiplies by sqrt(H_b) via
        # `Gemma4TextScaledWordEmbedding.embed_scale`, so we use its output as-is.
        base_embed_layer = self.base.get_input_embeddings()
        if input_ids is None:
            raise ValueError("Gemma4WithDrafter.forward requires `input_ids` (drafter consumes them).")
        embed = base_embed_layer(input_ids)
        inputs_embeds = torch.cat([embed, h_final], dim=-1)

        drafter_out = self.drafter(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            shared_kv_states=shared_kv,
        )

        return Gemma4JointOutput(
            logits=logits_base,
            drafter_logits=[drafter_out.logits],
            drafter_loss_weight=self.drafter_loss_weight,
            hidden_states=hidden_states,
        )

    # ------------------------------------------------------------------
    # Property pass-throughs (so VLM-specific code paths keep working)
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Module:
        return self.base.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.base.get_output_embeddings()

    @property
    def config(self):
        return self.base.config

    @property
    def vision_tower(self):
        return getattr(self.base, "vision_tower", None)

    @property
    def audio_tower(self):
        return getattr(self.base, "audio_tower", None)

    @property
    def language_model(self):
        return getattr(self.base, "language_model", None)

    def get_rope_index(self, *args, **kwargs):
        fn = getattr(self.base, "get_rope_index", None)
        if fn is None:
            raise AttributeError("base model does not expose `get_rope_index`")
        return fn(*args, **kwargs)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_pretrained(
        self,
        save_directory: str,
        checkpointer: Optional["Checkpointer"] = None,
        tokenizer: Any = None,
        **kwargs,
    ) -> None:
        """Save base and drafter as two HF-format sub-directories.

        Produces ``<save_directory>/base/`` and ``<save_directory>/drafter/``
        with HF-compatible artifacts. Each side can later be loaded back by HF
        ``from_pretrained`` independently (vLLM compatibility).
        """
        if checkpointer is None:
            raise ValueError(
                "Gemma4WithDrafter.save_pretrained requires `checkpointer`. The recipe "
                "should pass its `self.checkpointer` instance."
            )

        base_dir = os.path.join(save_directory, "base")
        drafter_dir = os.path.join(save_directory, "drafter")

        # Each sub-module already inherits HFCheckpointingMixin (via NeMo's
        # custom classes) and can be saved via Checkpointer.save_model.
        checkpointer.save_model(
            model=self.base,
            weights_path=base_dir,
            peft_config=kwargs.get("peft_config", None),
            tokenizer=tokenizer,
        )
        checkpointer.save_model(
            model=self.drafter,
            weights_path=drafter_dir,
            peft_config=None,
            tokenizer=tokenizer,
        )


__all__ = ["Gemma4JointOutput", "Gemma4WithDrafter"]
