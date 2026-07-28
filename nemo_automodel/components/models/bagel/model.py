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

"""Top-level BAGEL model.

The module supports Stage 1 understanding-only CE and Stage 2 joint
understanding and visual generation with flow-matching MSE. The VAE itself
stays outside this module tree; the recipe passes VAE-encoded latents to
``forward``.

Top-level module is ``nn.Module`` (not ``PreTrainedModel``) + mixed-in
``HFCheckpointingMixin`` — matches the llava-onevision pattern and avoids the
FSDP double-root issue that bites us with PreTrainedModel-derived roots.
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from nemo_automodel.components.models.bagel.attention_masks import create_sparse_mask
from nemo_automodel.components.models.bagel.configuration import (
    BagelBackendConfig,
    BagelConfig,
    resolve_bagel_backend,
)
from nemo_automodel.components.models.bagel.connector import BagelMultiModalProjector
from nemo_automodel.components.models.bagel.embeddings import BagelGridPositionEmbedding, BagelTimestepEmbedding
from nemo_automodel.components.models.bagel.modeling_qwen2_packed import Qwen2ForCausalLM
from nemo_automodel.components.models.bagel.modeling_siglip_navit import (
    SiglipVisionModel,
)
from nemo_automodel.components.models.bagel.state_dict_adapter import (
    BagelStateDictAdapter,
    load_bagel_checkpoint_state_dict,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)

logger = logging.getLogger(__name__)


def _stage_to_int(stage: Union[int, str]) -> int:
    """Normalize a BAGEL training stage value to ``1`` or ``2``."""
    if isinstance(stage, int) or str(stage).isdigit():
        stage_int = int(stage)
    else:
        stage_str = str(stage).lower()
        if stage_str == "stage1":
            stage_int = 1
        elif stage_str == "stage2":
            stage_int = 2
        else:
            raise ValueError(f"Unknown BAGEL stage {stage!r}; expected 1, 2, 'stage1', or 'stage2'.")
    if stage_int not in (1, 2):
        raise ValueError(f"Unknown BAGEL stage {stage!r}; expected 1 or 2.")
    return stage_int


def _prepare_config_for_stage(config: BagelConfig) -> None:
    """Apply BAGEL stage/checkpoint config fixes before module construction.

    AutoModel instantiates custom models as ``model_cls(config)`` and lets the
    common checkpointer load weights later.  BAGEL therefore needs the same
    stage-dependent config mutations that its direct ``from_pretrained`` path
    used to do before ``BagelModel`` is built.
    """
    stage = getattr(config, "stage", None)
    if stage is not None:
        stage_int = _stage_to_int(stage)
        config.stage = stage_int
        if stage_int == 1:
            config.visual_gen = False
            if getattr(config.text_config, "layer_module", None) is None:
                config.text_config.layer_module = "Qwen2DecoderLayer"
        else:
            config.visual_gen = True
            # Stage 2 must use the MoT decoder so ``*_moe_gen`` checkpoint
            # keys have slots in the module tree. Some serialized configs may
            # carry ``layer_module=null``.
            config.text_config.layer_module = "Qwen2MoTDecoderLayer"
    elif getattr(config.text_config, "layer_module", None) is None:
        config.text_config.layer_module = "Qwen2MoTDecoderLayer" if config.visual_gen else "Qwen2DecoderLayer"

    # BAGEL applies a select-layer offset so the effective ViT has
    # ``vit_config.num_hidden_layers + 1 + vit_select_layer`` layers. The
    # checkpoint carries the offset count, so apply this before constructing
    # the tower. Mark the config to avoid accidental double-application.
    if config.vision_config is not None:
        if not getattr(config, "_bagel_vit_select_layer_applied", False):
            effective_layers = config.vision_config.num_hidden_layers + 1 + config.vit_select_layer
            if effective_layers != config.vision_config.num_hidden_layers:
                logger.info(
                    "BagelConfig: adjusting vision_config.num_hidden_layers %d -> %d (vit_select_layer=%d).",
                    config.vision_config.num_hidden_layers,
                    effective_layers,
                    config.vit_select_layer,
                )
                config.vision_config.num_hidden_layers = effective_layers
            config._bagel_vit_select_layer_applied = True
        config.vision_config.rope = config.vit_rope


def _convert_patch_embedding_for_packed_vit(model: "BagelModel", config: BagelConfig) -> None:
    """Swap SigLIP patch embedding to Linear for BAGEL packed pixel inputs."""
    if not config.visual_und:
        return
    embeddings = model.vit_model.vision_model.embeddings
    if isinstance(embeddings.patch_embedding, nn.Conv2d):
        embeddings.convert_conv2d_to_linear(
            config.vision_config,
            meta=embeddings.patch_embedding.weight.is_meta,
        )


class BagelModel(nn.Module):
    """Plain container for the three BAGEL submodules.

    Attribute names (``language_model``, ``vit_model``, ``connector``,
    ``vit_pos_embed``) match the checkpoint layout so the state-dict adapter
    maps identity. There's no forward logic here - this class exists so that
    FSDP / state-dict tooling sees the expected tree structure without being
    confused by the ``HFCheckpointingMixin`` root.

    When ``config.visual_gen=True`` (Stage 2), we additionally attach the
    generation-side siblings (``time_embedder``, ``vae2llm``, ``llm2vae``,
    ``latent_pos_embed``) so the flow-matching head is ready to run. The VAE
    model itself is NOT owned here; the recipe keeps it separate
    (frozen, inference-only) and passes already-encoded latents into
    ``BagelForUnifiedMultimodal.forward``.
    """

    def __init__(self, config: BagelConfig, backend: Optional[BagelBackendConfig] = None) -> None:
        super().__init__()
        self.config = config
        self.backend = resolve_bagel_backend(backend)

        # Text backbone - always present.
        self.language_model = Qwen2ForCausalLM(config.text_config, backend=self.backend)

        # Understanding-side vision path. Built whenever the BAGEL config keeps
        # visual understanding enabled.
        if config.visual_und:
            if config.vision_config is None:
                raise ValueError(
                    "BagelConfig.visual_und=True but vision_config is None. "
                    "Provide a SiglipVisionConfig (or dict) for the ViT tower."
                )
            self.vit_model = SiglipVisionModel(config.vision_config)
            self.connector = BagelMultiModalProjector(
                in_dim=config.vision_config.hidden_size,
                out_dim=config.text_config.hidden_size,
                hidden_act=config.connector_act,
            )
            self.vit_pos_embed = BagelGridPositionEmbedding(
                max_num_patch_per_side=config.vit_max_num_patch_per_side,
                hidden_size=config.text_config.hidden_size,
            )

        # Generation-side: flow-matching latent path. Built only when
        # ``config.visual_gen=True``. The VAE model itself lives outside this
        # container; the recipe owns the frozen VAE and hands ``padded_latent``
        # to forward.
        if config.visual_gen:
            vae_cfg = config.vae_config or {}
            try:
                latent_channel = int(vae_cfg["z_channels"])
                downsample = int(vae_cfg["downsample"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    "visual_gen=True requires config.vae_config to carry "
                    "'z_channels' and 'downsample'. Recipe should load the VAE "
                    f"checkpoint first and populate these. (got {vae_cfg!r})"
                ) from e
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = latent_channel
            self.patch_latent_dim = config.latent_patch_size**2 * latent_channel
            self.time_embedder = BagelTimestepEmbedding(config.text_config.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, config.text_config.hidden_size)
            self.llm2vae = nn.Linear(config.text_config.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = BagelGridPositionEmbedding(
                max_num_patch_per_side=config.max_latent_size,
                hidden_size=config.text_config.hidden_size,
            )
            # Zero-init llm2vae so Stage 2 starts with the MSE head
            # contributing zero to the loss.
            nn.init.constant_(self.llm2vae.weight, 0.0)
            nn.init.constant_(self.llm2vae.bias, 0.0)


class BagelForUnifiedMultimodal(HFCheckpointingMixin, nn.Module):
    """BAGEL mixed-modal LLM wrapper for understanding and optional generation.

    ``visual_gen=False`` gives the Stage 1 understanding-only path. Stage 2
    sets ``visual_gen=True`` and uses the MoT ``*_moe_gen`` parameter siblings,
    VAE latents prepared by the recipe, and the flow-matching MSE head.
    """

    config_class = BagelConfig
    # BAGEL's served checkpoints are untied; the tie flag lives on the nested
    # text_config (aliased as llm_config), and the inner Qwen2 LM owns the head.
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    backend_config_resolver = staticmethod(resolve_bagel_backend)

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: BagelConfig, backend: Optional[BagelBackendConfig] = None) -> None:
        super().__init__()
        # Also covers the build_bagel_from_hf_backbones registry-bypass path, which
        # constructs this class directly. Reads the nested text_config tie flag via
        # the resolver's get_text_config fallback (BagelConfig has no top-level flag).
        reject_unsupported_tie_word_embeddings(type(self), config)
        _prepare_config_for_stage(config)
        self.config = config
        self.backend = resolve_bagel_backend(backend)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                "Resolved BAGEL backends: linear=%s, rms_norm=%s",
                self.backend.linear,
                self.backend.rms_norm,
            )
        self.model = BagelModel(config, backend=self.backend)
        _convert_patch_embedding_for_packed_vit(self.model, config)

        # Light state-dict adapter hook used by HFCheckpointingMixin /
        # Checkpointer for HF <-> NeMo renames. Identity today.
        self.state_dict_adapter = BagelStateDictAdapter(
            config=config,
            stage="stage1" if not config.visual_gen else "stage2",
        )

        # Convenience: cached scalars mirrored from the nested text config.
        self.hidden_size = config.text_config.hidden_size
        self.num_heads = config.text_config.num_attention_heads
        self.use_moe = "Mo" in getattr(config.text_config, "layer_module", "Qwen2DecoderLayer")

    def initialize_weights(self) -> None:
        """Initialize BAGEL weights after AM materializes a ``from_config`` model.

        ``BagelForUnifiedMultimodal`` is an ``nn.Module`` root, not a HF
        ``PreTrainedModel`` root. AM's meta-device ``from_config`` path
        materializes parameters after sharding and then calls this method.
        Delegate Qwen/SigLIP subtrees to their HF-style initializers, then
        initialize BAGEL-only connector and generation modules.
        """

        def _mark_hf_uninitialized(module: nn.Module) -> None:
            if hasattr(module, "_is_hf_initialized"):
                module._is_hf_initialized = False

        def _initialize_hf_subtree(module: nn.Module, name: str) -> None:
            module.apply(_mark_hf_uninitialized)
            if hasattr(module, "post_init"):
                module.post_init()
            elif hasattr(module, "init_weights"):
                module.init_weights()
            elif hasattr(module, "_init_weights"):
                module.apply(module._init_weights)
            else:
                raise AttributeError(f"{name} does not provide a HF-compatible weight initializer")

        _initialize_hf_subtree(self.model.language_model, "language_model")

        if self.config.visual_und:
            _initialize_hf_subtree(self.model.vit_model, "vit_model")

        def _init_bagel_module(module: nn.Module) -> None:
            if isinstance(module, BagelGridPositionEmbedding):
                module._init_weights()
            elif isinstance(module, nn.Linear):
                module.reset_parameters()

        bagel_only_modules = []
        for name in ("connector", "vit_pos_embed", "time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"):
            module = getattr(self.model, name, None)
            if module is not None:
                bagel_only_modules.append(module)
        for module in bagel_only_modules:
            module.apply(_init_bagel_module)

        if self.config.visual_gen:
            nn.init.constant_(self.model.llm2vae.weight, 0.0)
            nn.init.constant_(self.model.llm2vae.bias, 0.0)

    # ------------------------------------------------------------------
    # Checkpoint loading.
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *,
        stage: Union[int, str] = 1,
        strict: bool = False,
        **kwargs: Any,
    ) -> "BagelForUnifiedMultimodal":
        """Load a BAGEL-7B-MoT checkpoint directory into this class.

        Reads ``config.json`` via :meth:`BagelConfig.from_pretrained`, constructs
        an empty model, and then loads ``ema.safetensors`` filtered by
        :class:`BagelStateDictAdapter`. Stage 2 VAE weights are loaded by the
        recipe because the VAE is not owned by this module tree.

        Args:
            pretrained_model_name_or_path: Directory containing the HF-layout
                BAGEL checkpoint.
            stage: ``1`` (UND only) or ``2`` (UND + GEN). Strings
                ``"stage1"`` / ``"stage2"`` are also accepted.
            strict: If ``True``, raise on state-dict keys that don't match the
                adapter patterns. Defaults to ``False`` for compatibility with
                checkpoint sidecar files.
            **kwargs: Model overrides. ``backend`` is passed to the BAGEL model
                constructor; the remaining values are forwarded to
                ``BagelConfig.from_pretrained``.

        Returns:
            A fully-initialized ``BagelForUnifiedMultimodal`` with weights
            populated from disk. For Stage 1, ``visual_gen`` is forced off on
            the loaded config so the MoT gen-side path is left untouched.
        """
        path = pathlib.Path(pretrained_model_name_or_path)

        backend = kwargs.pop("backend", None)
        cfg = BagelConfig.from_pretrained(str(path), **kwargs)
        cfg.stage = stage
        stage_int = _stage_to_int(stage)

        model = cls(cfg, backend=backend)

        sd = load_bagel_checkpoint_state_dict(str(path), stage=stage_int, strict=strict)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            # vit_pos_embed.pos_embed is frozen-init'd; missing keys are
            # informational only with strict=False.
            logger.info("from_pretrained: %d missing key(s) after load (e.g. %s)", len(missing), missing[:5])
        if unexpected:
            logger.info("from_pretrained: %d unexpected key(s) after load (e.g. %s)", len(unexpected), unexpected[:5])

        return model

    # ------------------------------------------------------------------
    # Registry hook.
    # ------------------------------------------------------------------
    @classmethod
    def supports_config(cls, config: Any) -> bool:
        """Return ``True`` if this custom class supports ``config``."""
        return getattr(config, "model_type", None) == BagelConfig.model_type

    # ------------------------------------------------------------------
    # Convenience accessors used by the parallelizer / state-dict adapter.
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.model.language_model.get_output_embeddings()

    # ------------------------------------------------------------------
    # Forward (understanding + optional generation).
    # ------------------------------------------------------------------
    def forward(
        self,
        # --- always ---
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: Optional[List[torch.Tensor]] = None,
        split_lens: Optional[List[int]] = None,
        attn_modes: Optional[List[str]] = None,
        # --- understanding branch ---
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.Tensor] = None,
        # --- generation branch (Stage 2) ---
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.Tensor] = None,
        mse_loss_indexes: Optional[torch.Tensor] = None,
        # --- loss ---
        ce_loss_indexes: Optional[torch.Tensor] = None,
        packed_label_ids: Optional[torch.Tensor] = None,
        ce_loss_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run the BAGEL mixed-modal forward.

        Stage 1 (``visual_gen=False``) skips the flow-matching branch and MSE
        computation; Stage 2 activates both. ``ce_loss_weights`` is accepted
        for data-pipeline compatibility but not consumed here - CE is returned
        per-token (``reduction="none"``) and the trainer may apply weights
        downstream.

        Stage 2 inputs (``padded_latent``, ``patchified_vae_latent_shapes``,
        ``packed_latent_position_ids``, ``packed_vae_token_indexes``,
        ``packed_timesteps``, ``mse_loss_indexes``) are produced by the
        BAGEL collator when a pack contains t2i/edit samples. ``padded_latent``
        is the VAE-encoded latent tensor; recipe must call ``vae_model.encode``
        on the raw ``padded_images`` before forward; this module does not own
        the VAE.

        Returns:
            ``dict(ce=Tensor|None, mse=Tensor|None)`` - both can be ``None``
            when the pack has no samples of that loss type. Shape-stable with
            the BAGEL packed-training path.
        """
        mdl = self.model
        lm = mdl.language_model

        # --- build packed token embeddings ---
        packed_text_embedding = lm.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # --- MLM-style grouped routing: build the und/gen permutation ONCE ---
        # Reorder the packed sequence so und tokens occupy ``[:Lund]`` and gen tokens
        # ``[Lund:]`` (contiguous blocks). Every MoT layer then routes the *pointwise* path
        # (LN / MLP / QKV / O-proj) by a scalar slice boundary instead of per-layer
        # gather/scatter. Crucially — like MLM — attention still runs in ORIGINAL token order
        # (so its block-diagonal mask stays block-sparse): each attention layer restores the
        # original order via ``mot_inv`` before the kernel and re-groups via ``mot_perm``
        # after. The output is un-permuted at exit so loss/downstream stay in original order.
        mot_perm = None
        mot_inv = None
        if (
            self.model.backend.mot_grouped
            and self.use_moe
            and self.config.visual_gen
            and packed_vae_token_indexes is not None
        ):
            dev = packed_sequence.device
            und = packed_text_indexes
            if packed_vit_token_indexes is not None:
                und = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            gen = packed_vae_token_indexes
            covered = torch.zeros(sequence_length, dtype=torch.bool, device=dev)
            covered[und] = True
            covered[gen] = True
            rest = torch.nonzero(~covered, as_tuple=False).flatten()
            mot_perm = torch.cat([und, gen, rest])
            mot_inv = torch.empty(sequence_length, dtype=torch.long, device=dev)
            mot_inv[mot_perm] = torch.arange(sequence_length, device=dev)

        # --- attention mask (always ORIGINAL packed order; grouped mode restores order in-layer) ---
        if nested_attention_masks is None:
            if split_lens is None or attn_modes is None:
                raise ValueError(
                    "BagelForUnifiedMultimodal.forward: nested_attention_masks is None, so split_lens and "
                    "attn_modes are required to build a flex-attention BlockMask."
                )
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            attention_mask = create_block_mask(
                sparse_mask,
                B=1,
                H=self.num_heads,
                Q_LEN=seqlen,
                KV_LEN=seqlen,
                device=packed_text_embedding.device,
                BLOCK_SIZE=128,
                _compile=True,
            )
        else:
            attention_mask = nested_attention_masks

        # --- understanding branch: ViT tower + connector + 2D pos embed ---
        if self.config.visual_und and packed_vit_tokens is not None:
            if vit_token_seqlens is None or packed_vit_position_ids is None or packed_vit_token_indexes is None:
                raise ValueError(
                    "visual_und=True but one of vit_token_seqlens / packed_vit_position_ids / "
                    "packed_vit_token_indexes is None."
                )
            cu_seqlens = F.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
            max_seqlen = int(torch.max(vit_token_seqlens).item())
            packed_vit_token_embed = mdl.vit_model(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = mdl.connector(packed_vit_token_embed)
            vit_token_pos_emb = mdl.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            if packed_vit_token_embed.dtype != packed_sequence.dtype:
                packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        # --- generation branch: VAE latent -> flow-matching input (Stage 2) ---
        # Keep ``packed_latent_clean`` and ``noise`` around because the MSE
        # computation below uses them to form the
        # flow-matching velocity target ``v_t = noise - clean``.
        packed_latent_clean: Optional[torch.Tensor] = None
        noise: Optional[torch.Tensor] = None
        if self.config.visual_gen and padded_latent is not None:
            if (
                patchified_vae_latent_shapes is None
                or packed_latent_position_ids is None
                or packed_vae_token_indexes is None
                or packed_timesteps is None
            ):
                raise ValueError(
                    "visual_gen=True with padded_latent but one of "
                    "patchified_vae_latent_shapes / packed_latent_position_ids / "
                    "packed_vae_token_indexes / packed_timesteps is None."
                )
            p = mdl.latent_patch_size
            packed_latent_list: List[torch.Tensor] = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, : h * p, : w * p].reshape(mdl.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * mdl.latent_channel)
                packed_latent_list.append(latent)
            packed_latent_clean = torch.cat(packed_latent_list, dim=0)
            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = mdl.timestep_shift * packed_timesteps / (1 + (mdl.timestep_shift - 1) * packed_timesteps)
            packed_latent_tokens = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[
                :, None
            ] * noise
            packed_timestep_embeds = mdl.time_embedder(packed_timesteps)
            latent_token_pos_emb = mdl.latent_pos_embed(packed_latent_position_ids)
            packed_latent_tokens = mdl.vae2llm(packed_latent_tokens) + packed_timestep_embeds + latent_token_pos_emb
            if packed_latent_tokens.dtype != packed_sequence.dtype:
                packed_latent_tokens = packed_latent_tokens.to(packed_sequence.dtype)
            packed_sequence[packed_vae_token_indexes] = packed_latent_tokens

        # --- MoT index prep (only meaningful when use_moe=True) ---
        extra_inputs: Dict[str, torch.Tensor] = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs["packed_und_token_indexes"] = packed_und_token_indexes
            # Stage 2: route VAE tokens through the gen expert. Stage 1:
            # no VAE tokens — Qwen2Model.forward_train tolerates None and
            # treats the gen-side expert as dormant.
            extra_inputs["packed_gen_token_indexes"] = packed_vae_token_indexes if self.config.visual_gen else None
            if mot_perm is not None:
                # Threaded to every MoT layer so attention can restore/re-group token order.
                extra_inputs["mot_perm"] = mot_perm
                extra_inputs["mot_inv"] = mot_inv

        # --- apply the grouped permutation to the fully-populated sequence + RoPE ids ---
        # In grouped mode the MoT layers read only ``packed_und_token_indexes.shape[0]`` as
        # the und/gen slice boundary, so the (now-stale) index values are unused downstream.
        if mot_perm is not None:
            packed_sequence = packed_sequence[mot_perm]
            packed_position_ids = packed_position_ids[mot_perm]

        # --- LM forward ---
        # Route through the language_model's train/inference dispatcher: at
        # training time this calls Qwen2Model.forward_train; at eval, it can
        # call forward_inference.
        last_hidden_state = lm(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        # --- undo the grouped permutation so loss/downstream see original packed order ---
        if mot_inv is not None:
            last_hidden_state = last_hidden_state[mot_inv]

        # --- loss: CE (understanding) ---
        ce: Optional[torch.Tensor] = None
        if ce_loss_indexes is not None:
            if packed_label_ids is None:
                raise ValueError("ce_loss_indexes was provided but packed_label_ids is None.")
            packed_ce_preds = lm.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        # --- loss: MSE (flow-matching, Stage 2) ---
        # The per-token timestep list includes ``-inf`` sentinels for VAE
        # conditioning tokens (loss=0); after ``torch.sigmoid`` those become
        # 0 exactly, so
        # ``packed_timesteps > 0`` filters them out and only loss=1 positions
        # contribute to MSE.
        mse: Optional[torch.Tensor] = None
        if (
            self.config.visual_gen
            and mse_loss_indexes is not None
            and packed_latent_clean is not None
            and noise is not None
        ):
            packed_mse_preds = mdl.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean  # flow-matching velocity (data -> noise)
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        return {"mse": mse, "ce": ce}
