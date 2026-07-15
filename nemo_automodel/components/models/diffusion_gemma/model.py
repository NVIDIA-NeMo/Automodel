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

"""NeMo Automodel support for ``diffusion_gemma`` (block diffusion).

Architecture (design v2 item 1) — ONE shared parameter stack run twice:
    * Run the decoder layers once **causally** over the clean full sequence to
      build a per-layer read-only KV cache (the "encoder" KV). The text encoder
      is causal because ``use_bidirectional_attention == "vision"`` (not
      ``"all"``); a single causal pass over the clean full sequence reproduces
      the per-position KV that block-by-block inference builds.
    * Run the same layers once **bidirectionally** over the noised canvas (the
      response region), each layer concatenating ``[encoder_KV ; canvas_KV]`` on
      the key axis and using the block-causal training mask from
      ``attention_mask.build_block_diffusion_training_mask``.

A single shared stack (rather than tied-but-separate encoder/decoder modules)
keeps the model visible to AM's MoE FSDP grad-sync (``MoEFSDPSyncMixin`` /
``_iter_fsdp_modules`` assume a single ``model.layers`` stack with
``block.moe.experts``) and avoids FSDP2 double-sharding tied storage. The
``lm_head`` is tied to ``model.embed_tokens``.

Self-conditioning (decoder-only, Analog-Bits two-pass) is encapsulated in the
training forward so the recipe still calls ``model(**batch)`` once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
    DiffusionGemmaConfig,
    DiffusionGemmaTextConfig,
)
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaTextScaledWordEmbedding as ScaledWordEmbedding,
)

from nemo_automodel._transformers.model_capabilities import ModelCapabilities
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin

from .fsdp import register_diffusion_gemma_parallel_strategy
from .layers import (
    DiffusionGemmaMoEDecoderLayer,
    DiffusionGemmaRMSNorm,
    DiffusionGemmaSelfConditioning,
    DiffusionGemmaTextRotaryEmbedding,
    _build_moe_config,
)


@dataclass
class DiffusionGemmaOutput:
    """Training forward output.

    ``logits`` are the canvas-only (response) denoising logits ``[B, canvas_len, V]``.
    ``encoder_logits`` are the causal encoder's next-token logits over the clean
    full sequence ``[B, S, V]`` for the co-trained AR loss — ``None`` outside
    training (and when the AR loss is unused).
    """

    logits: "torch.Tensor"
    encoder_logits: "torch.Tensor | None" = None


def _make_causal_additive_mask(
    seq_len: int,
    *,
    padding_mask: torch.Tensor | None,
    sliding_window: int | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive causal (optionally sliding-window) mask for the encoder.

    Shape ``[B, 1, seq_len, seq_len]``; ``0`` keep, ``finfo.min`` masked.
    ``padding_mask`` is ``[B, seq_len]`` with ``True`` at padding positions.
    """
    q = torch.arange(seq_len, device=device)
    keep = q[:, None] >= q[None, :]  # causal: query attends to <= key
    if sliding_window is not None:
        keep = keep & ((q[:, None] - q[None, :]) < sliding_window)
    keep = keep[None, None].expand(batch_size, 1, seq_len, seq_len).clone()
    if padding_mask is not None:
        keep = keep & (~padding_mask)[:, None, None, :]
    additive = torch.zeros(keep.shape, dtype=dtype, device=device)
    additive.masked_fill_(~keep, torch.finfo(dtype).min)
    return additive


class DiffusionGemmaBackbone(nn.Module):
    """Single shared Gemma MoE transformer stack run causally then bidirectionally.

    Exposes ``layers`` (a ``ModuleDict`` keyed by string layer index),
    ``embed_tokens``, ``norm``, ``self_conditioning`` and ``rotary_emb``. The
    ``layers`` / ``embed_tokens`` names are what ``MoEFSDPSyncMixin`` and the
    FSDP2 sharding path key on.
    """

    def __init__(
        self,
        config: DiffusionGemmaTextConfig,
        backend: BackendConfig,
        moe_config: MoEConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.backend = backend
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.layer_types = config.layer_types

        self.moe_config = _build_moe_config(config, moe_config)

        self.embed_tokens = ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): DiffusionGemmaMoEDecoderLayer(config, layer_idx, self.moe_config, backend)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = DiffusionGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DiffusionGemmaTextRotaryEmbedding(config)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def _position_embeddings(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> dict:
        return {lt: self.rotary_emb(hidden_states, position_ids, lt) for lt in set(self.layer_types)}

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        padding_mask: torch.Tensor | None,
        return_hidden: bool = False,
    ):
        """Causal pass over the clean full sequence -> per-layer (K, V) cache.

        When ``return_hidden`` is True, also returns the final **normed** hidden
        states ``[B, S, H]`` (so the caller can produce the encoder's
        autoregressive logits for the co-trained AR loss). Default False keeps
        the KV-only contract used by inference and the parity/leakage tests.
        """
        inputs_embeds = self.embed_tokens(input_ids)
        seq_len = inputs_embeds.shape[1]
        batch_size = inputs_embeds.shape[0]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        full_mask = _make_causal_additive_mask(
            seq_len,
            padding_mask=padding_mask,
            sliding_window=None,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        sliding_mask = _make_causal_additive_mask(
            seq_len,
            padding_mask=padding_mask,
            sliding_window=self.config.sliding_window,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        masks = {"full_attention": full_mask, "sliding_attention": sliding_mask}
        position_embeddings = self._position_embeddings(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        encoder_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers.values():
            hidden_states, layer_kv = layer(
                hidden_states,
                position_embeddings=position_embeddings[layer.attention_type],
                attention_mask=masks[layer.attention_type],
                encoder_kv=None,
                padding_mask=padding_mask,
            )
            encoder_kv.append(layer_kv)
        if return_hidden:
            return encoder_kv, self.norm(hidden_states)
        return encoder_kv

    def decode(
        self,
        canvas_ids: torch.Tensor,
        *,
        encoder_kv: list[tuple[torch.Tensor, torch.Tensor]],
        decoder_position_ids: torch.Tensor,
        decoder_masks: dict,
        decoder_padding_mask: torch.Tensor | None = None,
        self_conditioning_logits: torch.Tensor | None = None,
        self_conditioning_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bidirectional pass over the noised canvas with cross-attention to the
        encoder KV cache. Returns the final (normed) hidden states.

        ``self_conditioning_mask`` (``[B]`` bool, training only) gates the self-cond
        branch PER EXAMPLE: examples with ``False`` get a zeroed soft-embedding
        (identical to the no-self-cond path), so a single always-on pass-1 can serve
        Google's per-example conditioned / zero-conditioned mix.
        """
        inputs_embeds = self.embed_tokens(canvas_ids)

        if self_conditioning_logits is not None:
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(self.embed_tokens.weight.dtype),
                self.embed_tokens.weight,
            ) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
            if self_conditioning_mask is not None:
                soft_embeddings = soft_embeddings * self_conditioning_mask.to(soft_embeddings.dtype)[:, None, None]
        else:
            soft_embeddings = torch.zeros_like(inputs_embeds)
        inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)

        position_embeddings = self._position_embeddings(inputs_embeds, decoder_position_ids)
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers.values()):
            hidden_states, _ = layer(
                hidden_states,
                position_embeddings=position_embeddings[layer.attention_type],
                attention_mask=decoder_masks[layer.attention_type],
                encoder_kv=encoder_kv[i],
                padding_mask=decoder_padding_mask,
            )
        return self.norm(hidden_states)

    def forward(
        self,
        *,
        mode: str,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        canvas_ids: torch.Tensor | None = None,
        encoder_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        decoder_position_ids: torch.Tensor | None = None,
        decoder_masks: dict | None = None,
        decoder_padding_mask: torch.Tensor | None = None,
        self_conditioning_logits: torch.Tensor | None = None,
        self_conditioning_mask: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        """Dispatch encode/decode through ``nn.Module.__call__`` for FSDP hooks.

        FSDP2 hooks are installed on module calls, not on arbitrary helper
        methods. The block-diffusion top-level forward must therefore enter the
        backbone via ``self.model(...)`` so root-owned parameters such as
        ``self_conditioning`` and the final ``norm`` are gathered before use.
        """
        if mode == "encode":
            if input_ids is None:
                raise ValueError("DiffusionGemmaBackbone.forward(mode='encode') requires input_ids.")
            if position_ids is None:
                raise ValueError("DiffusionGemmaBackbone.forward(mode='encode') requires position_ids.")
            return self.encode(
                input_ids,
                position_ids=position_ids,
                padding_mask=padding_mask,
                return_hidden=return_hidden,
            )
        if mode == "decode":
            if canvas_ids is None:
                raise ValueError("DiffusionGemmaBackbone.forward(mode='decode') requires canvas_ids.")
            if encoder_kv is None:
                raise ValueError("DiffusionGemmaBackbone.forward(mode='decode') requires encoder_kv.")
            if decoder_position_ids is None:
                raise ValueError("DiffusionGemmaBackbone.forward(mode='decode') requires decoder_position_ids.")
            if decoder_masks is None:
                raise ValueError("DiffusionGemmaBackbone.forward(mode='decode') requires decoder_masks.")
            return self.decode(
                canvas_ids,
                encoder_kv=encoder_kv,
                decoder_position_ids=decoder_position_ids,
                decoder_masks=decoder_masks,
                decoder_padding_mask=decoder_padding_mask,
                self_conditioning_logits=self_conditioning_logits,
                self_conditioning_mask=self_conditioning_mask,
            )
        raise ValueError(f"Unsupported DiffusionGemmaBackbone.forward mode: {mode!r}")


class DiffusionGemmaForBlockDiffusion(HFCheckpointingMixin, MoEFSDPSyncMixin, PreTrainedModel):
    """Block-diffusion Gemma MoE model for SFT.

    Inherits the AM checkpointing + MoE-FSDP machinery. The MoE backbone is
    reused from ``gemma4_moe``; the diffusion training forward and the two-pass
    self-conditioning are new. See module docstring for the single-shared-stack
    design.

    ``forward`` is the SFT **training** forward. A generation/inference loop
    (encode the prompt once, then iteratively denoise canvas blocks reusing the
    KV cache, with the self-conditioning recycling loop) is deferred; the
    ``model.encode`` / ``model.decode`` building blocks are the reusable pieces
    for it, and ``forward`` already accepts an explicit ``self_conditioning_logits``
    for the per-step inference contract.
    """

    config_class = DiffusionGemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # RoPE inv_freq must stay fp32: initialize_weights casts the model to bf16 and
    # nn.Module.to rounds floating buffers; cast_model_to_dtype restores keep-fp32
    # modules afterwards (see llama/rope_utils.py).
    _keep_in_fp32_modules = ["rotary_emb"]
    _no_split_modules = ["DiffusionGemmaMoEDecoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]

    @classmethod
    def get_capabilities(cls, config: "DiffusionGemmaConfig") -> "ModelCapabilities":
        """Parallelism support for the DiffusionGemma block-diffusion MoE.

        Single variant: FSDP2 + Expert Parallelism are supported (validated at
        EP=8). TP is unsupported for the custom MoE; CP/PP are not supported for
        this encoder-decoder block-diffusion path.
        """
        return ModelCapabilities(
            supports_tp=False,
            supports_cp=False,
            supports_pp=False,
            supports_ep=True,
        )

    @classmethod
    def from_config(
        cls,
        config: DiffusionGemmaConfig,
        moe_config: "MoEConfig | None" = None,
        backend: "BackendConfig | None" = None,
        **kwargs: Any,
    ) -> "DiffusionGemmaForBlockDiffusion":
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        moe_config: "MoEConfig | None" = None,
        backend: "BackendConfig | None" = None,
        canvas_length: int | None = None,
        self_conditioning: bool | None = None,
        freeze_router: bool | None = None,
        **kwargs: Any,
    ):
        # ``canvas_length`` is a declared field on the reference config (and round-trips),
        # so a YAML/from_pretrained override is written back onto it. The training-only
        # flags are NOT reference-config fields (it is a strict dataclass), so they live on
        # the model only: take the explicit kwarg, else read the config (returns the
        # default for an undeclared attr, or honors it if a future config declares it).
        if canvas_length is not None:
            config.canvas_length = canvas_length

        # This model only ever runs eager attention (``eager_attention_forward`` —
        # mask-driven, threading plain-tensor KV between the encode/decode passes);
        # it does not implement HF's sdpa/flash paths. Newer transformers validate
        # ``_attn_implementation`` in ``PreTrainedModel.__init__`` and would reject a
        # non-eager value for this class, so pin eager here (the BackendConfig.attn
        # setting governs NeMo's own kernels, not HF's attn dispatch).
        config._attn_implementation = "eager"
        if getattr(config, "text_config", None) is not None:
            config.text_config._attn_implementation = "eager"

        super().__init__(config)
        self.backend = backend or BackendConfig()
        text_config = config.text_config
        self.text_config = text_config
        self.canvas_length = int(getattr(config, "canvas_length", 256))
        self.self_conditioning = bool(
            self_conditioning if self_conditioning is not None else getattr(config, "self_conditioning", True)
        )
        self.freeze_router = bool(
            freeze_router if freeze_router is not None else getattr(config, "freeze_router", True)
        )
        self.final_logit_softcapping = text_config.final_logit_softcapping
        self.vocab_size = text_config.vocab_size

        self.model = DiffusionGemmaBackbone(text_config, self.backend, moe_config=moe_config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        # lm_head is tied to the shared embedding (HF resolves this via
        # _tied_weights_keys + get_output_embeddings).
        self.lm_head.weight = self.model.embed_tokens.weight

        # Expose moe_config for the MoE parallelizer assertion path.
        self.moe_config = self.model.moe_config

        if self.backend.enable_hf_state_dict_adapter:
            from nemo_automodel.shared.utils import dtype_from_str

            from .state_dict_adapter import DiffusionGemmaStateDictAdapter

            self.state_dict_adapter = DiffusionGemmaStateDictAdapter(
                text_config,
                self.model.moe_config,
                self.backend,
                dtype=dtype_from_str(getattr(text_config, "torch_dtype", None), torch.bfloat16),
            )

        if self.freeze_router:
            self.freeze_router_params()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def freeze_router_params(self) -> None:
        """Freeze the MoE router/gate (design v2 item 9).

        Sets ``train_gate=False`` and ``requires_grad=False`` on the gate's
        ``proj.weight`` and ``scale`` for every layer. Routing indices are
        already non-differentiable; ``per_expert_scale`` is folded into the
        (trainable) expert ``down_proj`` by the state-dict adapter, so the
        experts stay trainable. ``MoEFSDPSyncMixin`` keys on
        ``set_requires_gradient_sync``, never ``requires_grad``, so freezing
        the gate does not break grad-sync.
        """
        for layer in self.model.layers.values():
            gate = layer.moe.gate
            gate.train_gate = False
            for name in ("proj", "scale"):
                param = getattr(gate, name, None)
                if isinstance(param, nn.Module):
                    for p in param.parameters():
                        p.requires_grad_(False)
                elif isinstance(param, torch.Tensor):
                    param.requires_grad_(False)

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize grouped-expert parameters (other params init via HF post_init).

        ``dtype`` defaults to the model's **configured** ``torch_dtype`` rather than a
        hardcoded ``bfloat16``. The meta/FSDP init path
        (``checkpoint.checkpointing.initialize_model_weights``) calls this with no
        dtype; a blanket ``self.to(torch.bfloat16)`` would materialize the whole model
        in bf16 before the checkpoint loads, silently defeating the fp32 master weights
        that ``model.torch_dtype: float32`` configs request (leaving AdamW on bf16
        params). Honor the requested dtype instead.
        """
        if dtype is None:
            cfg_dtype = getattr(self.config, "torch_dtype", None)
            if isinstance(cfg_dtype, str):
                from nemo_automodel.shared.utils import dtype_from_str

                cfg_dtype = dtype_from_str(cfg_dtype, torch.bfloat16)
            dtype = cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.bfloat16
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            for layer in self.model.layers.values():
                layer.moe.init_weights(buffer_device)
        cast_model_to_dtype(self, dtype)

    def _softcap_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states).to(torch.float32)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        canvas_ids: torch.Tensor | None = None,
        self_conditioning_logits: torch.Tensor | None = None,
        encoder_position_ids: torch.Tensor | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
        decoder_position_ids: torch.Tensor | None = None,
        decoder_attention_mask: dict | None = None,
        decoder_padding_mask: torch.Tensor | None = None,
        do_self_conditioning: torch.Tensor | bool | None = None,
        **kwargs: Any,
    ) -> "DiffusionGemmaOutput":
        """Training forward — single shared stack run twice + two-pass self-cond.

        Args:
            input_ids: Clean full sequence (prompt + response), ``[B, S]``. Run
                causally to build the read-only encoder KV cache.
            canvas_ids: Noised response/canvas tokens, ``[B, canvas_len]``. Run
                bidirectionally with the block-causal mask.
            self_conditioning_logits: If given (inference / external loop), used
                directly and the two-pass logic is skipped. During training the
                two-pass scheme generates the self-cond signal internally.
            encoder_position_ids: Position ids for the encoder pass (``[B, S]``).
                Defaults to ``arange(S)``.
            encoder_padding_mask: ``True`` at padded encoder positions (``[B, S]``).
            decoder_position_ids: Position ids for the canvas (``[B, canvas_len]``).
                Must be the canvas tokens' **absolute** positions so their query
                RoPE aligns with the encoder key RoPE of the clean copies. In the
                v1 full-sequence-canvas layout (``canvas_len == S``) this is
                ``arange(S)`` (the default); a response-window canvas would use
                ``prefix_length + arange(canvas_len)`` per example.
            decoder_attention_mask: Dict ``{"full_attention", "sliding_attention"}``
                of additive block-causal masks (from
                ``build_block_diffusion_training_mask``). Required for training;
                built by the recipe's ``_forward_backward_step`` override.
            decoder_padding_mask: ``True`` at padded canvas positions
                (``[B, canvas_len]``). Used to keep padded rows out of MoE routing.
            do_self_conditioning: Per-example self-conditioning coins, a ``[B]``
                bool tensor (a scalar bool is broadcast). During training pass-1
                **always** runs (constant FSDP collectives every step -> no rank
                desync, and correct for ``local_batch_size > 1``); this mask gates,
                per example, whether pass-2 consumes the self-cond signal (``False``
                -> zeroed soft-embed, i.e. no self-cond). The recipe supplies it via
                ``_decide_self_conditioning``. Required during training (``None``
                would drop Google's per-example mix -> ``ValueError``); ignored
                outside training (eval / single pass).

        Returns:
            ``DiffusionGemmaOutput`` with canvas-only ``logits``
            (``[B, canvas_len, V]``, softcapped) and, during training,
            ``encoder_logits`` (``[B, S, V]``) for the co-trained AR loss.
        """
        if input_ids is None:
            raise ValueError("DiffusionGemmaForBlockDiffusion training forward requires input_ids (clean sequence).")
        if canvas_ids is None:
            raise ValueError("DiffusionGemmaForBlockDiffusion training forward requires canvas_ids (noised canvas).")
        if decoder_attention_mask is None:
            raise ValueError(
                "decoder_attention_mask (the block-causal mask dict) is required; build it via "
                "build_block_diffusion_training_mask in the recipe's _forward_backward_step override."
            )

        batch_size, enc_len = input_ids.shape
        canvas_len = canvas_ids.shape[1]
        device = input_ids.device

        if encoder_position_ids is None:
            encoder_position_ids = torch.arange(enc_len, device=device).unsqueeze(0).expand(batch_size, -1)
        if decoder_position_ids is None:
            if canvas_len != enc_len:
                raise ValueError(
                    "decoder_position_ids must be provided when canvas_len != enc_len; the default "
                    "arange(S) only applies to the v1 full-sequence-canvas layout (canvas_len == S)."
                )
            decoder_position_ids = torch.arange(canvas_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 1) Causal pass over the clean sequence -> read-only encoder KV cache.
        #    During training also keep the encoder hidden states to produce the
        #    co-trained autoregressive encoder logits (the SFT objective trains the
        #    shared backbone as a causal LM alongside the diffusion denoiser). The
        #    full-sequence lm_head over the large vocab is memory-heavy, so it is
        #    skipped outside training.
        encoder_logits = None
        if self.training:
            encoder_kv, encoder_hidden = self.model(
                mode="encode",
                input_ids=input_ids,
                position_ids=encoder_position_ids,
                padding_mask=encoder_padding_mask,
                return_hidden=True,
            )
            encoder_logits = self._softcap_logits(encoder_hidden)
        else:
            encoder_kv = self.model(
                mode="encode",
                input_ids=input_ids,
                position_ids=encoder_position_ids,
                padding_mask=encoder_padding_mask,
            )

        # 2) Self-conditioning (Analog-Bits), per-example. When the caller supplies
        #    self_conditioning_logits (inference / external loop) or self-cond is
        #    disabled / eval, no pass-1 runs. During training pass-1 ALWAYS runs
        #    (no_grad, same mask + canvas) so every step issues identical FSDP
        #    collectives regardless of the coins (no rank desync; correct for
        #    local_batch_size > 1); the per-example [B] mask (do_self_conditioning)
        #    then gates, per example, whether pass-2 consumes the self-cond signal
        #    (coin False -> zeroed soft-embed, identical to no self-cond). Only
        #    pass-2 backprops.
        sc_logits = self_conditioning_logits
        sc_mask = None
        if sc_logits is None and self.self_conditioning and self.training:
            if do_self_conditioning is None:
                raise ValueError(
                    "do_self_conditioning (a per-example [B] bool mask) must be supplied "
                    "during training; the recipe sets it via _decide_self_conditioning. "
                    "Leaving it None would drop Google's per-example self-conditioning mix."
                )
            sc_mask = (
                do_self_conditioning.to(device=device, dtype=torch.bool).reshape(-1)
                if torch.is_tensor(do_self_conditioning)
                else torch.full((batch_size,), bool(do_self_conditioning), dtype=torch.bool, device=device)
            )
            with torch.no_grad():
                pass1_hidden = self.model(
                    mode="decode",
                    canvas_ids=canvas_ids,
                    encoder_kv=encoder_kv,
                    decoder_position_ids=decoder_position_ids,
                    decoder_masks=decoder_attention_mask,
                    decoder_padding_mask=decoder_padding_mask,
                    self_conditioning_logits=None,
                )
                sc_logits = self._softcap_logits(pass1_hidden).detach()

        hidden_states = self.model(
            mode="decode",
            canvas_ids=canvas_ids,
            encoder_kv=encoder_kv,
            decoder_position_ids=decoder_position_ids,
            decoder_masks=decoder_attention_mask,
            decoder_padding_mask=decoder_padding_mask,
            self_conditioning_logits=sc_logits,
            self_conditioning_mask=sc_mask,
        )
        logits = self._softcap_logits(hidden_states)

        return DiffusionGemmaOutput(logits=logits, encoder_logits=encoder_logits)


ModelClass = DiffusionGemmaForBlockDiffusion

# Register the pure-FSDP2 (ep_size=1) parallelization strategy so that
# get_parallelization_strategy() selects it for this model. Done here (not in
# the package __init__) so registration runs alongside model construction.

register_diffusion_gemma_parallel_strategy()
