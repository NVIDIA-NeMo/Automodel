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

"""NemotronOmni (NemotronH_Nano_VL_V2) custom model for Nemo Automodel.

This model is a VLM (vision-language model) with:
- Vision encoder: RADIO v2.5-H (ViT-Huge, patch_size=16) -- loaded from HF
- Audio encoder: Parakeet (FastConformer-based) -- loaded from HF
- LLM: NemotronH (hybrid Mamba+Attention MoE) -- reuses nemotron_v3 custom implementation
- Projectors: MLP projectors for vision->LLM and audio->LLM

Architecture name: "NemotronH_Nano_VL_V2" (from config.json)
"""

import logging
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.nemotron_v3.model import (
    NemotronHForCausalLM as NemotronV3ForCausalLM,
)
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import NemotronOmniStateDictAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helper modules (vision projector, sound projector)
# These match the HF checkpoint exactly.
# ---------------------------------------------------------------------------


class SquaredReLU(nn.Module):
    """Squared ReLU activation: ReLU(x)^2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class VisionProjector(nn.Module):
    """MLP projector from vision encoder to LLM hidden space.

    HF checkpoint structure (mlp1):
        mlp1.0.weight  ->  RMSNorm weight  (vit_hidden_size * pixel_shuffle_factor^2,)
        mlp1.1.weight  ->  Linear1 weight  (projector_hidden_size, vit_hidden_size * pixel_shuffle_factor^2)
        mlp1.3.weight  ->  Linear2 weight  (llm_hidden_size, projector_hidden_size)

    Between linear1 and linear2 there is a SquaredReLU activation (index 2 in Sequential,
    but it has no weight).
    """

    def __init__(
        self,
        vit_hidden_size: int,
        projector_hidden_size: int,
        llm_hidden_size: int,
        downsample_ratio: float = 0.5,
    ):
        super().__init__()
        pixel_shuffle_channels = vit_hidden_size * int(1 / downsample_ratio) ** 2
        self.norm = RMSNorm(pixel_shuffle_channels, eps=1e-5)
        self.linear1 = nn.Linear(pixel_shuffle_channels, projector_hidden_size, bias=False)
        self.activation = SquaredReLU()
        self.linear2 = nn.Linear(projector_hidden_size, llm_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class SoundProjection(nn.Module):
    """MLP projector from sound encoder to LLM hidden space.

    HF checkpoint structure:
        sound_projection.norm.weight       -> RMSNorm weight  (sound_hidden_size,)
        sound_projection.linear1.weight    -> Linear1 weight  (projection_hidden_size, sound_hidden_size)
        sound_projection.linear2.weight    -> Linear2 weight  (llm_hidden_size, projection_hidden_size)
    """

    def __init__(
        self,
        sound_hidden_size: int,
        projection_hidden_size: int,
        llm_hidden_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.norm = RMSNorm(sound_hidden_size, eps=1e-5)
        self.linear1 = nn.Linear(sound_hidden_size, projection_hidden_size, bias=bias)
        self.activation = SquaredReLU()
        self.linear2 = nn.Linear(projection_hidden_size, llm_hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------


class NemotronOmniConfig(PretrainedConfig):
    """Configuration for the NemotronOmni (NemotronH_Nano_VL_V2) model.

    This wraps the HF config and provides easy access to sub-configs.
    """

    model_type = "NemotronH_Nano_VL_V2"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        sound_config=None,
        force_image_size=512,
        downsample_ratio=0.5,
        patch_size=16,
        template=None,
        ps_version="v2",
        image_tag_type="internvl",
        projector_hidden_size=20480,
        vit_hidden_size=1280,
        img_context_token_id=18,
        video_context_token_id=131081,
        sound_context_token_id=27,
        video_pruning_rate=0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.llm_config = llm_config
        self.sound_config = sound_config
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.patch_size = patch_size
        self.template = template
        self.ps_version = ps_version
        self.image_tag_type = image_tag_type
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.img_context_token_id = img_context_token_id
        self.video_context_token_id = video_context_token_id
        self.sound_context_token_id = sound_context_token_id
        self.video_pruning_rate = video_pruning_rate


# ---------------------------------------------------------------------------
# Model proxy for MoE parallelizer compatibility
# ---------------------------------------------------------------------------


class _ModelProxy:
    """Thin proxy so the MoE parallelizer can navigate model.model.moe_config
    and model.model -> get_text_module -> .layers without changing the weight
    hierarchy.

    The parallelizer (parallelizer.py) expects:
        model.model.moe_config           (for expert-count validation)
        model.model -> get_text_module()  (finds language_model attr) -> .layers

    By setting self.model = _ModelProxy(self.language_model) on the VLM:
        model.model.moe_config            -> language_model.model.moe_config  OK
        get_text_module(model.model)       -> model.model.language_model
                                           == language_model.model (NemotronV3Model)
                                           -> .layers                          OK
    """

    def __init__(self, llm: "NemotronV3ForCausalLM"):
        # llm is NemotronHForCausalLM which has .model = NemotronV3Model
        self.moe_config = llm.model.moe_config
        # Expose the inner NemotronV3Model as 'language_model' so that
        # get_text_module() can find it and access .layers
        self.language_model = llm.model


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class NemotronOmniForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """NemotronOmni VLM model for conditional generation (training).

    Wraps:
    - Vision encoder (RADIO v2.5-H) -- HF implementation via trust_remote_code
    - Audio encoder (Parakeet) -- HF implementation via trust_remote_code
    - Vision projector (MLP: RMSNorm -> Linear -> SquaredReLU -> Linear)
    - Sound projector (MLP: RMSNorm -> Linear -> SquaredReLU -> Linear)
    - Language model (NemotronH hybrid Mamba+Attention MoE) -- nemotron_v3 custom impl

    The LLM part reuses the nemotron_v3 implementation (NemotronHForCausalLM) which
    has custom DTensor parallelism for the Mamba+Attention hybrid MoE architecture.
    """

    @classmethod
    def from_config(
        cls,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Create model from config.

        Args:
            config: NemotronH_Nano_VL_V2 config (HF config with trust_remote_code)
            backend: Backend configuration
            **kwargs: Additional arguments

        Returns:
            NemotronOmniForConditionalGeneration instance
        """
        return cls(config, backend=backend, **kwargs)

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
            NemotronOmniForConditionalGeneration instance
        """
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Initialize NemotronOmniForConditionalGeneration.

        Args:
            config: NemotronH_Nano_VL_V2 config
            backend: Backend configuration
            **kwargs: Additional arguments
        """
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        # ---------------------------------------------------------------
        # Extract sub-configs
        # ---------------------------------------------------------------
        llm_config = config.llm_config
        vision_config = config.vision_config
        sound_config = getattr(config, "sound_config", None)

        # Store key VLM parameters
        self.force_image_size = getattr(config, "force_image_size", 512)
        self.patch_size = getattr(config, "patch_size", 16)
        self.downsample_ratio = getattr(config, "downsample_ratio", 0.5)
        self.ps_version = getattr(config, "ps_version", "v2")
        self.img_context_token_id = getattr(config, "img_context_token_id", 18)
        self.video_context_token_id = getattr(config, "video_context_token_id", 131081)
        self.sound_context_token_id = getattr(config, "sound_context_token_id", 27)

        self.num_image_token = int(
            (self.force_image_size // self.patch_size) ** 2
            * (self.downsample_ratio ** 2)
        )
        logger.info(f"NemotronOmni: num_image_token={self.num_image_token}")
        logger.info(f"NemotronOmni: ps_version={self.ps_version}")
        logger.info(f"NemotronOmni: img_context_token_id={self.img_context_token_id}")

        vit_hidden_size = getattr(config, "vit_hidden_size", 1280)
        projector_hidden_size = getattr(config, "projector_hidden_size", 20480)
        llm_hidden_size = llm_config.hidden_size

        # ---------------------------------------------------------------
        # 1. Language Model (reuses nemotron_v3 custom implementation)
        # ---------------------------------------------------------------
        logger.info("NemotronOmni: Creating NemotronV3 LLM backbone...")
        self.language_model = NemotronV3ForCausalLM(
            llm_config, backend=self.backend, **kwargs
        )
        logger.info(
            f"NemotronOmni: LLM created with {llm_config.num_hidden_layers} layers, "
            f"hidden_size={llm_config.hidden_size}, vocab_size={llm_config.vocab_size}"
        )

        # ---------------------------------------------------------------
        # 2. Vision Encoder (RADIO v2.5-H from HF)
        # ---------------------------------------------------------------
        logger.info("NemotronOmni: Creating RADIO vision encoder from HF config...")
        dtype = get_dtype(getattr(llm_config, "torch_dtype", None), torch.bfloat16)
        # FIX: Force timm to use eager (math) attention instead of fused SDPA
        # for the RADIO ViT. This ensures numerical parity with the HF model
        # which also uses eager attention. The timm Attention class reads this
        # global flag at __init__ time, so it must be set BEFORE model creation.
        from timm.layers.config import set_fused_attn as _timm_set_fused_attn
        _timm_set_fused_attn(False)
        self.vision_model = AutoModel.from_config(vision_config, trust_remote_code=True)
        _timm_set_fused_attn(True)  # Restore default for any subsequent timm usage
        # WAR for transformers issue 38358
        if hasattr(self.vision_model, "model") and hasattr(self.vision_model.model, "_init_weights"):
            self.vision_model.model._initialize_weights = self.vision_model.model._init_weights
        # Make preprocessor external (required by RADIO)
        if hasattr(self.vision_model, "radio_model"):
            self.vision_model.radio_model.make_preprocessor_external()
        self.vision_model = self.vision_model.to(dtype)

        # Convert RADIO buffers that are NOT in the HF checkpoint to
        # non-persistent so the DCP loader doesn't expect them on disk.
        self._make_missing_buffers_non_persistent(self.vision_model)
        logger.info("NemotronOmni: Vision encoder created (RADIO v2.5-H)")

        # ---------------------------------------------------------------
        # 3. Vision Projector (MLP: RMSNorm -> Linear -> SquaredReLU -> Linear)
        # ---------------------------------------------------------------
        self.vision_projector = VisionProjector(
            vit_hidden_size=vit_hidden_size,
            projector_hidden_size=projector_hidden_size,
            llm_hidden_size=llm_hidden_size,
            downsample_ratio=self.downsample_ratio,
        ).to(dtype)
        logger.info(
            f"NemotronOmni: Vision projector created "
            f"(vit_hidden={vit_hidden_size} -> proj_hidden={projector_hidden_size} -> llm_hidden={llm_hidden_size})"
        )

        # ---------------------------------------------------------------
        # 4. Audio Encoder (Parakeet from HF) + Sound Projector
        # ---------------------------------------------------------------
        if sound_config is not None:
            sound_hidden_size = getattr(sound_config, "hidden_size", 1024)
            sound_proj_hidden_size = getattr(sound_config, "projection_hidden_size", 4096)
            sound_proj_bias = getattr(sound_config, "projection_bias", False)

            logger.info("NemotronOmni: Creating Parakeet sound encoder...")
            try:
                from transformers import ParakeetEncoder, ParakeetEncoderConfig

                # Build ParakeetEncoderConfig from sound_config
                parakeet_config_dict = {
                    "attention_bias": getattr(sound_config, "attention_bias", False),
                    "hidden_size": sound_hidden_size,
                    "num_attention_heads": getattr(sound_config, "num_attention_heads", 8),
                    "num_hidden_layers": getattr(sound_config, "num_hidden_layers", 24),
                    "intermediate_size": getattr(sound_config, "intermediate_size", 4096),
                    "conv_kernel_size": getattr(sound_config, "conv_kernel_size", 9),
                    "convolution_bias": getattr(sound_config, "convolution_bias", False),
                    "subsampling_conv_channels": getattr(sound_config, "subsampling_conv_channels", 256),
                    "subsampling_conv_kernel_size": getattr(sound_config, "subsampling_conv_kernel_size", 3),
                    "subsampling_conv_stride": getattr(sound_config, "subsampling_conv_stride", 2),
                    "subsampling_factor": getattr(sound_config, "subsampling_factor", 8),
                    "num_mel_bins": getattr(sound_config, "num_mel_bins", 128),
                }
                parakeet_config = ParakeetEncoderConfig(**parakeet_config_dict)
                self.sound_encoder = ParakeetEncoder(parakeet_config).to(dtype)
                logger.info(f"NemotronOmni: Sound encoder created (hidden_size={sound_hidden_size})")
            except ImportError:
                logger.warning(
                    "NemotronOmni: ParakeetEncoder not available in transformers. "
                    "Sound encoder will not be loaded."
                )
                self.sound_encoder = None

            self.sound_projection = SoundProjection(
                sound_hidden_size=sound_hidden_size,
                projection_hidden_size=sound_proj_hidden_size,
                llm_hidden_size=llm_hidden_size,
                bias=sound_proj_bias,
            ).to(dtype)
            logger.info(
                f"NemotronOmni: Sound projector created "
                f"(sound_hidden={sound_hidden_size} -> proj_hidden={sound_proj_hidden_size} -> llm_hidden={llm_hidden_size})"
            )
        else:
            self.sound_encoder = None
            self.sound_projection = None
            logger.info("NemotronOmni: No sound config, audio encoder disabled.")

        # ---------------------------------------------------------------
        # 5. Model proxy for MoE parallelizer compatibility
        # ---------------------------------------------------------------
        # The MoE parallelizer (parallelizer.py) expects model.model.moe_config
        # and apply_ep navigates: model.model -> get_text_module() -> .layers.
        # We create a thin _ModelProxy that exposes these attributes:
        #   self.model.moe_config  -> language_model.model.moe_config
        #   self.model.language_model -> language_model.model (NemotronV3Model with .layers)
        self.model = _ModelProxy(self.language_model)
        logger.info("NemotronOmni: Model proxy created for parallelizer compatibility")

        # ---------------------------------------------------------------
        # 6. State dict adapter
        # ---------------------------------------------------------------
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = NemotronOmniStateDictAdapter(
                config=config,
                llm_config=llm_config,
                moe_config=self.language_model.model.moe_config,
                backend=self.backend,
                dtype=dtype,
            )
            logger.info("NemotronOmni: State dict adapter created")

    # ------------------------------------------------------------------
    # Buffer management helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_missing_buffers_non_persistent(module: nn.Module) -> None:
        """Convert persistent buffers that are NOT saved in HF checkpoints
        to non-persistent buffers.

        The RADIO vision encoder registers some buffers (e.g. ``summary_idxs``)
        as persistent, but the HF checkpoint does not contain them.  When the DCP
        loader builds its load plan it expects every persistent buffer to appear
        in the checkpoint and raises ``RuntimeError: Missing key`` otherwise.

        This method re-registers such buffers as non-persistent so they are
        kept at their init-time values and not expected on disk.
        """
        # Known buffers not in the HF RADIO checkpoint
        _NON_CHECKPOINT_BUFFERS = {"summary_idxs"}

        for name, sub in module.named_modules():
            for buf_name in list(sub._buffers.keys()):
                if buf_name in _NON_CHECKPOINT_BUFFERS:
                    buf = sub._buffers[buf_name]
                    # Re-register as non-persistent (keeps the tensor, removes
                    # it from state_dict())
                    sub.register_buffer(buf_name, buf, persistent=False)
                    logger.info(
                        f"NemotronOmni: Converted buffer '{name}.{buf_name}' "
                        f"to non-persistent (not in HF checkpoint)"
                    )

    # ------------------------------------------------------------------
    # Embedding access (required by VLM training infrastructure)
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        """Return the input embeddings from the language model."""
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the input embeddings of the language model."""
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Return the output embeddings (lm_head) from the language model."""
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings (lm_head) of the language model."""
        self.language_model.set_output_embeddings(new_embeddings)

    # ------------------------------------------------------------------
    # Vision feature extraction
    # ------------------------------------------------------------------

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        """Pixel shuffle for downsampling spatial resolution while increasing channels.

        Args:
            x: Input tensor [N, W, H, C]
            scale_factor: Downsampling ratio (default 0.5 = halve spatial dims)

        Returns:
            Shuffled tensor [N, W*scale, H*scale, C/(scale^2)]
        """
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract vision features from pixel values through RADIO + projector.

        Args:
            pixel_values: Image tensors [num_tiles, C, H, W]

        Returns:
            Vision embeddings [num_tiles, num_tokens, llm_hidden_size]
        """
        # Force vision model to eval mode for deterministic spectral reparam.
        # RADIO uses spectral reparameterization with power iteration that is
        # non-deterministic in train mode (random _u/_v init). Since the vision
        # tower is frozen during training, eval mode is correct and produces
        # reproducible outputs.
        was_training = self.vision_model.training
        self.vision_model.eval()
        vit_embeds = self.vision_model(pixel_values).features
        if was_training:
            self.vision_model.train()
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        vit_embeds = self.vision_projector(vit_embeds)

        return vit_embeds

    def extract_sound_feature(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract and project sound features from audio input.

        Args:
            input_features: Mel spectrogram features [batch, seq_len, feature_dim]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Sound embeddings projected to LLM hidden size
        """
        if self.sound_encoder is None:
            raise RuntimeError("Sound encoder not initialized.")
        outputs = self.sound_encoder(
            input_features=input_features,
            attention_mask=attention_mask,
        )
        sound_embeds = outputs.last_hidden_state
        sound_embeds = sound_embeds.to(dtype=torch.bfloat16)
        sound_embeds = self.sound_projection(sound_embeds)
        return sound_embeds

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        sound_features: Optional[torch.FloatTensor] = None,
        sound_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass for training.

        This follows the same pattern as the HF NemotronH_Nano_VL_V2.forward():
        1. Get text embeddings from LLM embed_tokens
        2. Extract vision features from pixel_values
        3. Replace image token embeddings with vision embeddings
        4. Run LLM forward pass
        5. Compute loss if labels provided

        Args:
            pixel_values: Image pixel values [num_tiles, C, H, W]
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs (unused, for API compat)
            image_flags: Flags indicating real images vs padding [num_tiles, 1]
            labels: Token IDs for loss computation [batch, seq_len]
            inputs_embeds: Pre-computed input embeddings (optional)
            use_cache: Whether to use caching (not used in training)
            **kwargs: Additional arguments

        Returns:
            CausalLMOutputWithPast with loss and logits
        """
        return_dict = return_dict if return_dict is not None else True

        # Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Process vision inputs
        if pixel_values is not None and image_flags is not None:
            image_flags = image_flags.squeeze(-1)

            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)

            selected = input_ids_flat == self.img_context_token_id

            vit_batch_size = pixel_values.shape[0]
            vit_embeds = self.extract_feature(pixel_values)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logger.info(
                    f"NemotronOmni: dynamic ViT batch size: {vit_batch_size}, "
                    f"images per sample: {vit_batch_size / B}, "
                    f"dynamic token length: {N}"
                )

            # Filter by image_flags (1 = real image, 0 = padding)
            vit_embeds = vit_embeds[image_flags == 1]

            try:
                inputs_embeds[selected] = (
                    inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                )
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                logger.warning(
                    f"Shape mismatch: {e}, "
                    f"inputs_embeds[selected].shape={inputs_embeds[selected].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                n_token = selected.sum()
                inputs_embeds[selected] = (
                    inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]
                )

            inputs_embeds = inputs_embeds.reshape(B, N, C)

        # --- Sound/audio token replacement ---
        has_sound = (
            sound_features is not None
            and self.sound_encoder is not None
            and self.sound_context_token_id is not None
        )
        if has_sound:
            B_s, N_s, C_s = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B_s * N_s, C_s)
            input_ids_flat_sound = input_ids.reshape(B_s * N_s)

            sound_selected = (input_ids_flat_sound == self.sound_context_token_id)
            num_sound_tokens = sound_selected.sum().item()

            if num_sound_tokens > 0:
                # Move sound features to correct device/dtype
                target_dtype = inputs_embeds.dtype
                sound_features = sound_features.to(dtype=target_dtype, device=inputs_embeds.device)
                if sound_attention_mask is not None:
                    sound_attention_mask = sound_attention_mask.to(device=inputs_embeds.device)

                # Extract and project sound features
                sound_embeds = self.extract_sound_feature(sound_features, sound_attention_mask)
                sound_embeds_flat = sound_embeds.reshape(-1, C_s)

                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    logger.info(
                        f"NemotronOmni: sound tokens: {num_sound_tokens}, "
                        f"sound_embeds shape: {sound_embeds.shape}, "
                        f"sound_features shape: {sound_features.shape}"
                    )

                try:
                    inputs_embeds[sound_selected] = (
                        inputs_embeds[sound_selected] * 0.0
                        + sound_embeds_flat.to(inputs_embeds.dtype)
                    )
                except Exception as e:
                    logger.warning(
                        f"Sound shape mismatch: {e}, "
                        f"inputs_embeds[sound_selected].shape={inputs_embeds[sound_selected].shape}, "
                        f"sound_embeds_flat.shape={sound_embeds_flat.shape}"
                    )
                    inputs_embeds[sound_selected] = (
                        inputs_embeds[sound_selected] * 0.0
                        + sound_embeds_flat[:num_sound_tokens].to(inputs_embeds.dtype)
                    )

                del sound_embeds, sound_embeds_flat

            inputs_embeds = inputs_embeds.reshape(B_s, N_s, C_s)

        # Forward through the LLM
        outputs = self.language_model(
            input_ids=None,  # We pass inputs_embeds instead
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        return outputs

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

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
            # Initialize LLM weights
            self.language_model.initialize_weights(buffer_device=buffer_device, dtype=dtype)

        # Vision model and projectors are loaded from checkpoint
        # Cast everything to target dtype
        cast_model_to_dtype(self, dtype)


ModelClass = NemotronOmniForConditionalGeneration
