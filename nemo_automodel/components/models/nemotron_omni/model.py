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
import os  # DEBUG: activation dump
import sys  # DEBUG: activation dump
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist  # DEBUG: activation dump
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig

# DEBUG: activation dump - directory to save activations
_DUMP_DIR = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemofw/users/huiyingl/nemotronomni/activation_dumps"  # DEBUG: activation dump

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


# DEBUG: activation dump - helper to save and log a tensor
def _dump_tensor(name: str, tensor: torch.Tensor, dump_dir: str = _DUMP_DIR) -> None:  # DEBUG: activation dump
    """Save a tensor to disk and print its stats. Only call on rank 0."""  # DEBUG: activation dump
    t = tensor.detach().cpu().float()  # DEBUG: activation dump
    path = os.path.join(dump_dir, f"{name}.pt")  # DEBUG: activation dump
    torch.save(t, path)  # DEBUG: activation dump
    print(f"[ACTIVATION DUMP] Saved {name}.pt  shape={tuple(t.shape)}  "  # DEBUG: activation dump
          f"min={t.min().item():.6f}  max={t.max().item():.6f}  mean={t.mean().item():.6f}")  # DEBUG: activation dump


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
        # DEBUG: activation dump v2 - forward call counter and LLM layer hooks
        # ---------------------------------------------------------------
        self._debug_forward_count = 0  # DEBUG: activation dump v2
        self._debug_llm_layer_outputs = {}  # DEBUG: activation dump v2
        self._debug_moe_routing = {}  # DEBUG: activation dump v2
        self._debug_hooks = []  # DEBUG: activation dump v2

        # DEBUG: activation dump v2 - register forward hooks on ALL LLM layers
        _num_layers = len(self.language_model.model.layers)  # DEBUG: activation dump v2
        for layer_idx in range(_num_layers):  # DEBUG: activation dump v2
            key = str(layer_idx)  # DEBUG: activation dump v2
            if key in self.language_model.model.layers:  # DEBUG: activation dump v2
                def _make_layer_hook(idx):  # DEBUG: activation dump v2
                    def hook_fn(module, input, output):  # DEBUG: activation dump v2
                        self._debug_llm_layer_outputs[idx] = output  # DEBUG: activation dump v2
                    return hook_fn  # DEBUG: activation dump v2
                h = self.language_model.model.layers[key].register_forward_hook(_make_layer_hook(layer_idx))  # DEBUG: activation dump v2
                self._debug_hooks.append(h)  # DEBUG: activation dump v2
        logger.info(f"DEBUG: Registered activation dump hooks on ALL {_num_layers} LLM layers")  # DEBUG: activation dump v2

        # DEBUG: activation dump v2 - register forward hooks on MoE gate modules for routing info
        # The hybrid_override_pattern: M=mamba, E=moe, *=attention
        # MoE layers have block_type=="moe" and their mixer is a MoE module with a gate
        _moe_layer_count = 0  # DEBUG: activation dump v2
        for layer_idx in range(_num_layers):  # DEBUG: activation dump v2
            key = str(layer_idx)  # DEBUG: activation dump v2
            block = self.language_model.model.layers[key]  # DEBUG: activation dump v2
            if hasattr(block, 'block_type') and block.block_type == "moe":  # DEBUG: activation dump v2
                gate_module = block.mixer.gate  # DEBUG: activation dump v2
                def _make_gate_hook(idx):  # DEBUG: activation dump v2
                    def gate_hook_fn(module, input, output):  # DEBUG: activation dump v2
                        # Gate.forward returns (weights, indices, aux_loss)  # DEBUG: activation dump v2
                        # input[0] is x (hidden states flattened to 2D)  # DEBUG: activation dump v2
                        weights, indices, aux_loss = output  # DEBUG: activation dump v2
                        # Recompute router logits from gate weight and input  # DEBUG: activation dump v2
                        x = input[0]  # DEBUG: activation dump v2
                        gate_precision = getattr(module, 'gate_precision', None)  # DEBUG: activation dump v2
                        if gate_precision is not None:  # DEBUG: activation dump v2
                            x_compute = x.to(dtype=gate_precision)  # DEBUG: activation dump v2
                            weight = module.weight.to(dtype=gate_precision)  # DEBUG: activation dump v2
                        else:  # DEBUG: activation dump v2
                            x_compute = x  # DEBUG: activation dump v2
                            weight = module.weight.to(dtype=x.dtype)  # DEBUG: activation dump v2
                        import torch.nn.functional as _F  # DEBUG: activation dump v2
                        bias = None  # DEBUG: activation dump v2
                        if hasattr(module, 'bias') and module.bias is not None:  # DEBUG: activation dump v2
                            bias = module.bias.to(dtype=x_compute.dtype)  # DEBUG: activation dump v2
                        router_logits = _F.linear(x_compute, weight, bias=bias)  # DEBUG: activation dump v2
                        self._debug_moe_routing[idx] = {  # DEBUG: activation dump v2
                            "router_logits": router_logits,  # DEBUG: activation dump v2
                            "top_indices": indices,  # DEBUG: activation dump v2
                            "top_weights": weights,  # DEBUG: activation dump v2
                        }  # DEBUG: activation dump v2
                    return gate_hook_fn  # DEBUG: activation dump v2
                h = gate_module.register_forward_hook(_make_gate_hook(layer_idx))  # DEBUG: activation dump v2
                self._debug_hooks.append(h)  # DEBUG: activation dump v2
                _moe_layer_count += 1  # DEBUG: activation dump v2
        logger.info(f"DEBUG: Registered MoE gate routing hooks on {_moe_layer_count} MoE layers")  # DEBUG: activation dump v2

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
        # DEBUG: activation dump - determine if we should dump
        _is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)  # DEBUG: activation dump
        _should_dump = _is_rank0 and self._debug_forward_count == 0  # DEBUG: activation dump

        if _should_dump:  # DEBUG: verify fused_attn state
            for name, mod_inst in self.vision_model.named_modules():  # DEBUG
                if hasattr(mod_inst, 'fused_attn'):  # DEBUG
                    print(f"[ACTIVATION DUMP] RADIO attn check: {name}.fused_attn = {mod_inst.fused_attn}")  # DEBUG
                    break  # DEBUG

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

        if _should_dump:  # DEBUG: activation dump
            _dump_tensor("radio_output", vit_embeds)  # DEBUG: activation dump

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        if _should_dump:  # DEBUG: activation dump
            _dump_tensor("pixel_shuffle_output", vit_embeds)  # DEBUG: activation dump

        vit_embeds = self.vision_projector(vit_embeds)

        if _should_dump:  # DEBUG: activation dump
            _dump_tensor("projector_output", vit_embeds)  # DEBUG: activation dump

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

        # DEBUG: activation dump - check if we should dump on this forward call
        _is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)  # DEBUG: activation dump
        _should_dump = _is_rank0 and self._debug_forward_count == 0  # DEBUG: activation dump

        if _should_dump:  # DEBUG: activation dump
            print(f"\n{'='*80}")  # DEBUG: activation dump
            print(f"[ACTIVATION DUMP] First forward pass on rank 0 - dumping activations to {_DUMP_DIR}")  # DEBUG: activation dump
            print(f"{'='*80}\n")  # DEBUG: activation dump
            # Save raw input data  # DEBUG: activation dump
            input_data = {}  # DEBUG: activation dump
            if input_ids is not None:  # DEBUG: activation dump
                input_data["input_ids"] = input_ids.detach().cpu()  # DEBUG: activation dump
            if attention_mask is not None:  # DEBUG: activation dump
                input_data["attention_mask"] = attention_mask.detach().cpu()  # DEBUG: activation dump
            if pixel_values is not None:  # DEBUG: activation dump
                input_data["pixel_values"] = pixel_values.detach().cpu().float()  # DEBUG: activation dump
            if image_flags is not None:  # DEBUG: activation dump
                input_data["image_flags"] = image_flags.detach().cpu()  # DEBUG: activation dump
            if labels is not None:  # DEBUG: activation dump
                input_data["labels"] = labels.detach().cpu()  # DEBUG: activation dump
            torch.save(input_data, os.path.join(_DUMP_DIR, "input_data.pt"))  # DEBUG: activation dump
            print(f"[ACTIVATION DUMP] Saved input_data.pt with keys: {list(input_data.keys())}")  # DEBUG: activation dump
            for k, v in input_data.items():  # DEBUG: activation dump
                if isinstance(v, torch.Tensor):  # DEBUG: activation dump
                    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")  # DEBUG: activation dump

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
                print(
                    f"NemotronOmni: dynamic ViT batch size: {vit_batch_size}, "
                    f"images per sample: {vit_batch_size / B}, "
                    f"dynamic token length: {N}"
                )

            # Filter by image_flags (1 = real image, 0 = padding)
            vit_embeds = vit_embeds[image_flags == 1]

            if _should_dump:  # DEBUG: activation dump
                _dump_tensor("vit_embeds_final", vit_embeds.reshape(-1, C))  # DEBUG: activation dump

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

        if _should_dump:  # DEBUG: activation dump
            _dump_tensor("inputs_embeds_after_vision", inputs_embeds)  # DEBUG: activation dump

        # DEBUG: activation dump v2 - clear layer outputs and routing info before LLM forward
        self._debug_llm_layer_outputs.clear()  # DEBUG: activation dump v2
        self._debug_moe_routing.clear()  # DEBUG: activation dump v2

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

        # DEBUG: activation dump v2 - save ALL LLM layer outputs, MoE routing, and logits
        if _should_dump:  # DEBUG: activation dump v2
            for layer_idx, layer_output in sorted(self._debug_llm_layer_outputs.items()):  # DEBUG: activation dump v2
                # layer_output may be a tensor or tuple  # DEBUG: activation dump v2
                if isinstance(layer_output, tuple):  # DEBUG: activation dump v2
                    t = layer_output[0]  # DEBUG: activation dump v2
                elif isinstance(layer_output, torch.Tensor):  # DEBUG: activation dump v2
                    t = layer_output  # DEBUG: activation dump v2
                else:  # DEBUG: activation dump v2
                    print(f"[ACTIVATION DUMP] WARNING: LLM layer {layer_idx} output type={type(layer_output)}, skipping")  # DEBUG: activation dump v2
                    continue  # DEBUG: activation dump v2
                _dump_tensor(f"llm_layer_{layer_idx}_output", t)  # DEBUG: activation dump v2

            # DEBUG: activation dump v2 - save MoE routing info for all MoE layers
            print(f"\n[ACTIVATION DUMP] Saving MoE routing info for {len(self._debug_moe_routing)} MoE layers...")  # DEBUG: activation dump v2
            for layer_idx, routing_data in sorted(self._debug_moe_routing.items()):  # DEBUG: activation dump v2
                _dump_tensor(f"moe_layer_{layer_idx}_router_logits", routing_data["router_logits"])  # DEBUG: activation dump v2
                _dump_tensor(f"moe_layer_{layer_idx}_top_indices", routing_data["top_indices"].float())  # DEBUG: activation dump v2
                _dump_tensor(f"moe_layer_{layer_idx}_top_weights", routing_data["top_weights"])  # DEBUG: activation dump v2

            if hasattr(outputs, "logits") and outputs.logits is not None:  # DEBUG: activation dump v2
                _dump_tensor("final_logits", outputs.logits)  # DEBUG: activation dump v2

            print(f"\n{'='*80}")  # DEBUG: activation dump v2
            print(f"[ACTIVATION DUMP] All activations saved to {_DUMP_DIR}")  # DEBUG: activation dump v2
            print(f"{'='*80}\n")  # DEBUG: activation dump v2

            self._debug_forward_count += 1  # DEBUG: activation dump v2
        else:  # DEBUG: activation dump v2
            self._debug_forward_count += 1  # DEBUG: activation dump v2

        # DEBUG: ckpt debug - dump weight tensors at step 50 to diagnose checkpoint saving bug
        # NOTE: All ranks must enter this block because state_dict() and full_tensor() are collectives
        _is_step50_all_ranks = self._debug_forward_count == 50  # DEBUG: ckpt debug
        if _is_step50_all_ranks:  # DEBUG: ckpt debug
            # Helper: extract tensor data without collectives (use _local_tensor for DTensors)  # DEBUG: ckpt debug
            def _safe_get_tensor(t):  # DEBUG: ckpt debug
                """Get tensor data without collectives. For DTensors, get local shard."""  # DEBUG: ckpt debug
                if hasattr(t, '_local_tensor'):  # DEBUG: ckpt debug
                    return t._local_tensor.detach().cpu().float(), True  # (tensor, is_shard)  # DEBUG: ckpt debug
                return t.detach().cpu().float(), False  # DEBUG: ckpt debug

            if _is_rank0:  # DEBUG: ckpt debug
                print(f"\n{'='*80}")  # DEBUG: ckpt debug
                print(f"[CKPT DEBUG] Step 50 weight dump - comparing live params vs state_dict()")  # DEBUG: ckpt debug
                print(f"{'='*80}\n")  # DEBUG: ckpt debug
                os.makedirs(_DUMP_DIR, exist_ok=True)  # DEBUG: ckpt debug

            # --- 1. Dump live parameter tensors (what the optimizer is actually updating) ---  # DEBUG: ckpt debug
            # lm_head weight (replicated, not DTensor)  # DEBUG: ckpt debug
            _lm_head_w = self.language_model.lm_head.weight  # DEBUG: ckpt debug
            _lm_head_full, _lm_is_shard = _safe_get_tensor(_lm_head_w)  # DEBUG: ckpt debug
            if _is_rank0:  # DEBUG: ckpt debug
                print(f"[CKPT DEBUG] lm_head.weight type={type(_lm_head_w)} dtype={_lm_head_w.dtype} is_shard={_lm_is_shard}")  # DEBUG: ckpt debug
                torch.save(_lm_head_full, os.path.join(_DUMP_DIR, "training_lm_head_step50.pt"))  # DEBUG: ckpt debug
                print(f"[CKPT DEBUG] training_lm_head_step50: shape={tuple(_lm_head_full.shape)} "  # DEBUG: ckpt debug
                      f"min={_lm_head_full.min().item():.6f} max={_lm_head_full.max().item():.6f} "  # DEBUG: ckpt debug
                      f"mean={_lm_head_full.mean().item():.6f} first5={_lm_head_full.flatten()[:5].tolist()}")  # DEBUG: ckpt debug

            # layer 0 mixer in_proj weight  # DEBUG: ckpt debug
            _layer0 = self.language_model.model.layers["0"]  # DEBUG: ckpt debug
            _layer0_in_proj = None  # DEBUG: ckpt debug
            if hasattr(_layer0, 'mixer') and hasattr(_layer0.mixer, 'in_proj'):  # DEBUG: ckpt debug
                _layer0_in_proj = _layer0.mixer.in_proj.weight  # DEBUG: ckpt debug
            elif hasattr(_layer0, 'mixer') and hasattr(_layer0.mixer, 'in_proj_weight'):  # DEBUG: ckpt debug
                _layer0_in_proj = _layer0.mixer.in_proj_weight  # DEBUG: ckpt debug
            if _layer0_in_proj is not None:  # DEBUG: ckpt debug
                _l0_full, _l0_is_shard = _safe_get_tensor(_layer0_in_proj)  # DEBUG: ckpt debug
                if _is_rank0:  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] layer0.mixer.in_proj type={type(_layer0_in_proj)} dtype={_layer0_in_proj.dtype} is_shard={_l0_is_shard}")  # DEBUG: ckpt debug
                    torch.save(_l0_full, os.path.join(_DUMP_DIR, "training_layer0_in_proj_step50.pt"))  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] training_layer0_in_proj_step50: shape={tuple(_l0_full.shape)} "  # DEBUG: ckpt debug
                          f"min={_l0_full.min().item():.6f} max={_l0_full.max().item():.6f} "  # DEBUG: ckpt debug
                          f"mean={_l0_full.mean().item():.6f} first5={_l0_full.flatten()[:5].tolist()}")  # DEBUG: ckpt debug
            else:  # DEBUG: ckpt debug
                if _is_rank0:  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] layer0 mixer attributes: {[n for n,_ in _layer0.mixer.named_parameters()][:10]}")  # DEBUG: ckpt debug

            # layer 1 expert gate_and_up_projs (DTensor under expert parallelism)  # DEBUG: ckpt debug
            _layer1 = self.language_model.model.layers["1"]  # DEBUG: ckpt debug
            if hasattr(_layer1, 'mixer') and hasattr(_layer1.mixer, 'experts'):  # DEBUG: ckpt debug
                _experts = _layer1.mixer.experts  # DEBUG: ckpt debug
                if hasattr(_experts, 'gate_and_up_projs'):  # DEBUG: ckpt debug
                    _expert0_up = _experts.gate_and_up_projs  # DEBUG: ckpt debug
                    _e0_full, _e0_is_shard = _safe_get_tensor(_expert0_up)  # DEBUG: ckpt debug
                    if _is_rank0:  # DEBUG: ckpt debug
                        print(f"[CKPT DEBUG] layer1.mixer.experts.gate_and_up_projs type={type(_expert0_up)} dtype={_expert0_up.dtype} is_shard={_e0_is_shard}")  # DEBUG: ckpt debug
                        torch.save(_e0_full, os.path.join(_DUMP_DIR, "training_expert0_up_proj_step50.pt"))  # DEBUG: ckpt debug
                        print(f"[CKPT DEBUG] training_expert0_up_proj_step50: shape={tuple(_e0_full.shape)} "  # DEBUG: ckpt debug
                              f"min={_e0_full.min().item():.6f} max={_e0_full.max().item():.6f} "  # DEBUG: ckpt debug
                              f"mean={_e0_full.mean().item():.6f} first5={_e0_full.flatten()[:5].tolist()}")  # DEBUG: ckpt debug
                        if hasattr(_expert0_up, 'placements'):  # DEBUG: ckpt debug
                            print(f"[CKPT DEBUG]   DTensor placements={_expert0_up.placements} device_mesh={_expert0_up.device_mesh}")  # DEBUG: ckpt debug
                else:  # DEBUG: ckpt debug
                    if _is_rank0:  # DEBUG: ckpt debug
                        print(f"[CKPT DEBUG] layer1 experts attributes: {[n for n,_ in _experts.named_parameters()][:10]}")  # DEBUG: ckpt debug
            else:  # DEBUG: ckpt debug
                if _is_rank0:  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] layer1 block_type={getattr(_layer1, 'block_type', 'unknown')}")  # DEBUG: ckpt debug

            # --- 2. ALL ranks call state_dict() together (it's a collective) ---  # DEBUG: ckpt debug
            if _is_rank0:  # DEBUG: ckpt debug
                print(f"\n[CKPT DEBUG] All ranks calling self.state_dict() (collective)...")  # DEBUG: ckpt debug
            try:  # DEBUG: ckpt debug
                _sd = self.state_dict()  # DEBUG: ckpt debug - ALL ranks must call this
                if _is_rank0:  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] state_dict() returned {len(_sd)} keys")  # DEBUG: ckpt debug
                    _sd_keys_to_check = [  # DEBUG: ckpt debug
                        "language_model.lm_head.weight",  # DEBUG: ckpt debug
                        "language_model.model.layers.0.mixer.in_proj.weight",  # DEBUG: ckpt debug
                        "language_model.model.layers.1.mixer.experts.gate_and_up_projs",  # DEBUG: ckpt debug
                    ]  # DEBUG: ckpt debug
                    for _sdk in _sd_keys_to_check:  # DEBUG: ckpt debug
                        if _sdk in _sd:  # DEBUG: ckpt debug
                            _sd_tensor = _sd[_sdk]  # DEBUG: ckpt debug
                            _sd_full, _sd_is_shard = _safe_get_tensor(_sd_tensor)  # DEBUG: ckpt debug
                            _sd_name = _sdk.replace(".", "_")  # DEBUG: ckpt debug
                            torch.save(_sd_full, os.path.join(_DUMP_DIR, f"statedict_{_sd_name}_step50.pt"))  # DEBUG: ckpt debug
                            print(f"[CKPT DEBUG] state_dict['{_sdk}'] type={type(_sd_tensor)} is_shard={_sd_is_shard}")  # DEBUG: ckpt debug
                            print(f"[CKPT DEBUG]   shape={tuple(_sd_full.shape)} "  # DEBUG: ckpt debug
                                  f"min={_sd_full.min().item():.6f} max={_sd_full.max().item():.6f} "  # DEBUG: ckpt debug
                                  f"mean={_sd_full.mean().item():.6f} first5={_sd_full.flatten()[:5].tolist()}")  # DEBUG: ckpt debug
                        else:  # DEBUG: ckpt debug
                            _similar = [k for k in _sd.keys() if _sdk.split(".")[-1] in k][:5]  # DEBUG: ckpt debug
                            print(f"[CKPT DEBUG] state_dict key '{_sdk}' NOT FOUND. Similar: {_similar}")  # DEBUG: ckpt debug

                    # Compare state_dict lm_head vs live param  # DEBUG: ckpt debug
                    if "language_model.lm_head.weight" in _sd:  # DEBUG: ckpt debug
                        _sd_lm_full, _ = _safe_get_tensor(_sd["language_model.lm_head.weight"])  # DEBUG: ckpt debug
                        if _sd_lm_full.shape == _lm_head_full.shape:  # DEBUG: ckpt debug
                            _diff = (_sd_lm_full - _lm_head_full).abs()  # DEBUG: ckpt debug
                            print(f"[CKPT DEBUG] COMPARE lm_head: live_param vs state_dict: "  # DEBUG: ckpt debug
                                  f"max_diff={_diff.max().item():.8f} mean_diff={_diff.mean().item():.8f}")  # DEBUG: ckpt debug
                        else:  # DEBUG: ckpt debug
                            print(f"[CKPT DEBUG] COMPARE lm_head: shape mismatch! live={tuple(_lm_head_full.shape)} sd={tuple(_sd_lm_full.shape)}")  # DEBUG: ckpt debug

                    # Print first 20 state_dict keys for reference  # DEBUG: ckpt debug
                    _all_keys = sorted(_sd.keys())  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] First 20 state_dict keys: {_all_keys[:20]}")  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] Last 10 state_dict keys: {_all_keys[-10:]}")  # DEBUG: ckpt debug

                del _sd  # DEBUG: ckpt debug - free memory
            except Exception as _e:  # DEBUG: ckpt debug
                if _is_rank0:  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] state_dict() FAILED: {_e}")  # DEBUG: ckpt debug
                    import traceback; traceback.print_exc()  # DEBUG: ckpt debug

            # --- 3. Test to_hf() on a small subset (no collectives needed for lm_head) ---  # DEBUG: ckpt debug
            if _is_rank0:  # DEBUG: ckpt debug
                print(f"\n[CKPT DEBUG] Testing state_dict_adapter.to_hf() conversion...")  # DEBUG: ckpt debug
                try:  # DEBUG: ckpt debug
                    _mini_sd = {  # DEBUG: ckpt debug
                        "language_model.lm_head.weight": self.language_model.lm_head.weight.detach().clone(),  # DEBUG: ckpt debug
                    }  # DEBUG: ckpt debug
                    if hasattr(self, 'state_dict_adapter'):  # DEBUG: ckpt debug
                        _hf_sd = self.state_dict_adapter.to_hf(dict(_mini_sd))  # DEBUG: ckpt debug
                        print(f"[CKPT DEBUG] to_hf() output keys: {list(_hf_sd.keys())}")  # DEBUG: ckpt debug
                        for _hk, _hv in _hf_sd.items():  # DEBUG: ckpt debug
                            _hv_full, _ = _safe_get_tensor(_hv)  # DEBUG: ckpt debug
                            print(f"[CKPT DEBUG]   '{_hk}': shape={tuple(_hv_full.shape)} "  # DEBUG: ckpt debug
                                  f"min={_hv_full.min().item():.6f} max={_hv_full.max().item():.6f} "  # DEBUG: ckpt debug
                                  f"first5={_hv_full.flatten()[:5].tolist()}")  # DEBUG: ckpt debug
                            if _hv_full.shape == _lm_head_full.shape:  # DEBUG: ckpt debug
                                _diff_hf = (_hv_full - _lm_head_full).abs()  # DEBUG: ckpt debug
                                print(f"[CKPT DEBUG]   COMPARE vs live param: max_diff={_diff_hf.max().item():.8f}")  # DEBUG: ckpt debug
                    else:  # DEBUG: ckpt debug
                        print(f"[CKPT DEBUG] No state_dict_adapter on model!")  # DEBUG: ckpt debug
                except Exception as _e:  # DEBUG: ckpt debug
                    print(f"[CKPT DEBUG] to_hf() test FAILED: {_e}")  # DEBUG: ckpt debug
                    import traceback; traceback.print_exc()  # DEBUG: ckpt debug

            if _is_rank0:  # DEBUG: ckpt debug
                print(f"\n{'='*80}")  # DEBUG: ckpt debug
                print(f"[CKPT DEBUG] Step 50 weight dump complete")  # DEBUG: ckpt debug
                print(f"{'='*80}\n")  # DEBUG: ckpt debug
            dist.barrier()  # DEBUG: ckpt debug - sync all ranks after debug dump

        # Debug: check for NaN in logits during validation
        if not self.training and hasattr(outputs, "logits") and outputs.logits is not None:
            if torch.isnan(outputs.logits).any():
                nan_count = torch.isnan(outputs.logits).sum().item()
                total = outputs.logits.numel()
                logger.warning(
                    f"NemotronOmni VAL: NaN in logits! {nan_count}/{total} elements. "
                    f"inputs_embeds has NaN: {torch.isnan(inputs_embeds).any().item()}"
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
