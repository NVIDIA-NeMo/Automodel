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

"""State dict adapter for NemotronOmni (NemotronH_Nano_Omni_Reasoning_V3) models.

Converts between HuggingFace checkpoint format and the custom Automodel format.

HF checkpoint key structure (from model.safetensors.index.json):
    # Vision encoder (RADIO) -- loaded as-is into self.vision_model
    vision_model.radio_model.model.blocks.{N}.{...}
    vision_model.radio_model.input_conditioner.norm_mean
    vision_model.radio_model.input_conditioner.norm_std
    vision_model.radio_model.model.patch_generator.{...}

    # Vision projector -- loaded into self.vision_projector
    HF:     mlp1.0.weight  (RMSNorm)
    Custom: vision_projector.norm.weight
    HF:     mlp1.1.weight  (Linear1)
    Custom: vision_projector.linear1.weight
    HF:     mlp1.3.weight  (Linear2)
    Custom: vision_projector.linear2.weight

    # Sound encoder (Parakeet) -- loaded into self.sound_encoder
    HF:     sound_encoder.encoder.{...}
    Custom: sound_encoder.{...}

    # Sound projector -- loaded into self.sound_projection
    HF:     sound_projection.norm.weight
    Custom: sound_projection.norm.weight
    HF:     sound_projection.linear1.weight
    Custom: sound_projection.linear1.weight
    HF:     sound_projection.linear2.weight
    Custom: sound_projection.linear2.weight

    # LLM (NemotronH) -- uses nemotron_v3 state_dict_adapter internally
    HF:     language_model.backbone.embeddings.weight
    Custom: language_model.model.embed_tokens.weight
    HF:     language_model.backbone.layers.{N}.{...}
    Custom: language_model.model.layers.{N}.{...}
    HF:     language_model.backbone.norm_f.weight
    Custom: language_model.model.norm.weight
    HF:     language_model.lm_head.weight
    Custom: language_model.lm_head.weight

    For MoE layers in the LLM:
    HF:     language_model.backbone.layers.{N}.mixer.experts.{E}.up_proj.weight   (split per-expert)
    HF:     language_model.backbone.layers.{N}.mixer.experts.{E}.down_proj.weight
    Custom: language_model.model.layers.{N}.mixer.experts.gate_and_up_projs       (merged)
    Custom: language_model.model.layers.{N}.mixer.experts.down_projs
"""

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key mapping tables
# ---------------------------------------------------------------------------

# Vision projector: HF mlp1 (nn.Sequential) -> our VisionProjector
_VISION_PROJ_HF_TO_CUSTOM = {
    "mlp1.0.weight": "vision_projector.norm.weight",
    "mlp1.1.weight": "vision_projector.linear1.weight",
    "mlp1.3.weight": "vision_projector.linear2.weight",
}
_VISION_PROJ_CUSTOM_TO_HF = {v: k for k, v in _VISION_PROJ_HF_TO_CUSTOM.items()}

# Sound encoder: HF has "sound_encoder.encoder.*", custom has "sound_encoder.*"
# This prefix stripping is needed because our model stores ParakeetEncoder directly
# as self.sound_encoder, while HF wraps it in a SoundEncoder class with .encoder attr.

# Sound projector: Keys are identical between HF and custom
# sound_projection.norm.weight, sound_projection.linear1.weight, sound_projection.linear2.weight


class NemotronOmniStateDictAdapter(StateDictAdapter):
    """State dict adapter for NemotronOmni (NemotronH_Nano_Omni_Reasoning_V3) models.

    Handles conversion between HF checkpoint format and custom Automodel format.

    The adapter delegates LLM key conversion to NemotronV3StateDictAdapter
    (which handles backbone->model renaming, norm_f->norm, embeddings->embed_tokens,
    and MoE expert merging) and handles vision/audio components directly.
    """

    def __init__(
        self,
        config,
        llm_config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the state dict adapter.

        Args:
            config: Top-level NemotronOmni config
            llm_config: LLM sub-config (NemotronHConfig)
            moe_config: MoE configuration
            backend: Backend configuration
            dtype: Target dtype
        """
        self.config = config
        self.llm_config = llm_config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype

        # Create the LLM state dict adapter (handles expert merging etc.)
        self._llm_adapter = NemotronV3StateDictAdapter(
            config=llm_config,
            moe_config=moe_config,
            backend=backend,
            dtype=dtype,
        )

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint state dict to custom Automodel format.

        Steps:
        1. Separate HF state dict into: vision_model, mlp1, sound_encoder,
           sound_projection, language_model components
        2. Convert vision projector keys (mlp1.* -> vision_projector.*)
        3. Convert sound encoder keys (sound_encoder.encoder.* -> sound_encoder.*)
        4. Pass language_model keys through NemotronV3StateDictAdapter
        5. Merge everything back

        Args:
            hf_state_dict: HuggingFace format state dict
            device_mesh: Optional device mesh for distributed expert loading
            **kwargs: Additional arguments

        Returns:
            Custom format state dict
        """
        result = {}
        llm_state_dict = {}
        debug_counts = {
            "vision_model": 0,
            "vision_projector": 0,
            "sound_encoder": 0,
            "sound_projection": 0,
            "llm": 0,
            "other": 0,
        }

        for key in list(hf_state_dict.keys()):
            value = hf_state_dict.pop(key)

            # 1. Vision model keys (pass through as-is)
            if key.startswith("vision_model."):
                result[key] = value
                debug_counts["vision_model"] += 1

            # 2. Vision projector keys (mlp1.* -> vision_projector.*)
            elif key.startswith("mlp1."):
                if key in _VISION_PROJ_HF_TO_CUSTOM:
                    new_key = _VISION_PROJ_HF_TO_CUSTOM[key]
                    result[new_key] = value
                    debug_counts["vision_projector"] += 1
                    logger.debug(f"  Vision proj: {key} -> {new_key}")
                else:
                    logger.warning(f"  Unknown vision projector key: {key}")
                    result[key] = value
                    debug_counts["other"] += 1

            # 3. Sound encoder keys
            # HF: sound_encoder.encoder.* -> Custom: sound_encoder.*
            # Because our model stores ParakeetEncoder directly as self.sound_encoder
            elif key.startswith("sound_encoder.encoder."):
                new_key = "sound_encoder." + key[len("sound_encoder.encoder."):]
                result[new_key] = value
                debug_counts["sound_encoder"] += 1

            # 4. Sound projection keys (pass through as-is)
            elif key.startswith("sound_projection."):
                result[key] = value
                debug_counts["sound_projection"] += 1

            # 5. LLM keys (language_model.*)
            # Strip "language_model." prefix, pass to NemotronV3StateDictAdapter,
            # then re-add "language_model." prefix
            elif key.startswith("language_model."):
                stripped_key = key[len("language_model."):]
                llm_state_dict[stripped_key] = value
                debug_counts["llm"] += 1

            else:
                logger.warning(f"  Unknown key prefix: {key}")
                result[key] = value
                debug_counts["other"] += 1

        # Convert LLM keys using NemotronV3StateDictAdapter
        logger.info(
            f"NemotronOmni state_dict_adapter.from_hf: "
            f"Passing {len(llm_state_dict)} LLM keys to NemotronV3StateDictAdapter"
        )
        converted_llm = self._llm_adapter.from_hf(llm_state_dict, device_mesh=device_mesh, **kwargs)

        # Re-add "language_model." prefix
        for key, value in converted_llm.items():
            result[f"language_model.{key}"] = value

        # Debug summary
        logger.info(
            f"NemotronOmni state_dict_adapter.from_hf summary: "
            f"vision_model={debug_counts['vision_model']}, "
            f"vision_projector={debug_counts['vision_projector']}, "
            f"sound_encoder={debug_counts['sound_encoder']}, "
            f"sound_projection={debug_counts['sound_projection']}, "
            f"llm={debug_counts['llm']} (converted to {len(converted_llm)}), "
            f"other={debug_counts['other']}, "
            f"total_output={len(result)}"
        )

        # Print sample keys for verification
        if logger.isEnabledFor(logging.DEBUG):
            sample_keys = sorted(result.keys())[:20]
            logger.debug(f"  Sample output keys: {sample_keys}")

        return result

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert custom Automodel state dict to HF format.

        Steps:
        1. Separate state dict into components
        2. Convert vision projector keys back (vision_projector.* -> mlp1.*)
        3. Convert sound encoder keys back (sound_encoder.* -> sound_encoder.encoder.*)
        4. Pass LLM keys through NemotronV3StateDictAdapter.to_hf
        5. Merge everything back

        Args:
            state_dict: Custom format state dict
            exclude_key_regex: Optional regex pattern to exclude keys
            **kwargs: Additional arguments

        Returns:
            HuggingFace format state dict
        """
        hf_result = {}
        llm_state_dict = {}

        for fqn in list(state_dict.keys()):
            tensor = state_dict.pop(fqn)

            if exclude_key_regex and re.match(exclude_key_regex, fqn):
                continue

            # Vision model (pass through)
            if fqn.startswith("vision_model."):
                hf_result[fqn] = tensor

            # Vision projector (custom -> HF)
            elif fqn.startswith("vision_projector."):
                if fqn in _VISION_PROJ_CUSTOM_TO_HF:
                    hf_result[_VISION_PROJ_CUSTOM_TO_HF[fqn]] = tensor
                else:
                    hf_result[fqn] = tensor

            # Sound encoder (custom -> HF: add "encoder." prefix)
            elif fqn.startswith("sound_encoder."):
                new_key = "sound_encoder.encoder." + fqn[len("sound_encoder."):]
                hf_result[new_key] = tensor

            # Sound projection (pass through)
            elif fqn.startswith("sound_projection."):
                hf_result[fqn] = tensor

            # LLM keys (strip language_model. prefix for NemotronV3)
            elif fqn.startswith("language_model."):
                stripped = fqn[len("language_model."):]
                llm_state_dict[stripped] = tensor

            else:
                hf_result[fqn] = tensor

        # Convert LLM keys to HF format
        converted_llm = self._llm_adapter.to_hf(
            llm_state_dict, exclude_key_regex=exclude_key_regex, **kwargs
        )

        # Re-add "language_model." prefix
        for key, value in converted_llm.items():
            hf_result[f"language_model.{key}"] = value

        return hf_result

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: Any, **kwargs
    ) -> list[tuple[str, Any]]:
        """Convert a single tensor from custom format to HF format.

        Args:
            fqn: Fully qualified name of the tensor
            tensor: The tensor to convert
            **kwargs: Additional arguments

        Returns:
            List of (fqn, tensor) tuples in HF format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        # Vision model (pass through)
        if fqn.startswith("vision_model."):
            new_fqn = fqn

        # Vision projector
        elif fqn.startswith("vision_projector."):
            new_fqn = _VISION_PROJ_CUSTOM_TO_HF.get(fqn, fqn)

        # Sound encoder
        elif fqn.startswith("sound_encoder."):
            new_fqn = "sound_encoder.encoder." + fqn[len("sound_encoder."):]

        # Sound projection (pass through)
        elif fqn.startswith("sound_projection."):
            new_fqn = fqn

        # LLM keys
        elif fqn.startswith("language_model."):
            stripped = fqn[len("language_model."):]
            llm_results = self._llm_adapter.convert_single_tensor_to_hf(
                stripped, tensor, **kwargs
            )
            # Re-add language_model. prefix
            results = [(f"language_model.{k}", v) for k, v in llm_results]
            if exclude_key_regex:
                results = [(k, v) for k, v in results if not re.match(exclude_key_regex, k)]
            return results

        else:
            new_fqn = fqn

        result = [(new_fqn, tensor)]
        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
        return result
