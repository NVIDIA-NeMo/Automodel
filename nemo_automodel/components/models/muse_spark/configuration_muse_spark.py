# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native runtime configuration for the nested Muse Spark Open Hugging Face release."""

from __future__ import annotations

import copy
import math
from typing import Any

from transformers import PretrainedConfig

from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig


class MuseSparkConfig(MuseSpark1xOmniConfig):
    """Read/write the nested HF schema while exposing the native flat text contract.

    The native backbone and distributed machinery consume the inherited text
    attributes. Hugging Face serialization restores the nested release schema;
    a decoder-prefix override preserves the explicit RoPE/window schedule.
    """

    model_type = "muse_spark"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        audio_config: dict[str, Any] | None = None,
        out_hidden_size: int | None = None,
        projector_hidden_size: int = 4096,
        projector_hidden_act: str = "gelu",
        image_token_id: int = 200092,
        video_token_id: int = 200091,
        audio_token_id: int = 200038,
        **kwargs: Any,
    ) -> None:
        text = {
            "model_type": "muse_spark_text",
            "hidden_size": 8192,
            "intermediate_size": 16384,
            "num_hidden_layers": 62,
            "vocab_size": 202048,
            "num_attention_heads": 128,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "moe_hidden_size": 4096,
            "moe_intermediate_size": 4096,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "num_shared_experts": 0,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "residual_gate_temperature": 0.3,
            "qk_scale_factor": 5.4730064863875,
            "output_multiplier": 0.17677669529663687,
            "routed_scaling_factor": 1.0,
            "final_logit_softcapping": 20.0,
            "max_position_embeddings": 1048576,
            "sliding_window": 2048,
            "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
            "attention_bias": False,
            "attention_dropout": 0.0,
            "tie_word_embeddings": False,
            "bos_token_id": 200000,
            "eos_token_id": 200001,
            "pad_token_id": 200018,
        }
        text.update(copy.deepcopy(text_config or {}))
        layers = int(text["num_hidden_layers"])
        text.setdefault("mlp_layer_types", ["sparse"] * layers)
        text.setdefault(
            "layer_types",
            ["full_attention" if (layers - i - 1) % 4 == 0 else "sliding_attention" for i in range(layers)],
        )
        theta = float(text["rope_parameters"]["rope_theta"])
        text.setdefault("layer_rope_theta", [0.0 if (layers - i - 1) % 4 == 0 else theta for i in range(layers)])
        if text["hidden_act"] != "silu" or text["num_shared_experts"] != 0:
            raise ValueError("Muse Spark Open requires SwiGLU routed experts without shared experts")
        if text["mlp_layer_types"] != ["sparse"] * layers:
            raise ValueError("Muse Spark Open currently requires every text layer to be sparse")
        if text["rope_parameters"]["rope_type"] != "default":
            raise ValueError("Muse Spark Open currently requires default text RoPE")
        if len(text["layer_types"]) != layers or len(text["layer_rope_theta"]) != layers:
            raise ValueError("Muse Spark Open text attention and RoPE schedules must match num_hidden_layers")
        if any(kind not in ("full_attention", "sliding_attention") for kind in text["layer_types"]):
            raise ValueError("Unsupported Muse Spark Open text attention layer type")
        if any(value not in (0, theta) for value in text["layer_rope_theta"]):
            raise ValueError("Muse Spark Open supports one shared nonzero text RoPE theta")
        if int(text["moe_hidden_size"]) < 1 or int(text["hidden_size"]) % int(text["moe_hidden_size"]):
            raise ValueError("Muse Spark Open moe_hidden_size must divide hidden_size")
        if float(text["output_multiplier"]) <= 0 or float(text["residual_gate_temperature"]) <= 0:
            raise ValueError("Muse Spark Open output multiplier and residual gate temperature must be positive")

        vision = {
            "model_type": "muse_spark_vision",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_attention_heads": 16,
            "num_hidden_layers": 50,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-5,
            "merge_size": 2,
            "patch_size": 14,
            "patch_temporal": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "max_position_embeddings": 1024,
            "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0},
        }
        vision.update(copy.deepcopy(vision_config or {}))
        vision.setdefault(
            "layer_types",
            [
                "full_attention" if (i + 1) % 4 == 0 or i == vision["num_hidden_layers"] - 1 else "window_attention"
                for i in range(vision["num_hidden_layers"])
            ],
        )
        if len(vision["layer_types"]) != vision["num_hidden_layers"] or any(
            kind not in ("window_attention", "full_attention") for kind in vision["layer_types"]
        ):
            raise ValueError("Muse Spark Open vision layer_types must describe every vision layer")
        if vision["hidden_act"] != "gelu" or vision["rope_parameters"]["rope_type"] != "axial":
            raise ValueError("Muse Spark Open vision requires GELU and axial RoPE")
        if projector_hidden_act != "gelu":
            raise ValueError("Muse Spark Open vision adapter requires GELU")
        expected_output = vision["hidden_size"] * vision["merge_size"] ** 2
        if out_hidden_size is not None and out_hidden_size != expected_output:
            raise ValueError("Muse Spark Open out_hidden_size must equal vision hidden_size * merge_size**2")

        audio = {
            "model_type": "muse_spark_audio",
            "hidden_sizes": [256, 512, 1024, 2048],
            "depths": [2, 2, 2, 2],
            "patch_sizes": [16, 6, 5, 4],
            "num_hidden_layers": 4,
            "conv_kernel_size": 7,
            "hidden_act": "gelu",
            "mlp_ratio": 4.0,
            "rms_norm_eps": 1e-6,
            "out_hidden_size": text["hidden_size"],
        }
        audio.update(copy.deepcopy(audio_config or {}))
        if audio["hidden_act"] != "gelu" or audio["mlp_ratio"] != 4 or audio["rms_norm_eps"] != 1e-6:
            raise ValueError("Muse Spark Open audio requires GELU, mlp_ratio=4 and rms_norm_eps=1e-6")
        if audio["out_hidden_size"] != text["hidden_size"]:
            raise ValueError("Muse Spark Open audio output width must equal the text hidden size")
        if not (
            len(audio["hidden_sizes"])
            == len(audio["depths"])
            == len(audio["patch_sizes"])
            == audio["num_hidden_layers"]
        ):
            raise ValueError("Muse Spark Open audio stage arrays must match num_hidden_layers")

        self._hf_text_config = text
        self._hf_vision_config = vision
        self.audio_config = audio
        self.out_hidden_size = expected_output
        self.projector_hidden_size = projector_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self._output_multiplier = float(text["output_multiplier"])
        self._qk_scale_factor = float(text["qk_scale_factor"])
        native = MuseSpark1xOmniConfig()
        moe = native.moe | {
            "topk": text["num_experts_per_tok"],
            "num_experts": text["num_local_experts"],
            "override_routed_hidden_dim": text["moe_intermediate_size"],
            "compression_factor": text["hidden_size"] // text["moe_hidden_size"],
            "route_scale": text["routed_scaling_factor"],
        }
        values = {
            key: text[key]
            for key in (
                "vocab_size",
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "rms_norm_eps",
                "attention_bias",
                "attention_dropout",
                "tie_word_embeddings",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
            )
        }
        values.update(
            num_hidden_layers=layers,
            schedule_num_hidden_layers=layers,
            rms_post_norm_eps=text["post_norm_eps"],
            rope_theta=theta,
            residual_gate_temperature=text["residual_gate_temperature"],
            ffn_dim_multiplier=text["intermediate_size"] / text["hidden_size"],
            output_soft_cap_temp=text["final_logit_softcapping"],
            global_attn_cfg=[text["sliding_window"]] * 3 + [0],
            moe=moe,
            vision_config=vision,
            audio_perception_modality_args={
                "encoder_family": "wav_ae",
                "input_mode": "wav",
                "use_patch_encoder": False,
                "patchify_sizes": audio["patch_sizes"],
                "hidden_sizes": audio["hidden_sizes"],
                "layers_per_stage": audio["depths"],
                "kernel_size": audio["conv_kernel_size"],
                "patch_size": math.prod(audio["patch_sizes"]),
            },
            image_token_id=image_token_id,
            patch_token_id=image_token_id,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
            architectures=["MuseSparkForConditionalGeneration"],
        )
        values.update(kwargs)
        super().__init__(**values)
        if self.num_hidden_layers > layers:
            raise ValueError("Muse Spark Open cannot extend the checkpoint's explicit text layer schedule")
        self.layer_types = text["layer_types"][: self.num_hidden_layers]
        self.layer_sliding_windows = [
            int(text["sliding_window"]) if kind == "sliding_attention" else 0 for kind in self.layer_types
        ]
        self.sliding_window = int(text["sliding_window"])
        self.no_rope_layers = [not self.layer_uses_rope(i) for i in range(self.num_hidden_layers)]

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], **kwargs: Any
    ) -> MuseSparkConfig | tuple[MuseSparkConfig, dict[str, Any]]:
        """Apply prefix overrides before constructing the attention schedules."""
        # PretrainedConfig normally applies overrides after __init__. The layer
        # count must participate in construction so all schedules stay aligned.
        if "num_hidden_layers" in kwargs:
            config_dict = dict(config_dict, num_hidden_layers=kwargs.pop("num_hidden_layers"))
        return super().from_dict(config_dict, **kwargs)

    def layer_uses_rope(self, layer_idx: int) -> bool:
        """Return the release's explicit per-layer RoPE choice."""
        return self._hf_text_config["layer_rope_theta"][layer_idx] != 0

    @property
    def output_multiplier(self) -> float:
        """Return the release's output scaling factor."""
        return self._output_multiplier

    @property
    def router_output_multiplier(self) -> float:
        """Return the matching MetaP router scaling factor."""
        return self._output_multiplier

    @property
    def scale_query_by(self) -> float:
        """Return the release's post-normalization query scale."""
        return self._qk_scale_factor

    @property
    def vision_navit_config(self) -> dict[str, Any]:
        """Translate the HF vision schema for the native NaViT implementation."""
        vision = self._hf_vision_config
        return vision | {
            "norm_eps": vision["layer_norm_eps"],
            "downsample_factor": vision["merge_size"],
            "image_size": vision["pos_emb_height"] * vision["patch_size"],
            "position_grid_h": vision["pos_emb_height"],
            "position_grid_w": vision["pos_emb_width"],
            "rope_theta": vision["rope_parameters"]["rope_theta"],
            "use_rope": True,
            "vision_output_dim": self.out_hidden_size,
            "projector_input_dim": self.projector_hidden_size,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the HF nested schema, including native decoder-prefix overrides."""
        output = super().to_dict()
        common_keys = set(PretrainedConfig().to_dict())
        native_keys = set(MuseSpark1xOmniConfig().to_dict()) - common_keys
        for key in native_keys | {"_hf_text_config", "_hf_vision_config", "_output_multiplier", "_qk_scale_factor"}:
            output.pop(key, None)
        text = copy.deepcopy(self._hf_text_config)
        text.update(
            {
                key: getattr(self, key)
                for key in (
                    "vocab_size",
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "max_position_embeddings",
                    "rms_norm_eps",
                    "attention_bias",
                    "attention_dropout",
                    "tie_word_embeddings",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "intermediate_size",
                )
            }
        )
        text["layer_types"] = self.layer_types[: self.num_hidden_layers]
        text["layer_rope_theta"] = text["layer_rope_theta"][: self.num_hidden_layers]
        text["mlp_layer_types"] = text["mlp_layer_types"][: self.num_hidden_layers]
        output.update(
            model_type=self.model_type,
            text_config=text,
            vision_config=copy.deepcopy(self._hf_vision_config),
            audio_config=copy.deepcopy(self.audio_config),
            image_token_id=self.patch_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            use_audio_perception=self.use_audio_perception,
        )
        return output
