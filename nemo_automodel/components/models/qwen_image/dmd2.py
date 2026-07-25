# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen-Image model ownership for Model Optimizer DMD2 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import nn

from nemo_automodel.shared.import_utils import safe_import_from

_MODELOPT_QWEN_IMAGE_ERROR = (
    "Qwen-Image DMD2 requires a Model Optimizer build with the Diffusers 0.37+ "
    "Qwen-Image plugin fixes. Install it with `uv sync --extra dmd2`."
)


class _DMDConfigWithGAN(Protocol):
    """DMD configuration fields consumed by the Qwen-Image adapter."""

    gan_loss_weight_gen: float


class _FeatureCapture(Protocol):
    """Callable contract for Model Optimizer's Qwen-Image feature hook."""

    def __call__(
        self,
        model: nn.Module,
        *,
        feature_indices: list[int],
        h_lat: int,
        w_lat: int,
    ) -> None:
        """Attach or resize feature capture for one latent resolution."""
        ...


@dataclass(frozen=True)
class _QwenImageModelOptAPI:
    """Model Optimizer symbols owned by the Qwen-Image integration."""

    pipeline_cls: type
    discriminator_cls: type[nn.Module]
    attach_feature_capture: _FeatureCapture


class QwenImageDMD2Adapter:
    """Own Qwen-Image-specific DMD2 calls, topology, and attention behavior."""

    def __init__(
        self,
        *,
        guidance: float | None = None,
        gan_feature_indices: list[int] | tuple[int, ...] = (30,),
        gan_num_blocks: int = 60,
        gan_inner_dim: int = 3072,
    ) -> None:
        """Create the declaratively configured Qwen-Image adapter.

        Args:
            guidance: Optional Qwen-Image guidance embedding value.
            gan_feature_indices: Transformer block indices captured by the GAN.
            gan_num_blocks: Number of Qwen-Image transformer blocks.
            gan_inner_dim: Hidden dimension consumed by the ImageDiT discriminator.
        """
        self.guidance = guidance
        self.gan_feature_indices = tuple(int(index) for index in gan_feature_indices)
        self.gan_num_blocks = int(gan_num_blocks)
        self.gan_inner_dim = int(gan_inner_dim)
        self._modelopt_api: _QwenImageModelOptAPI | None = None

    def require_modelopt_dependencies(self) -> None:
        """Resolve the optional Model Optimizer Qwen-Image API."""
        pipeline_available, pipeline_cls = safe_import_from(
            "modelopt.torch.fastgen.plugins.qwen_image",
            "QwenImageDMDPipeline",
            msg=_MODELOPT_QWEN_IMAGE_ERROR,
        )
        hook_available, attach_feature_capture = safe_import_from(
            "modelopt.torch.fastgen.plugins.qwen_image",
            "attach_feature_capture",
            msg=_MODELOPT_QWEN_IMAGE_ERROR,
        )
        shape_update_available, _ = safe_import_from(
            "modelopt.torch.fastgen.plugins.qwen_image",
            "update_feature_capture_shape",
            msg=_MODELOPT_QWEN_IMAGE_ERROR,
        )
        discriminator_available, discriminator_cls = safe_import_from(
            "modelopt.torch.fastgen.discriminators",
            "Discriminator_ImageDiT",
            msg=_MODELOPT_QWEN_IMAGE_ERROR,
        )
        if not all(
            (
                pipeline_available,
                hook_available,
                shape_update_available,
                discriminator_available,
            )
        ):
            raise ImportError(_MODELOPT_QWEN_IMAGE_ERROR)

        self._modelopt_api = _QwenImageModelOptAPI(
            pipeline_cls=pipeline_cls,
            discriminator_cls=discriminator_cls,
            attach_feature_capture=attach_feature_capture,
        )

    @property
    def parallel_model_class_name(self) -> str:
        """Return the model class key used by AutoModel parallelization."""
        return "QwenImageTransformer2DModel"

    @staticmethod
    def checkpoint_transformer_blocks(model: nn.Module) -> int:
        """Apply the model-owned Qwen-Image activation-checkpoint boundary."""
        from nemo_automodel.components.models.qwen_image.fsdp import checkpoint_qwen_image_transformer_blocks

        return checkpoint_qwen_image_transformer_blocks(model)

    @staticmethod
    def validate_transformer(model: nn.Module, *, name: str) -> None:
        """Require the Diffusers Qwen-Image transformer contract."""
        candidate = getattr(model, "module", model)
        class_names = {cls.__name__ for cls in type(candidate).__mro__}
        if "QwenImageTransformer2DModel" not in class_names:
            raise TypeError(f"DMD2 {name} must be a QwenImageTransformer2DModel, got {type(candidate).__name__}.")

    def configure_transformer(
        self,
        model: nn.Module,
        *,
        name: str,
        attention_backend: str | None,
    ) -> None:
        """Validate a Qwen-Image transformer and set its attention backend."""
        self.validate_transformer(model, name=name)
        if attention_backend is not None:
            candidate = getattr(model, "module", model)
            candidate.set_attention_backend(attention_backend)

    def validate_dmd_config(self, config: _DMDConfigWithGAN) -> None:
        """Validate Qwen-Image GAN geometry when the GAN branch is enabled."""
        if config.gan_loss_weight_gen <= 0:
            return
        if not self.gan_feature_indices:
            raise ValueError("GAN-enabled Qwen-Image DMD2 requires at least one feature index.")
        invalid_indices = [index for index in self.gan_feature_indices if index < 0 or index >= self.gan_num_blocks]
        if invalid_indices:
            raise ValueError(
                f"Qwen-Image GAN feature indices {invalid_indices} are outside [0, {self.gan_num_blocks})."
            )
        if self.gan_inner_dim < 2:
            raise ValueError(f"Qwen-Image gan_inner_dim must be at least 2, got {self.gan_inner_dim}.")

    @staticmethod
    def normalize_text_mask(
        mask: torch.Tensor | None,
        *,
        attention_backend: str | None,
        prompt_kind: Literal["positive", "negative"],
    ) -> torch.Tensor | None:
        """Normalize a Qwen-Image prompt mask for the selected attention backend.

        Args:
            mask: Tensor of shape ``[sequence]`` or ``[batch, sequence]`` with
                one for valid tokens and zero for padding, or ``None``.
            attention_backend: Configured Qwen-Image attention backend.
            prompt_kind: Whether the mask belongs to positive or negative text.

        Returns:
            The input mask for non-Flash backends, or ``None`` for an all-valid
            Flash-attention sequence.

        Raises:
            ValueError: If Flash attention receives a padded sequence.
        """
        if mask is None or attention_backend != "flash":
            return mask
        if not torch.all(mask == 1):
            raise ValueError(
                f"Qwen-Image flash attention cannot consume a padded {prompt_kind}-prompt mask; "
                "use trimmed embeddings or set model.attention_backend=null."
            )
        return None

    def build_discriminator(
        self,
        config: _DMDConfigWithGAN,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module | None:
        """Build the Qwen-Image ImageDiT discriminator when GAN is enabled."""
        if config.gan_loss_weight_gen <= 0:
            return None
        self.validate_dmd_config(config)
        api = self._require_api()
        discriminator = api.discriminator_cls(
            feature_indices=set(self.gan_feature_indices),
            num_blocks=self.gan_num_blocks,
            inner_dim=self.gan_inner_dim,
        )
        discriminator.to(device=device, dtype=dtype)
        discriminator.train()
        return discriminator

    def attach_feature_capture(
        self,
        teacher: nn.Module,
        *,
        height: int,
        width: int,
    ) -> None:
        """Attach Qwen-Image GAN feature capture for a latent resolution."""
        self._require_api().attach_feature_capture(
            teacher,
            feature_indices=list(self.gan_feature_indices),
            h_lat=height,
            w_lat=width,
        )

    def build_pipeline(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        fake_score: nn.Module,
        config: object,
        discriminator: nn.Module | None,
    ) -> object:
        """Build Model Optimizer's Qwen-Image DMD2 pipeline."""
        return self._require_api().pipeline_cls(
            student=student,
            teacher=teacher,
            fake_score=fake_score,
            config=config,
            discriminator=discriminator,
            guidance=self.guidance,
        )

    def _require_api(self) -> _QwenImageModelOptAPI:
        """Return resolved Model Optimizer symbols or fail on lifecycle misuse."""
        if self._modelopt_api is None:
            raise RuntimeError("Call require_modelopt_dependencies() before using the Qwen-Image DMD2 adapter.")
        return self._modelopt_api
