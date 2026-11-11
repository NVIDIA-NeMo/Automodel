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

"""Praxis world-action policy: the video backbone plus the action expert.

One call denoises a chunk-causal window of ``K`` chunks: the backbone predicts the video
velocity of the noised latents and hands its encoder taps to the action expert, which
predicts the action velocity. The frozen Wan2.2 VAE and the Qwen text encoder are not part
of this module; their outputs are the inputs here.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
from torch import Tensor, nn

from nemo_automodel.components.distributed.parallelizer import ModelParallelizer
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.models.praxis.action_expert import ActionExpert
from nemo_automodel.components.models.praxis.backbone import EncoderFeatures, MoeTrainingSettings, VideoBackbone
from nemo_automodel.components.models.praxis.config import PraxisConfig
from nemo_automodel.components.models.praxis.hub import load_checkpoint_tensors, resolve_repo
from nemo_automodel.components.models.praxis.parallelize import PraxisModelParallelizer
from nemo_automodel.components.models.praxis.state_dict_adapter import PraxisStateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.processing_utils import ProcessorMixin


@dataclass
class PraxisOutput:
    """Outputs of one denoising step.

    Attributes:
        velocity: Video velocity of shape [batch, channels, K * F, h, w], ``None`` when not decoded.
        action_velocity: Action velocity of shape [batch, K * F, action_patch_dim].
        features: The encoder features the action expert read.
    """

    velocity: Tensor | None
    action_velocity: Tensor
    features: EncoderFeatures


def praxis_backend(dispatcher: Literal["torch", "deepep", "hybridep", "uccl_ep"] = "torch") -> BackendConfig:
    """The backend the Praxis checkpoint runs with: torch linears, flex attention, grouped-GEMM experts.

    Args:
        dispatcher: Token exchange of the routed experts under expert parallelism (``distributed.ep_size > 1``):
            the DTensor all-gather of ``"torch"``, or DeepEP, HybridEP or UCCL-EP. Without expert parallelism
            the experts run locally whatever the value.
    """
    return BackendConfig(
        attn="flex",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch_mm" if torch.cuda.is_available() else "torch",
        dispatcher=dispatcher,
        # One token dispatcher per MoE layer: the encoder and decoder towers dispatch different token counts, and a
        # HybridEP buffer shared across layers would be reallocated inside one forward, which invalidates the
        # dispatch handles the earlier layers' backward still needs.
        dispatcher_share_token_dispatcher=False,
        gate_precision=torch.float32,
        enable_hf_state_dict_adapter=True,
    )


def max_tokens_per_window(config: PraxisConfig) -> int:
    """The longest token sequence one training window produces in either tower.

    The decoder runs on the noised chunks' latents one token per latent pixel, ``patch_size[1] * patch_size[2]``
    times the encoder's video tokens, so with its registers it is the longest sequence of a window and the
    per-rank capacity a token dispatcher needs for one window per micro-step (``local_batch_size`` 1; larger
    local batches multiply it).

    Args:
        config: The checkpoint configuration: window geometry, latent stride and register count.

    Returns:
        Number of tokens.
    """
    video, chunking = config.video, config.chunking
    latent_frames = chunking.n_chunks * chunking.latents_per_chunk
    latent_pixels = (video.height // video.vae_stride[1]) * (video.width // video.vae_stride[2])
    return config.backbone.num_registers + latent_frames * latent_pixels


def _per_layer_moe_training(
    config: PraxisConfig, moe_training: MoeTrainingSettings | None
) -> MoeTrainingSettings | None:
    """Divide the balance-loss weight by the number of MoE layers.

    ``MoeTrainingSettings.aux_loss_coeff`` is the weight of the mean over the MoE layers, as in the
    reference trainer; the shared Automodel router adds one term per layer, so each layer gets the
    weight divided by the layer count.
    """
    if moe_training is None or moe_training.aux_loss_coeff == 0:
        return moe_training
    n_moe = sum(
        1
        for tower in (config.backbone.encoder, config.backbone.decoder)
        for i in range(tower.depth)
        if i not in tower.dense_blocks
    )
    return replace(moe_training, aux_loss_coeff=moe_training.aux_loss_coeff / max(1, n_moe))


class CheckpointConfigExporter:
    """Write the source checkpoint's ``config.json`` next to consolidated weights.

    The checkpointer's metadata hook; a Praxis checkpoint carries no tokenizer or generation config, and its
    ``config.json`` is copied verbatim so that the consolidated directory loads in the reference runtime.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path

    def validate(
        self, *, tokenizer: PreTrainedTokenizerBase | ProcessorMixin | None, original_model_path: str | None
    ) -> None:
        """Check that the source config exists before any rank writes."""
        if not self._config_path.is_file():
            raise FileNotFoundError(f"{self._config_path} is missing")

    def save(
        self,
        *,
        hf_metadata_dir: str,
        tokenizer: PreTrainedTokenizerBase | ProcessorMixin | None,
        original_model_path: str | None,
        v4_compatible: bool,
    ) -> None:
        """Copy the source ``config.json`` into ``hf_metadata_dir``."""
        shutil.copyfile(self._config_path, Path(hf_metadata_dir) / "config.json")


class PraxisPolicy(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Video backbone and action expert of a Praxis checkpoint."""

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True

    # The model-owned parallelization: FSDP2, expert parallelism when configured, activation checkpointing.
    parallelizer: ClassVar[ModelParallelizer] = PraxisModelParallelizer()

    def __init__(
        self,
        config: PraxisConfig,
        backend: BackendConfig | None = None,
        moe_training: MoeTrainingSettings | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        backend = backend or praxis_backend()
        if backend.dispatcher == "hybridep" and backend.dispatcher_hybridep_max_tokens_per_rank is None:
            # Size HybridEP's process-global buffer for the longest sequence a window produces: the towers dispatch
            # different token counts, and a buffer sized by the first (encoder) layer would be reallocated inside
            # the forward, which invalidates the dispatch handles the earlier layers' backward still needs.
            backend = replace(backend, dispatcher_hybridep_max_tokens_per_rank=max_tokens_per_window(config))
        self.backend = backend
        dtype = dtype_from_str(config.torch_dtype, torch.bfloat16)
        self.dtype_ = dtype
        self.backbone = VideoBackbone(
            config.backbone, self.backend, dtype, _per_layer_moe_training(config, moe_training)
        )
        self.action_expert = ActionExpert(config.action_expert)
        # Parameters follow the checkpoint dtype; buffers (router biases, norm scales) keep their own.
        for parameter in self.parameters():
            parameter.data = parameter.data.to(dtype)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = PraxisStateDictAdapter(config.backbone.moe.expert_hidden)
        # The checkpoint directory the weights derive from; set by from_pretrained.
        self.pretrained_path: Path | None = None

    @classmethod
    def from_config(cls, config: PraxisConfig, backend: BackendConfig | None = None) -> PraxisPolicy:
        """Build the model with uninitialized weights on the current default device."""
        return cls(config, backend)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        backend: BackendConfig | None = None,
        device: str | torch.device = "cpu",
        moe_training: MoeTrainingSettings | None = None,
    ) -> PraxisPolicy:
        """Build the model from a checkpoint directory or its Hugging Face repo.

        With a real ``device`` the checkpoint tensors are assigned strictly and the model is returned in
        eval mode with gradients disabled. With ``device="meta"`` only the architecture is built; a
        training recipe shards it first and then fills it through the checkpointer
        (:meth:`~nemo_automodel.components.checkpoint.checkpointing.Checkpointer.load_base_model`).

        Args:
            pretrained_model_name_or_path: Checkpoint directory with ``config.json``, the safetensors
                shards and ``model.safetensors.index.json``, or the Hugging Face repo id of one
                (``runwayml/praxis-1-droid``), downloaded to the Hub cache.
            backend: Backend selection; defaults to :func:`praxis_backend`.
            device: Device of the loaded tensors, or ``"meta"`` for the architecture alone.
            moe_training: Router load-balancing terms for training (both zero at inference).

        Returns:
            The model; :attr:`pretrained_path` names the checkpoint directory.
        """
        root = resolve_repo(pretrained_model_name_or_path)
        config = PraxisConfig.from_pretrained(root)
        with torch.device("meta"):
            model = cls(config, backend, moe_training)
        model.pretrained_path = root
        model.name_or_path = str(pretrained_model_name_or_path)
        if torch.device(device).type == "meta":
            return model
        native = model.state_dict_adapter.from_hf(load_checkpoint_tensors(root, device))
        missing, unexpected = model.load_state_dict(native, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing {sorted(missing)[:10]}, unexpected {sorted(unexpected)[:10]}"
            )
        model.place_rotary_tables(torch.device(device))
        return model.eval().requires_grad_(False)

    def _get_consolidated_hf_metadata_exporter(
        self, *, tokenizer: PreTrainedTokenizerBase | ProcessorMixin | None, original_model_path: str | None
    ) -> CheckpointConfigExporter:
        """The checkpointer's metadata hook: the consolidated directory carries the source checkpoint's config."""
        if self.pretrained_path is None:
            raise ValueError("the model has no pretrained_path; build it with PraxisPolicy.from_pretrained")
        return CheckpointConfigExporter(self.pretrained_path / "config.json")

    def place_rotary_tables(self, device: torch.device) -> None:
        """Move the rotary frequency tables to ``device``.

        The tables are plain attributes, not buffers, so module casts and sharding leave them alone;
        they are built once on the CPU and only moved, never recomputed on the accelerator.
        """
        for rope in (self.action_expert.rope, self.backbone.encoder.core.rope, self.backbone.decoder.core.rope):
            rope.theta = rope.theta.to(device)

    @property
    def device(self) -> torch.device:
        return self.backbone.encoder.registers.device

    def moe_layers(self) -> list[MoE]:
        """The MoE feed-forwards of both towers in module order."""
        return [m for m in self.modules() if isinstance(m, MoE)]

    def update_moe_gate_bias(self) -> None:
        """Move every router's selection bias toward balanced expert load (call once per optimizer step)."""
        with torch.no_grad():
            for layer in self.moe_layers():
                if layer.gate.bias_update_factor > 0:
                    layer.gate.update_bias()

    def forward(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        noised_actions: Tensor,
        action_timesteps: Tensor,
        proprio: Tensor,
        rope_shifts: tuple[int, int, int, int] = (0, 0, 0, 0),
        decode: bool = True,
        backbone_grad: bool = True,
    ) -> PraxisOutput:
        """Denoise one window of ``K`` chunks of ``F`` latent frames.

        Args:
            text: Text-encoder states of shape [1, text_tokens, text_dim].
            anchor: Anchor latent of shape [1, channels, 1, h, w].
            clean: Clean copies of chunks ``0 .. K-2`` of shape [1, channels, (K-1) * F, h, w], or ``None``.
            noised: Noised chunks of shape [1, channels, K * F, h, w].
            timesteps: Video flow time per chunk, shape [1, K].
            noised_actions: Noised action tokens of shape [1, K * F, action_patch_dim], chunk-major.
            action_timesteps: Action flow time per chunk, shape [1, K].
            proprio: Normalized boundary pose per chunk, shape [1, K, proprio_patch_dim].
            rope_shifts: ``(encoder anchor, encoder text, encoder video, decoder video)`` rotary offsets.
            decode: Run the decoder tower for the video velocity.
            backbone_grad: Build the autograd graph through the backbone. ``False`` runs it under
                ``torch.no_grad`` for runs that train only the action expert.

        Returns:
            :class:`PraxisOutput`.
        """
        with torch.set_grad_enabled(backbone_grad and torch.is_grad_enabled()):
            out = self.backbone(text, anchor, clean, noised, timesteps, decode=decode, rope_shifts=rope_shifts)
        action_velocity = self.action_expert(out.features, noised_actions, action_timesteps, proprio)
        return PraxisOutput(velocity=out.velocity, action_velocity=action_velocity, features=out.features)


ModelClass = PraxisPolicy
