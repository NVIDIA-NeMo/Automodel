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
"""Frozen Inkling 2 target for DSpark training.

The released draft reads the target's normalized embedding and eight target
features. Its checkpoint labels a feature by the decoder layer it enters: label
``L`` is the partial attention-residual block that layer ``L`` receives (the
output of layer ``L - 1``), and label ``num_hidden_layers`` is the post-final-norm
hidden state. The generic ``HFDSparkTargetModel`` captures them through
``Inkling2ForCausalLM.get_dspark_target_feature_modules`` at DSpark ids
``label - 1`` (see ``Inkling2DSparkReleasedConfig.dspark_target_layer_ids``).

What the target needs here is its construction: 384 routed experts per layer do
not fit on one GPU, so the frozen target loads through the expert-parallel / FSDP
infrastructure the recipe composes, like the other MoE DSpark targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.config import DistributedSetup


@dataclass(frozen=True)
class Inkling2DSparkTargetConfig:
    """Construction settings of the frozen Inkling 2 DSpark target.

    Attributes:
        target_path: Hub id or local directory of the target checkpoint.
        trust_remote_code: Forwarded to the checkpoint loaders.
    """

    target_path: str
    trust_remote_code: bool = False

    def build(
        self,
        *,
        device: torch.device,
        compute_dtype: torch.dtype,
        distributed_setup: DistributedSetup,
        load_metadata_with_gloo: bool = False,
    ) -> nn.Module:
        """Load the pretrained target through the supplied EP/FSDP infrastructure.

        The backend is the one the Inkling 2 fine-tuning recipes train with: FlexAttention,
        Torch linear and norm layers, and ``torch_mm`` experts behind the HybridEP dispatcher.

        Args:
            device: Resolved execution device; the sharded target requires CUDA.
            compute_dtype: Precision used to load and run the frozen target.
            distributed_setup: Runtime parallelism configuration composed by the recipe.
            load_metadata_with_gloo: Read checkpoint metadata over Gloo.

        Returns:
            The pretrained ``Inkling2ForConditionalGeneration``, frozen and in eval mode.

        Raises:
            RuntimeError: If ``device`` is not CUDA.
        """
        # The transformers bridge imports model configs during registration.
        from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
        from nemo_automodel.components.models.common import BackendConfig

        if device.type != "cuda":
            raise RuntimeError(
                "The Inkling 2 DSpark target requires CUDA: it loads through the expert-parallel / FSDP path."
            )
        model = NeMoAutoModelForImageTextToText.from_pretrained(
            self.target_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=compute_dtype,
            distributed_setup=distributed_setup,
            load_metadata_with_gloo=load_metadata_with_gloo,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            backend=BackendConfig(
                attn="flex",
                linear="torch",
                rms_norm="torch",
                experts="torch_mm",
                dispatcher="hybridep",
                enable_hf_state_dict_adapter=True,
                enable_fsdp_optimizations=True,
            ),
        )
        model.requires_grad_(False)
        return model.eval()


__all__ = ["Inkling2DSparkTargetConfig"]
