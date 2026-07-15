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

"""NeMo AutoModel wrapper for the Inkling multimodal MoE model.

The wrapper reuses the HuggingFace ``InklingForConditionalGeneration`` in full
(vision/audio towers, attention with short convolutions, relative-position
logits, norms, embeddings, lm_head) and swaps only the sparse feed-forward of
each decoder layer for an expert-parallel :class:`InklingMoE`. This keeps the
non-MoE numerics bit-identical to transformers while giving the routed experts
grouped-GEMM compute and EP sharding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.models.inkling.configuration_inkling import InklingConfig
from transformers.models.inkling.modeling_inkling import (
    InklingForConditionalGeneration as HFInklingForConditionalGeneration,
)
from transformers.models.inkling.modeling_inkling import InklingMoE as HFInklingMoE

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .layers import InklingMoE, build_inkling_moe_config
from .state_dict_adapter import InklingStateDictAdapter


class InklingForConditionalGeneration(HFCheckpointingMixin, HFInklingForConditionalGeneration, MoEFSDPSyncMixin):
    """Inkling VLM with expert-parallel MoE feed-forwards."""

    # Preserve HF's fp32-pinned short-conv modules and additionally pin the router
    # correction bias: bf16 rounding of either flips numerics / expert selection.
    _keep_in_fp32_modules_strict = [
        "attn_sconv",
        "mlp_sconv",
        "k_sconv",
        "v_sconv",
        "e_score_correction_bias",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: InklingConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> "InklingForConditionalGeneration":
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "InklingForConditionalGeneration":
        config = InklingConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: InklingConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> None:
        backend = backend or BackendConfig()

        # Propagate the requested top-level dtype to the nested sub-configs so the
        # HF towers and our MoE parameters are constructed in a consistent dtype.
        top_dtype = getattr(config, "torch_dtype", None)
        if top_dtype is not None:
            for sub_cfg in vars(config).values():
                if sub_cfg is not config and hasattr(sub_cfg, "torch_dtype"):
                    sub_cfg.torch_dtype = top_dtype

        super().__init__(config)

        self.backend = backend
        # Router scoring is selection-sensitive; keep it in fp32 unless overridden.
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32

        text_config = config.text_config
        self.moe_config = moe_config or build_inkling_moe_config(text_config, backend)
        # Exposed on the inner model too so the parallelizer can discover it.
        self.model.moe_config = self.moe_config

        # Swap each sparse decoder layer's HF MoE for the expert-parallel InklingMoE.
        for layer in self.model.language_model.layers:
            if isinstance(layer.mlp, HFInklingMoE):
                layer.mlp = InklingMoE(text_config, backend, moe_config=self.moe_config)

        model_dtype = get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = InklingStateDictAdapter(
                text_config,
                self.moe_config,
                self.backend,
                dtype=model_dtype,
            )

    def update_moe_gate_bias(self) -> None:
        """Inkling uses a trained correction bias, so gate-bias updates are a no-op."""
        return
