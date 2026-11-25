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

"""One-depth shared-last-layer ERNIE 5.5 multi-token prediction training.

Semantics reference (independent implementation): NVIDIA-NeMo/Megatron-Bridge
60f7921aa13bff30d801c6b37c063a29786e3626, ERNIE modeling_ernie55/mtp.py,
Apache-2.0. Reuses AutoModel's actual decoder block, norms and MultiMax.
The generic MTPModule consumes post-norm 3-D hidden states; this envelope instead
needs the final 4-D streams and a second call to a single registered decoder body.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed import MeshContext, MoEParallelizerConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.mtp import MTPConfig, shift_packed_tensor
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config, mtp_decoder_config
from nemo_automodel.components.models.ernie5_5.layers import Ernie5_5HyperHead, Ernie5_5RMSNorm
from nemo_automodel.components.models.ernie5_5.model import (
    Ernie5_5_MoeForCausalLM,
    Ernie5_5AttentionInputs,
    Ernie5_5Block,
)
from nemo_automodel.components.models.ernie5_5.mtp_checkpoint_adapter import Ernie5_5MTPCheckpointAdapter
from nemo_automodel.components.models.ernie5_5.mtp_router import Ernie5_5MTPRouterState
from nemo_automodel.components.models.ernie5_5.parallelization import Ernie5_5ModelParallelizer
from nemo_automodel.shared.utils import dtype_from_str


@dataclass
class Ernie5_5MTPOutput(CausalLMOutputWithPast):
    """Standard main logits plus one depth's post-MultiMax auxiliary logits.

    Attributes:
        logits: Main tensor of shape [batch, sequence, vocab].
        mtp_per_depth_logits: One tensor of shape [batch, sequence, vocab], or None
            for decoder-only evaluation. The standard SFT recipe applies the
            same global main-token denominator to main and auxiliary losses.
        mtp_loss_scaling_factor: Auxiliary-loss coefficient; no loss is computed here.
    """

    mtp_per_depth_logits: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float | None = None


class Ernie5_5MTPModelParallelizer(Ernie5_5ModelParallelizer):
    """Require the MTP envelope to belong to the outer FSDP root.

    The inherited flow gives the FP32 hyper-head its own unit. The remaining
    five owned tensors live outside the decoder root and require outer wrapping.
    """

    def _prepare_model(self, model: nn.Module, mesh_context: MeshContext) -> None:
        """Reject CP and uncovered MTP ownership before the executor mutates state.

        Raises:
            ValueError: If CP is enabled, outer wrapping is disabled, or the base parallelizer rejects the topology.
        """
        super()._prepare_model(model, mesh_context)
        if mesh_context.cp_size != 1:
            raise ValueError("ERNIE 5.5 MTP training requires CP1")
        moe = mesh_context.moe_parallel_config or MoEParallelizerConfig()
        if not moe.wrap_outer_model:
            raise ValueError("ERNIE MTP requires wrap_outer_model=True to shard all eight owned parameters")

    def _apply_moe(self, model: nn.Module, mesh_context: MeshContext) -> nn.Module:
        """Check that every trainable MTP parameter was claimed by an FSDP unit.

        Raises:
            RuntimeError: If the outer model is not an FSDP unit or a trainable MTP parameter remains unsharded.
        """
        model = super()._apply_moe(model, mesh_context)
        uncovered = [
            name
            for name, parameter in model.named_parameters()
            if name.startswith("mtp.") and parameter.requires_grad and not isinstance(parameter, DTensor)
        ]
        if not isinstance(model, FSDPModule):
            raise RuntimeError("ERNIE MTP requires the outer model to be an FSDP unit")
        if uncovered:
            raise RuntimeError(f"ERNIE MTP parameters are outside FSDP units: {uncovered}")
        return model


class Ernie5_5MTPEnvelope(nn.Module):
    """Own exactly eight trainable MTP tensors and one persistent router buffer.

    No decoder, embedding or vocabulary module is registered or cached here.
    Resolve the shared body at its decoder path after loading or FSDP wrapping.
    """

    def __init__(self, config: Ernie5_5Config) -> None:
        super().__init__()
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        self.enorm = Ernie5_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.hnorm = Ernie5_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
        self.hyper_head = Ernie5_5HyperHead(config.hc_mult, config.hidden_size, eps=config.rms_norm_eps)
        self.final_norm = Ernie5_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.router_state = Ernie5_5MTPRouterState(config.n_routed_experts)

    def forward(
        self,
        hidden_streams: torch.Tensor,
        *,
        next_embeddings: torch.Tensor,
        shared_body: Ernie5_5Block,
        inputs: Ernie5_5AttentionInputs,
    ) -> torch.Tensor:
        """Fuse the next token with each stream and call the shared body normally.

        Args:
            hidden_streams: Final decoder tensor [batch, sequence, streams, hidden].
            next_embeddings: Shifted embeddings [batch, sequence, hidden]; no wrap.
            shared_body: Registered last decoder block, passed afresh after wrapping.
            inputs: Same position/mask layout as the decoder's original query slots.

        Returns:
            MTP head input [batch, sequence, hidden] in the projection dtype. The
            residual streams round after fusion and after each hyper-connection.
        """
        embedding = self.e_proj(self.enorm(next_embeddings)).unsqueeze(-2)
        previous = self.h_proj(self.hnorm(hidden_streams))
        streams = shared_body(previous + embedding, inputs, routing_state=self.router_state)
        return self.final_norm(self.hyper_head(streams))

    @torch.no_grad()
    def init_weights(self, std: float) -> None:
        """Initialize all owned state, including after meta materialization."""
        for norm in (self.enorm, self.hnorm, self.final_norm):
            norm.reset_parameters()
        nn.init.normal_(self.e_proj.weight, std=std)
        nn.init.normal_(self.h_proj.weight, std=std)
        self.hyper_head.init_weights(std)
        self.router_state.reset_parameters()


class Ernie5_5MTPForCausalLM(Ernie5_5_MoeForCausalLM):
    """Train one shared-last-layer MTP depth with full parameters at CP1 and PP1.

    Each batch contains one unpadded document. Main logits use the decoder's
    existing computation and MultiMax. The SFT recipe divides both main and
    auxiliary CE sums by the same global main-token count, preserving the
    configured auxiliary coefficient per token. Packed inputs, CP, PP, LoRA
    and KV caching are unsupported.
    """

    _keep_in_fp32_modules_strict = [
        *Ernie5_5_MoeForCausalLM._keep_in_fp32_modules_strict,
        "mtp.hyper_head.",
        "mtp.router_state.",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallelism and input layouts admitted by the MTP implementation."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True
        supports_thd: bool = False

    supports_thd = False
    _owns_cp_attention = False
    parallelizer = Ernie5_5MTPModelParallelizer()

    def __init__(
        self,
        config: Ernie5_5Config,
        backend: BackendConfig | None = None,
        *,
        mtp_loss_scaling_factor: float = 0.1,
        logits_dtype: torch.dtype | None = None,
    ) -> None:
        """Build one MTP depth using the decoder's shared body and vocabulary head.

        Args:
            config: One-depth shared-last-layer configuration; it is not modified.
            backend: Supported attention, expert and dispatcher implementations.
            mtp_loss_scaling_factor: Weight of the auxiliary token objective.
            logits_dtype: Output dtype of both head uses; None keeps the head's weight dtype.
        """
        # The native architecture alias selects this training class. HF exports
        # retain the released model-family name; the MTP recipe selects the alias
        # again when loading the exported checkpoint.
        model_config = copy.deepcopy(config)
        if model_config.architectures and model_config.architectures[0] == "Ernie5_5MTPForCausalLM":
            model_config.architectures = ["Ernie5_5_MoeForCausalLM"]
        super().__init__(mtp_decoder_config(model_config), backend, logits_dtype=logits_dtype)
        self.config = model_config
        self.mtp_config = MTPConfig(num_layers=1, layer_pattern="*", loss_scaling_factor=mtp_loss_scaling_factor)
        self.mtp = Ernie5_5MTPEnvelope(self.config)
        # The shared EP expert group cannot be re-unsharded by two AC
        # recomputations. Match upstream's repeated-MTP exclusion without
        # registering the body under a second mtp.layers path.
        self.model.layers[-1]._nemo_disable_activation_checkpointing = True
        cast_model_to_dtype(self, dtype_from_str(self.config.dtype, torch.bfloat16))
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Ernie5_5MTPCheckpointAdapter(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
    ) -> Ernie5_5MTPOutput:
        """Run the decoder and optionally the one-depth shared-body MTP path.

        Args:
            input_ids: Integer tensor [1, sequence] containing one unpadded document.
            attention_mask: Optional all-true tensor [1, sequence].
            padding_mask: Optional all-false tensor [1, sequence].
            position_ids: Optional integer tensor [1, sequence] equal to 0..sequence-1.
            use_cache: Must be False; there is no KV cache.

        Returns:
            Main and optional MTP logits [1, sequence, vocab]; neither includes loss.
            Both use the shared head's ``logits_dtype``, or its weight dtype when
            that constructor argument is None.

        Raises:
            ValueError: If KV caching, CP, padded or multi-document inputs are requested.
            TypeError: If packing metadata such as seq_lens or packed_seq_ids is supplied.
        """
        if use_cache or self.cp_mesh is not None:
            raise ValueError("ERNIE 5.5 MTP training requires no KV cache and CP1")
        self._validate_single_document(input_ids, attention_mask, padding_mask, position_ids)
        if self.training:
            self._require_frozen_indexers()
        enabled = self.training
        callback = partial(self._forward_mtp, input_ids=input_ids) if enabled else None
        # The normal inner-module call is required: its FSDP unit owns the
        # decoder norm and MultiMax, including the latter's second use in MTP.
        result = self.model(input_ids, mtp_forward=callback)
        logits, auxiliary = (result[0], [result[1]]) if enabled else (result, None)
        return Ernie5_5MTPOutput(
            logits=logits,
            mtp_per_depth_logits=auxiliary,
            mtp_loss_scaling_factor=self.mtp_config.loss_scaling_factor,
        )

    def _forward_mtp(
        self, streams: torch.Tensor, inputs: Ernie5_5AttentionInputs, *, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Run MTP inside the active inner FSDP forward, without a cached body.

        Args:
            streams: Final decoder streams [batch, sequence, streams, hidden].
            inputs: Decoder query-slot positions and causal masks.
            input_ids: Original integer tokens [batch, sequence].

        Returns:
            Post-MultiMax logits [batch, sequence, vocab]. The rolled token-0
            tail is routed like any document position, matching the reference;
            its loss has no target. Both correction biases remain frozen.
        """
        shifted_embeddings = self.model.embed_tokens(shift_packed_tensor(input_ids, depth=1))
        hidden = self.mtp(
            streams,
            next_embeddings=shifted_embeddings,
            shared_body=self.model.layers[-1],
            inputs=inputs,
        )
        return self.model.multimax(self.model.lm_head(hidden))

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """Initialize the production decoder and every MTP-owned tensor."""
        super().initialize_weights(buffer_device, dtype)
        self.mtp.init_weights(self.config.initializer_range)
