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

"""Native Step-5 Preview text and vision-language training models."""

from dataclasses import dataclass, replace
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
    shard_batch_contiguous,
)
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5.configuration import (
    ConfigValue,
    Step5BackendConfig,
    Step5Config,
    Step5VLConfig,
)
from nemo_automodel.components.models.step5.cp import gather_metadata
from nemo_automodel.components.models.step5.layers import Step5DecoderLayer, Step5RMSNorm
from nemo_automodel.components.models.step5.parallelization import PARALLELIZER
from nemo_automodel.components.models.step5.sparse import region_metadata
from nemo_automodel.components.models.step5.state_dict_adapter import Step5StateDictAdapter
from nemo_automodel.components.models.step5.vision import Step5VisionEncoder
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str


class Step5TextModel(nn.Module):
    """Embedding, decoder stack and final normalization."""

    def __init__(
        self, config: Step5Config, backend: BackendConfig, moe_config: MoEConfig, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.config = config
        self.csa_forward_full_blocks = isinstance(backend, Step5BackendConfig) and backend.csa_forward_full_blocks
        self.csa_backward_full_blocks = isinstance(backend, Step5BackendConfig) and backend.csa_backward_full_blocks
        self.swa_varlen = isinstance(backend, Step5BackendConfig) and backend.swa_varlen and backend.attn == "fa4"
        self.moe_config = moe_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleDict(
            {
                str(i): Step5DecoderLayer(config, i, backend, moe_config, dtype=dtype)
                for i in range(config.num_hidden_layers)
            }
        )
        self.norm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps, compile_norm=backend.compile_norm)

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
        packed_documents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the decoder stack with FP32 residuals.

        Args:
            embeddings: Tensor of shape [batch, local_sequence, hidden].
            positions: Tensor of shape [batch, local_sequence].
            padding_mask: Boolean tensor of shape [batch, local_sequence], true for padding.
            packed_documents: Optional independent document IDs [batch, local_sequence],
                positive for real tokens and zero for padding.

        Returns:
            Tensor of shape [batch, local_sequence, hidden]: FP32 residuals on
            intermediate pipeline stages, or normalized output on the final stage.
        """
        first_layer = next(iter(self.layers.values()))
        cp_group = first_layer.self_attn.cp_group
        compute_dtype = first_layer.self_attn.qkv_proj.weight.dtype
        fields = (positions, padding_mask.to(positions.dtype))
        if packed_documents is not None:
            if packed_documents.shape != positions.shape:
                raise ValueError("Step5 packed document IDs must match position_ids")
            fields = (*fields, packed_documents)
        metadata = gather_metadata(torch.stack(fields, dim=-1), cp_group)
        global_positions, global_padding = metadata[..., 0], metadata[..., 1].bool()
        if packed_documents is not None:
            # Check once on global metadata, including boundaries crossing CP ranks.
            # The attention kernels then consume only linear-size compact metadata.
            documents = metadata[..., 2]
            valid = ~global_padding
            starts = valid.clone()
            starts[:, 1:] &= global_padding[:, :-1] | (documents[:, 1:] != documents[:, :-1])
            same_document = valid[:, 1:] & valid[:, :-1] & ~starts[:, 1:]
            invalid = (valid & (documents <= 0)) | (global_padding & (documents != 0))
            invalid |= starts & (global_positions != 0)
            invalid[:, 1:] |= same_document & (global_positions[:, 1:] != global_positions[:, :-1] + 1)
            if invalid.any():
                raise ValueError("Step5 packed document IDs, padding and position_ids disagree")
        sparse_metadata = None
        if self.config.sparse_config.enabled:
            query_start = 0 if cp_group is None else dist.get_rank(cp_group) * positions.shape[1]
            sparse_metadata = region_metadata(
                global_positions,
                global_padding,
                query_start=query_start,
                queries=positions.shape[1],
                region_size=self.config.sparse_config.region_block_size,
                topk=self.config.sparse_config.topk,
                window=self.config.sliding_window if self.swa_varlen else None,
                csa_forward_full_blocks=self.csa_forward_full_blocks,
                csa_backward_full_blocks=self.csa_backward_full_blocks,
            )
        residual = embeddings.float()
        for layer in self.layers.values():
            residual = layer(
                residual,
                positions,
                padding_mask,
                global_positions,
                global_padding,
                compute_dtype=compute_dtype,
                sparse_metadata=sparse_metadata,
            )
        # Pipeline boundaries retain the original FP32 residual stream. Only
        # the final stage applies the output norm in the projection dtype.
        return residual if self.norm is None else self.norm(residual.to(compute_dtype))


class Step5ForCausalLM(HFCheckpointingMixin, PreTrainedModel, MoEFSDPSyncMixin):
    """Step-5 causal LM with native grouped experts and model-owned CP."""

    parallelizer = PARALLELIZER
    config_class = Step5Config
    base_model_prefix = "model"
    _pp_keep_self_forward: bool = True
    _pp_return_hidden_states_supported: bool = True
    _pp_return_hidden_states: bool = False
    _owns_cp_attention = True
    _owns_packed_attention = True
    packed_mask_type = "document_ids"
    tie_word_embeddings_support = TieSupport.UNTIED_ONLY
    _no_split_modules = ["Step5DecoderLayer"]
    _keep_in_fp32_modules = ["rotary_emb"]
    _keep_in_fp32_modules_strict = [
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
        "q_norm",
        "k_norm",
        "w_proj",
        "ssmax_s",
        "moe.gate",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallelism supported by the model's explicit hooks."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    def __init__(
        self,
        config: Step5Config | Step5VLConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
    ) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        text = config.text_config if isinstance(config, Step5VLConfig) else config
        self.backend = (
            replace(backend, gate_precision=torch.float32)
            if backend is not None
            else BackendConfig(
                attn="fa4",
                linear="torch",
                rms_norm="torch_fp32",
                experts="torch_mm",
                dispatcher="torch",
                gate_precision=torch.float32,
            )
        )
        dtype = dtype_from_str(text.dtype, torch.bfloat16)
        self.cp_mesh: DeviceMesh | None = None
        moe_config = moe_config or MoEConfig(
            dim=text.hidden_size,
            inter_dim=text.intermediate_size,
            moe_inter_dim=text.moe_intermediate_size,
            n_routed_experts=text.moe_num_experts,
            n_shared_experts=0,
            n_activated_experts=text.moe_top_k,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid_with_bias",
            route_scale=text.moe_router_scaling_factor,
            norm_topk_prob=text.norm_expert_weight,
            force_e_score_correction_bias=text.use_moe_router_bias,
            swiglu_limit=7.0,
            swiglu_clamp_after_activation=True,
            apply_router_weight_after_down=True,
            router_weights_fp32=True,
            dtype=dtype,
            gate_dtype=torch.float32,
        )
        if not moe_config.swiglu_clamp_after_activation:
            raise ValueError("Step5 requires post-SiLU clamping in its MoEConfig")
        self.model = Step5TextModel(text, self.backend, moe_config, dtype=dtype)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text.hidden_size, text.vocab_size, bias=False, dtype=dtype
        )
        self.state_dict_adapter = Step5StateDictAdapter(text, moe_config, self.backend, dtype)
        self.init_weights()
        # Establish FP32 norm storage even under Transformers' BF16 default dtype,
        # before distributed checkpoint loading can round source FP32 weights.
        cast_model_to_dtype(self, dtype)

    @classmethod
    def from_config(
        cls,
        config: Step5Config | Step5VLConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
    ) -> "Step5ForCausalLM":
        """Construct a model through the AutoModel registry."""
        return cls(config, moe_config=moe_config, backend=backend)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize all trainable tensors for checkpoint-free construction."""
        del buffer_device
        # Model-owned projections expose weight tensors for torch/TE linears.
        # Expert tensors are initialized once by MoE, without a second full fill.
        for name, parameter in self.model.named_parameters():
            if parameter.ndim >= 2 and ".moe." not in name:
                nn.init.normal_(parameter, std=0.02)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=0.02)
        for module in self.model.modules():
            if isinstance(module, (Step3p5RMSNorm, nn.LayerNorm)):
                module.reset_parameters()
        for layer in self.model.layers.values():
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer.ssmax_s.zero_()
            if layer.moe is not None:
                layer.moe.init_weights(layer.self_attn.qkv_proj.weight.device)

    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Initialize after meta allocation and apply precision-aware casting."""
        self.init_weights(buffer_device)
        # Construction establishes storage dtypes before FSDP captures parameters.
        # Recasting sharded parameters here replaces FSDP's cached storage.
        if not any(isinstance(parameter, DTensor) for parameter in self.parameters()):
            cast_model_to_dtype(self, dtype)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the language embedding module."""
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        """Return the untied language-model head."""
        return self.lm_head

    def _is_pipeline_parallel_stage(self) -> bool:
        """Identify a stage after AutoPipeline removes modules owned by other stages."""
        return self.model.embed_tokens is None or self.lm_head is None

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Declare stage I/O without rounding the FP32 residual at a pipeline boundary.

        Args:
            is_first: Whether this stage receives token IDs.
            microbatch_size: Number of sequences in each pipeline microbatch.
            seq_len: Local sequence length after any context-parallel sharding.
            dtype: Compute dtype for the final normalized hidden states or logits.

        Returns:
            Input and output metadata tuples. First-stage input IDs have shape
            [microbatch_size, seq_len] and dtype int64. Intermediate residuals
            have shape [microbatch_size, seq_len, hidden] and dtype float32.
            Final outputs use the compute dtype with shape
            [microbatch_size, seq_len, hidden] for fused loss, or
            [microbatch_size, seq_len, vocab] for projected logits.
        """
        hidden = self.config.hidden_size
        if is_first:
            inputs = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs = (torch.empty(microbatch_size, seq_len, hidden, device="meta", dtype=torch.float32),)
        if self.lm_head is None:
            width, output_dtype = hidden, torch.float32
        else:
            width = hidden if self._pp_return_hidden_states else self.config.vocab_size
            output_dtype = dtype
        outputs = (torch.empty(microbatch_size, seq_len, width, device="meta", dtype=output_dtype),)
        return inputs, outputs

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, torch.Tensor],
        *,
        num_chunks: int = 1,
    ) -> dict[str, ContextParallelSharder | torch.Tensor]:
        """Choose the framework's contiguous sharder.

        Args:
            batch: Mapping with [batch, global_sequence] token and label tensors,
                optional [batch, global_sequence, hidden] embeddings and image tensors.
            num_chunks: Number of pipeline microbatches; each uses the same sequence sharder.

        Returns:
            Mapping containing the model's ContextParallelSharder.
        """
        del num_chunks
        packed_documents = batch.get("_packed_seq_ids")
        attention_mask = batch.get("attention_mask")
        if packed_documents is None and attention_mask is not None and attention_mask.ndim == 2:
            if (attention_mask > 1).any():
                packed_documents = attention_mask
        if packed_documents is not None and "position_ids" not in batch:
            raise ValueError("Step5 packed inputs require document-local position_ids")
        updates: dict[str, ContextParallelSharder | torch.Tensor] = {
            "cp_sharder": ContextParallelSharder(
                shard_batch=partial(shard_batch_contiguous, extra_seq_keys={"_packed_seq_ids": 1}),
                local_token_global_indices=contiguous_local_indices,
            )
        }
        if packed_documents is not None:
            updates["_packed_seq_ids"] = packed_documents
        if "padding_mask" not in batch and "attention_mask" not in batch:
            primary = batch["input_ids"] if "input_ids" in batch else batch["inputs_embeds"]
            updates["padding_mask"] = torch.zeros(primary.shape[:2], device=primary.device, dtype=torch.bool)
        return updates

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
        labels: torch.Tensor | None = None,
        _packed_seq_ids: torch.Tensor | None = None,
        **kwargs: ConfigValue | torch.Tensor,
    ) -> CausalLMOutputWithPast | torch.Tensor:
        """Evaluate all local query positions.

        Args:
            input_ids: Int64 tensor of shape [batch, local_sequence], or the FP32
                residual tensor [batch, local_sequence, hidden] on later pipeline stages.
            inputs_embeds: Optional tensor of shape [batch, local_sequence, hidden].
            position_ids: Optional tensor of shape [batch, local_sequence], resetting at document starts.
            attention_mask: Optional tensor of shape [batch, local_sequence], one for valid tokens.
            padding_mask: Optional bool tensor of shape [batch, local_sequence], true for padding.
            logits_to_keep: Last-token count or tensor of sequence indices to project.
            output_hidden_states: Return final hidden states for native fused CE.
            labels: Optional tensor of shape [batch, local_sequence] for serial HF-style shifted CE.
            _packed_seq_ids: Optional [batch, local_sequence] document IDs from native packing.
            **kwargs: Recipe metadata. BSHD inputs are used even for concatenated documents.

        Returns:
            Output with logits [batch, selected_sequence, vocab], optional final
            hidden states [batch, local_sequence, hidden], and scalar loss.
            Intermediate stages put FP32 residuals in the logits field; the
            final stage returns a hidden-state tensor when fused PP loss is enabled.
        """
        if kwargs.get("qkv_format", "bshd") != "bshd":
            raise ValueError("Step5 uses BSHD with position resets for packing; qkv_format must be 'bshd'")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Step5 attention_mask must have shape [batch, sequence]")
        if _packed_seq_ids is None and attention_mask is not None and (attention_mask > 1).any():
            _packed_seq_ids = attention_mask
        if _packed_seq_ids is not None and position_ids is None:
            raise ValueError("Step5 packed inputs require document-local position_ids")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Step5 requires input_ids or inputs_embeds")
            inputs_embeds = input_ids if self.model.embed_tokens is None else self.model.embed_tokens(input_ids)
        if self.model.embed_tokens is None and (
            inputs_embeds.dtype != torch.float32
            or inputs_embeds.ndim != 3
            or inputs_embeds.shape[-1] != self.config.hidden_size
        ):
            raise ValueError("Step5 pipeline stages require FP32 residuals of shape [batch, sequence, hidden]")
        batch, sequence = inputs_embeds.shape[:2]
        if position_ids is None:
            offset = 0 if self.cp_mesh is None else self.cp_mesh.get_local_rank() * sequence
            position_ids = torch.arange(offset, offset + sequence, device=inputs_embeds.device).expand(batch, -1)
        if padding_mask is None:
            padding_mask = (
                torch.zeros((batch, sequence), dtype=torch.bool, device=inputs_embeds.device)
                if attention_mask is None
                else ~attention_mask.bool()
            )
        hidden = self.model(inputs_embeds, position_ids, padding_mask, _packed_seq_ids)
        if self._pp_return_hidden_states:
            return hidden
        if self.lm_head is None:
            return CausalLMOutputWithPast(logits=hidden)
        out = compute_lm_head_logits(
            self.lm_head, hidden, logits_to_keep, is_thd=False, output_hidden_states=output_hidden_states
        )
        if labels is not None:
            if self.cp_mesh is not None and self.cp_mesh.size() > 1:
                raise ValueError("Use the native recipe's globally shifted loss with CP")
            # HF-style labels are unshifted. Never train a document's last
            # token to predict the next document, or predict from/into padding.
            valid_targets = ~padding_mask[:, :-1] & ~padding_mask[:, 1:] & (position_ids[:, 1:] != 0)
            shifted_labels = labels[:, 1:].masked_fill(~valid_targets, -100)
            out.loss = F.cross_entropy(
                out.logits[:, :-1].float().flatten(0, 1), shifted_labels.flatten(), ignore_index=-100
            )
        return out


class Step5ForConditionalGeneration(Step5ForCausalLM):
    """Step-5 language model with the compatible Step perception encoder."""

    config_class = Step5VLConfig

    @dataclass(frozen=True)
    class ModelCapabilities(Step5ForCausalLM.ModelCapabilities):
        """Vision preprocessing and pipeline microbatching are not yet integrated."""

        supports_pp: bool = False

    def __init__(
        self, config: Step5VLConfig, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None
    ) -> None:
        super().__init__(config, moe_config, backend)
        self.vision_model = Step5VisionEncoder(config.vision_config)
        self.vit_large_projector = nn.Linear(
            config.vision_config.width * 4, config.hidden_size, bias=config.projector_bias
        )
        cast_model_to_dtype(self, dtype_from_str(config.text_config.dtype, torch.bfloat16))

    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Initialize both towers after meta allocation, then apply compute dtype."""
        self.init_weights(buffer_device)
        self.vision_model.reset_parameters()
        nn.init.normal_(self.vit_large_projector.weight, std=0.02)
        if self.vit_large_projector.bias is not None:
            nn.init.zeros_(self.vit_large_projector.bias)
        if not any(isinstance(parameter, DTensor) for parameter in self.parameters()):
            cast_model_to_dtype(self, dtype)

    def _encode_images(self, pixels: torch.Tensor, *, compute_dtype: torch.dtype) -> torch.Tensor:
        """Encode overview or crop images.

        Args:
            pixels: Tensor of shape [images, channels, height, width].
            compute_dtype: Dtype of the completed language embedding projection.

        Returns:
            Tensor of shape [images, image_tokens, hidden].
        """
        pixels = pixels.to(compute_dtype)
        features = self.vision_model(pixels, downsample=True)
        return self.vit_large_projector(features)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        patch_pixel_values: torch.Tensor | None = None,
        num_patches: torch.Tensor | list[int] | None = None,
        **kwargs: ConfigValue | torch.Tensor,
    ) -> CausalLMOutputWithPast:
        """Inject projected crop/overview tokens and run the text backbone.

        Args:
            input_ids: Tensor of shape [batch, local_sequence].
            inputs_embeds: Optional tensor of shape [batch, local_sequence, hidden].
            pixel_values: Optional tensor of shape [images, channels, height, width].
            patch_pixel_values: Optional tensor of shape [crops, channels, height, width].
            num_patches: Crop counts of shape [images], or equivalent list.
            **kwargs: Tensor and scalar arguments from Step5ForCausalLM.forward.

        Returns:
            Language output with logits of shape [batch, local_sequence, vocab].
        """
        if pixel_values is not None:
            if input_ids is None or inputs_embeds is not None:
                raise ValueError("Image injection requires input_ids and no precomputed inputs_embeds")
            inputs_embeds = self.model.embed_tokens(input_ids)
            overviews = self._encode_images(
                pixel_values.reshape(-1, *pixel_values.shape[-3:]), compute_dtype=inputs_embeds.dtype
            )
            crops = (
                None
                if patch_pixel_values is None
                else self._encode_images(
                    patch_pixel_values.reshape(-1, *patch_pixel_values.shape[-3:]), compute_dtype=inputs_embeds.dtype
                )
            )
            counts = (
                [0] * overviews.shape[0]
                if num_patches is None
                else (num_patches.tolist() if isinstance(num_patches, torch.Tensor) else num_patches)
            )
            if len(counts) != overviews.shape[0] or sum(counts) != (0 if crops is None else crops.shape[0]):
                raise ValueError("Image and crop counts do not match Step5 visual inputs")
            ordered = []
            offset = 0
            for index, count in enumerate(counts):
                if count:
                    ordered.append(crops[offset : offset + count].flatten(0, 1))
                ordered.append(overviews[index])
                offset += count
            visual = torch.cat(ordered)
            mask = input_ids == self.config.image_token_id
            group = None if self.cp_mesh is None else self.cp_mesh.get_group()
            global_mask = gather_metadata(mask, group)
            if int(global_mask.sum().item()) != visual.shape[0]:
                raise ValueError("Step5 image placeholder count does not match projected visual tokens")
            sequence = input_ids.shape[1]
            rank = 0 if group is None else dist.get_rank(group)
            token_numbers = global_mask.flatten().long().cumsum(0).view_as(global_mask) - 1
            local_numbers = token_numbers[:, rank * sequence : (rank + 1) * sequence][mask]
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = visual[local_numbers].to(inputs_embeds.dtype)
        elif patch_pixel_values is not None:
            raise ValueError("Crop inputs require overview pixel_values")
        return super().forward(input_ids, inputs_embeds=inputs_embeds, **kwargs)


ModelClass = Step5ForCausalLM
