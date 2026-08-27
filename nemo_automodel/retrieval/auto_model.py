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

"""Auto-model entry points for retrieval and embedding tasks."""

import gc
import logging
from typing import TYPE_CHECKING, List, Optional

import torch
from huggingface_hub import constants as hf_constants
from torch.nn.attention import SDPBackend
from transformers import PreTrainedModel

from nemo_automodel._transformers.auto_model import (
    _reject_separate_distributed_kwargs,
    _resolve_distributed_setup,
)
from nemo_automodel._transformers.infrastructure import apply_model_infrastructure, instantiate_infrastructure
from nemo_automodel._transformers.kernel_patches import (
    DEFAULT_ATTN_IMPLEMENTATION,
    _patch_attention,
    _patch_liger_kernel,
)
from nemo_automodel.components.distributed.config import DistributedSetup

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.utils.compile_utils import CompileConfig


logger = logging.getLogger(__name__)


class _NeMoAutoModelForRetrievalBase:
    """Private shared base for encoder auto-models.

    Subclasses set ``_ENCODER_CLS_NAME`` to select the concrete encoder class
    from ``nemo_automodel.retrieval.modeling``.
    """

    _ENCODER_CLS_NAME: str | None = None  # "BiEncoderModel" or "CrossEncoderModel"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: List[SDPBackend] | None = None,
        torch_dtype="auto",
        distributed_setup: DistributedSetup | None = None,
        device_mesh: Optional["DeviceMesh"] = None,
        compile_config: Optional["CompileConfig"] = None,
        peft_config: dict | None = None,
        **kwargs,
    ) -> PreTrainedModel:
        """Load an encoder model with infrastructure (FSDP, PEFT, kernel patching, etc.).

        This method builds an encoder via the subclass's ``_ENCODER_CLS_NAME``,
        applies kernel patching, and then applies all infrastructure (FSDP,
        checkpointing, etc.) through ``apply_model_infrastructure()``.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier.
            attn_implementation: Attention implementation to use (e.g.,
                ``"flash_attention_2"``, ``"sdpa"``, ``"eager"``).
                Defaults to ``DEFAULT_ATTN_IMPLEMENTATION``
                (``"flash_attention_2"`` when flash-attn is installed, otherwise ``"sdpa"``).
            use_liger_kernel: Whether to apply Liger kernel optimizations.
            use_sdpa_patching: Whether to apply SDPA patching.
            sdpa_method: SDPA backend methods to use.
            torch_dtype: Data type passed to the underlying model initialization.
            distributed_setup: Resolved distributed topology and policy object.
            device_mesh: Pre-created Hugging Face-style device mesh. NeMo wraps it
                in a topology-only ``DistributedSetup`` internally.
            compile_config: Configuration for torch.compile.
            peft_config: PEFT/LoRA configuration dictionary.
            **kwargs: Additional arguments passed to the encoder's ``build()`` method.

        Returns:
            Encoder model instance with loaded weights and all infrastructure applied.

        Notes:
            If kernel patching fails, the method retries with adjusted parameters.
        """
        _reject_separate_distributed_kwargs(kwargs)
        from nemo_automodel.retrieval import modeling as _enc_mod

        encoder_cls = getattr(_enc_mod, cls._ENCODER_CLS_NAME)

        if attn_implementation == "ffpa":
            from nemo_automodel.components.attention.ffpa_attention import register_ffpa_attention

            register_ffpa_attention()

        logger.info(f"Loading {cls.__name__} from {pretrained_model_name_or_path}")

        def _retry(**override):
            return cls.from_pretrained(
                pretrained_model_name_or_path,
                attn_implementation=attn_implementation,
                use_liger_kernel=override.get("use_liger_kernel", use_liger_kernel),
                use_sdpa_patching=override.get("use_sdpa_patching", use_sdpa_patching),
                sdpa_method=sdpa_method,
                torch_dtype=torch_dtype,
                distributed_setup=distributed_setup,
                device_mesh=device_mesh,
                compile_config=compile_config,
                peft_config=peft_config,
                **kwargs,
            )

        build_kwargs = dict(kwargs)
        build_kwargs.pop("tp_size", None)
        build_kwargs.pop("cp_size", None)
        build_kwargs.pop("has_packed_sequence", None)

        setup = _resolve_distributed_setup(
            distributed_setup=distributed_setup,
            device_mesh=device_mesh,
        )
        mesh = setup.mesh_context
        distributed_config = setup.strategy_config
        moe_parallel_config = setup.moe_parallel_config
        activation_checkpointing = setup.activation_checkpointing

        model_wrapper, autopipeline, parallelize_fn, qat_quantizer = instantiate_infrastructure(
            distributed_config=distributed_config,
            pipeline_config=None,
            qat_config=None,
            moe_parallel_config=moe_parallel_config,
            activation_checkpointing=activation_checkpointing,
            device=torch.device("cuda", torch.cuda.current_device()),
            mesh=mesh,
        )

        device = torch.cuda.current_device()

        model = encoder_cls.build(
            model_name_or_path=pretrained_model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            **build_kwargs,
        )

        try:
            if use_liger_kernel:
                logger.info("Applying Liger kernel patching to encoder")
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logger.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        try:
            if use_sdpa_patching:
                logger.info("Applying SDPA patching to encoder")
                model = _patch_attention(model, sdpa_method)  # noqa: F821
        except Exception:
            logger.warning("Retrying without SDPA patching.")
            del model
            gc.collect()
            return _retry(use_sdpa_patching=False)

        model = apply_model_infrastructure(
            model=model,  # noqa: F821
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            is_meta_device=False,
            device=device,
            model_wrapper=model_wrapper,
            mesh=mesh,
            peft_config=peft_config,
            quantization_config=None,
            fp8_config=None,
            qat_quantizer=qat_quantizer,
            loss_fn=None,
            autopipeline=autopipeline,
            parallelize_fn=parallelize_fn,
            compile_config=compile_config,
            load_base_model=False,  # encoder_cls.build already loads weights
            cache_dir=build_kwargs.get("cache_dir", hf_constants.HF_HUB_CACHE),
        )

        return model


class NeMoAutoModelBiEncoder(_NeMoAutoModelForRetrievalBase):
    """NeMo AutoModel for bi-encoder embedding tasks with full infrastructure support.

    Wraps ``BiEncoderModel.build()`` with kernel patching, PEFT, FSDP, and
    other distributed infrastructure via ``apply_model_infrastructure()``.

    Examples:
    --------
    >>> model = NeMoAutoModelBiEncoder.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> model = NeMoAutoModelBiEncoder.from_pretrained(
    ...     "meta-llama/Llama-3.2-1B",
    ...     pooling="cls",
    ...     l2_normalize=False,
    ...     distributed_setup=distributed_setup,
    ... )
    """

    _ENCODER_CLS_NAME = "BiEncoderModel"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        pooling: str | None = None,
        l2_normalize: bool | None = None,
        do_distributed_inbatch_negative: bool = False,
        detach_distributed_inbatch_negatives: bool = True,
        **kwargs,
    ) -> PreTrainedModel:
        """Load a bi-encoder model with infrastructure.

        Accepts all arguments from ``_NeMoAutoModelForRetrievalBase.from_pretrained``
        plus the bi-encoder-specific parameters below. Sentence Transformers export
        metadata is derived from effective model, tokenizer, and collator settings.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier.
            pooling: Pooling strategy (``'avg'``, ``'cls'``, ``'last'``, etc.). When omitted, standard
                Sentence Transformers metadata is restored when available, otherwise defaults to ``'avg'``.
            l2_normalize: Whether to L2-normalize embeddings. When omitted, the standard Sentence Transformers
                module stack is restored when available, otherwise defaults to ``True``.
            do_distributed_inbatch_negative: Whether to gather passages across ranks for distributed in-batch
                negatives during training.
            detach_distributed_inbatch_negatives: Whether to detach remote passage embeddings in distributed
                in-batch-negative losses. Set to false for full cross-rank gradient flow.
            **kwargs: Forwarded to ``_NeMoAutoModelForRetrievalBase.from_pretrained``.

        Returns:
            BiEncoderModel instance with loaded weights and all infrastructure applied.
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            pooling=pooling,
            l2_normalize=l2_normalize,
            do_distributed_inbatch_negative=do_distributed_inbatch_negative,
            detach_distributed_inbatch_negatives=detach_distributed_inbatch_negatives,
            **kwargs,
        )


class NeMoAutoModelCrossEncoder(_NeMoAutoModelForRetrievalBase):
    """NeMo AutoModel for cross-encoder scoring tasks with full infrastructure support.

    Wraps ``CrossEncoderModel.build()`` with kernel patching, PEFT, FSDP, and
    other distributed infrastructure via ``apply_model_infrastructure()``.

    Examples:
    --------
    >>> model = NeMoAutoModelCrossEncoder.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> model = NeMoAutoModelCrossEncoder.from_pretrained(
    ...     "meta-llama/Llama-3.2-1B",
    ...     distributed_setup=distributed_setup,
    ... )
    """

    _ENCODER_CLS_NAME = "CrossEncoderModel"


__all__ = ["NeMoAutoModelBiEncoder", "NeMoAutoModelCrossEncoder"]
