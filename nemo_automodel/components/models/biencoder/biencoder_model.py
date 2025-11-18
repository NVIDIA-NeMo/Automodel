# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import gc
import logging
from typing import List, Optional

from torch.nn.attention import SDPBackend

from nemo_automodel import __version__
from nemo_automodel._transformers.auto_model import (
    _BaseNeMoAutoModelClass,
    _patch_attention,
    _patch_liger_kernel,
)

from .llama_bidirectional_model import BiencoderModel

logger = logging.getLogger(__name__)


class NeMoAutoModelBiencoder(_BaseNeMoAutoModelClass):
    """
    NeMo AutoModel Biencoder with custom kernel support.

    This class extends _BaseNeMoAutoModelClass to provide biencoder functionality
    with NeMo AutoModel architecture, including support for Liger kernels and
    SDPA patching optimizations. The parent class handles all kernel patching
    (_patch_liger_kernel, _patch_attention), while this class only customizes
    the model initialization to use BiencoderModel.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        share_encoder=True,
        add_linear_pooler=False,
        out_dimension=None,
        do_gradient_checkpointing=False,
        train_n_passages: int = 1,
        eval_negative_size: int = 0,
        pooling: str = "avg",
        l2_normalize: bool = True,
        t: float = 1.0,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: Optional[List[SDPBackend]] = None,
        **kwargs,
    ):
        """
        Load a biencoder model from pretrained weights.

        This method uses BiencoderModel.build to initialize the model,
        then applies kernel patching from the parent class (_patch_liger_kernel,
        _patch_attention). If patching fails, the method retries with adjusted
        parameters.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier
            share_encoder: Whether to share encoder weights between query and passage
            add_linear_pooler: Whether to add a linear pooler layer
            out_dimension: Output dimension for linear pooler
            do_gradient_checkpointing: Whether to enable gradient checkpointing
            train_n_passages: Number of passages per query during training
            eval_negative_size: Number of negative samples during evaluation
            pooling: Pooling strategy ('avg', 'cls', 'last', etc.)
            l2_normalize: Whether to L2 normalize embeddings
            t: Temperature for scaling similarity scores
            use_liger_kernel: Whether to apply Liger kernel optimizations
            use_sdpa_patching: Whether to apply SDPA patching
            sdpa_method: SDPA backend methods to use
            **kwargs: Additional arguments passed to BiencoderModel.build

        Returns:
            BiencoderModel instance with loaded and patched weights

        Notes:
            If kernel patching fails, the partially constructed model is
            deleted and the method recurses once with use_liger_kernel=False
            or use_sdpa_patching=False
        """
        logger.info(f"Loading NeMoAutoModelBiencoder from {pretrained_model_name_or_path}")

        def _retry(**override):
            """Internal helper to re-enter this function with patched parameters."""
            return cls.from_pretrained(
                pretrained_model_name_or_path,
                share_encoder=share_encoder,
                add_linear_pooler=add_linear_pooler,
                out_dimension=out_dimension,
                do_gradient_checkpointing=do_gradient_checkpointing,
                train_n_passages=train_n_passages,
                eval_negative_size=eval_negative_size,
                pooling=pooling,
                l2_normalize=l2_normalize,
                t=t,
                use_liger_kernel=override.get("use_liger_kernel", use_liger_kernel),
                use_sdpa_patching=override.get("use_sdpa_patching", use_sdpa_patching),
                sdpa_method=sdpa_method,
                **kwargs,
            )

        # Use BiencoderModel.build to initialize model with base encoders
        hf_kwargs = {"attn_implementation": "flash_attention_2"}
        kwargs.update(hf_kwargs)
        model = BiencoderModel.build(
            model_name_or_path=pretrained_model_name_or_path,
            share_encoder=share_encoder,
            add_linear_pooler=add_linear_pooler,
            out_dimension=out_dimension if out_dimension is not None else 768,
            do_gradient_checkpointing=do_gradient_checkpointing,
            train_n_passages=train_n_passages,
            eval_negative_size=eval_negative_size,
            pooling=pooling,
            l2_normalize=l2_normalize,
            t=t,
            **kwargs,
        )

        # Apply kernel patching from parent class
        try:
            if use_liger_kernel:
                logger.info("Applying Liger kernel patching to query encoder")
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logger.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        try:
            if use_sdpa_patching:
                logger.info("Applying SDPA patching to BiencoderModel")
                model = _patch_attention(model, sdpa_method)  # noqa: F821
            model.config.update({"nemo_version": __version__})
            return model
        except Exception:
            logger.warning("Retrying without SDPA patching.")
            del model
            gc.collect()
            return _retry(use_sdpa_patching=False)
