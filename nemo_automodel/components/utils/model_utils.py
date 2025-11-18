# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import inspect
import logging
import os
from contextlib import contextmanager

from nemo_automodel.shared.import_utils import safe_import

HAVE_TORCHAO, torch_ao = safe_import("torchao")

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _supports_logits_to_keep(model: nn.Module) -> bool:
    """
    Check if the model supports logits_to_keep.

    Args:
        model (nn.Module): The model to check.

    Returns:
        bool: True if the model supports logits_to_keep, False otherwise.
    """
    if callable(getattr(model, "forward", None)):
        return "logits_to_keep" in set(inspect.signature(model.forward).parameters.keys())
    else:
        return False


def _supports_seq_lens(model: nn.Module) -> bool:
    """
    Check if the model supports seq_lens.
    """
    if callable(getattr(model, "forward", None)):
        return "seq_lens" in set(inspect.signature(model.forward).parameters.keys())
    else:
        return False


def _get_model_param_stats(model: nn.Module) -> tuple[int, int, float]:
    """
    Get the number of trainable parameters and the L2 norm of the model.

    Args:
        model: Model to analyze

    Returns:
        total_params: int
        trainable_params: int
        local_sq_norm: float
    """
    total_params = 0
    trainable_params = 0
    local_sq_norm = 0.0

    for p in model.parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
        try:
            local_sq_norm += float(p.detach().float().norm(2).item() ** 2)
        except Exception:
            pass
    return total_params, trainable_params, local_sq_norm


def resolve_trust_remote_code(pretrained_model_name_or_path):
    """
    Whitelist NVIDIA models to allow remote code execution.

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model.

    Returns:
        bool: True if the model should be loaded with trust_remote_code, False otherwise.
    """
    if not pretrained_model_name_or_path:
        return False
    # pretrained_model_name_or_path can be something like nvidia/NVIDIA-Nemotron-Nano-9B-v2
    return not os.path.isdir(pretrained_model_name_or_path) and pretrained_model_name_or_path.startswith("nvidia/")


def print_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Print the number of trainable parameters in the model.

    Args:
        model: Model to analyze

    Returns:
        trainable_params: int
        total_params: int
    """
    total_params, trainable_params, local_sq_norm = _get_model_param_stats(model)

    try:
        # TODO(@akoumparouli): make this sharding aware.
        local_sq_norm = float(local_sq_norm**0.5)
        trainable_pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

        logging.info("Model summary:")
        logging.info("--------------------------------")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters percentage: {trainable_pct:.2f}%")
        logging.info(f"Param L2 norm: {local_sq_norm:.4f}")
        logging.info("--------------------------------")
    except Exception:
        logging.info("Model summary: <unavailable>")

    return trainable_params, total_params


def _freeze_module_by_attribute_and_patterns(model, attribute_name, name_patterns):
    """Helper function to freeze parameters by attribute name and name patterns.

    Args:
        model: The model to apply freezing to.
        attribute_name: Name of the model attribute to freeze (e.g., 'vision_tower').
        name_patterns: List of patterns to match in module names.
    """
    # Freeze by attribute name
    if hasattr(model, attribute_name):
        for param in getattr(model, attribute_name).parameters():
            param.requires_grad = False

    # Freeze by name patterns
    for name, module in model.named_modules():
        if any(pattern in name.lower() for pattern in name_patterns):
            for param in module.parameters():
                param.requires_grad = False


def apply_parameter_freezing(model, freeze_config):
    """Apply parameter freezing based on configuration.

    Args:
        model: The model to apply freezing to.
        freeze_config: Configuration dict specifying what to freeze.

    freeze_config can contain:
        - freeze_embeddings: bool (default True)
        - freeze_vision_tower: bool (default False)
        - freeze_language_model: bool (default False)
    """
    freeze_embeddings = freeze_config.get("freeze_embeddings", True)
    freeze_vision_tower = freeze_config.get("freeze_vision_tower", True)
    freeze_audio_tower = freeze_config.get("freeze_audio_tower", False)
    freeze_language_model = freeze_config.get("freeze_language_model", False)

    # Freeze embeddings
    if freeze_embeddings:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.requires_grad = False

    # Freeze vision tower
    if freeze_vision_tower:
        _freeze_module_by_attribute_and_patterns(model, "vision_tower", ["vision", "visual", "image_encoder"])

    # Freeze audio tower
    if freeze_audio_tower:
        _freeze_module_by_attribute_and_patterns(model, "audio_tower", ["audio", "audio_encoder"])

    # Freeze language model backbone
    if freeze_language_model:
        _freeze_module_by_attribute_and_patterns(model, "language_model", ["language", "text", "llm"])


def squeeze_input_for_thd(input_ids, position_ids, padding_mask, attn_kwargs, seqlens_padding_value=-1000):
    """
    Squeeze batch dimension and prepare inputs for THD (total, hidden, depth) format.

    This function removes the batch dimension from input tensors and processes attention
    kwargs for use with Transformer Engine's THD format. It's typically used when the
    batch has already been converted to THD format (with batch_size=1 as a placeholder
    dimension) and that dimension needs to be removed.

    The function performs three key operations:
    1. Removes the batch dimension (dim 0) from input tensors
    2. Filters out padding values from cumulative sequence length tensors
    3. Converts max_seqlen from tensor to scalar if needed

    Args:
        input_ids (torch.Tensor): Input token IDs with shape [1, total_tokens] or
            [1, total_tokens, hidden_dim]. The first dimension will be squeezed.
        position_ids (torch.Tensor): Position IDs with shape [1, total_tokens].
            The first dimension will be squeezed.
        padding_mask (torch.Tensor): Padding mask with shape [1, total_tokens].
            The first dimension will be squeezed.
        attn_kwargs (dict): Dictionary of attention-related tensors. May contain:
            - cu_seqlens: Cumulative sequence lengths [1, num_seqs+1]
            - cu_seqlens_padded: Cumulative padded sequence lengths [1, num_seqs+1]
            - max_seqlen: Maximum sequence length (tensor or int)
            - Other attention parameters (will be squeezed if tensors)
        seqlens_padding_value (int): Sentinel value used to indicate padding in
            cu_seqlens and cu_seqlens_padded tensors. These values will be filtered
            out. Default: -1000.

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Input IDs with batch dimension removed [total_tokens]
                or [total_tokens, hidden_dim]
            - position_ids (torch.Tensor): Position IDs with batch dimension removed [total_tokens]
            - padding_mask (torch.Tensor): Padding mask with batch dimension removed [total_tokens]
            - attn_kwargs (dict): Updated attention kwargs with:
                - Batch dimensions removed from all tensor values
                - Padding values filtered from cu_seqlens and cu_seqlens_padded
                - max_seqlen converted to scalar if it was a tensor

    Example:
        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [1, 5]
        >>> position_ids = torch.tensor([[0, 1, 2, 3, 4]])  # [1, 5]
        >>> padding_mask = torch.tensor([[False, False, False, False, False]])  # [1, 5]
        >>> attn_kwargs = {
        ...     'cu_seqlens': torch.tensor([[0, 3, 5, -1000]]),  # [1, 4] with padding
        ...     'cu_seqlens_padded': torch.tensor([[0, 3, 5, -1000]]),
        ...     'max_seqlen': torch.tensor([3])
        ... }
        >>> ids, pos, mask, kwargs = squeeze_input_for_thd(
        ...     input_ids, position_ids, padding_mask, attn_kwargs
        ... )
        >>> ids.shape
        torch.Size([5])
        >>> kwargs['cu_seqlens']  # Padding value filtered out
        tensor([0, 3, 5])
        >>> kwargs['max_seqlen']  # Converted to scalar
        3

    Note:
        This function modifies attn_kwargs in-place. If you need to preserve the original
        dictionary, pass a copy.
    """
    input_ids = input_ids.squeeze(0)
    position_ids = position_ids.squeeze(0)
    if padding_mask is not None:
        padding_mask = padding_mask.squeeze(0)
    for key, value in attn_kwargs.items():
        if isinstance(value, torch.Tensor):
            attn_kwargs[key] = value.squeeze(0)
        if key in ["cu_seqlens", "cu_seqlens_padded"]:
            attn_kwargs[key] = value[value != seqlens_padding_value].contiguous()
        if key == "max_seqlen" and isinstance(value, torch.Tensor):
            attn_kwargs[key] = value.item()

    return input_ids, position_ids, padding_mask, attn_kwargs


# taken and edited from https://github.com/huggingface/transformers/blob/32a58e31463e238c967207bf73772490c353551a/src/transformers/integrations/accelerate.py#L53-L158
@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.

    Example:

    ```python
    import torch.nn as nn
    from nemo_automodel.components.utils.model_utils import init_empty_weights

    with init_empty_weights():
        tst = nn.Linear(100, 100)  # on `cuda` device
    ```
    """
    device = torch.device("meta")
    fp8_parameter_mapping = {
        "_linear_mm_config": "linear_mm_config",
        "_dtype": "dtype",
        "_precomputed_scale": "precomputed_scale",
    }
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            if HAVE_TORCHAO and isinstance(
                module._parameters[name], torch_ao.float8.fsdp_utils.WeightWithDynamicFloat8CastTensor
            ):
                kwargs = {}
                for k in module._parameters[name].__dict__:
                    if k in fp8_parameter_mapping:
                        kwargs[fp8_parameter_mapping[k]] = getattr(module._parameters[name], k)
            else:
                kwargs = module._parameters[name].__dict__
                kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
