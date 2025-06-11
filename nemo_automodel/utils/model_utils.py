import torch.nn as nn
import logging
from nemo_automodel.utils.dist_utils import get_rank_safe

logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.

    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if get_rank_safe() == 0:
        print("--------------------------------")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {all_param:,}")
        print(
            f"Trainable parameters percentage: {100 * trainable_params / all_param:.2f}%"
        )
        print("--------------------------------")

    return trainable_params, all_param


def apply_parameter_freezing(model, freeze_config):
    """
    Apply parameter freezing based on configuration.

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
    freeze_language_model = freeze_config.get("freeze_language_model", False)

    # Freeze embeddings
    if freeze_embeddings:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.requires_grad = False

    # Freeze vision tower
    if freeze_vision_tower:
        if hasattr(model, "vision_tower"):
            for param in model.vision_tower.parameters():
                param.requires_grad = False
        # Alternative patterns for different VLM architectures
        for name, module in model.named_modules():
            if any(
                pattern in name.lower()
                for pattern in ["vision", "visual", "image_encoder"]
            ):
                for param in module.parameters():
                    param.requires_grad = False

    # Freeze language model backbone
    if freeze_language_model:
        if hasattr(model, "language_model"):
            for param in model.language_model.parameters():
                param.requires_grad = False
        # Alternative patterns
        for name, module in model.named_modules():
            if any(pattern in name.lower() for pattern in ["language", "text", "llm"]):
                for param in module.parameters():
                    param.requires_grad = False

    # Log freezing info
    print_trainable_parameters(model)
