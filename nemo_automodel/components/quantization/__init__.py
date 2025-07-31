from .fp8 import (
    apply_fp8_to_model, 
    precompute_fp8_scales_for_fsdp,
    verify_fp8_conversion,
    HAVE_TORCHAO
)
from .fp8 import FP8Config

# Import Float8LinearConfig only if available
if HAVE_TORCHAO:
    from .fp8 import Float8LinearConfig

__all__ = [
    "apply_fp8_to_model", 
    "precompute_fp8_scales_for_fsdp",
    "verify_fp8_conversion",
    "HAVE_TORCHAO",
    "FP8Config"
] 

# Add Float8LinearConfig to exports only if available
if HAVE_TORCHAO:
    __all__.append("Float8LinearConfig") 