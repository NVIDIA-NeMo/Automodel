import torch
import torch.nn as nn
from typing import List, Tuple, Iterable, Dict, Any, Set
from dist_utils import print0

# Manual LoRA implementation to avoid PEFT wrapper issues
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(base.weight.new_zeros((r, base.in_features)))
        self.B = nn.Parameter(base.weight.new_zeros((base.out_features, r)))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.t() @ self.B.t() * self.scale

def collect_attention_targets(model: nn.Module) -> Set[str]:
    """
    Collect target module names for LoRA adaptation.
    Focuses on attention projection layers.
    """
    TARGETS = {"to_q", "to_k", "to_v", "to_out", "to_out.0"}
    found_targets = set()
    
    for name, module in model.named_modules():
        module_name = name.split(".")[-1]
        
        # Check for direct linear layers with target names
        if isinstance(module, nn.Linear) and module_name in TARGETS:
            found_targets.add(module_name)
        
        # Check for sequential modules (common in diffusion models)
        if isinstance(module, nn.Sequential) and name.endswith("to_out"):
            if len(module) > 0 and isinstance(module[0], nn.Linear):
                found_targets.add("to_out.0")
    
    # Fallback to basic targets if sequential not found
    if "to_out.0" not in found_targets and "to_out" in found_targets:
        found_targets.add("to_out")
    
    print0(f"[INFO] Found attention targets: {sorted(found_targets)}")
    return found_targets

def wan_install_and_materialize_lora(transformer: nn.Module, rank: int, alpha: int, dropout: float = 0.05) -> int:
    """
    Install LoRA on WAN transformer using manual patching approach.
    This avoids PEFT wrapper issues that can change the forward signature.
    """
    # Target modules for LoRA adaptation
    target_suffixes = {"to_q", "to_k", "to_v", "to_out"}
    
    lora_count = 0
    modules_to_replace = []
    
    # First pass: collect modules to replace
    for name, module in list(transformer.named_modules()):
        module_name = name.split(".")[-1]
        
        # Handle direct linear layers
        if isinstance(module, nn.Linear) and module_name in target_suffixes:
            modules_to_replace.append((name, module, module_name))
        
        # Handle sequential modules (e.g., to_out that contains [Linear, Dropout])
        elif isinstance(module, nn.Sequential) and name.endswith("to_out"):
            if len(module) > 0 and isinstance(module[0], nn.Linear):
                # Replace the first linear layer in the sequence
                modules_to_replace.append((name, module, "to_out_seq"))
    
    # Second pass: replace modules
    for full_name, module, module_type in modules_to_replace:
        try:
            if module_type == "to_out_seq":
                # For sequential modules, replace the first linear layer
                original_linear = module[0]
                lora_linear = LoRALinear(original_linear, r=rank, alpha=alpha, dropout=dropout)
                module[0] = lora_linear
                lora_count += 1
                print0(f"[INFO] Replaced {full_name}[0] with LoRA")
            else:
                # For direct linear layers
                parent_name = ".".join(full_name.split(".")[:-1])
                if parent_name:
                    parent = transformer.get_submodule(parent_name)
                else:
                    parent = transformer
                
                lora_linear = LoRALinear(module, r=rank, alpha=alpha, dropout=dropout)
                setattr(parent, module_type, lora_linear)
                lora_count += 1
                print0(f"[INFO] Replaced {full_name} with LoRA")
                
        except Exception as e:
            print0(f"[WARNING] Failed to replace {full_name}: {e}")
    
    print0(f"[INFO] Installed LoRA on {lora_count} modules")
    return lora_count

def collect_wan_lora_parameters(transformer: nn.Module) -> List[nn.Parameter]:
    """
    Collect LoRA parameters from the transformer.
    After manual LoRA installation, this finds all LoRA A and B matrices.
    This version is designed to work properly with FSDP's use_orig_params=True.
    """
    params: List[nn.Parameter] = []
    
    # First pass: identify all parameters and their types
    base_params = []
    lora_params = []
    
    for name, module in transformer.named_modules():
        if isinstance(module, LoRALinear):
            # LoRA module found - collect its parameters
            lora_params.extend([module.A, module.B])
            # Base parameters in LoRA modules
            base_params.append(module.base.weight)
            if module.base.bias is not None:
                base_params.append(module.base.bias)
        else:
            # Regular module - all parameters are base parameters
            for param in module.parameters(recurse=False):
                base_params.append(param)
    
    # Second pass: set requires_grad appropriately
    # Freeze all base parameters
    for p in base_params:
        p.requires_grad = False
    
    # Enable gradients for LoRA parameters only
    lora_modules_found = 0
    for name, module in transformer.named_modules():
        if isinstance(module, LoRALinear):
            # Enable gradients for LoRA parameters
            module.A.requires_grad = True
            module.B.requires_grad = True
            params.extend([module.A, module.B])
            lora_modules_found += 1
            
            # Double-check base weights are frozen
            module.base.weight.requires_grad = False
            if module.base.bias is not None:
                module.base.bias.requires_grad = False
    
    print0(f"[INFO] Collected {len(params)} LoRA parameters from {lora_modules_found} LoRA modules")
    print0(f"[INFO] Froze {len(base_params)} base parameters")
    
    # Verify parameter shapes
    if params:
        shapes = [f"{p.shape}" for p in params[:5]]  # Show first 5
        print0(f"[INFO] Sample LoRA parameter shapes: {shapes}")
    
    if len(params) == 0:
        print0("[WARNING] No LoRA parameters found! Check if LoRA installation was successful.")
        # Debug: show what modules we found
        module_types = [(name, type(module).__name__) for name, module in transformer.named_modules()]
        print0(f"[DEBUG] Found modules: {module_types[:10]}...")  # Show first 10
    
    return params

# Alternative PEFT-based implementation (use if manual approach has issues)
def wan_install_lora_with_peft(transformer: nn.Module, rank: int, alpha: int) -> int:
    """
    Alternative implementation using PEFT with proper configuration.
    Use this if the manual approach doesn't work for your specific model.
    """
    try:
        from peft import LoraConfig, get_peft_model
        
        # Collect target modules
        targets = collect_attention_targets(transformer)
        if not targets:
            raise RuntimeError("No suitable target modules found for LoRA")
        
        # Create LoRA config with neutral task type
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=list(targets),
            bias="none",
            task_type=None,  # Neutral task type to avoid HF-specific wrappers
        )
        
        # Apply PEFT
        peft_model = get_peft_model(transformer, config)
        
        # Verify forward signature wasn't changed
        import inspect
        sig = inspect.signature(peft_model.forward)
        print0(f"[INFO] PEFT model forward signature: {sig}")
        
        return len(targets)
        
    except ImportError:
        raise RuntimeError("PEFT not available. Install with: pip install peft")
    except Exception as e:
        print0(f"[ERROR] PEFT LoRA installation failed: {e}")
        raise

def collect_peft_lora_parameters(transformer) -> List[nn.Parameter]:
    """
    Collect LoRA parameters from PEFT-wrapped model.
    """
    try:
        params = []
        for name, param in transformer.named_parameters():
            if param.requires_grad and "lora_" in name:
                params.append(param)
        
        print0(f"[INFO] Collected {len(params)} PEFT LoRA parameters")
        
        if len(params) == 0:
            print0("[WARNING] No PEFT LoRA parameters found!")
            # Print all parameter names for debugging
            all_params = [(n, p.requires_grad) for n, p in transformer.named_parameters()]
            print0(f"[DEBUG] All parameters: {all_params[:10]}...")  # Show first 10
        
        return params
        
    except Exception as e:
        print0(f"[ERROR] Failed to collect PEFT LoRA parameters: {e}")
        return []

# Backwards compatibility functions
def wan_has_add_lora(transformer: nn.Module) -> bool:
    """Check if transformer has built-in add_lora method."""
    return hasattr(transformer, "add_lora") and callable(getattr(transformer, "add_lora"))

def broadcast_params(params: Iterable[nn.Parameter], world_size: int, src: int = 0):
    """Broadcast parameters across distributed processes."""
    if world_size <= 1:
        return
    import torch.distributed as dist
    for p in params:
        dist.broadcast(p.data, src=src)

def allreduce_grads(params: Iterable[nn.Parameter], world_size: int):
    """All-reduce gradients across distributed processes."""
    if world_size <= 1:
        return
    import torch.distributed as dist
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)