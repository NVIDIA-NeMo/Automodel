import torch, torch.nn as nn
from typing import List, Tuple, Iterable, Dict, Any
from .dist_utils import print0

# Robust helpers to interact with WAN 2.2 LoRA in diffusers

def wan_has_add_lora(transformer: nn.Module) -> bool:
    return hasattr(transformer, "add_lora") and callable(getattr(transformer, "add_lora"))

def wan_install_and_materialize_lora(transformer: nn.Module, rank: int, alpha: int) -> int:
    """
    Preferred: call WAN's own method to attach & materialize LoRA modules.
    Returns number of processors (or blocks) updated (best-effort).
    """
    if not wan_has_add_lora(transformer):
        raise RuntimeError("WAN transformer lacks `.add_lora(rank, alpha)`; cannot materialize trainable LoRA.")

    # Let the model create real nn.Parameter LoRA weights internally.
    transformer.add_lora(rank=rank, alpha=alpha)   # WAN 2.2 implements this
    # Count approx processors if available
    if hasattr(transformer, "attn_processors"):
        n = len(getattr(transformer, "attn_processors", {}))
        print0(f"[INFO] Installed {n} WAN LoRA processors")
        return n
    print0("[INFO] Installed WAN LoRA (count unknown)")
    return 0

def _iter_named_modules(obj: Any) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in getattr(obj, "named_modules", lambda: [])():
        yield name, module

def _looks_like_lora_module(name: str, module: nn.Module) -> bool:
    n = name.lower()
    cls = module.__class__.__name__.lower()
    return ("lora" in n) or ("lora" in cls) or n.endswith("_lora") or n.startswith("lora_")

def collect_wan_lora_parameters(transformer: nn.Module) -> List[nn.Parameter]:
    """
    After `add_lora`, WAN adds actual LoRA submodules (nn.Module) with nn.Parameters.
    We walk the transformer, pick modules that look like LoRA, and gather parameters.
    """
    params: List[nn.Parameter] = []

    # Freeze base by default; only LoRA trainable
    for p in transformer.parameters():
        p.requires_grad = False

    found = 0
    for mod_name, mod in _iter_named_modules(transformer):
        if _looks_like_lora_module(mod_name, mod):
            for p in mod.parameters(recurse=True):
                p.requires_grad = True
                params.append(p)
            found += 1

    print0(f"[INFO] Collected {len(params)} LoRA parameters from {found} LoRA submodules")
    return params

def broadcast_params(params: Iterable[nn.Parameter], world_size: int, src: int = 0):
    if world_size <= 1:
        return
    import torch.distributed as dist
    for p in params:
        dist.broadcast(p.data, src=src)

def allreduce_grads(params: Iterable[nn.Parameter], world_size: int):
    if world_size <= 1:
        return
    import torch.distributed as dist
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)
