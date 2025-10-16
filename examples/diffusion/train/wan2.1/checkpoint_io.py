import os

import torch
from dist_utils import is_main_process, print0
from lora_utils import LoRALinear  # Add this import
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _swap_to_base(pipe, transformer_names, model_map):
    old = {}
    for name in transformer_names:
        old[name] = getattr(pipe, name)
        setattr(pipe, name, model_map[name]["base_transformer"])
    return old


def _restore_fsdp(pipe, transformer_names, old):
    for name in transformer_names:
        setattr(pipe, name, old[name])


# Keep your existing save_lora_checkpoint for reference/compatibility
def save_lora_checkpoint(pipe, model_map, transformer_names, optimizer, scheduler, output_dir: str, step: int):
    # ... existing implementation ...
    pass


def load_lora_checkpoint(pipe, model_map, transformer_names, optimizer, scheduler, ckpt_path: str) -> int:
    # ... existing implementation ...
    pass


# Add the new manual LoRA checkpoint functions
def save_manual_lora_checkpoint(pipe, model_map, transformer_names, optimizer, scheduler, output_dir: str, step: int):
    """Save manually installed LoRA weights and training state."""
    if not is_main_process():
        return

    ckpt = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt, exist_ok=True)

    # Extract LoRA state dict from each transformer
    for name in transformer_names:
        base_transformer = model_map[name]["base_transformer"]

        lora_state_dict = {}

        with FSDP.summon_full_params(base_transformer, writeback=False):
            for module_name, module in base_transformer.named_modules():
                if isinstance(module, LoRALinear):
                    # Save both A and B matrices with their full path
                    lora_state_dict[f"{module_name}.lora_A"] = module.A.detach().cpu()
                    lora_state_dict[f"{module_name}.lora_B"] = module.B.detach().cpu()
                    lora_state_dict[f"{module_name}.lora_rank"] = module.r
                    lora_state_dict[f"{module_name}.lora_alpha"] = module.scale * module.r

        # Save LoRA weights
        lora_path = os.path.join(ckpt, f"{name}_lora_weights.pt")
        torch.save(lora_state_dict, lora_path)
        print0(f"[INFO] Saved {len(lora_state_dict) // 4} LoRA modules for {name} to {lora_path}")

    # Save training state
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, os.path.join(ckpt, "training_state.pt"))
    print0(f"[INFO] Manual LoRA checkpoint saved at step {step}")


def load_manual_lora_checkpoint(pipe, model_map, transformer_names, optimizer, scheduler, ckpt_path: str) -> int:
    """Load manually installed LoRA weights and training state."""
    if not os.path.exists(ckpt_path):
        print0(f"[WARNING] Checkpoint {ckpt_path} not found")
        return 0

    # Load LoRA weights for each transformer
    for name in transformer_names:
        lora_path = os.path.join(ckpt_path, f"{name}_lora_weights.pt")
        if not os.path.exists(lora_path):
            print0(f"[WARNING] LoRA weights for {name} not found at {lora_path}")
            continue

        lora_state_dict = torch.load(lora_path, map_location="cpu")
        base_transformer = model_map[name]["base_transformer"]

        # Load LoRA parameters
        loaded_count = 0
        for module_name, module in base_transformer.named_modules():
            if isinstance(module, LoRALinear):
                a_key = f"{module_name}.lora_A"
                b_key = f"{module_name}.lora_B"
                if a_key in lora_state_dict and b_key in lora_state_dict:
                    module.A.data.copy_(lora_state_dict[a_key])
                    module.B.data.copy_(lora_state_dict[b_key])
                    loaded_count += 1

        print0(f"[INFO] Loaded {loaded_count} LoRA modules for {name}")

    # Load training state
    state_path = os.path.join(ckpt_path, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state.get("step", 0))
        print0(f"[INFO] Loaded training state from step {step}")
        return step

    return 0
