import os, torch
from typing import Dict, List
from .dist_utils import is_main_process, print0

def _swap_to_base(pipe, transformer_names, model_map):
    old = {}
    for name in transformer_names:
        old[name] = getattr(pipe, name)
        setattr(pipe, name, model_map[name]["base_transformer"])
    return old

def _restore_fsdp(pipe, transformer_names, old):
    for name in transformer_names:
        setattr(pipe, name, old[name])

def save_lora_checkpoint(pipe, model_map, transformer_names, optimizer, scheduler, output_dir: str, step: int):
    if not is_main_process():
        return
    ckpt = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt, exist_ok=True)

    old = _swap_to_base(pipe, transformer_names, model_map)
    try:
        pipe.save_lora_weights(ckpt)
        print0(f"[INFO] Saved LoRA weights to {ckpt}")
    except Exception as e:
        print0(f"[ERROR] Failed to save LoRA weights: {e}")
    finally:
        _restore_fsdp(pipe, transformer_names, old)

    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, os.path.join(ckpt, "training_state.pt"))
    print0(f"[INFO] Checkpoint saved at step {step}")

def load_lora_checkpoint(pipe, model_map, transformer_names, optimizer, scheduler, ckpt_path: str) -> int:
    if not os.path.exists(ckpt_path):
        print0(f"[WARNING] Checkpoint {ckpt_path} not found")
        return 0
    state_path = os.path.join(ckpt_path, "training_state.pt")
    if not os.path.exists(state_path):
        print0(f"[WARNING] Training state not found at {state_path}")
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    old = _swap_to_base(pipe, transformer_names, model_map)
    try:
        pipe.load_lora_weights(ckpt_path)
        print0(f"[INFO] Loaded LoRA weights from {ckpt_path}")
    except Exception as e:
        print0(f"[ERROR] Failed to load LoRA weights: {e}")
    finally:
        _restore_fsdp(pipe, transformer_names, old)

    return int(state.get("step", 0))
