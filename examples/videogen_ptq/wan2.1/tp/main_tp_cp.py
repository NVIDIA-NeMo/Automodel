import argparse

from dist_utils import print0
from trainer_tp_cp import WanI2VLoRATrainer


def parse_args():
    p = argparse.ArgumentParser("WAN 2.2 I2V LoRA (TP+CP hybrid)")
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--boundary_ratio", type=float, default=0.5)
    p.add_argument("--meta_folder", type=str, required=True)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--validate_every", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="./wan_i2v_outputs")
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--train_transformer_2", action="store_true", help="Train transformer_2 instead of transformer")

    # TP+CP specific arguments
    p.add_argument("--tp_size", type=int, default=None, help="Tensor parallelism size (auto-detect if None)")
    p.add_argument("--cp_size", type=int, default=None, help="Context parallelism size (auto-detect if None)")

    return p.parse_args()


def main():
    a = parse_args()

    # Validate parallelism arguments
    if a.tp_size is not None and a.tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if a.cp_size is not None and a.cp_size <= 0:
        raise ValueError("cp_size must be positive")

    print0("[INFO] Starting WAN I2V LoRA training with TP+CP")
    print0(f"[INFO] Tensor Parallelism size: {a.tp_size if a.tp_size else 'auto'}")
    print0(f"[INFO] Context Parallelism size: {a.cp_size if a.cp_size else 'auto'}")
    print0(f"[INFO] Training transformer: {'transformer_2' if a.train_transformer_2 else 'transformer'}")

    t = WanI2VLoRATrainer(
        model_id=a.model_id,
        lora_rank=a.lora_rank,
        lora_alpha=a.lora_alpha,
        learning_rate=a.learning_rate,
        boundary_ratio=a.boundary_ratio,
        train_transformer_2=a.train_transformer_2,
        tp_size=a.tp_size,
        cp_size=a.cp_size,
    )
    t.train(
        meta_folder=a.meta_folder,
        num_epochs=a.num_epochs,
        batch_size=a.batch_size,
        save_every=a.save_every,
        log_every=a.log_every,
        validate_every=a.validate_every,
        output_dir=a.output_dir,
        resume_checkpoint=a.resume_checkpoint,
    )


if __name__ == "__main__":
    main()

# Example usage:
#
# Auto-configure TP+CP (recommended):
# torchrun --nproc-per-node=8 main_tp_cp.py --meta_folder /path/to/meta --batch_size 1
#
# Manual TP+CP configuration:
# torchrun --nproc-per-node=8 main_tp_cp.py --meta_folder /path/to/meta --tp_size 2 --cp_size 4 --batch_size 1
#
# TP only (no CP):
# torchrun --nproc-per-node=4 main_tp_cp.py --meta_folder /path/to/meta --tp_size 4 --cp_size 1 --batch_size 1
#
# CP only (no TP):
# torchrun --nproc-per-node=4 main_tp_cp.py --meta_folder /path/to/meta --tp_size 1 --cp_size 4 --batch_size 1
