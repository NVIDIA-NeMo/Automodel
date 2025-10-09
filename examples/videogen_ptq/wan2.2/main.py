import argparse

from wan_i2v_lora.trainer import WanI2VLoRATrainer


def parse_args():
    p = argparse.ArgumentParser("WAN 2.2 I2V LoRA (FSDP hybrid)")
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
    return p.parse_args()


def main():
    a = parse_args()
    t = WanI2VLoRATrainer(
        model_id=a.model_id,
        lora_rank=a.lora_rank,
        lora_alpha=a.lora_alpha,
        learning_rate=a.learning_rate,
        boundary_ratio=a.boundary_ratio,
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

# torchrun --nproc-per-node=8 main.py --meta_folder /path/to/data --batch_size 1
