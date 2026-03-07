"""
Standalone HF fine-tuning script for Gemma 4 2B-IT on MedPix-VQA.

No dependencies on the nemo_automodel repo — uses only transformers, datasets,
and standard PyTorch.

Usage (single GPU):
    python examples/vlm_finetune/gemma4/finetune_hf.py
"""

from __future__ import annotations

import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "/workspace/eevee-4-e2b-it_vv1"
OUTPUT_DIR = "vlm_checkpoints/gemma4_2b_it_hf"
DATASET_ID = "mmoukouba/MedPix-VQA"

LR = 1e-5
MAX_STEPS = 1000
GRAD_ACCUM_STEPS = 8
MAX_LENGTH = 1024
LOG_EVERY = 1
SAVE_EVERY = 100


def build_conversation(example):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": example["answer"]},
            ],
        },
    ]


def make_collate_fn(processor):
    def collate_fn(examples):
        conversations = []
        images = []
        for ex in examples:
            conversations.append(build_conversation(ex))
            images.append([ex["image_id"]])

        texts = processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False,
        )

        batch = processor(
            text=texts, images=images, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_LENGTH,
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        for i, convo in enumerate(conversations):
            user_text = processor.apply_chat_template(
                [convo[0]], tokenize=False, add_generation_prompt=True,
            )
            user_len = processor.tokenizer(
                user_text, add_special_tokens=False, return_tensors="pt",
            )["input_ids"].shape[1]
            labels[i, :user_len] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    device = torch.device("cuda")

    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    print("Loading model...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.config.use_cache = False
    model.to(device)

    # Freeze vision tower and embeddings
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False
    if hasattr(model.model, "embed_tokens"):
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable)
    print(f"Trainable: {trainable_count:,} / {total:,} ({100*trainable_count/total:.1f}%)", flush=True)

    print("Loading dataset...", flush=True)
    train_ds = load_dataset(DATASET_ID, split="train[:1000]")

    dataloader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0,
        pin_memory=True, collate_fn=make_collate_fn(processor),
    )

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01, betas=(0.9, 0.95))

    print(f"Starting training for {MAX_STEPS} steps (grad_accum={GRAD_ACCUM_STEPS})...", flush=True)
    model.train()
    step = 0
    optimizer.zero_grad()

    while step < MAX_STEPS:
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)

            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or step == MAX_STEPS - 1:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % LOG_EVERY == 0:
                mem = torch.cuda.max_memory_allocated() / (1024**3)
                print(
                    f"step {step} | loss {outputs.loss.item():.4f} | "
                    f"mem {mem:.2f} GiB",
                    flush=True,
                )

            if step > 0 and step % SAVE_EVERY == 0:
                save_path = f"{OUTPUT_DIR}/step_{step}"
                model.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}", flush=True)

            step += 1
            if step >= MAX_STEPS:
                break

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
