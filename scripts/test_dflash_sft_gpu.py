"""Minimal GPU smoke test for DFlashSFTRecipe core logic.

Tests the three novel pieces without the full distributed recipe stack:
  1. Target forward + hidden-state extraction
  2. Anchor-block masking
  3. Draft forward conditioned on target hidden states
  4. DFlashDecayLoss forward + backward

Run:
    python scripts/test_dflash_sft_gpu.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


# --- Inlined helpers (avoid importing the full recipe which needs torchao) ---

def _build_target_layer_ids(num_target_layers, num_draft_layers):
    if num_draft_layers == 1:
        return [int(num_target_layers // 2)]
    start, end = 1, int(num_target_layers) - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def _extract_context_features(hidden_states, layer_ids):
    offset = 1
    return torch.cat([hidden_states[lid + offset] for lid in layer_ids], dim=-1)


def _get_target_embeddings(model):
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embed is None:
        embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if head is None:
        head = getattr(model, "lm_head", None)
    return embed, head


class DFlashDecayLoss(torch.nn.Module):
    def __init__(self, loss_gamma=7.0):
        super().__init__()
        self.loss_gamma = float(loss_gamma)

    def forward(self, logits, target_ids, block_mask):
        B, T, V = logits.shape
        nll = F.cross_entropy(logits.reshape(-1, V), target_ids.reshape(-1), reduction="none").reshape(B, T)
        w = torch.exp(-torch.arange(T, device=logits.device, dtype=nll.dtype) / self.loss_gamma)
        weights = w.unsqueeze(0) * block_mask.to(nll.dtype)
        loss = (nll * weights).sum() / weights.sum().clamp_min(1e-8)
        return loss

DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"
TARGET_ID = "Qwen/Qwen3-4B"
DTYPE = torch.bfloat16
DEVICE = "cuda"
SEQ_LEN = 256
BATCH_SIZE = 1


def main():
    print(f"Loading tokenizer from {TARGET_ID} ...")
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"Loading draft model {DRAFT_ID} ...")
    draft = AutoModel.from_pretrained(DRAFT_ID, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE)
    draft.train()

    print(f"Loading target model {TARGET_ID} (frozen) ...")
    target = AutoModelForCausalLM.from_pretrained(TARGET_ID, torch_dtype=DTYPE).to(DEVICE)
    target.eval()
    target.requires_grad_(False)

    target_embed, target_head = _get_target_embeddings(target)

    # --- layer IDs ---
    draft_cfg = getattr(draft, "config", None)
    num_tgt = getattr(draft_cfg, "num_target_layers", None)
    num_hid = getattr(draft_cfg, "num_hidden_layers", None)
    if num_tgt and num_hid:
        layer_ids = _build_target_layer_ids(int(num_tgt), int(num_hid))
    else:
        layer_ids = [target.config.num_hidden_layers // 2]
    print(f"  layer_ids = {layer_ids}")

    # --- block size ---
    block_size = getattr(draft_cfg, "block_size", None) or 16
    print(f"  block_size = {block_size}")

    # --- dummy input ---
    prompt = "The quick brown fox jumps over the lazy dog. " * 10
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=SEQ_LEN, padding="max_length")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # --- anchor masking ---
    mask_token_id = tok.mask_token_id
    if mask_token_id is None:
        tok.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tok.mask_token_id

    valid_len = int(attention_mask.sum(dim=1).min().item())
    max_start = max(1, valid_len - block_size)
    start = torch.randint(1, max_start + 1, (1,), device=DEVICE).item()
    print(f"  anchor start = {start}, valid_len = {valid_len}")

    block_output_ids = input_ids.new_full((BATCH_SIZE, block_size), mask_token_id)
    block_output_ids[:, 0] = input_ids[:, start]
    block_targets = input_ids[:, start + 1 : start + block_size]
    block_mask = attention_mask[:, start + 1 : start + block_size].float()

    # --- target forward (no grad) ---
    print("Running target forward ...")
    with torch.no_grad():
        out = target(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True, use_cache=False)
        target_hidden = _extract_context_features(out.hidden_states, layer_ids)[:, :start, :]
    print(f"  target_hidden shape: {target_hidden.shape}")

    # --- draft forward ---
    noise_embedding = target_embed(block_output_ids)
    position_ids = torch.arange(start + block_size, device=DEVICE).unsqueeze(0).expand(BATCH_SIZE, -1)

    print("Running draft forward ...")
    draft_hidden = draft(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        use_cache=False,
        is_causal=False,
    )
    if not torch.is_tensor(draft_hidden):
        draft_hidden = getattr(draft_hidden, "last_hidden_state", draft_hidden[0])
    print(f"  draft_hidden shape: {draft_hidden.shape}")

    # --- loss + backward ---
    logits = target_head(draft_hidden[:, -block_size + 1:, :])
    print(f"  logits shape: {logits.shape}")

    # paper default gamma for block_size 16
    gamma = {16: 7.0, 10: 5.0, 8: 4.0}.get(block_size, block_size / 2.0)
    loss_fn = DFlashDecayLoss(loss_gamma=gamma)
    loss = loss_fn(logits=logits, target_ids=block_targets, block_mask=block_mask)
    print(f"  loss = {loss.item():.4f}  (gamma={gamma})")

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in draft.parameters() if p.grad is not None]
    print(f"  grad norms (first 5): {[f'{g:.4f}' for g in grad_norms[:5]]}")
    assert loss.item() > 0, "loss is zero"
    assert len(grad_norms) > 0, "no gradients"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
