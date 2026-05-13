"""Minimal GPU smoke test for DFlashSFTRecipe core logic.

Tests the four novel pieces without the full distributed recipe stack:
  1. Target forward + hidden-state extraction
  2. Anchor-block masking (single-block and multi-block)
  3. Draft forward conditioned on target hidden states
  4. DFlashDecayLoss forward + backward

Run:
    python scripts/test_dflash_sft_gpu.py
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from nemo_automodel.components.loss.dllm_loss import DFlashDecayLoss
from nemo_automodel.recipes.dllm.strategy import DFlashStrategy

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


DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"
TARGET_ID = "Qwen/Qwen3-4B"
DTYPE = torch.bfloat16
DEVICE = "cuda"
SEQ_LEN = 256
BATCH_SIZE = 1
NUM_BLOCKS = 4  # multi-block test


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
    gamma = {16: 7.0, 10: 5.0, 8: 4.0}.get(block_size, block_size / 2.0)
    print(f"  block_size = {block_size}, gamma = {gamma}")

    # --- dummy input ---
    prompt = "The quick brown fox jumps over the lazy dog. " * 10
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=SEQ_LEN, padding="max_length")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # --- mask token ---
    mask_token_id = tok.mask_token_id
    if mask_token_id is None:
        tok.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tok.mask_token_id

    loss_fn = DFlashDecayLoss(loss_gamma=gamma)

    # =========================================================
    # Test 1: single-block forward + backward
    # =========================================================
    print("\n--- Test 1: single-block ---")

    valid_len = int(attention_mask.sum(dim=1).min().item())
    max_start = max(1, valid_len - block_size)
    start = int(torch.randint(1, max_start + 1, (1,), device=DEVICE).item())
    print(f"  anchor start = {start}, valid_len = {valid_len}")

    block_output_ids = input_ids.new_full((BATCH_SIZE, block_size), mask_token_id)
    block_output_ids[:, 0] = input_ids[:, start]
    block_targets = input_ids[:, start + 1 : start + block_size]
    block_mask = attention_mask[:, start + 1 : start + block_size].float()

    with torch.no_grad():
        out = target(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True, use_cache=False)
        target_hidden = _extract_context_features(out.hidden_states, layer_ids)[:, :start, :]
    print(f"  target_hidden shape: {target_hidden.shape}")

    noise_embedding = target_embed(block_output_ids)
    position_ids = torch.arange(start + block_size, device=DEVICE).unsqueeze(0).expand(BATCH_SIZE, -1)

    draft.zero_grad()
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

    logits = target_head(draft_hidden[:, -block_size + 1:, :])
    num_tokens = int(block_mask.sum().item())
    loss_result = loss_fn(logits=logits, target_ids=block_targets, block_mask=block_mask,
                          num_tokens=num_tokens)
    loss = loss_result.total_loss
    print(f"  loss = {loss.item():.4f}")
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in draft.parameters() if p.grad is not None]
    print(f"  grad norms (first 5): {[f'{g:.4f}' for g in grad_norms[:5]]}")
    assert loss.item() > 0, "single-block loss is zero"
    assert len(grad_norms) > 0, "single-block: no gradients"
    print("  PASSED")

    # =========================================================
    # Test 2: multi-block sparse attention forward + backward
    # =========================================================
    print(f"\n--- Test 2: multi-block (N={NUM_BLOCKS}) ---")

    # Use DFlashStrategy._sample_anchor_blocks logic directly.
    avail = valid_len - NUM_BLOCKS * block_size
    assert avail > 0, f"Sequence too short for {NUM_BLOCKS} blocks (valid_len={valid_len})"
    perm = torch.randperm(avail, device=DEVICE)[:NUM_BLOCKS].sort().values
    starts = (perm + torch.arange(NUM_BLOCKS, device=DEVICE) * block_size + 1).tolist()
    starts = [int(s) for s in starts]
    ctx_len = starts[-1]
    n = len(starts)
    print(f"  starts = {starts}, ctx_len = {ctx_len}")

    boi_list, bt_list, bm_list = [], [], []
    for s in starts:
        boi = input_ids.new_full((BATCH_SIZE, block_size), mask_token_id)
        boi[:, 0] = input_ids[:, s]
        boi_list.append(boi)
        bt_list.append(input_ids[:, s + 1 : s + block_size])
        bm_list.append(attention_mask[:, s + 1 : s + block_size].float())

    mb_block_output_ids = torch.cat(boi_list, dim=1)
    mb_block_targets = torch.cat(bt_list, dim=1)
    mb_block_mask = torch.cat(bm_list, dim=1)

    with torch.no_grad():
        out = target(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True, use_cache=False)
        mb_target_hidden = _extract_context_features(out.hidden_states, layer_ids)[:, :ctx_len, :]
    print(f"  mb_target_hidden shape: {mb_target_hidden.shape}")

    mb_noise_embedding = target_embed(mb_block_output_ids)

    ctx_pos = torch.arange(ctx_len, device=DEVICE)
    block_pos = torch.cat([torch.arange(s, s + block_size, device=DEVICE) for s in starts])
    mb_position_ids = torch.cat([ctx_pos, block_pos]).unsqueeze(0).expand(BATCH_SIZE, -1)

    attn_mask = DFlashStrategy._build_block_attention_mask(
        starts, block_size, ctx_len, mb_noise_embedding.dtype, DEVICE
    )
    print(f"  attn_mask shape: {attn_mask.shape}")

    draft.zero_grad()
    mb_draft_hidden = draft(
        target_hidden=mb_target_hidden,
        noise_embedding=mb_noise_embedding,
        position_ids=mb_position_ids,
        attention_mask=attn_mask,
        use_cache=False,
        is_causal=False,
    )
    if not torch.is_tensor(mb_draft_hidden):
        mb_draft_hidden = getattr(mb_draft_hidden, "last_hidden_state", mb_draft_hidden[0])
    print(f"  mb_draft_hidden shape: {mb_draft_hidden.shape}")

    pred = torch.cat(
        [mb_draft_hidden[:, b * block_size + 1 : (b + 1) * block_size, :]
         for b in range(n)],
        dim=1,
    )
    mb_logits = target_head(pred)
    print(f"  mb_logits shape: {mb_logits.shape}")

    mb_num_tokens = int(mb_block_mask.sum().item())
    mb_loss_result = loss_fn(
        logits=mb_logits,
        target_ids=mb_block_targets,
        block_mask=mb_block_mask,
        num_tokens=mb_num_tokens,
        block_size=block_size,
    )
    mb_loss = mb_loss_result.total_loss
    print(f"  loss = {mb_loss.item():.4f}  (block_size={block_size}, N={n})")
    mb_loss.backward()
    mb_grad_norms = [p.grad.norm().item() for p in draft.parameters() if p.grad is not None]
    print(f"  grad norms (first 5): {[f'{g:.4f}' for g in mb_grad_norms[:5]]}")
    assert mb_loss.item() > 0, "multi-block loss is zero"
    assert len(mb_grad_norms) > 0, "multi-block: no gradients"
    print("  PASSED")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
