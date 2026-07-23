# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashAttention-2 EAGLE-3 packing parity + speed check on GPU.

Random-initialises four documents of length 600 / 700 / 800 / 9000 (the last is
truncated to ``SEQ_LENGTH``) and trains them once each, ``micro_batch_size=1``
(the recipe default; FA2 position-id packing requires batch size 1), under two
layouts:

  * no packing: four ``[1, SEQ_LENGTH]`` rows, one padded document each;
  * packing: greedily packed into ``[1, SEQ_LENGTH]`` rows (here two: the three
    short docs share one row, the truncated long doc fills another).

The target (FA2, the path under test) isolates documents from per-document
position_ids; the draft runs eager (the recipe default) in fp32, fed the target's
bf16 aux/logits cast up at the boundary.

Both layouts supervise the identical (document, position, TTT-step) triples, so
the valid-token-weighted global loss and accumulated draft gradients match within
bf16 FA2 tolerance, while packing drops the padding compute and halves the steps.
Reports the speedup and the loss / gradient deltas.

Run directly for the numbers (``pytest -s`` or ``python``); needs one GPU with a
working flash-attn build.
"""

from __future__ import annotations

import importlib.util
import time

import pytest
import torch

from nemo_automodel.components.datasets.llm.eagle3 import _pack_collate, build_packed_eagle3_dataset
from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel

_HAS_FA = importlib.util.find_spec("flash_attn") is not None

from transformers import LlamaConfig, LlamaForCausalLM

SEQ_LENGTH = 4096
# Four rollouts of the same ascending length profile (9000 is truncated to
# SEQ_LENGTH), enough documents to exercise multi-row packing.
DOC_LENS = [
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
] * 4
HIDDEN = 512
VOCAB = 2048
TARGET_LAYERS = 8  # deep enough for the default aux ids [1, 3, 4]
TTT_STEPS = 7
# Target runs the FA2 path under test in bf16 (FA2 has no fp32 kernel); the draft
# stays fp32 so its fp32-RoPE q/k stay dtype-consistent and parity isn't muddied
# by extra draft-side bf16 noise. Target aux/logits are cast to fp32 at the boundary.
TARGET_DTYPE = torch.bfloat16
DRAFT_DTYPE = torch.float32


def _make_documents() -> list[dict[str, list[int]]]:
    """Four random documents (input_ids + all-ones loss_mask), pre-truncated to T."""
    torch.manual_seed(7)
    docs = []
    for length in DOC_LENS:
        eff = min(length, SEQ_LENGTH)
        ids = torch.randint(0, VOCAB, (eff,)).tolist()
        docs.append({"input_ids": ids, "loss_mask": [1] * eff})
    return docs


def _build_target() -> HFEagle3TargetModel:
    config = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=TARGET_LAYERS,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LENGTH,
        attn_implementation="flash_attention_2",
    )
    target = LlamaForCausalLM(config).to(device="cuda", dtype=TARGET_DTYPE).eval()
    target.requires_grad_(False)
    return HFEagle3TargetModel(target)


def _build_trainer() -> Eagle3TrainerModule:
    """Identical draft init for both layouts (fixed seed); eager fp32 draft."""
    torch.manual_seed(123)
    draft_config = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LENGTH,
    )
    draft_config.draft_vocab_size = VOCAB  # full vocab -> every position supervised
    draft_config.target_hidden_size = HIDDEN
    # Draft stays on eager (the recipe default): its FA2 path upcasts q/k via fp32
    # RoPE, which then mismatches the bf16 value cache. The target is the FA2 path
    # under test here.
    draft_config.attn_implementation = "eager"
    draft = LlamaEagle3DraftModel(draft_config).to(device="cuda", dtype=DRAFT_DTYPE)
    selected_token_ids = torch.arange(VOCAB, dtype=torch.long, device="cuda")
    selected_token_mask = torch.ones(VOCAB, dtype=torch.bool, device="cuda")
    return Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=TTT_STEPS,
    ).to("cuda")


def _to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}


def _padded_batches(docs: list[dict[str, list[int]]]) -> list[dict[str, torch.Tensor]]:
    """Layout A: one padded ``[1, SEQ_LENGTH]`` row per document."""
    batches = []
    for doc in docs:
        ids = doc["input_ids"]
        input_ids = torch.zeros(1, SEQ_LENGTH, dtype=torch.long)
        loss_mask = torch.zeros(1, SEQ_LENGTH, dtype=torch.long)
        attention_mask = torch.zeros(1, SEQ_LENGTH, dtype=torch.long)
        input_ids[0, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        loss_mask[0, : len(ids)] = 1
        attention_mask[0, : len(ids)] = 1
        batches.append(_to_cuda({"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}))
    return batches


def _packed_batches(docs: list[dict[str, list[int]]]) -> list[dict[str, torch.Tensor]]:
    """Layout B: greedily packed rows, one ``[1, SEQ_LENGTH]`` batch per pack."""
    packs = build_packed_eagle3_dataset(docs, packed_sequence_size=SEQ_LENGTH, pad_token_id=0)
    return [_to_cuda(_pack_collate([pack])) for pack in packs]


def _step(target_wrapper, trainer, batch, *, packed: bool):
    """One micro-batch training step; seq_lens present -> packed path."""
    kwargs = {}
    if packed:
        kwargs = {
            "position_ids": batch["position_ids"],
            "seq_lens": batch["seq_lens"],
            "doc_remaining": batch["doc_remaining"],
        }
    target_batch = target_wrapper.generate_batch(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        loss_mask=batch["loss_mask"],
        **kwargs,
    )
    inputs = target_batch.to_trainer_inputs()
    # Bridge the bf16 target -> fp32 draft: cast the float supervision tensors.
    for key in ("aux_hidden_states", "target_logits", "target_probs"):
        if inputs.get(key) is not None:
            inputs[key] = inputs[key].float()
    return trainer(**inputs)


def _run_layout(target_wrapper, trainer, batches, *, packed: bool):
    """Accumulate a valid-token-weighted global loss + gradients over all batches.

    Each micro-batch returns a mean loss over its own valid tokens; backprop
    ``loss * valid`` so ``.grad`` accumulates the gradient of the summed CE, then
    divide by the global valid count. With identical supervision the two layouts
    therefore land on the same global loss and gradients.
    """
    trainer.zero_grad(set_to_none=True)
    total_ce = 0.0
    total_valid = 0
    for batch in batches:
        metrics = _step(target_wrapper, trainer, batch, packed=packed)
        valid = metrics.valid_tokens
        (metrics.loss * valid).backward()
        total_ce += (metrics.loss.detach() * valid).item()
        total_valid += int(valid.item())
    global_loss = total_ce / max(total_valid, 1)
    grads = {
        n: (p.grad.detach() / max(total_valid, 1)).clone() for n, p in trainer.named_parameters() if p.grad is not None
    }
    return global_loss, grads, total_valid


def _time_layout(target_wrapper, trainer, batches, *, packed: bool, warmup: int = 2, iters: int = 8) -> float:
    """Median wall-clock seconds for one full pass (fwd+bwd) over all the docs."""
    timings = []
    for it in range(warmup + iters):
        trainer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch in batches:
            metrics = _step(target_wrapper, trainer, batch, packed=packed)
            metrics.loss.backward()
        torch.cuda.synchronize()
        if it >= warmup:
            timings.append(time.perf_counter() - t0)
    timings.sort()
    return timings[len(timings) // 2]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.skipif(not _HAS_FA, reason="requires flash-attn")
def test_eagle3_packing_fa2_parity_and_speed():
    docs = _make_documents()
    target_wrapper = _build_target()

    padded = _padded_batches(docs)
    packed = _packed_batches(docs)

    # --- Parity: fresh (identically-initialised) trainer per layout. ---
    trainer_a = _build_trainer()
    loss_a, grads_a, valid_a = _run_layout(target_wrapper, trainer_a, padded, packed=False)

    trainer_b = _build_trainer()
    loss_b, grads_b, valid_b = _run_layout(target_wrapper, trainer_b, packed, packed=True)

    assert valid_a == valid_b, f"valid-token counts differ: {valid_a} vs {valid_b}"
    assert set(grads_a) == set(grads_b)

    loss_abs = abs(loss_a - loss_b)
    loss_rel = loss_abs / max(abs(loss_a), 1e-8)

    max_grad_rel = 0.0
    max_grad_name = ""
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        denom = ga.abs().max().clamp_min(1e-6)
        rel = ((ga - gb).abs().max() / denom).item()
        if rel > max_grad_rel:
            max_grad_rel, max_grad_name = rel, name

    # --- Speed: median full-pass wall-clock for each layout. ---
    t_padded = _time_layout(target_wrapper, trainer_a, padded, packed=False)
    t_packed = _time_layout(target_wrapper, trainer_b, packed, packed=True)

    real_tokens = sum(min(length, SEQ_LENGTH) for length in DOC_LENS)

    print("\n=== EAGLE-3 FA2 packing parity & speed (micro_batch_size=1) ===")
    print(f"docs (truncated)      : {[min(length, SEQ_LENGTH) for length in DOC_LENS]}  (real tokens={real_tokens})")
    print(f"no-packing steps      : {len(padded)} rows x {SEQ_LENGTH} = {len(padded) * SEQ_LENGTH} tok (padding waste)")
    print(f"packing    steps      : {len(packed)} rows x {SEQ_LENGTH} = {len(packed) * SEQ_LENGTH} tok")
    print(f"valid supervised toks : {valid_a} (both layouts)")
    print(f"loss  no-packing      : {loss_a:.6f}")
    print(f"loss  packing         : {loss_b:.6f}")
    print(f"loss  |abs| / rel     : {loss_abs:.3e} / {loss_rel:.3e}")
    print(f"max grad rel diff     : {max_grad_rel:.3e}  ({max_grad_name})")
    print(f"full-pass no-packing  : {t_padded * 1e3:.2f} ms ({len(padded)} steps)")
    print(f"full-pass packing     : {t_packed * 1e3:.2f} ms ({len(packed)} steps)")
    print(f"speedup (padded/packed): {t_padded / t_packed:.2f}x")

    # bf16 FA2 target: parity is approximate. Tolerances are loose but would blow
    # up by orders of magnitude on a real bug (e.g. cross-document leakage).
    assert loss_rel < 5e-2, f"loss relative diff too large: {loss_rel:.3e}"
    assert max_grad_rel < 2e-1, f"grad relative diff too large: {max_grad_rel:.3e} ({max_grad_name})"


if __name__ == "__main__":
    test_eagle3_packing_fa2_parity_and_speed()
