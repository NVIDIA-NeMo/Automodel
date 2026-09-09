# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Non-degeneracy tests for the diffusion_gemma recipe's ACTUAL response-window mask.

The mask leakage/structure unit tests in
``tests/unit_tests/models/diffusion_gemma/test_diffusion_gemma_mask.py`` build
hand-crafted masks. This file instead drives the recipe's own
``_build_response_window`` (the code that runs in ``_forward_backward_step``)
using the ``seq_length`` / ``block_size`` from the *shipped* example configs, and
asserts the resulting mask is **non-degenerate**:

* the encoder is NOT fully masked (the v1 full-sequence-canvas shortcut with
  ``seq_length == block_size`` degenerated to plain bidirectional denoising —
  encoder fully masked — which is exactly what this guards against);
* a config whose response spans more than one block produces >= 2 response
  blocks with cross-block clean conditioning;
* the prompt prefix is visible to every response query;
* the strict ``>`` offset-block-causal boundary holds (block i is masked at its
  own clean encoder start).
"""

from pathlib import Path

import torch
import yaml

from nemo_automodel.recipes.dllm.strategy import BlockDiffusionStrategy
from nemo_automodel.recipes.dllm.train_ft import DiffusionGemmaSFTRecipe

REPO_ROOT = Path(__file__).resolve().parents[4]
SFT_CONFIG = REPO_ROOT / "examples" / "dllm_sft" / "diffusion_gemma_sft.yaml"


def _config_dims(path: Path) -> tuple[int, int, int]:
    """Return ``(seq_length, block_size, sliding_window_or_default)`` from a config."""
    cfg = yaml.safe_load(path.read_text())
    seq_length = int(cfg["dataset"]["seq_length"])
    block_size = int(cfg["dllm"]["block_size"])
    return seq_length, block_size


def _make_recipe(block_size: int, sliding_window: int = 1024) -> DiffusionGemmaSFTRecipe:
    """A bare recipe carrying only the attributes ``_build_response_window`` reads.

    Avoids ``setup()`` (which needs a real model + checkpoint); the mask path
    itself only depends on these four attributes.
    """
    recipe = DiffusionGemmaSFTRecipe.__new__(DiffusionGemmaSFTRecipe)
    recipe.dllm_block_size = block_size
    recipe.canvas_length = block_size
    recipe.block_diffusion_sliding_window = sliding_window

    class _DC:
        autocast_dtype = None

    recipe.distributed_config = _DC()
    return recipe


def _single_turn_batch(seq_length: int, prefix_len: int):
    """One single-turn example: clean ids + a contiguous supervised suffix.

    ``prefix_len`` (the prompt length) is deliberately NOT block-aligned so the
    test catches any code that assumes a block-aligned prompt boundary.
    """
    clean = torch.arange(1, seq_length + 1, dtype=torch.long).unsqueeze(0)
    noisy = clean.clone()
    loss_mask = torch.zeros(1, seq_length, dtype=torch.long)
    loss_mask[0, prefix_len:] = 1
    # Corrupt a couple of supervised positions so noise_mask is non-empty.
    noise_mask = torch.zeros(1, seq_length, dtype=torch.bool)
    noise_mask[0, prefix_len] = True
    noise_mask[0, seq_length - 1] = True
    noisy[0, prefix_len] = 0
    noisy[0, seq_length - 1] = 0
    p_mask = torch.ones(1, seq_length, dtype=torch.float32)
    return clean, noisy, noise_mask, loss_mask, p_mask


def _attended(window: dict) -> torch.Tensor:
    """Boolean keep-mask [B, Lq, key_len] from the recipe's additive full mask."""
    add = window["decoder_attention_mask"]["full_attention"]
    return add[:, 0] > torch.finfo(add.dtype).min / 2


def test_split_prompt_response_boundary_drives_window():
    """The recipe slices on the single-turn prompt boundary from split_prompt_response."""
    seq_length, _ = _config_dims(SFT_CONFIG)
    prefix_len = 17
    _, _, _, loss_mask, _ = _single_turn_batch(seq_length, prefix_len)
    prefix_lengths, response_mask = BlockDiffusionStrategy.split_prompt_response(
        torch.zeros(1, seq_length, dtype=torch.long), loss_mask
    )
    assert prefix_lengths.item() == prefix_len
    assert response_mask[0, :prefix_len].sum() == 0
    assert response_mask[0, prefix_len:].all()


def test_sft_config_response_spans_multiple_blocks():
    """The shipped SFT config must train real blocks: >= 2 response blocks.

    With a non-block-aligned prompt the response still has to cover more than one
    diffusion block, otherwise the canvas degenerates to a single bidirectional
    block (the v1 failure mode).
    """
    seq_length, block_size = _config_dims(SFT_CONFIG)
    prefix_len = block_size // 2 + 3  # not a multiple of block_size
    recipe = _make_recipe(block_size)
    clean, noisy, noise_mask, loss_mask, p_mask = _single_turn_batch(seq_length, prefix_len)

    window = recipe._build_response_window(clean, noisy, noise_mask, loss_mask, p_mask)

    response_length = seq_length - prefix_len
    num_blocks = (response_length + block_size - 1) // block_size
    assert num_blocks >= 2, f"SFT config response must span >= 2 blocks, got {num_blocks}"

    canvas_ids = window["canvas_ids"]
    assert canvas_ids.shape == (1, response_length)
    # decoder positions are the response tokens' absolute positions.
    expected_positions = torch.arange(prefix_len, seq_length)
    assert torch.equal(window["decoder_position_ids"][0], expected_positions)


def test_encoder_not_fully_masked():
    """The encoder KV must be reachable — the degenerate v1 case masked it entirely.

    A later response block has to attend at least one clean encoder column
    (prompt or an earlier clean response block).
    """
    seq_length, block_size = _config_dims(SFT_CONFIG)
    prefix_len = block_size // 2 + 3
    recipe = _make_recipe(block_size)
    batch = _single_turn_batch(seq_length, prefix_len)
    window = recipe._build_response_window(*batch)

    enc_len = seq_length
    keep = _attended(window)  # [1, Lq, enc_len + R]
    encoder_cols = keep[0, :, :enc_len]
    assert encoder_cols.any(), "encoder is fully masked: canvas degenerated to plain bidirectional denoising"
    # The last response query (last block) must see clean context.
    assert encoder_cols[-1].any(), "last response block sees no clean encoder column"


def test_prompt_visible_to_all_response_blocks():
    """Every response query attends every clean prompt column (always-visible prefix)."""
    seq_length, block_size = _config_dims(SFT_CONFIG)
    prefix_len = block_size // 2 + 3
    recipe = _make_recipe(block_size)
    batch = _single_turn_batch(seq_length, prefix_len)
    window = recipe._build_response_window(*batch)

    keep = _attended(window)
    prompt_cols = keep[0, :, :prefix_len]  # encoder columns [0, prefix_len)
    assert prompt_cols.all(), "prompt prefix must be visible to every response query"


def test_strict_offset_block_causal_boundary():
    """Block i is masked at its own clean encoder start (strict ``>``, not ``>=``).

    Response block i covers canvas positions [i*bs, (i+1)*bs); the clean encoder
    copy of its first token is at absolute position prefix_len + i*bs. A leaky
    ``>=`` mask would expose it.
    """
    seq_length, block_size = _config_dims(SFT_CONFIG)
    prefix_len = block_size // 2 + 3
    recipe = _make_recipe(block_size)
    batch = _single_turn_batch(seq_length, prefix_len)
    window = recipe._build_response_window(*batch)

    keep = _attended(window)
    response_length = seq_length - prefix_len
    num_blocks = (response_length + block_size - 1) // block_size
    for i in range(num_blocks):
        q = i * block_size  # first query of block i
        own_start = prefix_len + i * block_size  # clean encoder copy of block i's first token
        assert not keep[0, q, own_start], f"LEAKAGE: block {i} query attends its own clean encoder start"
        if i > 0:
            # An earlier clean response column (prev block start) must be visible.
            assert keep[0, q, prefix_len + (i - 1) * block_size], f"block {i} cannot see earlier clean block {i - 1}"


def test_canvas_block_diagonal_no_cross_block():
    """Canvas self-attention is block-diagonal: block i never attends canvas block j != i."""
    seq_length, block_size = _config_dims(SFT_CONFIG)
    prefix_len = block_size // 2 + 3
    recipe = _make_recipe(block_size)
    batch = _single_turn_batch(seq_length, prefix_len)
    window = recipe._build_response_window(*batch)

    keep = _attended(window)
    enc_len = seq_length
    response_length = seq_length - prefix_len
    canvas = keep[0, :, enc_len : enc_len + response_length]  # [Lq, R]
    q_block = torch.arange(response_length) // block_size
    kv_block = torch.arange(response_length) // block_size
    cross_block = q_block[:, None] != kv_block[None, :]
    assert not canvas[cross_block].any(), "canvas block i must not attend canvas block j != i"


def test_per_example_varying_prompt_lengths():
    """Batched examples with different prompt lengths each get their own window.

    Exercises the per-example mask assembly (the path that pads shorter responses
    and keeps every query row well-defined).
    """
    seq_length, block_size = _config_dims(SFT_CONFIG)
    recipe = _make_recipe(block_size)

    prefixes = [block_size // 2 + 3, block_size + 7]
    clean = torch.arange(1, seq_length + 1, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    noisy = clean.clone()
    loss_mask = torch.zeros(2, seq_length, dtype=torch.long)
    noise_mask = torch.zeros(2, seq_length, dtype=torch.bool)
    for b, pfx in enumerate(prefixes):
        loss_mask[b, pfx:] = 1
        noise_mask[b, pfx] = True
    p_mask = torch.ones(2, seq_length, dtype=torch.float32)

    window = recipe._build_response_window(clean, noisy, noise_mask, loss_mask, p_mask)

    canvas_len = seq_length - min(prefixes)
    assert window["canvas_ids"].shape == (2, canvas_len)
    # No query row in either example may be fully masked (would NaN the softmax).
    keep = _attended(window)
    assert keep.any(dim=2).all(), "a query row is fully masked (all -inf)"
    # The example with the shorter response has padded canvas rows that are
    # dropped from the loss mask.
    longer_resp = seq_length - prefixes[0]
    shorter_resp = seq_length - prefixes[1]
    assert window["loss_mask"][1, shorter_resp:].sum() == 0
    assert window["loss_mask"][0, :longer_resp].sum() > 0


def test_attention_mask_limits_response_window_and_padding_masks():
    """Tail padding must not become canvas tokens routed through MoE under EP."""
    seq_length = 16
    block_size = 4
    recipe = _make_recipe(block_size)

    clean = torch.arange(1, seq_length + 1, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    noisy = clean.clone()
    prefix_len = 4
    effective_lengths = torch.tensor([12, 8])
    attention_mask = torch.zeros(2, seq_length, dtype=torch.long)
    loss_mask = torch.zeros(2, seq_length, dtype=torch.long)
    noise_mask = torch.zeros(2, seq_length, dtype=torch.bool)
    for b, effective_len in enumerate(effective_lengths.tolist()):
        attention_mask[b, :effective_len] = 1
        loss_mask[b, prefix_len:effective_len] = 1
        noise_mask[b, prefix_len:effective_len] = True
    p_mask = torch.ones(2, seq_length, dtype=torch.float32)

    window = recipe._build_response_window(clean, noisy, noise_mask, loss_mask, p_mask, attention_mask=attention_mask)

    longest_response = int(effective_lengths.max().item()) - prefix_len
    shorter_response = int(effective_lengths.min().item()) - prefix_len
    assert window["canvas_ids"].shape == (2, longest_response)
    assert torch.equal(window["encoder_padding_mask"], attention_mask.logical_not())
    assert not window["decoder_padding_mask"][0].any()
    assert not window["decoder_padding_mask"][1, :shorter_response].any()
    assert window["decoder_padding_mask"][1, shorter_response:].all()
    assert window["loss_mask"][1, shorter_response:].sum() == 0
