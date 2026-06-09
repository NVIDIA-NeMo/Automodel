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

"""Tests for dLLM strategies (MDLMStrategy, HybridStrategy, DFlashStrategy) and get_dllm_strategy."""

import types

import pytest
import torch

from nemo_automodel.components.loss.dllm_loss import DFlashDecayLoss, MDLMCrossEntropyLoss
from nemo_automodel.recipes.dllm.strategy import (
    DLLM_STRATEGIES,
    DFlashStrategy,
    HybridStrategy,
    MDLMStrategy,
    _build_target_layer_ids,
    get_dllm_strategy,
)


def test_get_dllm_strategy_rejects_unknown_mode():
    """Unknown mode must raise a clear ValueError (recipe entry point relies on this)."""
    with pytest.raises(ValueError, match="Unknown dllm.mode"):
        get_dllm_strategy("unknown")


def test_get_dllm_strategy_resolves_dflash():
    """Registry happy-path for the flagship DFlash strategy — a typo in the
    DLLM_STRATEGIES dict would only surface at smoke time without this test."""
    assert isinstance(get_dllm_strategy("dflash"), DFlashStrategy)


def test_every_registered_strategy_has_valid_normalization_mode():
    """Strategy contract: ``normalization_mode`` selects the loss denominator in
    ``_run_train_optim_step``; a typo (e.g. ``"supervize"``) raises
    ``Invalid normalization_mode`` at runtime. Iterating the registry catches
    this at test time for every strategy."""
    for name, cls in DLLM_STRATEGIES.items():
        mode = cls().normalization_mode
        assert mode in ("supervised", "noise"), f"strategy {name!r}: invalid normalization_mode {mode!r}"


def test_build_target_layer_ids_even_spacing():
    assert _build_target_layer_ids(num_target_layers=12, num_draft_layers=1) == [6]
    assert _build_target_layer_ids(num_target_layers=12, num_draft_layers=3) == [1, 5, 9]


# ---------------------------------------------------------------------------
# MDLMStrategy tests
# ---------------------------------------------------------------------------


class TestMDLMStrategy:
    @pytest.fixture
    def strategy(self):
        return MDLMStrategy()

    def test_apply_corruption_uses_uniform(self, strategy):
        """MDLM always uses uniform corruption (p_mask constant per sequence)."""
        torch.manual_seed(42)
        B, L = 4, 32
        input_ids = torch.randint(0, 100, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)
        _, _, p_mask = strategy.apply_corruption(
            input_ids,
            loss_mask,
            mask_token_id=999,
            eps=0.001,
            block_size=None,
            half_life_ratio=None,
        )
        for b in range(B):
            assert (p_mask[b] == p_mask[b, 0]).all()

    def test_prepare_batch_sets_noisy_input_ids(self, strategy):
        """MDLM sets input_ids to noisy tokens and removes attention_mask (bidirectional)."""
        batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4)}
        noisy = torch.ones(2, 4, dtype=torch.long) * 999
        noise_mask = torch.ones(2, 4, dtype=torch.bool)
        clean = torch.zeros(2, 4, dtype=torch.long)

        result = strategy.prepare_batch(batch, noisy, noise_mask, clean)
        assert (result["input_ids"] == noisy).all()
        assert "attention_mask" not in result

    def test_pre_step_stashes_corruption_sidecars(self, strategy):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "loss_mask": torch.tensor([[1, 0, 1], [0, 1, 1]]),
        }

        def apply_corruption(input_ids, loss_mask):
            return input_ids + 100, loss_mask.bool(), torch.full(input_ids.shape, 0.5)

        recipe = types.SimpleNamespace(_apply_corruption=apply_corruption)

        num_noise, num_supervised = strategy.pre_step(recipe, [batch])

        assert num_noise == 4
        assert num_supervised == 4
        assert torch.equal(batch["_noisy_input_ids"], torch.tensor([[101, 102, 103], [104, 105, 106]]))
        assert torch.equal(batch["_noise_mask"], batch["loss_mask"].bool())
        assert torch.equal(batch["_clean_input_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]]))

    def test_forward_backward_delegates_to_recipe_step(self, strategy):
        calls = []

        def forward_backward_step(*args, **kwargs):
            calls.append((args, kwargs))

        recipe = types.SimpleNamespace(_forward_backward_step=forward_backward_step)
        batch = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
        loss_buffer = []

        strategy.forward_backward(
            recipe,
            1,
            batch,
            loss_buffer=loss_buffer,
            num_diffusion_tokens=7,
            num_ar_tokens=3,
            num_batches=2,
            is_train=False,
        )

        args, kwargs = calls[0]
        assert args == (1, batch)
        assert kwargs == {
            "loss_buffer": loss_buffer,
            "num_diffusion_tokens": 7,
            "num_ar_tokens": 3,
            "num_batches": 2,
            "is_train": False,
        }


# ---------------------------------------------------------------------------
# LLaDA-specific integration tests
# ---------------------------------------------------------------------------


class TestLLaDAIntegration:
    """Tests specific to LLaDA model integration with MDLM strategy."""

    LLADA_MASK_TOKEN_ID = 126336

    def test_corruption_with_llada_mask_token(self):
        """Corrupted positions get LLaDA's mask token; uncorrupted positions are unchanged."""
        torch.manual_seed(42)
        strategy = MDLMStrategy()
        B, L = 2, 16
        input_ids = torch.randint(0, 1000, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)

        noisy, noise_mask, p_mask = strategy.apply_corruption(
            input_ids,
            loss_mask,
            mask_token_id=self.LLADA_MASK_TOKEN_ID,
            eps=0.001,
            block_size=None,
            half_life_ratio=None,
        )
        assert (noisy[noise_mask] == self.LLADA_MASK_TOKEN_ID).all()
        assert (noisy[~noise_mask] == input_ids[~noise_mask]).all()

    def test_prepare_batch_passes_extra_keys_for_recipe_filtering(self):
        """Strategy keeps extra collator keys (input_lengths); the recipe filters
        them against the LLaDA forward signature (which does not accept **kwargs)."""
        strategy = MDLMStrategy()
        batch = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4),
            "input_lengths": torch.tensor([3, 4]),  # extra key from collator
        }
        noisy = torch.ones(2, 4, dtype=torch.long) * 126336
        noise_mask = torch.ones(2, 4, dtype=torch.bool)
        clean = torch.zeros(2, 4, dtype=torch.long)

        result = strategy.prepare_batch(batch, noisy, noise_mask, clean)
        assert (result["input_ids"] == noisy).all()
        assert "attention_mask" not in result
        assert "input_lengths" in result  # passed through; recipe filters it


# ---------------------------------------------------------------------------
# HybridStrategy tests
# ---------------------------------------------------------------------------


class TestHybridStrategy:
    @pytest.fixture
    def strategy(self):
        return HybridStrategy()

    def test_create_loss_fn_reads_alpha_from_config(self, strategy):
        assert strategy.create_loss_fn({"ar_loss_alpha": 0.3}).alpha == 0.3
        assert strategy.create_loss_fn({}).alpha == 1.0  # default

    def test_apply_corruption_uniform_when_no_block_size(self, strategy):
        """block_size=None should select uniform corruption (constant p_mask per row)."""
        torch.manual_seed(42)
        B, L = 2, 16
        input_ids = torch.randint(0, 100, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)
        _, _, p_mask = strategy.apply_corruption(
            input_ids,
            loss_mask,
            mask_token_id=999,
            eps=0.001,
            block_size=None,
            half_life_ratio=None,
        )
        for b in range(B):
            assert torch.allclose(p_mask[b], p_mask[b, 0].expand_as(p_mask[b]))

    def test_apply_corruption_blockwise_when_block_size_set(self, strategy):
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (2, 16))
        loss_mask = torch.ones(2, 16, dtype=torch.long)

        noisy, noise_mask, p_mask = strategy.apply_corruption(
            input_ids,
            loss_mask,
            mask_token_id=999,
            eps=0.001,
            block_size=4,
            half_life_ratio=None,
        )

        assert noisy.shape == input_ids.shape
        assert noise_mask.shape == input_ids.shape
        assert p_mask.shape == input_ids.shape

    def test_prepare_batch_passes_clean_input_ids(self, strategy):
        """Hybrid models receive clean tokens plus a masked_indices sidecar."""
        batch = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4),
            "use_cache": True,
        }
        noisy = torch.full((2, 4), 100, dtype=torch.long)
        noise_mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
        clean = torch.arange(8, dtype=torch.long).reshape(2, 4)

        result = strategy.prepare_batch(batch, noisy, noise_mask, clean)

        assert (result["input_ids"] == clean).all()
        assert (result["masked_indices"] == noise_mask).all()
        assert (result["labels"] == clean).all()
        assert result["skip_loss"] is True
        assert "attention_mask" not in result
        assert "use_cache" not in result


# ---------------------------------------------------------------------------
# DFlashStrategy — anchor-block sampling (CPU, no model loading)
# ---------------------------------------------------------------------------

MASK_ID = 999
BLOCK_SIZE = 16


def test_dflash_strategy_defaults():
    """Lock load-bearing DFlashStrategy constructor defaults.

    These defaults encode deliberate design decisions (paper §4.2 + the
    safety guard rails surfaced by the original PR review):

    - ``overlap_anchors=True``: paper-default per-sample independent anchor
      sampling (gap #1 fix). Flipping to False silently reverts to the legacy
      batch-shared stars-and-bars sampler.
    - ``block_size=0``: sentinel meaning "read block_size from the draft
      model's config". Any non-zero default would override the draft config.
    - ``num_blocks_per_sample=1``: safe default; production yaml must opt
      into the paper's 512 explicitly. A larger default would silently OOM
      smaller GPUs.
    - ``attention_backend="sdpa"``: dense fallback that works everywhere;
      production yaml must opt into ``flex_attention`` for N=512.
    """
    s = DFlashStrategy()
    assert s.overlap_anchors is True
    assert s.block_size == 0
    assert s.num_blocks_per_sample == 1
    assert s.attention_backend == "sdpa"


def test_dflash_strategy_placeholder_methods():
    strategy = DFlashStrategy()
    assert isinstance(strategy.create_loss_fn({}), MDLMCrossEntropyLoss)

    batch = {"input_ids": torch.tensor([[1, 2]])}
    assert strategy.prepare_batch(batch, None, None, None) is batch

    torch.manual_seed(12)
    input_ids = torch.randint(0, 100, (2, 8))
    loss_mask = torch.ones(2, 8, dtype=torch.long)
    noisy, noise_mask, p_mask = strategy.apply_corruption(
        input_ids,
        loss_mask,
        mask_token_id=999,
        eps=0.001,
        block_size=None,
        half_life_ratio=None,
    )
    assert noisy.shape == input_ids.shape
    assert noise_mask.shape == input_ids.shape
    assert p_mask.shape == input_ids.shape


def _make_recipe(mask_token_id=MASK_ID):
    """Minimal recipe stub with the fields DFlashStrategy methods need."""
    return types.SimpleNamespace(mask_token_id=mask_token_id)


def _make_strategy(block_size=BLOCK_SIZE, overlap_anchors=False):
    """DFlashStrategy stub with block_size set; defaults to the
    non-overlapping sampler for backward compatibility with existing tests.
    """
    s = DFlashStrategy()
    s.block_size = block_size
    s.overlap_anchors = overlap_anchors
    return s


class _FakeTargetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = types.SimpleNamespace(embed_tokens=torch.nn.Embedding(16, 4))
        self.lm_head = torch.nn.Linear(4, 16)
        self.config = types.SimpleNamespace(num_hidden_layers=12)
        self.eval_called = False
        self.to_device = None

    def eval(self):
        self.eval_called = True
        return super().eval()

    def get_input_embeddings(self):
        return None

    def get_output_embeddings(self):
        return None

    def to(self, device):
        self.to_device = device
        return super().to(device)


class _FakeTokenizer:
    mask_token_id = None

    def add_special_tokens(self, special_tokens):
        assert special_tokens == {"mask_token": "<|MASK|>"}
        self.mask_token_id = 321


class _RecordingDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs["noise_embedding"]


def test_dflash_setup_extra_resolves_fake_target_and_config(monkeypatch):
    fake_target = _FakeTargetModel()
    fake_tokenizer = _FakeTokenizer()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: fake_target)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: fake_tokenizer)

    draft_cfg = types.SimpleNamespace(block_size=8, num_target_layers=12, num_hidden_layers=3)
    draft = types.SimpleNamespace(config=draft_cfg)
    recipe = types.SimpleNamespace(
        cfg={
            "dflash": {
                "target_model_id": "fake-target",
                "target_torch_dtype": "float32",
                "block_size": 0,
                "num_blocks_per_sample": 7,
                "attention_backend": "flex_attention",
                "overlap_anchors": False,
                "use_fused_linear_ce": False,
                "ce_chunk_size": 17,
            },
            "dataset": {"seq_length": 128},
        },
        mask_token_id=None,
        dist_env=types.SimpleNamespace(device=torch.device("cpu")),
        model_parts=[draft],
    )

    strategy = DFlashStrategy()
    strategy.setup_extra(recipe)

    assert recipe.mask_token_id == 321
    assert strategy.target_model is fake_target
    assert fake_target.eval_called is True
    assert fake_target.to_device == torch.device("cpu")
    assert all(not parameter.requires_grad for parameter in fake_target.parameters())
    assert strategy.target_embed is fake_target.model.embed_tokens
    assert strategy.target_head is fake_target.lm_head
    assert strategy.block_size == 8
    assert strategy.layer_ids == [1, 5, 9]
    assert strategy.num_blocks_per_sample == 7
    assert strategy.attention_backend == "flex_attention"
    assert draft_cfg._attn_implementation == "flex_attention"
    assert strategy.overlap_anchors is False
    assert strategy.fixed_ctx_len == 128
    assert strategy.use_fused_linear_ce is False
    assert isinstance(strategy.dflash_loss_fn, DFlashDecayLoss)
    assert strategy.dflash_loss_fn.loss_gamma == 4.0
    assert strategy.dflash_loss_fn.chunk_size == 17


def test_dflash_setup_extra_requires_target_model_id():
    recipe = types.SimpleNamespace(cfg={"dflash": {}}, mask_token_id=5)
    with pytest.raises(ValueError, match="dflash.target_model_id"):
        DFlashStrategy().setup_extra(recipe)


def test_dflash_run_target_forward_concatenates_selected_hidden_states():
    class Target(torch.nn.Module):
        def forward(self, **kwargs):
            hidden_states = [torch.full((2, 5, 3), float(i)) for i in range(5)]
            return types.SimpleNamespace(hidden_states=hidden_states)

    strategy = DFlashStrategy()
    strategy.target_model = Target()
    strategy.layer_ids = [0, 2]

    hidden = strategy._run_target_forward(
        input_ids=torch.ones(2, 5, dtype=torch.long),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
        start=3,
    )

    assert hidden.shape == (2, 3, 6)
    assert torch.equal(hidden[..., :3], torch.ones(2, 3, 3))
    assert torch.equal(hidden[..., 3:], torch.full((2, 3, 3), 3.0))


def test_dflash_sample_anchor_block_uses_loss_mask():
    torch.manual_seed(4)
    strategy = _make_strategy(block_size=4)
    recipe = _make_recipe()
    input_ids = torch.arange(16, dtype=torch.long).view(2, 8)
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    loss_mask = torch.zeros(2, 8, dtype=torch.long)

    start, block_output_ids, block_targets, block_mask = strategy._sample_anchor_block(
        recipe,
        input_ids,
        attention_mask,
        loss_mask=loss_mask,
    )

    assert 1 <= start <= 4
    assert torch.equal(block_output_ids[:, 0], input_ids[:, start])
    assert block_targets.shape == (2, 3)
    assert block_mask.sum().item() == 0.0


def test_dflash_pre_step_stashes_target_and_anchor_tensors():
    strategy = _make_strategy(block_size=4, overlap_anchors=False)
    strategy.num_blocks_per_sample = 2

    def run_target_forward(input_ids, attention_mask, start):
        return torch.ones(input_ids.size(0), start, 3, device=input_ids.device)

    strategy._run_target_forward = run_target_forward
    recipe = types.SimpleNamespace(mask_token_id=MASK_ID, dist_env=types.SimpleNamespace(device=torch.device("cpu")))
    batch = {
        "input_ids": torch.arange(32, dtype=torch.long).view(2, 16),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "loss_mask": torch.ones(2, 16, dtype=torch.long),
    }

    num_noise, num_supervised = strategy.pre_step(recipe, [batch])

    assert num_noise == 12
    assert num_supervised == 12
    assert batch["_dflash_anchor_positions"].shape == (2, 2)
    assert batch["_dflash_block_keep"].all()
    assert batch["_dflash_target_hidden"].shape == (2, 16, 3)
    assert batch["_dflash_block_output_ids"].shape == (2, 8)
    assert batch["_dflash_block_targets"].shape == (2, 6)
    assert batch["_dflash_block_mask"].sum().item() == 12.0


@pytest.mark.parametrize("use_fused_linear_ce", [False, True])
def test_dflash_forward_backward_uses_precomputed_multiblock_tensors(use_fused_linear_ce):
    draft = _RecordingDraft()
    strategy = _make_strategy(block_size=3, overlap_anchors=False)
    strategy.num_blocks_per_sample = 2
    strategy.fixed_ctx_len = 6
    strategy.attention_backend = "sdpa"
    strategy.use_fused_linear_ce = use_fused_linear_ce
    strategy.target_embed = torch.nn.Embedding(16, 5)
    strategy.target_head = torch.nn.Linear(5, 11)
    strategy.dflash_loss_fn = DFlashDecayLoss(loss_gamma=2.0, use_fused_linear_ce=use_fused_linear_ce, chunk_size=2)
    recipe = types.SimpleNamespace(
        dist_env=types.SimpleNamespace(device=torch.device("cpu")),
        model_parts=[draft],
        distributed_config=types.SimpleNamespace(defer_fsdp_grad_sync=True, autocast_dtype=None),
        te_fp8=None,
        device_mesh=None,
        _dllm_loss_buffer=[],
        _dflash_correct_per_pos_buffer=[],
        _dflash_count_per_pos_buffer=[],
    )
    batch = {
        "_dflash_anchor_positions": torch.tensor([[1, 3]], dtype=torch.long),
        "_dflash_block_keep": torch.tensor([[True, True]]),
        "_dflash_target_hidden": torch.ones(1, 4, 2),
        "_dflash_block_output_ids": torch.tensor([[2, 15, 15, 4, 15, 15]], dtype=torch.long),
        "_dflash_block_targets": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "_dflash_block_mask": torch.ones(1, 4),
    }
    loss_buffer = []

    strategy.forward_backward(
        recipe,
        0,
        batch,
        loss_buffer=loss_buffer,
        num_diffusion_tokens=4,
        num_batches=1,
        is_train=False,
    )

    assert len(loss_buffer) == 1
    assert len(recipe._dllm_loss_buffer) == 1
    assert len(recipe._dflash_correct_per_pos_buffer) == 1
    assert len(recipe._dflash_count_per_pos_buffer) == 1
    assert recipe._dflash_correct_per_pos_buffer[0].shape == (2,)
    assert recipe._dflash_count_per_pos_buffer[0].shape == (2,)
    assert draft.last_kwargs["target_hidden"].shape == (1, 6, 2)
    assert draft.last_kwargs["noise_embedding"].shape == (1, 6, 5)
    assert draft.last_kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5]]
    assert draft.last_kwargs["attention_mask"].shape == (1, 1, 6, 12)


def test_dflash_forward_backward_fallback_skips_mask_for_single_sdpa_block():
    draft = _RecordingDraft()
    strategy = _make_strategy(block_size=3, overlap_anchors=False)
    strategy.num_blocks_per_sample = 1
    strategy.attention_backend = "sdpa"
    strategy.use_fused_linear_ce = False
    strategy.target_embed = torch.nn.Embedding(32, 5)
    strategy.target_head = torch.nn.Linear(5, 32)
    strategy.dflash_loss_fn = DFlashDecayLoss(loss_gamma=2.0)
    strategy._run_target_forward = lambda input_ids, attention_mask, start: torch.ones(input_ids.size(0), start, 2)
    recipe = types.SimpleNamespace(
        mask_token_id=31,
        dist_env=types.SimpleNamespace(device=torch.device("cpu")),
        model_parts=[draft],
        distributed_config=types.SimpleNamespace(defer_fsdp_grad_sync=True, autocast_dtype=None),
        te_fp8=None,
        device_mesh=None,
        _dllm_loss_buffer=[],
        _dflash_correct_per_pos_buffer=[],
        _dflash_count_per_pos_buffer=[],
    )
    batch = {
        "input_ids": torch.arange(16, dtype=torch.long).view(1, 16),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
        "loss_mask": torch.ones(1, 16, dtype=torch.long),
    }

    strategy.forward_backward(
        recipe,
        0,
        batch,
        loss_buffer=[],
        num_diffusion_tokens=2,
        num_batches=1,
        is_train=False,
    )

    assert "attention_mask" not in draft.last_kwargs


class TestDFlashSampleAnchorBlocks:
    """Tests for the non-overlapping (legacy) _sample_anchor_blocks path.

    Returns the per-sample 5-tuple
    ``(anchor_positions [B,N], block_keep_mask [B,N], block_output_ids,
    block_targets, block_mask)``.
    """

    def _make_inputs(self, seq_len, batch_size=2):
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attn = torch.ones(batch_size, seq_len, dtype=torch.long)
        return input_ids, attn

    def test_shapes(self):
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        for n in (1, 4):
            input_ids, attn = self._make_inputs(128)
            ap, keep, boi, bt, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=n)
            assert ap.shape == (2, n)
            assert keep.shape == (2, n)
            assert boi.shape == (2, n * 8)
            assert bt.shape == (2, n * 7)
            assert bm.shape == (2, n * 7)

    def test_blocks_are_non_overlapping(self):
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        for _ in range(10):
            ap, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
            starts = ap[0].tolist()  # batch-shared in non-overlap mode
            assert starts == sorted(starts)
            for i in range(len(starts) - 1):
                assert starts[i + 1] >= starts[i] + s.block_size, f"blocks overlap: {starts}"

    def test_blocks_fit_in_sequence(self):
        seq_len = 64
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(seq_len)
        for _ in range(10):
            ap, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
            assert (ap >= 1).all() and (ap + s.block_size <= seq_len).all()

    def test_anchor_token_is_clean(self):
        """First token of each kept block must be the real token at its anchor."""
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        ap, keep, boi, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=3)
        B, n = ap.shape
        for b in range(B):
            for i in range(n):
                if keep[b, i]:
                    assert boi[b, i * s.block_size] == input_ids[b, ap[b, i]]

    def test_non_anchor_tokens_are_mask(self):
        """All positions after the anchor in each block should be MASK_ID."""
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        ap, _, boi, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=3)
        n = ap.shape[1]
        for b in range(n):
            noise_slice = boi[:, b * s.block_size + 1 : (b + 1) * s.block_size]
            assert (noise_slice == MASK_ID).all()

    def test_loss_mask_zeros_block_mask(self):
        """block_mask must be zero wherever loss_mask is zero."""
        torch.manual_seed(7)
        B, L, bs = 2, 64, 8
        s = _make_strategy(block_size=bs)
        recipe = _make_recipe()
        input_ids = torch.randint(0, 100, (B, L))
        attn = torch.ones(B, L, dtype=torch.long)
        # Zero the entire loss_mask — every predicted position should be masked out.
        loss_mask = torch.zeros(B, L, dtype=torch.long)
        _, _, _, _, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=3, loss_mask=loss_mask)
        assert bm.sum().item() == 0


class TestDFlashSampleAnchorBlocksOverlapping:
    """Tests for the paper-default per-sample overlap_anchors=True sampler.

    Each sample draws ``num_blocks`` anchors independently (Appendix A.1), so
    anchor_positions is ``[B, N]`` with potentially different rows.
    """

    def _make_inputs(self, seq_len, batch_size=2):
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attn = torch.ones(batch_size, seq_len, dtype=torch.long)
        return input_ids, attn

    def test_anchors_in_valid_range(self):
        """Every anchor must satisfy 1 <= a <= valid_len - block_size."""
        seq_len, bs = 64, 8
        s = _make_strategy(block_size=bs, overlap_anchors=True)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(seq_len)
        for _ in range(10):
            ap, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=16)
            assert (ap >= 1).all() and (ap <= seq_len - bs).all()

    def test_can_exceed_non_overlapping_cap(self):
        """N > seq_len // block_size succeeds (impossible in non-overlap mode)."""
        seq_len, bs = 32, 8  # non-overlap cap = 4
        s = _make_strategy(block_size=bs, overlap_anchors=True)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(seq_len)
        ap, keep, boi, bt, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=20)
        assert ap.shape == (2, 20)
        assert boi.shape == (2, 20 * bs)

    def test_per_sample_diversity(self):
        """Different samples should (with high probability) get different anchors."""
        torch.manual_seed(0)
        s = _make_strategy(block_size=8, overlap_anchors=True)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(256, batch_size=2)
        ap, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=16)
        # Two independently-sampled rows of 16 anchors should not be identical.
        assert not torch.equal(ap[0], ap[1])

    def test_anchor_token_is_clean_per_sample(self):
        """Each kept block's first token is the real token at that sample's anchor."""
        s = _make_strategy(block_size=8, overlap_anchors=True)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        ap, keep, boi, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=10)
        B, n = ap.shape
        for b in range(B):
            for i in range(n):
                if keep[b, i]:
                    assert boi[b, i * s.block_size] == input_ids[b, ap[b, i]]

    def test_min_token_filter_drops_short_samples(self):
        """Samples with < 2*block_size supervised tokens get block_keep_mask=False."""
        bs = 8
        s = _make_strategy(block_size=bs, overlap_anchors=True)
        recipe = _make_recipe()
        torch.manual_seed(1)
        B, L = 2, 128
        input_ids = torch.randint(0, 100, (B, L))
        attn = torch.ones(B, L, dtype=torch.long)
        # Sample 0 long enough, sample 1 too short (< 2*bs valid tokens).
        attn[1, 2 * bs - 1 :] = 0
        ap, keep, _, _, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
        assert keep[0].all()  # long sample kept
        assert (~keep[1]).all()  # short sample fully dropped
        assert bm[1].sum().item() == 0  # short sample contributes no loss
