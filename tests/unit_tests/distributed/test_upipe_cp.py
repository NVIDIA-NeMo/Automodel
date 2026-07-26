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

"""CPU-only tests for the UPipe head layout, RoPE adapter, and config validation.

The head permutation is the part most likely to be silently wrong -- a bad
permutation still produces plausible-looking logits -- so it is checked against
an independently derived query-to-KV pairing rather than against itself.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel.upipe.layout import (
    invert_permutation,
    upipe_head_permutation,
)
from nemo_automodel.components.distributed.context_parallel.upipe.validation import validate_upipe_attention

# (n_heads, n_kv_heads, cp_size) triples UPipe must accept.
VALID_SHAPES = [
    (32, 8, 1),
    (32, 8, 2),
    (32, 8, 4),
    (32, 8, 8),
    (16, 4, 1),
    (16, 4, 2),
    (16, 4, 4),
    (16, 16, 4),  # MHA
    (64, 8, 8),
    (40, 8, 4),
]


class TestHeadPermutation:
    @pytest.mark.parametrize("n_heads,n_kv_heads,cp_size", VALID_SHAPES)
    def test_is_a_permutation(self, n_heads, n_kv_heads, cp_size):
        perm = upipe_head_permutation(n_heads, n_kv_heads, cp_size)
        assert sorted(perm) == list(range(n_heads))

    @pytest.mark.parametrize("n_heads,n_kv_heads,cp_size", VALID_SHAPES)
    def test_round_trips_through_its_inverse(self, n_heads, n_kv_heads, cp_size):
        perm = upipe_head_permutation(n_heads, n_kv_heads, cp_size)
        inverse = invert_permutation(perm)
        assert [perm[inverse[i]] for i in range(n_heads)] == list(range(n_heads))

        out = torch.arange(n_heads).view(1, 1, n_heads, 1)
        permuted = out[:, :, perm, :][:, :, inverse, :]
        assert torch.equal(permuted, out)

    @pytest.mark.parametrize("n_heads,n_kv_heads,cp_size", VALID_SHAPES)
    def test_each_slot_lands_on_the_kv_head_the_stage_holds(self, n_heads, n_kv_heads, cp_size):
        """The permutation's whole job: reconcile the staged pairing with GQA.

        Stage ``s`` rank ``r`` computes against KV head ``(s // g) * cp + r``.
        Whatever logical head the permutation assigns to that slot must be one
        whose GQA group is exactly that KV head, or the model is wrong.
        """
        perm = upipe_head_permutation(n_heads, n_kv_heads, cp_size)
        gqa_ratio = n_heads // n_kv_heads

        for stage in range(n_heads // cp_size):
            for rank in range(cp_size):
                slot = stage * cp_size + rank
                kv_head_held_by_stage = (stage // gqa_ratio) * cp_size + rank
                assert perm[slot] // gqa_ratio == kv_head_held_by_stage

    def test_mha_needs_no_permutation(self):
        assert upipe_head_permutation(16, 16, 4) == list(range(16))

    def test_single_rank_needs_no_permutation(self):
        assert upipe_head_permutation(32, 8, 1) == list(range(32))

    def test_matches_the_reference_ordering_when_ulysses_equals_gqa_ratio(self):
        """Agrees with the Untied Ulysses reference in the case the reference gets right.

        The reference rebuilds this list per forward with a formula that only
        holds for ``ulysses_degree == gqa_ratio``; pinning that case guards
        against drift in the shared regime.
        """
        n_heads, n_kv_heads, cp_size = 32, 8, 8
        gqa_ratio = n_heads // n_kv_heads

        reference = []
        stage_idx = []
        for stage in range(n_heads // cp_size):
            if stage == 0 or stage // gqa_ratio != (stage - 1) // gqa_ratio:
                stage_idx = [(stage + i) * gqa_ratio for i in range(cp_size)]
            else:
                stage_idx = [idx + 1 for idx in stage_idx]
            reference.extend(stage_idx)

        assert upipe_head_permutation(n_heads, n_kv_heads, cp_size) == reference

    @pytest.mark.parametrize(
        "n_heads,n_kv_heads,cp_size",
        [(32, 8, 3), (30, 7, 2), (32, 8, 16)],
    )
    def test_rejects_indivisible_shapes(self, n_heads, n_kv_heads, cp_size):
        with pytest.raises(ValueError):
            upipe_head_permutation(n_heads, n_kv_heads, cp_size)


class TestRopeTableAdapter:
    @staticmethod
    def _adapter():
        pytest.importorskip("triton")
        from nemo_automodel.components.distributed.context_parallel.upipe.rotary import (
            rope_tables_from_position_embeddings,
        )

        return rope_tables_from_position_embeddings

    def test_takes_the_first_half_and_drops_the_batch_axis(self):
        adapt = self._adapter()
        head_dim, seq = 64, 8
        half = torch.randn(seq, head_dim // 2)
        # HuggingFace duplicates the halves because rotate_half spans the full width.
        cos = torch.cat([half, half], dim=-1).unsqueeze(0)
        sin = torch.cat([half, half], dim=-1).unsqueeze(0)

        out_cos, out_sin = adapt(cos, sin, head_dim)

        assert out_cos.shape == (seq, head_dim // 2)
        assert out_sin.shape == (seq, head_dim // 2)
        assert torch.equal(out_cos, half)
        assert out_cos.is_contiguous() and out_sin.is_contiguous()

    def test_accepts_tables_without_a_batch_axis(self):
        adapt = self._adapter()
        cos = torch.randn(8, 64)
        out_cos, _ = adapt(cos, cos.clone(), 64)
        assert out_cos.shape == (8, 32)

    def test_rejects_tables_that_are_too_narrow(self):
        adapt = self._adapter()
        with pytest.raises(ValueError, match="need at least"):
            adapt(torch.randn(8, 16), torch.randn(8, 16), 64)

    def test_rejects_wrong_rank(self):
        adapt = self._adapter()
        with pytest.raises(ValueError, match="2D or 3D"):
            adapt(torch.randn(2, 2, 8, 64), torch.randn(2, 2, 8, 64), 64)


class TestValidation:
    BASE = dict(n_heads=32, n_kv_heads=8, head_dim=64, cp_size=8, require_flash_attn=False)

    def test_accepts_a_supported_configuration(self):
        validate_upipe_attention(**self.BASE)

    @pytest.mark.parametrize("n_heads,n_kv_heads,cp_size", VALID_SHAPES)
    def test_accepts_every_shape_the_permutation_supports(self, n_heads, n_kv_heads, cp_size):
        validate_upipe_attention(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=64,
            cp_size=cp_size,
            require_flash_attn=False,
        )

    def test_rejects_heads_not_divisible_by_cp_size(self):
        with pytest.raises(ValueError, match="num_attention_heads"):
            validate_upipe_attention(**{**self.BASE, "n_heads": 12, "n_kv_heads": 4, "cp_size": 8})

    def test_rejects_kv_heads_not_divisible_by_cp_size(self):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            validate_upipe_attention(**{**self.BASE, "n_kv_heads": 4, "cp_size": 8})

    def test_rejects_odd_head_dim(self):
        with pytest.raises(ValueError, match="even head_dim"):
            validate_upipe_attention(**{**self.BASE, "head_dim": 65})

    def test_rejects_head_dim_beyond_flash_attention_limit(self):
        with pytest.raises(ValueError, match="head_dim <= 256"):
            validate_upipe_attention(**{**self.BASE, "head_dim": 512})

    def test_rejects_tensor_parallelism(self):
        with pytest.raises(ValueError, match="tensor parallelism"):
            validate_upipe_attention(**{**self.BASE, "tp_size": 2})

    def test_rejects_non_torch_rope(self):
        with pytest.raises(ValueError, match="rope='torch'"):
            validate_upipe_attention(**{**self.BASE, "rope_backend": "quack"})

    def test_rejects_rope_fusion(self):
        with pytest.raises(ValueError, match="rope_fusion=False"):
            validate_upipe_attention(**{**self.BASE, "rope_fusion": True})

    def test_rejects_compiled_attention(self):
        with pytest.raises(ValueError, match="compile_attn=False"):
            validate_upipe_attention(**{**self.BASE, "compile_attn": True})

    def test_reports_missing_flash_attn(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.context_parallel.upipe.validation.flash_attn_available",
            lambda: False,
        )
        with pytest.raises(ValueError, match="flash-attn"):
            validate_upipe_attention(**{**self.BASE, "require_flash_attn": True})


class TestRuntimeValidation:
    @staticmethod
    def _validate(**overrides):
        from nemo_automodel.components.distributed.context_parallel.upipe.validation import validate_upipe_runtime

        kwargs = dict(has_peft=False, is_packed=False, has_non_trailing_pad=False)
        kwargs.update(overrides)
        return validate_upipe_runtime(**kwargs)

    def test_accepts_plain_causal_training(self):
        self._validate()

    def test_rejects_peft(self):
        with pytest.raises(ValueError, match="PEFT"):
            self._validate(has_peft=True)

    def test_rejects_packed_sequences(self):
        with pytest.raises(ValueError, match="packed"):
            self._validate(is_packed=True)

    def test_rejects_non_trailing_padding(self):
        with pytest.raises(ValueError, match="right padding"):
            self._validate(has_non_trailing_pad=True)


class TestPaddingDetection:
    """Right padding is safe under a causal mask; left and interior padding are not."""

    @staticmethod
    def _detect(rows):
        from nemo_automodel.components.distributed.context_parallel.upipe.validation import has_non_trailing_padding

        return has_non_trailing_padding(torch.tensor(rows))

    def test_unpadded_batch_is_fine(self):
        assert self._detect([[1, 1, 1, 1]]) is False

    def test_right_padding_is_fine(self):
        assert self._detect([[1, 1, 0, 0], [1, 1, 1, 0]]) is False

    def test_fully_padded_row_is_fine(self):
        assert self._detect([[0, 0, 0, 0]]) is False

    def test_left_padding_is_flagged(self):
        assert self._detect([[0, 0, 1, 1]]) is True

    def test_interior_padding_is_flagged(self):
        assert self._detect([[1, 0, 1, 1]]) is True

    def test_flags_a_batch_where_only_one_row_is_left_padded(self):
        assert self._detect([[1, 1, 0, 0], [0, 1, 1, 1]]) is True

    def test_handles_missing_and_degenerate_masks(self):
        from nemo_automodel.components.distributed.context_parallel.upipe.validation import has_non_trailing_padding

        assert has_non_trailing_padding(None) is False
        assert has_non_trailing_padding(torch.tensor([[1]])) is False
        assert has_non_trailing_padding(torch.ones(2, 3, 4)) is False


class TestBackendConfig:
    def test_upipe_is_a_selectable_attention_backend(self):
        from nemo_automodel.components.models.common.utils import BackendConfig

        assert BackendConfig(attn="upipe").attn == "upipe"


class TestCpHook:
    """The hook must claim CP only for UPipe, so every other backend keeps the generic path."""

    @staticmethod
    def _hook_result(attn_backend, batch=None):
        from nemo_automodel.components.models.common.utils import BackendConfig
        from nemo_automodel.components.models.llama.model import LlamaForCausalLM

        model = object.__new__(LlamaForCausalLM)
        torch.nn.Module.__init__(model)
        model.backend = BackendConfig(attn=attn_backend)
        return LlamaForCausalLM.prepare_model_inputs_for_cp(model, batch if batch is not None else {})

    def test_returns_a_sharder_for_upipe(self):
        result = self._hook_result("upipe")
        assert "cp_sharder" in result
        assert result["cp_sharder"].local_token_global_indices is not None

    @pytest.mark.parametrize("attn_backend", ["sdpa", "te", "flex", "eager"])
    def test_defers_to_the_framework_for_other_backends(self, attn_backend):
        assert self._hook_result(attn_backend) == {}

    def test_accepts_an_all_ones_attention_mask(self):
        batch = {"attention_mask": torch.ones(2, 8, dtype=torch.long)}
        assert "cp_sharder" in self._hook_result("upipe", batch)

    def test_accepts_a_right_padded_batch(self):
        batch = {"attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
        assert "cp_sharder" in self._hook_result("upipe", batch)

    def test_rejects_a_left_padded_batch(self):
        batch = {"attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])}
        with pytest.raises(ValueError, match="right padding"):
            self._hook_result("upipe", batch)

    def test_reads_an_inverted_padding_mask_when_there_is_no_attention_mask(self):
        # padding_mask marks pads, so it is the complement of attention_mask.
        batch = {"padding_mask": torch.tensor([[True, False, False, False]])}
        with pytest.raises(ValueError, match="right padding"):
            self._hook_result("upipe", batch)

    def test_rejects_packed_sequences(self):
        with pytest.raises(ValueError, match="packed"):
            self._hook_result("upipe", {"qkv_format": "thd"})

    def test_left_padded_batch_is_fine_for_other_backends(self):
        batch = {"attention_mask": torch.tensor([[0, 1, 1, 1]])}
        assert self._hook_result("sdpa", batch) == {}


class TestCapabilityGate:
    """UPipe owns its CP transport, so the CP gate must admit it the way it admits TE and magi."""

    @staticmethod
    def _supports(attn, cp_size=1):
        from nemo_automodel._transformers.capabilities import ModelSupports

        class _BackendModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backend = SimpleNamespace(attn=attn)
                self.config = SimpleNamespace()

            def forward(self, **kwargs):
                raise NotImplementedError

        # ModelSupports holds the model weakly, so hand it back to keep it alive.
        model = _BackendModel()
        return model, ModelSupports(model, SimpleNamespace(cp_size=cp_size))

    def test_cp_gate_admits_upipe(self):
        _model, supports = self._supports("upipe", cp_size=8)
        assert supports.supports_cp is True

    def test_cp_gate_still_rejects_a_non_cp_backend(self):
        _model, supports = self._supports("flex", cp_size=8)
        assert supports.supports_cp is False

    def test_upipe_does_not_claim_sequence_packing(self):
        # UPipe cannot see document boundaries; admitting it here would let a
        # packed run through and silently attend across documents.
        _model, supports = self._supports("upipe", cp_size=8)
        assert supports.supports_cp_with_sequence_packing is False
