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

"""Numerical parity of the native Beam model against the Hugging Face implementation."""

import glob
import os
import tempfile

import pytest
import torch

from nemo_automodel.components.models.beam.layers import (
    _cp_local_positions,
    is_moe_layer,
    packed_sdpa_mask,
    ssmax_query_scale,
    window_size_for_backend,
)
from nemo_automodel.components.models.beam.model import BeamForCausalLM, beam_moe_config
from nemo_automodel.components.models.beam.parallelization import PARALLELIZER, BeamModelParallelizer
from nemo_automodel.components.moe.layers import MLP, MoE

from .utils import hf_model_and_checkpoint, native_from_hf

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA backend")


class TestStructure:
    def test_layer_flavours(self, beam_config, backend_config):
        model = BeamForCausalLM(beam_config, backend=backend_config)
        layers = model.model.layers
        assert [layer.layer_type for layer in layers.values()] == beam_config.layer_types
        assert isinstance(layers["0"].mlp, MLP) and all(isinstance(layers[str(i)].mlp, MoE) for i in (1, 2, 3))
        assert [layers[str(i)].self_attn.use_rope for i in range(4)] == [True, False, True, False]
        assert [layers[str(i)].self_attn.use_ssmax for i in range(4)] == [False, True, False, True]
        assert [layers[str(i)].self_attn.sliding_window for i in range(4)] == [8, None, 8, None]
        assert [is_moe_layer(beam_config, i) for i in range(4)] == [False, True, True, True]
        assert set(model.model.rotary_emb) == {"sliding_attention"}

    def test_ssmax_scale_matches_hf(self):
        from transformers.models.beam.modeling_beam import compute_attention_scaling

        pos = torch.arange(0, 5000, 7).unsqueeze(0)
        ours = ssmax_query_scale(pos)
        theirs = compute_attention_scaling(pos)[:, 0, :, 0]
        torch.testing.assert_close(ours, theirs, atol=0, rtol=0)

    def test_rejects_pre_beam_config(self, beam_config, backend_config):
        """A flat (Reflection-style) rope_parameters dict must not silently turn NoPE + SSMax into RoPE."""
        beam_config.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        with pytest.raises(ValueError, match="keyed by layer type"):
            BeamForCausalLM(beam_config, backend=backend_config)

    def test_router_matches_hf_semantics(self, beam_config, backend_config):
        model = BeamForCausalLM(beam_config, backend=backend_config)
        gate = model.model.layers["1"].mlp.gate
        assert gate.score_func == "sigmoid_with_bias" and gate.weight.dtype == torch.float32
        assert gate.e_score_correction_bias is not None and gate.bias_update_factor == 0.0
        assert gate.n_groups == 2 and gate.topk_groups == 1 and gate.route_scale == 2.5 and gate.norm_topk_prob

    def test_window_convention_per_backend(self):
        assert window_size_for_backend(8, "sdpa") == (8, 0)
        assert window_size_for_backend(8, "te") == (7, 0)
        assert window_size_for_backend(None, "te") is None

    def test_packed_sdpa_mask(self):
        cu = torch.tensor([0, 3, 5], dtype=torch.int32)
        m = packed_sdpa_mask(cu, 5, None)[0, 0]
        assert m.tolist() == [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, False, False, True, False],
            [False, False, False, True, True],
        ]
        mw = packed_sdpa_mask(cu, 5, 2)[0, 0]
        assert mw[2].tolist() == [False, True, True, False, False]  # window of 2 tokens incl. current

    def test_cp_local_positions_zigzag(self):
        """Full-length position_ids are reduced to this rank's head/tail chunks (2*cp_size chunks); local ones pass through."""
        pos = torch.arange(16).unsqueeze(0)
        assert _cp_local_positions(pos, 8, 2, 0).tolist() == [[0, 1, 2, 3, 12, 13, 14, 15]]
        assert _cp_local_positions(pos, 8, 2, 1).tolist() == [[4, 5, 6, 7, 8, 9, 10, 11]]
        assert _cp_local_positions(pos, 4, 4, 1).tolist() == [[2, 3, 12, 13]]
        assert torch.equal(_cp_local_positions(pos, 16, 1, 0), pos)  # cp_size 1
        local = pos[:, :8]
        assert torch.equal(_cp_local_positions(local, 8, 2, 1), local)  # already sharded by the CP sharder

    def test_hf_state_dict_names_match_module_tree(self, beam_config, backend_config):
        model = BeamForCausalLM(beam_config, backend=backend_config)
        names = set(model.state_dict())
        for key in (
            "model.layers.0.self_attn.gate_bias",
            "model.layers.0.self_attn.gate_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.mlp.gate.weight",
            "model.layers.1.mlp.gate.e_score_correction_bias",
            "model.layers.1.mlp.experts.gate_and_up_projs",
            "lm_head.weight",
        ):
            assert key in names, key
        assert not any("attn_output_layernorm" in k or "mlp_output_layernorm" in k for k in names)


@pytest.mark.usefixtures("fp32_matmul")
class TestParityWithHF:
    @pytest.mark.parametrize(
        "backend_name,device,top_k,normalize,dropout",
        [
            ("backend_config", "cpu", 2, True, 0.0),
            pytest.param("backend_config", "cuda", 2, True, 0.0, marks=requires_cuda),
            pytest.param(
                "flex_backend_config",
                "cuda",
                2,
                True,
                0.0,
                marks=[
                    requires_cuda,
                    pytest.mark.runtime_budget(
                        30, reason="Cold CUDA FlexAttention forward compilation (12-20 s measured on H200)."
                    ),
                ],
            ),
            pytest.param("te_backend_config", "cuda", 2, True, 0.0, marks=requires_cuda),
            pytest.param("te_full_backend_config", "cuda", 2, True, 0.0, marks=requires_cuda),
            ("backend_config", "cpu", 1, True, 0.0),
            ("backend_config", "cpu", 1, False, 0.0),
            ("backend_config", "cpu", 2, True, 0.5),
        ],
    )
    def test_logits_match_hf(self, beam_config, request, backend_name, device, top_k, normalize, dropout):
        cfg = beam_config
        if top_k == 1:
            cfg.num_experts, cfg.num_experts_per_tok, cfg.n_group, cfg.topk_group = 4, 1, 1, 1
        cfg.norm_topk_prob, cfg.attention_dropout = normalize, dropout
        device = torch.device(device)
        hf, sd = hf_model_and_checkpoint(cfg, device)
        native = native_from_hf(cfg, request.getfixturevalue(backend_name), sd, device)
        ids = torch.randint(0, cfg.vocab_size, (2, 24), generator=torch.Generator().manual_seed(11)).to(device)
        with torch.no_grad():
            ref, out = hf(ids, use_cache=False).logits.float(), native(ids).logits.float()
        torch.testing.assert_close(out, ref, atol=2e-4, rtol=1e-4)
        assert out.abs().max() <= cfg.final_logit_softcapping

    def test_packed_thd_matches_hf(self, beam_config, backend_config):
        """Packed input on the SDPA backend (block-diagonal mask fallback) vs HF's packed forward."""
        device = torch.device("cpu")
        hf, sd = hf_model_and_checkpoint(beam_config, device)
        native = native_from_hf(beam_config, backend_config, sd, device)
        lens = (7, 13)
        torch.manual_seed(2)
        seqs = [torch.randint(0, beam_config.vocab_size, (1, n), device=device) for n in lens]
        ids = torch.cat(seqs, dim=1)
        pos = torch.cat([torch.arange(n, device=device) for n in lens]).unsqueeze(0)
        cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32, device=device)
        with torch.no_grad():
            ref = hf(input_ids=ids, position_ids=pos, attention_mask=None, use_cache=False).logits.float()
            out = native(ids, position_ids=pos, qkv_format="thd", cu_seqlens=cu, max_seqlen=max(lens)).logits.float()
        torch.testing.assert_close(out, ref, atol=2e-4, rtol=1e-4)

    def test_registry_resolves_native_class(self, beam_config, backend_config):
        from nemo_automodel._transformers.registry import ModelRegistry

        assert ModelRegistry.has_custom_model("BeamForCausalLM")
        assert ModelRegistry.resolve_custom_model_cls("BeamForCausalLM", beam_config) is BeamForCausalLM


@requires_cuda
class TestTEBackend:
    """TransformerEngine backend: fp32 parity (unfused attention) and native thd packing (bf16)."""

    def test_bf16_packed_thd_matches_unpacked(self, beam_config, te_backend_config):
        """TE's native thd path: packed sequences must match the same sequences run separately (bf16)."""
        device = torch.device("cuda")
        cfg = beam_config
        cfg.torch_dtype = torch.bfloat16
        hf, sd = hf_model_and_checkpoint(cfg, device)
        native = native_from_hf(cfg, te_backend_config, sd, device).to(torch.bfloat16)
        lens = (8, 16)
        torch.manual_seed(2)
        seqs = [torch.randint(0, cfg.vocab_size, (1, n), device=device) for n in lens]
        with torch.no_grad():
            separate = [native(s).logits.float()[0] for s in seqs]
            ids = torch.cat(seqs, dim=1)
            pos = torch.cat([torch.arange(n, device=device) for n in lens]).unsqueeze(0)
            cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32, device=device)
            packed = native(
                ids, position_ids=pos, qkv_format="thd", cu_seqlens=cu, max_seqlen=max(lens)
            ).logits.float()[0]
        for i, n in enumerate(lens):
            sl = slice(int(cu[i]), int(cu[i + 1]))
            torch.testing.assert_close(packed[sl], separate[i], atol=5e-2, rtol=5e-2)
            # boundary check: the second sequence must not see the first (would differ by O(1))
            assert (packed[sl] - separate[i]).abs().max() < 0.2


def _padded_batch(cfg, device, side, n_valid=15, seq_len=24, seed=3):
    """One unpadded row plus one row with ``seq_len - n_valid`` pads on the given side; positions count valid tokens."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    seq = torch.randint(0, cfg.vocab_size, (n_valid,), generator=g).to(device)
    pads_a = torch.randint(0, cfg.vocab_size, (seq_len - n_valid,), generator=g).to(device)
    pads_b = torch.randint(0, cfg.vocab_size, (seq_len - n_valid,), generator=g).to(device)
    other = torch.randint(0, cfg.vocab_size, (seq_len,), generator=g).to(device)

    def build(pads):
        if side == "left":
            row, mask_row = torch.cat([pads, seq]), torch.cat([torch.zeros_like(pads), torch.ones_like(seq)])
        else:
            row, mask_row = torch.cat([seq, pads]), torch.cat([torch.ones_like(seq), torch.zeros_like(pads)])
        ids = torch.stack([other, row])
        mask = torch.stack([torch.ones(seq_len, dtype=torch.long, device=device), mask_row.long()])
        return ids, mask, (mask.cumsum(-1) - 1).clamp(min=0), mask_row.bool()

    return seq, build(pads_a), build(pads_b)


def _padding_checks(cfg, backend, device, side):
    """Padding invariance (pad token ids must not move valid logits) and padded-vs-alone parity."""
    hf, sd = hf_model_and_checkpoint(cfg, device)
    native = native_from_hf(cfg, backend, sd, device)
    seq, (ids_a, mask, pos, valid), (ids_b, _, _, _) = _padded_batch(cfg, device, side)
    with torch.no_grad():
        alone = native(
            seq.unsqueeze(0), position_ids=torch.arange(seq.numel(), device=device).unsqueeze(0)
        ).logits.float()[0]
        la = native(ids_a, attention_mask=mask, position_ids=pos).logits.float()[1][valid]
        lb = native(ids_b, attention_mask=mask, position_ids=pos).logits.float()[1][valid]
        ref = hf(input_ids=ids_a, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()[1][valid]
    assert torch.equal(la, lb), (
        f"{side} padding: changing pad token ids moved valid logits by {(la - lb).abs().max():.3e}"
    )
    torch.testing.assert_close(la, alone, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(la, ref, atol=2e-4, rtol=1e-4)


class TestPaddingMasks:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_sdpa_honours_padding(self, beam_config, backend_config, side):
        _padding_checks(beam_config, backend_config, torch.device("cpu"), side)

    @requires_cuda
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_te_honours_padding(self, beam_config, te_backend_config, side):
        _padding_checks(beam_config, te_backend_config, torch.device("cuda"), side)

    @requires_cuda
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_flex_rejects_masked_batches(self, beam_config, flex_backend_config, side):
        """The shared FlexAttention has no padding mask; Beam refuses any masked batch on flex (no value inspection)."""
        device = torch.device("cuda")
        hf, sd = hf_model_and_checkpoint(beam_config, device)
        native = native_from_hf(beam_config, flex_backend_config, sd, device)
        _, (ids, mask, pos, _), _ = _padded_batch(beam_config, device, side)
        with pytest.raises(NotImplementedError, match="does not accept an attention_mask"):
            with torch.no_grad():
                native(ids, attention_mask=mask, position_ids=pos)
        with torch.no_grad():  # unmasked batches keep working on flex
            native(ids, position_ids=pos)


def _non_bf16_representable(shape, generator, scale=0.05):
    """fp32 values that bf16 cannot hold exactly (an offset far below bf16's 8-bit mantissa resolution)."""
    values = torch.randn(shape, generator=generator) * scale
    values = values + 2.0**-12
    assert (values.to(torch.bfloat16).float() != values).float().mean() > 0.9
    return values


def _set_fp32_router(hf_model, seed=11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for layer in hf_model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.weight.copy_(_non_bf16_representable(layer.mlp.gate.weight.shape, g))
                layer.mlp.gate.e_score_correction_bias.copy_(
                    _non_bf16_representable(layer.mlp.gate.e_score_correction_bias.shape, g, 0.1)
                )


class TestFp32Router:
    def test_declares_fp32_router_contract(self, beam_config, backend_config):
        assert BeamForCausalLM._keep_in_fp32_modules_strict == ["mlp.gate.weight", "e_score_correction_bias"]
        model = BeamForCausalLM(beam_config, backend=backend_config)
        assert model.backend.gate_precision == torch.float32  # defaulted from None
        gate = model.model.layers["1"].mlp.gate
        assert gate.weight.dtype == torch.float32 and gate.e_score_correction_bias.dtype == torch.float32
        assert gate.gate_precision == torch.float32 and gate.router_weights_fp32

    def test_rejects_lower_gate_precision(self, beam_config, backend_config):
        from dataclasses import replace

        with pytest.raises(ValueError, match="gate_precision must be float32"):
            BeamForCausalLM(beam_config, backend=replace(backend_config, gate_precision=torch.bfloat16))
        assert BeamForCausalLM(beam_config, backend=replace(backend_config, gate_precision=torch.float32)) is not None

    def test_router_values_survive_bf16_model_cast(self, beam_config, backend_config):
        """cast_model_to_dtype (what initialize_weights uses) must keep the pinned fp32 values bit-exact."""
        from nemo_automodel.components.models.common.utils import cast_model_to_dtype

        model = BeamForCausalLM(beam_config, backend=backend_config)
        g = torch.Generator(device="cpu").manual_seed(5)
        with torch.no_grad():
            for i in ("1", "2", "3"):
                gate = model.model.layers[i].mlp.gate
                gate.weight.copy_(_non_bf16_representable(gate.weight.shape, g))
                gate.e_score_correction_bias.copy_(_non_bf16_representable(gate.e_score_correction_bias.shape, g, 0.1))
        before = {k: v.detach().clone() for k, v in model.state_dict().items() if "mlp.gate." in k}
        cast_model_to_dtype(model, torch.bfloat16)
        after = model.state_dict()
        for key, value in before.items():
            assert after[key].dtype == torch.float32, key
            assert torch.equal(after[key], value), key
        assert model.model.layers["1"].self_attn.q_proj.weight.dtype == torch.bfloat16
        assert model.model.layers["0"].mlp.down_proj.weight.dtype == torch.bfloat16

    @requires_cuda
    def test_from_pretrained_bf16_keeps_fp32_router_values(self, beam_config, backend_config):
        """Production init path (NeMoAutoModelForCausalLM, bf16 model dtype): the router loads fp32 values untouched."""
        from safetensors.torch import load_file
        from transformers.models.beam import BeamForCausalLM as HFBeam

        from nemo_automodel import NeMoAutoModelForCausalLM

        torch.manual_seed(0)
        hf = HFBeam(beam_config)
        _set_fp32_router(hf)
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d, safe_serialization=True)
            saved = {}
            for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
                saved.update(load_file(f))
            model = NeMoAutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16, backend=backend_config)
        assert type(model).__name__ == "BeamForCausalLM"
        sd = model.state_dict()
        for layer in ("1", "2", "3"):
            for name in ("weight", "e_score_correction_bias"):
                key = f"model.layers.{layer}.mlp.gate.{name}"
                assert sd[key].dtype == torch.float32, key
                assert torch.equal(sd[key].cpu(), saved[key]), key
        assert sd["model.layers.1.self_attn.q_proj.weight"].dtype == torch.bfloat16

    def test_isolated_router_parity_with_bf16_activations(self, beam_config):
        """HF router vs native Gate: identical fp32 weights/bias and identical bf16 activations -> same expert sets."""
        from transformers.models.beam.modeling_beam import BeamTopKRouter

        from nemo_automodel.components.moe.layers import Gate

        g = torch.Generator(device="cpu").manual_seed(7)
        weight = _non_bf16_representable((beam_config.num_experts, beam_config.hidden_size), g, 0.5)
        bias = _non_bf16_representable((beam_config.num_experts,), g, 0.1)
        hf_router = BeamTopKRouter(beam_config)
        native_gate = Gate(beam_moe_config(beam_config, torch.bfloat16), gate_precision=torch.float32)
        with torch.no_grad():
            hf_router.weight.copy_(weight)
            hf_router.e_score_correction_bias.copy_(bias)
            native_gate.weight.copy_(weight)
            native_gate.e_score_correction_bias.copy_(bias)
        assert native_gate.weight.dtype == torch.float32 and hf_router.weight.dtype == torch.float32
        x = (torch.randn(4096, beam_config.hidden_size, generator=g) * 2).to(torch.bfloat16)
        with torch.no_grad():
            _, hf_weights, hf_idx = hf_router(x)
            out = native_gate(x, torch.ones(4096, dtype=torch.bool), None)
        nat_weights, nat_idx = out[0], out[1]
        hf_idx_sorted, order = hf_idx.sort(dim=-1)
        nat_idx_sorted, nat_order = nat_idx.sort(dim=-1)
        assert torch.equal(hf_idx_sorted, nat_idx_sorted), (
            f"expert sets differ on {(hf_idx_sorted != nat_idx_sorted).any(-1).sum()} tokens"
        )
        # HF applies routed_scaling_factor after aggregation, the native gate on the weights: undo it before comparing.
        nat_w = (nat_weights.float() / beam_config.routed_scaling_factor).gather(1, nat_order)
        torch.testing.assert_close(nat_w, hf_weights.float().gather(1, order), atol=1e-6, rtol=1e-5)

    def test_beam_parallelizer_registered(self, beam_config, backend_config):
        from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer

        model = BeamForCausalLM(beam_config, backend=backend_config)
        parallelizer = get_model_parallelizer(model)
        assert parallelizer is PARALLELIZER and isinstance(parallelizer, BeamModelParallelizer)
        # the MoE/EP path only routes FSDP units through model sidecars that opt in
        assert parallelizer._customizes_moe_fsdp
