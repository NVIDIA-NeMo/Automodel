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

"""Native-only tests for the diffusion_gemma block-diffusion model.

These exercise construction, the single-shared-stack two-pass training forward,
the END-TO-END leakage invariant on the composed forward (design v2 item 2),
the frozen router, and the state-dict adapter round-trip. They do NOT need the
transformers 5.8 fork (unlike the parity test), but they DO need torch + the
``gemma4_moe`` MoE backend, so they only run where those import (the training
container), not on a stock-transformers login node.
"""

import importlib.util

import pytest
import torch

# The model reuses the fork's leaf layers + config (transformers.models.diffusion_gemma)
# and the gemma4_moe MoE backend (transformers.models.gemma4); both ship in the 5.8-dev
# fork wheel, so gate on the fork being importable.
_FORK_AVAILABLE = importlib.util.find_spec("transformers.models.diffusion_gemma") is not None
_GEMMA4_AVAILABLE = importlib.util.find_spec("transformers.models.gemma4") is not None

pytestmark = pytest.mark.skipif(
    not (_FORK_AVAILABLE and _GEMMA4_AVAILABLE),
    reason="transformers.models.diffusion_gemma (5.8-dev fork) / gemma4 backend not available",
)


def _tiny_model(self_conditioning=True, freeze_router=True):
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion

    text_cfg = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=1,
        sliding_window=4096,
        layer_types=["sliding_attention", "full_attention"],
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=16,
    )
    # self_conditioning/freeze_router are model-construction flags (not strict fork-config
    # fields), so they are passed to the model, not the config.
    config = DiffusionGemmaConfig(text_config=text_cfg, vision_config=None, canvas_length=4)
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )
    model = DiffusionGemmaForBlockDiffusion(
        config, backend=backend, self_conditioning=self_conditioning, freeze_router=freeze_router
    ).to(torch.float32)
    # Initialize the grouped-expert params (torch.empty otherwise).
    for layer in model.model.layers.values():
        torch.nn.init.normal_(layer.moe.experts.gate_and_up_projs, std=0.02)
        torch.nn.init.normal_(layer.moe.experts.down_projs, std=0.02)
    return model, config


def _masks(model, batch_size, seq_len, block_size, device, dtype=torch.float32):
    from nemo_automodel.components.models.diffusion_gemma.attention_mask import (
        build_block_diffusion_training_mask,
    )

    mask_full, mask_sliding = build_block_diffusion_training_mask(
        prefix_lengths=0,
        response_length=seq_len,
        enc_len=seq_len,
        block_size=block_size,
        sliding_window=model.text_config.sliding_window,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    return {"full_attention": mask_full, "sliding_attention": mask_sliding}


def test_construct_and_forward_shape():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    model.train()
    b, s, blk = 2, 8, 4
    vocab = 64
    clean = torch.randint(0, vocab, (b, s))
    canvas = clean.clone()
    canvas[:, blk:] = torch.randint(0, vocab, (b, s - blk))  # "corrupt" later blocks
    masks = _masks(model, b, s, blk, clean.device)

    out = model(input_ids=clean, canvas_ids=canvas, decoder_attention_mask=masks, do_self_conditioning=True)
    assert out.logits.shape == (b, s, vocab)
    assert torch.isfinite(out.logits).all()


def test_tie_weights_restores_lm_head_alias():
    model, _ = _tiny_model()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    assert model.lm_head.weight is not model.model.embed_tokens.weight

    model.tie_weights()

    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_top_forward_enters_backbone_call_for_fsdp_hooks():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    model.train()
    b, s, blk = 1, 8, 4
    vocab = 64
    clean = torch.randint(0, vocab, (b, s))
    canvas = torch.randint(0, vocab, (b, s))
    masks = _masks(model, b, s, blk, clean.device)

    calls = 0

    def count_backbone_call(_module, _inputs):
        nonlocal calls
        calls += 1

    handle = model.model.register_forward_pre_hook(count_backbone_call)
    try:
        model(input_ids=clean, canvas_ids=canvas, decoder_attention_mask=masks, do_self_conditioning=True)
    finally:
        handle.remove()

    assert calls == 3


def test_padding_masks_reach_moe_routing():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    model.train()
    b, s, blk = 2, 8, 4
    vocab = 64
    clean = torch.randint(0, vocab, (b, s))
    canvas = torch.randint(0, vocab, (b, s))
    masks = _masks(model, b, s, blk, clean.device)
    encoder_padding_mask = torch.tensor(
        [[False, False, False, False, False, False, True, True], [False, False, False, True, True, True, True, True]]
    )
    decoder_padding_mask = torch.tensor(
        [[False, False, False, False, True, True, True, True], [False, False, True, True, True, True, True, True]]
    )

    seen_masks = []
    for layer in model.model.layers.values():

        def fake_moe(x, padding_mask=None, cp_mesh=None, *, gate_input=None):
            seen_masks.append(padding_mask)
            return torch.zeros_like(x)

        layer.moe.forward = fake_moe

    model(
        input_ids=clean,
        canvas_ids=canvas,
        encoder_padding_mask=encoder_padding_mask,
        decoder_attention_mask=masks,
        decoder_padding_mask=decoder_padding_mask,
        do_self_conditioning=False,
    )

    # Training self-conditioning always runs the decoder twice (a no-grad pass-1
    # that produces the self-cond signal, then the real pass-2); do_self_conditioning
    # only gates whether pass-2 *consumes* that signal, not whether pass-1 runs. So a
    # 2-layer model issues encode(2) + decode-pass-1(2) + decode-pass-2(2) = 6 MoE
    # calls, ordered [enc, enc, dec, dec, dec, dec].
    assert len(seen_masks) == 6
    assert all(mask is encoder_padding_mask for mask in seen_masks[:2])
    assert all(mask is decoder_padding_mask for mask in seen_masks[2:])


def test_frozen_router_no_grad():
    torch.manual_seed(0)
    model, _ = _tiny_model(freeze_router=True)
    model.train()
    b, s, blk = 1, 8, 4
    vocab = 64
    clean = torch.randint(0, vocab, (b, s))
    canvas = torch.randint(0, vocab, (b, s))
    masks = _masks(model, b, s, blk, clean.device)

    out = model(input_ids=clean, canvas_ids=canvas, decoder_attention_mask=masks, do_self_conditioning=False)
    out.logits.float().sum().backward()

    for layer in model.model.layers.values():
        gate = layer.moe.gate
        assert gate.proj.weight.grad is None, "frozen router proj.weight received a gradient"
        assert gate.scale.grad is None, "frozen router scale received a gradient"
        # Experts must stay trainable and receive gradients.
        assert layer.moe.experts.gate_and_up_projs.grad is not None
        assert not gate.proj.weight.requires_grad
        assert not gate.scale.requires_grad
        assert layer.moe.experts.gate_and_up_projs.requires_grad


def test_two_pass_pass1_is_no_grad():
    """The two-pass self-cond must backprop only through pass-2.

    With self-conditioning on, the loss should depend on the self-conditioning
    MLP weights (used in pass-2). Pass-1 runs under no_grad, so it must not
    create a second, leaking gradient path — we assert the graph is finite and
    the self_conditioning module receives a gradient (it is used in pass-2).
    """
    torch.manual_seed(0)
    model, _ = _tiny_model(self_conditioning=True, freeze_router=False)
    model.train()
    b, s, blk = 2, 8, 4
    vocab = 64
    clean = torch.randint(0, vocab, (b, s))
    canvas = torch.randint(0, vocab, (b, s))
    masks = _masks(model, b, s, blk, clean.device)

    out = model(input_ids=clean, canvas_ids=canvas, decoder_attention_mask=masks, do_self_conditioning=True)
    out.logits.float().sum().backward()

    sc = model.model.self_conditioning
    assert sc.gate_proj.weight.grad is not None, "self-conditioning MLP got no gradient (pass-2 should use it)"
    assert torch.isfinite(sc.gate_proj.weight.grad).all()


def test_no_leakage_on_composed_forward():
    """END-TO-END leakage (design v2 item 2): a response block's logits must not
    change when its OWN clean copy in ``input_ids`` is perturbed, but MUST change
    when an EARLIER clean block is perturbed.

    With the full-sequence-canvas layout (prefix 0), canvas block 1 (positions
    ``[blk, 2*blk)``) may attend clean encoder block 0 and its own noised block,
    but NOT the clean encoder copy of block 1 (strict ``block_q > block_kv``). So:

    * perturbing a clean token in block 1 -> block-1 logits UNCHANGED (no leak);
    * perturbing a clean token in block 0 -> block-1 logits CHANGE (live path).

    We compose the two passes explicitly (a leaky pass-1 would poison the
    self-conditioning signal fed to pass-2).
    """
    torch.manual_seed(0)
    model, _ = _tiny_model(self_conditioning=True, freeze_router=True)
    model.eval()
    b, s, blk = 1, 8, 4  # 2 blocks of size 4
    vocab = 64
    clean = torch.randint(0, vocab, (b, s))
    canvas = torch.randint(0, vocab, (b, s))
    masks = _masks(model, b, s, blk, clean.device)

    def run(clean_ids):
        # Compose the two passes explicitly: pass-1 (no self-cond) -> detached
        # logits -> pass-2 conditioned on them.
        enc_pos = torch.arange(s).unsqueeze(0)
        with torch.no_grad():
            kv = model.model.encode(clean_ids, position_ids=enc_pos, padding_mask=None)
            h1 = model.model.decode(
                canvas, encoder_kv=kv, decoder_position_ids=enc_pos, decoder_masks=masks, self_conditioning_logits=None
            )
            sc = model._softcap_logits(h1).detach()
            h2 = model.model.decode(
                canvas, encoder_kv=kv, decoder_position_ids=enc_pos, decoder_masks=masks, self_conditioning_logits=sc
            )
            return model._softcap_logits(h2)

    base = run(clean)
    block1 = slice(blk, 2 * blk)

    # Perturb block 1's OWN clean token -> block-1 logits must be UNCHANGED.
    clean_own = clean.clone()
    clean_own[0, blk] = (clean_own[0, blk] + 1) % vocab
    diff_own = (run(clean_own)[:, block1] - base[:, block1]).abs().max().item()
    assert diff_own < 1e-5, f"LEAKAGE: block 1 saw its own clean copy (diff={diff_own})"

    # Perturb block 0's clean token -> block-1 logits SHOULD change (live path).
    clean_prev = clean.clone()
    clean_prev[0, 0] = (clean_prev[0, 0] + 1) % vocab
    diff_prev = (run(clean_prev)[:, block1] - base[:, block1]).abs().max().item()
    assert diff_prev > 1e-6, "expected block 1 to depend on the earlier clean block 0"


def test_state_dict_adapter_round_trip_native():
    """Native -> HF -> native key/value round-trip (no fork needed)."""
    model, _ = _tiny_model(freeze_router=False)
    adapter = model.state_dict_adapter
    # The adapter downcasts converted weights to its dtype (the ckpt's bf16 in
    # real use, which is correct). Run it in fp32 here so the round-trip is
    # exact and validates the transpose/fold/rename LOGIC, not bf16 rounding.
    adapter.dtype = torch.float32

    native_sd = {k: v.clone() for k, v in model.state_dict().items()}
    hf_sd = adapter.to_hf(native_sd)
    back = adapter.from_hf(hf_sd)

    # Every native parameter must round-trip (buffers like rope inv_freq are
    # non-persistent and absent from state_dict).
    native_params = {n for n, _ in model.named_parameters()}
    for name in native_params:
        if name == "lm_head.weight":
            continue  # tied; reconstructed from embed on from_hf
        assert name in back, f"missing after round-trip: {name}"
        max_diff = (native_sd[name].float() - back[name].float()).abs().max().item()
        assert max_diff == 0.0, f"value drift for {name}: {max_diff}"

    # HF side must be under model.decoder.* and carry per_expert_scale.
    assert any(k.startswith("model.decoder.layers.0.experts.") for k in hf_sd)
    assert any(k.endswith(".router.per_expert_scale") for k in hf_sd)
    assert "lm_head.weight" not in hf_sd  # tied, omitted from the HF checkpoint
