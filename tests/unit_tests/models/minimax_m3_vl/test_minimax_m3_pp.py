# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Pipeline-parallel plumbing tests for the M3 VL wrapper.

These exercise the model-side PP hooks without a process group: the FQN
rewrite (against the framework's real auto-generated names), the stage
predicate, the shape metas, and partial-stage forwards (first / middle /
last) built by nulling modules the way the framework's splitter does.
"""

import copy

import pytest
import torch

from nemo_automodel.components.distributed.pipelining.functional import generate_hf_model_fqn_per_model_part
from nemo_automodel.components.distributed.pipelining.hf_utils import MULTIMODAL_SUFFIXES

NUM_LAYERS = 3


def _auto_fqns(num_stages: int):
    """Reproduce what the framework generates for M3 (language_model property ->
    nested `model.language_model.` text prefix; vision under `model.`)."""
    return generate_hf_model_fqn_per_model_part(
        num_stages=num_stages,
        num_layers=NUM_LAYERS,
        include_embeddings=True,
        include_lm_head=True,
        include_rotary_emb=True,
        include_multimodal_encoders=False,
        extra_module_fqns=[f"model.{s}" for s in MULTIMODAL_SUFFIXES],
        fqn_prefix="model.language_model.",
        lm_head_fqn="lm_head",
    )


def _prune_layers(model, keep_keys):
    for k in list(model.model.layers.keys()):
        if k not in keep_keys:
            del model.model.layers[k]


def _make_image_inputs(model, grid_thw=((1, 4, 4),)):
    cfg = model.config.vision_config
    patch_dim = cfg.num_channels * cfg.img_token_compression_config["temporal_patch_size"] * cfg.patch_size**2
    n_patches = sum(t * h * w for t, h, w in grid_thw)
    n_tokens = n_patches // (cfg.img_token_compression_config["spatial_merge_size"] ** 2)
    torch.manual_seed(1)
    pixel_values = torch.randn(n_patches, patch_dim)
    return pixel_values, [list(g) for g in grid_thw], n_tokens


def test_customize_rewrites_to_real_module_paths(vlm_model):
    raw = _auto_fqns(num_stages=2)
    fixed = vlm_model.customize_pipeline_stage_modules(raw, layers_prefix="model.language_model.")
    flat = [n for stage in fixed for n in stage]

    # Text stack rewritten to model.* (never the nested model.language_model.*).
    assert "model.embed_tokens" in flat
    assert "model.norm" in flat
    assert "model.rotary_emb" in flat
    assert all(f"model.layers.{i}" in flat for i in range(NUM_LAYERS))  # all layers present
    assert all(not n.startswith("model.language_model.") for n in flat)

    # Vision tower is a top-level sibling, not under model.*
    assert "vision_tower" in flat
    assert "model.vision_tower" not in flat
    assert "lm_head" in flat

    # Every rewritten FQN must resolve to a real submodule (the splitter nulls by
    # exactly these names) -- non-existent multimodal suffixes are the only allowed
    # extras and must NOT be real-path collisions.
    real = dict(vlm_model.named_modules())
    for n in flat:
        if n in real:
            continue
        # The only tolerated misses are extra multimodal suffixes M3 doesn't have.
        assert n in MULTIMODAL_SUFFIXES, f"rewritten FQN {n!r} is not a real module path"


def test_is_pp_stage_full_model(vlm_model):
    assert vlm_model._is_pipeline_parallel_stage() is False


def test_pipeline_stage_metas(vlm_model):
    h = vlm_model.config.text_config.hidden_size
    v = vlm_model.config.text_config.vocab_size
    ins, outs = vlm_model.get_pipeline_stage_metas(is_first=True, microbatch_size=2, seq_len=5, dtype=torch.float32)
    assert ins[0].shape == (2, 5) and ins[0].dtype == torch.long
    assert outs[0].shape == (2, 5, v)  # full model owns lm_head
    ins2, _ = vlm_model.get_pipeline_stage_metas(is_first=False, microbatch_size=2, seq_len=5, dtype=torch.float32)
    assert ins2[0].shape == (2, 5, h)

    # Logits meta must follow lm_head's own dtype, not the passed model dtype, so
    # it stays correct if lm_head is ever kept in fp32 while the model runs bf16.
    ins3, outs3 = vlm_model.get_pipeline_stage_metas(is_first=False, microbatch_size=2, seq_len=5, dtype=torch.bfloat16)
    assert ins3[0].dtype == torch.bfloat16  # inter-stage hidden uses the model dtype
    assert outs3[0].dtype == vlm_model.lm_head.weight.dtype  # logits follow lm_head (float32 here)
    assert outs3[0].dtype != torch.bfloat16


def test_first_stage_forward_returns_hidden(vlm_model):
    m = copy.deepcopy(vlm_model)
    m.model.norm = None
    m.lm_head = None
    _prune_layers(m, {"0"})
    assert m._is_pipeline_parallel_stage() is True

    b, s, h = 2, 16, vlm_model.config.text_config.hidden_size
    ids = torch.randint(0, 120, (b, s))  # no image-token ids
    out = m(ids)
    assert out.shape == (b, s, h)


def test_middle_stage_forward_consumes_and_returns_hidden(vlm_model):
    m = copy.deepcopy(vlm_model)
    m.model.embed_tokens = None
    m.model.norm = None
    m.lm_head = None
    _prune_layers(m, {"1"})
    assert m._is_pipeline_parallel_stage() is True

    b, s, h = 2, 16, vlm_model.config.text_config.hidden_size
    hidden_in = torch.randn(b, s, h)  # previous stage output in the input_ids slot
    out = m(hidden_in)
    assert out.shape == (b, s, h)


def test_stage0_consumes_pp_media_chunks(vlm_model):
    """Under PP, stage 0 receives media via _vlm_*_chunks (not in the batch); the
    chunk path must reproduce the direct-media path so vision is actually spliced."""
    from .conftest import IMAGE_TOKEN_INDEX

    pixel_values, grid_thw, n_tokens = _make_image_inputs(vlm_model)
    grid_tensor = torch.tensor(grid_thw, dtype=torch.long)
    ids = torch.randint(2, 99, (1, 24))
    ids[0, 5 : 5 + n_tokens] = IMAGE_TOKEN_INDEX

    # Reference: full model, media passed directly in the call (non-PP path).
    with torch.no_grad():
        direct = vlm_model(ids, pixel_values=pixel_values, image_grid_thw=grid_thw)

    # Stage 0: keep embed_tokens + all layers, but route through the chunk path
    # with pixel_values omitted from the call, as the PP schedule does.
    m = copy.deepcopy(vlm_model)
    m._vlm_pixel_values_chunks = [pixel_values]
    m._vlm_image_grid_hws_chunks = [grid_tensor]
    m._vlm_chunk_idx = 0
    with torch.no_grad():
        via_chunks = m(ids)  # no pixel_values in the call -> must come from chunks

    assert m._vlm_chunk_idx == 1, "media chunk cursor did not advance"
    torch.testing.assert_close(via_chunks, direct, rtol=1e-4, atol=1e-4)

    # Guard against the regression: without the chunk plumbing the image positions
    # would stay as raw embed_tokens(image_token_index) and diverge from `direct`.
    m_blind = copy.deepcopy(vlm_model)
    with torch.no_grad():
        no_media = m_blind(ids)  # no chunks, no pixel_values -> no splicing
    assert not torch.allclose(no_media, direct, rtol=1e-3, atol=1e-3)


def test_mtp_under_pp_raises_on_forward(vlm_model):
    """The forward MTP-under-PP guard is keyed on config (not the nullable mtp
    module), so it fires even when a manual module_fqns split bypasses the
    customize_pipeline_stage_modules hook."""
    m = copy.deepcopy(vlm_model)
    m.config.text_config.num_mtp_modules = 1  # config declares MTP...
    m.model.norm = None
    m.lm_head = None
    _prune_layers(m, {"0"})  # ...and this is a partial PP stage
    assert m._is_pipeline_parallel_stage() is True
    with pytest.raises(NotImplementedError, match="MTP"):
        m(torch.randint(0, 120, (1, 8)))


def test_last_stage_forward_returns_logits(vlm_model):
    m = copy.deepcopy(vlm_model)
    m.model.embed_tokens = None
    _prune_layers(m, {"2"})  # keeps norm + lm_head
    assert m._is_pipeline_parallel_stage() is True

    b, s = 2, 16
    h = vlm_model.config.text_config.hidden_size
    v = vlm_model.config.text_config.vocab_size
    hidden_in = torch.randn(b, s, h)
    out = m(hidden_in)
    assert out.shape == (b, s, v)
