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

"""Stage-3 tests: vision tower + projector/merger + VLM wrapper.

Parity uses an independent CPU transcription of the sglang vision forward
(``minimax_vl_common``): Conv3d patch embed, pre_layrnorm, axis-split 3D RoPE,
bidirectional attention, GELU projector, spatial patch merger.
"""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLVisionConfig
from nemo_automodel.components.models.minimax_m3_vl.vision_encoder import MiniMaxM3VisionModel

from .conftest import IMAGE_TOKEN_INDEX, VISION_CONFIG


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_vision_rope_freqs(vm, grid_thw, device):
    """Independently recompute the axis-split 3D RoPE per-token freqs."""
    head_dim = vm.config.hidden_size // vm.config.num_attention_heads
    rope_dims = 2 * (head_dim // 2)
    axis_dim = int(2 * ((rope_dims // 3) // 2))
    theta = vm.config.rope_theta
    m = vm.spatial_merge_size
    inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
    out = []
    for grid_t, grid_h, grid_w in grid_thw:
        tpf = grid_h * grid_w
        tpos = torch.arange(grid_t).unsqueeze(1).expand(-1, tpf).flatten()
        hpos = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w)
        hpos = hpos.reshape(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3)
        hpos = hpos.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()
        wpos = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1)
        wpos = wpos.reshape(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3)
        wpos = wpos.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()
        seq_t = torch.arange(max(grid_t, 1), dtype=inv_freq.dtype)
        seq_hw = torch.arange(max(grid_h, grid_w), dtype=inv_freq.dtype)
        ft = torch.outer(seq_t, inv_freq)
        fhw = torch.outer(seq_hw, inv_freq)
        out.append(torch.cat([ft[tpos], fhw[hpos], fhw[wpos]], dim=-1))
    return torch.cat(out, dim=0).to(device)


def _ref_vision(model, pixel_values, grid_thw):
    """Independent reference forward of the vision tower (reads model weights)."""
    vt = model.vision_tower
    vm = vt.vision_model
    cfg = vm.config
    H, D = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    tps, ps, C = vm.embeddings.temporal_patch_size, cfg.patch_size, cfg.num_channels

    px = pixel_values.float().reshape(-1, C, tps, ps, ps)
    w = vm.embeddings.patch_embedding.weight.float()
    x = F.conv3d(px, w, stride=(tps, ps, ps)).reshape(px.shape[0], -1)  # [N, hidden]
    x = F.layer_norm(
        x, (cfg.hidden_size,), vm.pre_layrnorm.weight.float(), vm.pre_layrnorm.bias.float(), cfg.layer_norm_eps
    )

    freqs = _ref_vision_rope_freqs(vm, grid_thw, x.device)
    cos, sin = freqs.cos().repeat(1, 2), freqs.sin().repeat(1, 2)
    rope_dim = cos.shape[-1]

    for layer in vm.encoder.layers:
        seq = x.shape[0]
        h = F.layer_norm(
            x, (cfg.hidden_size,), layer.layer_norm1.weight.float(), layer.layer_norm1.bias.float(), cfg.layer_norm_eps
        )
        q = F.linear(h, layer.self_attn.q_proj.weight.float(), layer.self_attn.q_proj.bias.float()).view(seq, H, D)
        k = F.linear(h, layer.self_attn.k_proj.weight.float(), layer.self_attn.k_proj.bias.float()).view(seq, H, D)
        v = F.linear(h, layer.self_attn.v_proj.weight.float(), layer.self_attn.v_proj.bias.float()).view(seq, H, D)
        for t in (q, k):
            t[..., :rope_dim] = t[..., :rope_dim] * cos[:, None, :] + _rotate_half(t[..., :rope_dim]) * sin[:, None, :]
        qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        ao = F.scaled_dot_product_attention(qh, kh, vh, scale=D**-0.5).squeeze(0).transpose(0, 1).reshape(seq, -1)
        x = x + F.linear(ao, layer.self_attn.out_proj.weight.float(), layer.self_attn.out_proj.bias.float())
        h = F.layer_norm(
            x, (cfg.hidden_size,), layer.layer_norm2.weight.float(), layer.layer_norm2.bias.float(), cfg.layer_norm_eps
        )
        h = F.linear(h, layer.mlp.fc1.weight.float(), layer.mlp.fc1.bias.float())
        h = F.gelu(h)
        x = x + F.linear(h, layer.mlp.fc2.weight.float(), layer.mlp.fc2.bias.float())

    # projector
    p = vt.multi_modal_projector
    x = F.linear(x, p.linear_1.weight.float(), p.linear_1.bias.float())
    x = F.gelu(x)
    x = F.linear(x, p.linear_2.weight.float(), p.linear_2.bias.float())
    # patch merger
    mg = vt.patch_merge_mlp
    x = x.reshape(x.shape[0] // (mg.spatial_merge_size**2), -1)
    x = F.linear(x, mg.linear_1.weight.float(), mg.linear_1.bias.float())
    x = F.gelu(x)
    x = F.linear(x, mg.linear_2.weight.float(), mg.linear_2.bias.float())
    return x


def _make_image_inputs(model, grid_thw=((1, 4, 4),)):
    cfg = model.config.vision_config
    patch_dim = cfg.num_channels * cfg.img_token_compression_config["temporal_patch_size"] * cfg.patch_size**2
    n_patches = sum(t * h * w for t, h, w in grid_thw)
    merge = cfg.img_token_compression_config["spatial_merge_size"] ** 2
    n_tokens = n_patches // merge
    torch.manual_seed(1)
    pixel_values = torch.randn(n_patches, patch_dim)
    return pixel_values, [list(g) for g in grid_thw], n_tokens


def test_vision_tower_parity(vlm_model):
    pixel_values, grid_thw, _ = _make_image_inputs(vlm_model)
    with torch.no_grad():
        mine = vlm_model.vision_tower(pixel_values, grid_thw).float()
        ref = _ref_vision(vlm_model, pixel_values, grid_thw)
    assert mine.shape == ref.shape
    assert torch.allclose(mine, ref, atol=1e-4), (mine - ref).abs().max().item()


def test_vlm_forward_finite_and_text_only(vlm_model):
    vocab = vlm_model.config.text_config.vocab_size
    pixel_values, grid_thw, n_tokens = _make_image_inputs(vlm_model)
    ids = torch.randint(2, 99, (1, 16))
    ids[0, 3 : 3 + n_tokens] = IMAGE_TOKEN_INDEX
    with torch.no_grad():
        logits = vlm_model(ids, pixel_values=pixel_values, image_grid_thw=grid_thw)
        text_only = vlm_model(torch.randint(2, 99, (1, 16)))
    assert logits.shape == (1, 16, vocab) and torch.isfinite(logits).all()
    assert text_only.shape == (1, 16, vocab) and torch.isfinite(text_only).all()


def test_vlm_splices_vision_features_via_reference(vlm_model):
    """Full VLM logits match feeding reference-vision embeds through the (verified) text path."""
    pixel_values, grid_thw, n_tokens = _make_image_inputs(vlm_model)
    ids = torch.randint(2, 99, (1, 16))
    ids[0, 5 : 5 + n_tokens] = IMAGE_TOKEN_INDEX
    with torch.no_grad():
        mine = vlm_model(ids, pixel_values=pixel_values, image_grid_thw=grid_thw).float()
        embeds = vlm_model.model.embed_tokens(ids).clone()
        embeds[ids == IMAGE_TOKEN_INDEX] = _ref_vision(vlm_model, pixel_values, grid_thw).to(embeds.dtype)
        ref = vlm_model.lm_head(vlm_model.model(None, inputs_embeds=embeds)).float()
    assert torch.allclose(mine, ref, atol=1e-4), (mine - ref).abs().max().item()


class _FakeCPMesh:
    """Minimal CP submesh stand-in (no real process group) for the shard path."""

    def __init__(self, size: int, rank: int):
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank

    def get_group(self):
        return None


def test_prepare_model_inputs_for_cp_is_sharder_only(vlm_model):
    """The CP hook is sharder-only: it consumes nothing and returns a
    ContextParallelismSharder whose shard_batch is the aux-only round-robin shard.
    Embedding + vision splice + sequence shard now run inside forward, so the raw
    batch (input_ids, pixel_values) is left intact for the forward to consume."""
    from nemo_automodel.components.distributed.cp_sharder import (
        ContextParallelismSharder,
        round_robin_local_indices,
        shard_batch_aux_only,
    )

    assert hasattr(vlm_model, "prepare_model_inputs_for_cp")
    pixel_values, grid_thw, n_tokens = _make_image_inputs(vlm_model)
    ids = torch.randint(2, 99, (1, 16))
    ids[0, 5 : 5 + n_tokens] = IMAGE_TOKEN_INDEX
    cp_batch = {"input_ids": ids, "pixel_values": pixel_values, "image_grid_thw": grid_thw}
    prepared = vlm_model(_pre_embed_only=True, _cp_batch=cp_batch)

    assert set(prepared) == {"cp_sharder"}
    sharder = prepared["cp_sharder"]
    assert isinstance(sharder, ContextParallelismSharder)
    assert sharder.shard_batch is shard_batch_aux_only
    assert sharder.local_token_global_indices is round_robin_local_indices
    # nothing consumed: the raw streams stay for the forward
    assert cp_batch["input_ids"] is ids and cp_batch["pixel_values"] is pixel_values


def test_forward_shards_sequence_in_forward_under_cp(vlm_model):
    """With a CP mesh installed, forward embeds the full sequence then keeps this
    rank's round-robin chunk pair, so the text model runs on (and the logits span)
    the local shard length ``padded / cp_size`` instead of the full sequence."""
    vocab = vlm_model.config.text_config.vocab_size
    ids = torch.randint(2, 99, (1, 16))  # already a multiple of 2*cp -> no pad
    try:
        vlm_model.cp_mesh = _FakeCPMesh(2, 0)
        with torch.no_grad():
            logits = vlm_model(ids)
    finally:
        vlm_model.cp_mesh = None
    assert logits.shape == (1, 8, vocab) and torch.isfinite(logits).all()


def test_forward_no_shard_without_cp_mesh(vlm_model):
    """Default (no CP mesh) forward keeps the full sequence length."""
    vocab = vlm_model.config.text_config.vocab_size
    ids = torch.randint(2, 99, (1, 16))
    with torch.no_grad():
        logits = vlm_model(ids)
    assert logits.shape == (1, 16, vocab)


def test_vlm_adapter_roundtrip_and_naming(vlm_model):
    adapter = vlm_model.state_dict_adapter
    native = {k: v.clone() for k, v in vlm_model.state_dict().items()}
    hf = adapter.to_hf(native)
    assert any(k.startswith("language_model.model.layers.") for k in hf)
    assert any(k.startswith("language_model.lm_head") for k in hf)
    assert any(k.startswith("vision_tower.vision_model.encoder.layers.") for k in hf)
    assert "multi_modal_projector.linear_1.weight" in hf  # top-level in HF layout
    assert "patch_merge_mlp.linear_1.weight" in hf
    assert any(k.endswith("block_sparse_moe.experts.0.w1.weight") for k in hf)

    back = adapter.from_hf(hf)
    assert set(back.keys()) == set(native.keys())
    for key in native:
        assert torch.allclose(native[key].float(), back[key].float(), atol=1e-6), key


def test_video_grid_rejected(vlm_model):
    # Image-only support: grid_t beyond vision_segment_max_frames must fail loudly
    # (video temporal segmentation is not implemented).
    vc = vlm_model.config.vision_config
    patch_dim = vc.num_channels * vc.img_token_compression_config["temporal_patch_size"] * vc.patch_size**2
    grid_t = vc.vision_segment_max_frames + 1
    pixel_values = torch.randn(grid_t * 4 * 4, patch_dim)
    with pytest.raises(AssertionError):
        vlm_model.vision_tower(pixel_values, [[grid_t, 4, 4]])


def test_patch_merge_bias_independent():
    # patch_merge_bias is a separate flag from multimodal_projector_bias.
    vc = MiniMaxM3VLVisionConfig(**VISION_CONFIG)
    vm = MiniMaxM3VisionModel(
        vc, text_hidden_size=64, projector_hidden_size=64, multimodal_projector_bias=True, patch_merge_bias=False
    )
    assert vm.multi_modal_projector.linear_1.bias is not None
    assert vm.patch_merge_mlp.linear_1.bias is None
