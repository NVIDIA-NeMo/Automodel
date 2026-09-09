# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch


class _FakeCPMesh:
    def __init__(self, rank: int, size: int = 2):
        self._rank = rank
        self._size = size

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def _batch():
    neg = torch.finfo(torch.float32).min
    mask = torch.full((1, 1, 3, 8), neg)
    mask[..., :5] = 0
    for i in range(3):
        mask[..., i, 5 + i] = 0
    return {
        "input_ids": torch.arange(5)[None],
        "canvas_ids": torch.arange(20, 23)[None],
        "encoder_position_ids": torch.arange(5)[None],
        "decoder_position_ids": torch.arange(10, 13)[None],
        "encoder_padding_mask": torch.zeros((1, 5), dtype=torch.bool),
        "decoder_padding_mask": torch.zeros((1, 3), dtype=torch.bool),
        "encoder_labels": torch.tensor([[1, 2, 3, 4, -100]]),
        "decoder_attention_mask": {"full_attention": mask, "sliding_attention": mask.clone()},
    }


def test_mixed_stream_cp_layout_and_bias_padding():
    from nemo_automodel.components.models.diffusion_gemma.cp import shard_diffusion_gemma_batch

    shards = []
    for rank in range(2):
        batch = _batch()
        _, sharded, layout = shard_diffusion_gemma_batch(_FakeCPMesh(rank), None, batch, padding_token_id=99)
        shards.append(sharded)
        assert layout.original_seq_len == 3
        assert layout.padded_seq_len == 4
        assert sharded["input_ids"].shape == (1, 6)
        assert sharded["canvas_ids"].shape == (1, 2)
        assert sharded["decoder_attention_mask"]["full_attention"].shape == (1, 1, 2, 12)

    def restore(key):
        values = torch.cat([s[key] for s in shards], dim=1)
        indices = torch.cat([s["cp_encoder_indices" if key != "canvas_ids" else "cp_canvas_indices"] for s in shards])
        return values.index_select(1, torch.argsort(indices))

    assert restore("input_ids").tolist() == [[0, 1, 2, 3, 4, 99, 99, 99, 99, 99, 99, 99]]
    assert restore("canvas_ids").tolist() == [[20, 21, 22, 99]]

    bias = shards[0]["decoder_attention_mask"]["full_attention"]
    # Global bias is replicated. Pad query row 3 has a valid own-canvas key and
    # all newly padded non-diagonal locations remain masked.
    assert bias[0, 0, 1, 8].item() == 0
    assert bias[0, 0, 1, 11].item() == -1.0e4
    assert bias.isfinite().all()
