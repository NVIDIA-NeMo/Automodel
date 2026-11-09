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

"""Train a checkpoint-loaded FP32 VLM with single-rank mixed-precision attention."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.models.common import BackendConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention kernels")


@pytest.fixture
def single_rank_group(tmp_path):
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1, timeout=timedelta(seconds=120)
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _save_checkpoint(path: Path) -> None:
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=["full_attention"],
        num_nextn_predict_layers=0,
        pad_token_id=0,
        torch_dtype="float32",
        attn_implementation="sdpa",
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=128,
        intermediate_size=256,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=128,
        attn_implementation="sdpa",
    )
    config = Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        tie_word_embeddings=False,
    )
    Qwen3_5ForConditionalGeneration(config).save_pretrained(path)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_loaded_single_rank_vlm_updates_both_towers(tmp_path, single_rank_group, checkpointing):
    torch.manual_seed(42)
    checkpoint = tmp_path / "checkpoint"
    _save_checkpoint(checkpoint)
    # Exercise FA4 on Blackwell and FA2 on other supported CUDA hosts.
    blackwell = torch.cuda.get_device_capability()[0] >= 10
    attention = "flash_attention_4" if blackwell else "flash_attention_2"
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        str(checkpoint),
        dtype=torch.float32,
        attn_implementation=attention,
        backend=BackendConfig(attn="fa4" if blackwell else "sdpa", linear="torch", rms_norm="torch", rope_fusion=False),
        distributed_setup=DistributedSetup.build(
            strategy=FSDP2Config(), world_size=1, activation_checkpointing=checkpointing
        ),
    ).train()
    buffers = {name: buffer.clone() for name, buffer in model.named_buffers() if buffer.is_floating_point()}
    weights = (model.model.visual.patch_embed.proj.weight, model.model.language_model.embed_tokens.weight)
    original_weights = [parameter.detach().clone() for parameter in weights]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    inputs = dict(
        input_ids=torch.tensor([[3, 62, 60, 60, 60, 60, 63, 5]], device="cuda"),
        attention_mask=torch.ones(1, 8, device="cuda", dtype=torch.long),
        position_ids=torch.arange(8, device="cuda").expand(3, 1, 8),
        pixel_values=torch.randn(4, 12, device="cuda", dtype=torch.bfloat16),
        image_grid_thw=torch.tensor([[1, 2, 2]], device="cuda"),
    )
    for _ in range(2):
        logits = model(**inputs).logits
        assert logits.dtype == torch.bfloat16
        loss = logits.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
        optimizer.zero_grad()

    current_buffers = dict(model.named_buffers())
    for name, original in buffers.items():
        assert current_buffers[name].dtype == torch.float32
        torch.testing.assert_close(current_buffers[name], original, rtol=0, atol=0)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    for parameter, original in zip(weights, original_weights):
        assert not torch.equal(parameter, original)
