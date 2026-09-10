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

"""Image boundaries preserve Engram masks and select modality-aware expert routes."""

import torch
from PIL import Image

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.models.deepseek_v41.processing import IMAGE_PLACEHOLDER, DeepseekV41Processor
from tests.unit_tests.models.deepseek_v41.test_vision import _config, _tokenizer


def test_image_masks_reach_engram_and_modality_routing() -> None:
    tokenizer = _tokenizer()
    config = DeepseekV41Config(
        text_config=DeepseekV41TextConfig(
            vocab_size=32,
            hidden_size=8,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=32,
            qk_rope_head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            o_groups=1,
            hc_mult=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            candidate_source_layer_id=-1,
            engram_layer_ids=[0],
            engram_num_embeddings=[101],
            engram_vocab_size=11,
            engram_max_ngram_size=2,
            engram_n_heads=1,
            engram_head_dim=32,
            engram_compressed_vocab_size=len(tokenizer),
            dtype="float32",
        ),
        vision_config=_config().vision_config,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER),
        dtype="float32",
    )
    model = DeepseekV41ForCausalLM(
        config,
        tokenizer=tokenizer,
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch"),
    )
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    layer = model.model.layers["0"]
    with torch.no_grad():
        layer.mlp.gate.weight.zero_()
        layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor([10.0, 9.0, 0.0, 0.0]))
        layer.mlp.gate.bias_vl.copy_(torch.tensor([0.0, 0.0, 10.0, 9.0]))
    batch = DeepseekV41Processor(tokenizer, config)(
        f"one {IMAGE_PLACEHOLDER} two", Image.new("RGB", (4, 6), "blue"), return_tensors="pt"
    )
    observed = {}

    def record_hash_mask(_module, args, kwargs):
        """Capture the bool [batch, sequence] input mask without changing it."""
        observed["hash_mask"] = kwargs["token_mask"].detach().clone()

    def record_engram(_module, args, kwargs, output):
        """Compare [batch, sequence, streams, hidden] before and after memory writes."""
        observed["engram_mask"] = kwargs["token_mask"].detach().clone()
        image_mask = batch["vision_token_types"] >= 0
        torch.testing.assert_close(output[image_mask], args[0][image_mask], rtol=0, atol=0)

    def record_routes(_module, _args, output):
        """Capture selected expert IDs [tokens, topk] without replacing gate outputs."""
        observed["routes"] = output[1].detach().clone()

    handles = [
        model.model.engram_hash.register_forward_pre_hook(record_hash_mask, with_kwargs=True),
        layer.engram.register_forward_hook(record_engram, with_kwargs=True),
        layer.mlp.gate.register_forward_hook(record_routes),
    ]
    try:
        result = model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(result.logits).all()
    text_mask = batch["vision_token_types"] < 0
    torch.testing.assert_close(observed["hash_mask"], text_mask, rtol=0, atol=0)
    torch.testing.assert_close(observed["engram_mask"], text_mask, rtol=0, atol=0)
    routes = observed["routes"].sort(-1).values
    torch.testing.assert_close(routes[text_mask.flatten()], torch.tensor([0, 1]).expand(int(text_mask.sum()), 2))
    torch.testing.assert_close(routes[~text_mask.flatten()], torch.tensor([2, 3]).expand(int((~text_mask).sum()), 2))
