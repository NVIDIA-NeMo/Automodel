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

"""Packed text boundaries must not override HF image/video frame boundaries."""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.vision_utils import get_vision_attention_seqlens

from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing_for_models
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration


def _model(
    *, attn: str = "sdpa", device: str = "cpu", dtype: torch.dtype = torch.float32, hybrid: bool = False
) -> Qwen3_5ForConditionalGeneration:
    torch.manual_seed(42)
    hidden_size = 16 if device == "cpu" else 128
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2 if hybrid else 1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=hidden_size // 2,
        layer_types=["linear_attention", "full_attention"] if hybrid else ["full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        pad_token_id=0,
        torch_dtype="float32",
        attn_implementation="sdpa",
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=hidden_size,
        attn_implementation="sdpa" if device == "cpu" else "flash_attention_2",
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
    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", rope_fusion=False)
    model = Qwen3_5ForConditionalGeneration(config, backend=backend).to(device=device).eval()
    model.model.language_model.init_weights(buffer_device=torch.device(device))
    cast_model_to_dtype(model, dtype)
    return model


def check_packed_media_parity(
    media: str,
    precomputed: bool,
    *,
    attn: str = "sdpa",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    pack_sizes: tuple[int, ...] = (2,),
    hybrid: bool = False,
) -> None:
    """Compare packed logits and all parameter gradients to independent documents."""
    model = _model(attn=attn, device=device, dtype=dtype, hybrid=hybrid)
    reference = _model(device=device, dtype=dtype, hybrid=hybrid)
    # TE's attention bookkeeping has no counterpart in the SDPA reference.
    reference.load_state_dict(
        {key: value for key, value in model.state_dict().items() if not key.endswith("._extra_state")}
    )
    if attn == "te":
        from transformer_engine.pytorch.attention import DotProductAttention

        full_attention = model.model.language_model.layers[str(int(hybrid))].self_attn
        assert isinstance(full_attention.attn_module, DotProductAttention)
    contract = configure_packing_for_models([model])
    documents = []
    for document_index in range(sum(pack_sizes)):
        tokens = [3 + document_index]
        document = {}
        for kind, token, frames, pixels_key in (
            ("image", 60, 1, "pixel_values"),
            ("video", 61, 2, "pixel_values_videos"),
        ):
            if media in (kind, "both"):
                tokens.extend([62] + [token] * (frames * 4) + [63])
                # The collator stores media in BF16; use identical inputs in the reference.
                document[pixels_key] = torch.randn(frames * 4, 12, dtype=torch.bfloat16)
                document[f"{kind}_grid_thw"] = torch.tensor([[frames, 2, 2]])
        tokens.extend([5] * (document_index + 1))
        document.update(
            input_ids=torch.tensor(tokens),
            labels=torch.tensor(tokens),
            attention_mask=torch.full((len(tokens),), document_index + 1),
            position_ids=torch.arange(len(tokens)).expand(3, -1),
        )
        documents.append(document)

    packs = []
    offset = 0
    for count in pack_sizes:
        group = documents[offset : offset + count]
        for index, document in enumerate(group):
            document["attention_mask"].fill_(index + 1)
        packs.append(
            {
                key: torch.cat([document[key] for document in group], dim=-1 if key == "position_ids" else 0)
                for key in group[0]
            }
        )
        offset += count
    length = max(pack["input_ids"].numel() for pack in packs)
    # Uneven documents/rows exercise both token padding and -1 boundary padding.
    batch = neat_packed_vlm_collater(packs, packing=contract, max_length=length + 2)
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    documents = [{key: value.to(device) for key, value in document.items()} for document in documents]
    original_boundaries = batch["cu_seqlens"].clone()
    assert original_boundaries.ndim == 2
    if precomputed:
        for kind in ("image", "video"):
            if media in (kind, "both"):
                cu_seqlens, _ = get_vision_attention_seqlens(batch[f"{kind}_grid_thw"], model.config.vision_config)
                batch[f"{kind}_cu_seqlens"] = cu_seqlens
                batch[f"{kind}_max_seqlen"] = 4

    actual = model(**batch).logits[batch["_packed_seq_ids"] > 0]
    expected = torch.cat(
        [
            reference(
                **{
                    key: value.unsqueeze(1 if key == "position_ids" else 0)
                    if key in ("input_ids", "position_ids")
                    else value
                    for key, value in document.items()
                    if key not in ("labels", "attention_mask")
                }
            ).logits.squeeze(0)
            for document in documents
        ],
        dim=0,
    )
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (0.03, 3e-4)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(batch["cu_seqlens"], original_boundaries)

    upstream = torch.randn_like(actual) / actual.numel()
    actual.backward(upstream)
    expected.backward(upstream)
    for name, param in model.named_parameters():
        reference_grad = reference.get_parameter(name).grad
        if reference_grad is None:
            assert param.grad is None, name
        else:
            assert param.grad is not None, name
            torch.testing.assert_close(
                param.grad,
                reference_grad,
                rtol=1e-4 if dtype == torch.float32 else rtol,
                atol=2e-6 if dtype == torch.float32 else atol,
                msg=lambda message: f"{name}: {message}",
            )


@pytest.mark.parametrize("media", ["image", "video", "both", "text"])
@pytest.mark.parametrize("precomputed", [False, True])
def test_packed_media_matches_independent_documents(media: str, precomputed: bool) -> None:
    check_packed_media_parity(media, precomputed)
