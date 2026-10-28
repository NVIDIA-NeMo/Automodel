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

"""Offline, real-CUDA generation coverage with a randomly initialized tiny Wan."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def tiny_wan_path(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

    torch.manual_seed(123)
    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "a": 3, "cat": 4}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    text_encoder = UMT5EncoderModel(
        UMT5Config(vocab_size=8, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4, dropout_rate=0)
    )
    transformer = diffusers.WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=16,
        ffn_dim=64,
        num_layers=1,
    )
    vae = diffusers.AutoencoderKLWan(
        base_dim=4,
        z_dim=4,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )
    pipe = diffusers.WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=diffusers.UniPCMultistepScheduler(flow_shift=3.0),
    )
    model_path = tmp_path / "tiny-wan"
    pipe.save_pretrained(model_path)
    return model_path


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_wan_load_and_generate(tiny_wan_path, tmp_path, monkeypatch, cpu_offload):
    """Exercise real loading, text encoding, CUDA denoising and VAE decoding."""
    import examples.diffusion.generate.generate as gen

    cfg = SimpleNamespace(
        model=SimpleNamespace(pretrained_model_name_or_path=str(tiny_wan_path)),
        inference=SimpleNamespace(
            dtype="bfloat16",
            prompts=["a cat"],
            num_inference_steps=2,
            guidance_scale=1.0,
            height=16,
            width=16,
            pipeline_kwargs=SimpleNamespace(to_dict=lambda: {"num_frames": 5, "max_sequence_length": 8}),
        ),
        vae=SimpleNamespace(enable_cpu_offload=cpu_offload),
        output=SimpleNamespace(output_dir=str(tmp_path / "output")),
        seed=7,
    )
    pipe = gen.load_pipeline(cfg, None)
    expected_device = "cpu" if cpu_offload else "cuda"
    for name in ("text_encoder", "transformer", "vae"):
        assert next(getattr(pipe, name).parameters()).device.type == expected_device, name

    gen.apply_optimizations(pipe, cfg)
    assert pipe._execution_device.type == "cuda"

    # Only file encoding is replaced; all pipeline modules execute real forwards.
    videos = []
    monkeypatch.setattr(
        "diffusers.utils.export_to_video", lambda frames, *args, **kwargs: videos.append(np.asarray(frames))
    )
    gen.run_inference(pipe, cfg, is_rank0=True)
    assert len(videos) == 1
    assert videos[0].shape == (5, 16, 16, 3)
    assert np.isfinite(videos[0]).all()
    if cpu_offload:
        pipe.remove_all_hooks()
