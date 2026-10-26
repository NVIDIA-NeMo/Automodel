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

"""Local K3 processor, standard collation and unshifted packing inputs."""

import base64
import copy
import json
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from tokenizers import AddedToken
from transformers import AutoProcessor

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn, pad_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper
from nemo_automodel.components.datasets.vlm.fake_image import inject_fake_image_into_conversation
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.kimi_k3.processing import KimiK3ImageProcessor, KimiK3Processor
from nemo_automodel.components.models.kimi_k3.tokenization import TikTokenTokenizer


@pytest.fixture
def processor(tmp_path):
    vocab = tmp_path / "tiktoken.model"
    vocab.write_text("".join(f"{base64.b64encode(bytes([i])).decode()} {i}\n" for i in range(256)))
    names = [
        "[BOS]",
        "[EOS]",
        "[PAD]",
        "[UNK]",
        "<|media_pad|>",
        "<|media_begin|>",
        "<|media_content|>",
        "<|media_end|>",
        "<|im_end|>",
        "<|im_user|>",
        "<|im_assistant|>",
        "<|close|>",
    ]
    tokenizer = TikTokenTokenizer(
        str(vocab),
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        added_tokens_decoder={256 + i: AddedToken(name, special=True) for i, name in enumerate(names)},
    )
    return KimiK3Processor(KimiK3ImageProcessor(), tokenizer)


def _chat(image=None):
    content = [{"type": "text", "text": "Describe."}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    return [{"role": "user", "content": content}, {"role": "assistant", "content": "Red."}]


def test_local_processor_load_save_and_checkpoint_remote_metadata(processor, tmp_path):
    from nemo_automodel._transformers import registry

    assert "kimi_k3" in registry._CUSTOM_PROCESSOR_REGISTRATIONS
    processor.save_pretrained(tmp_path)
    # Published checkpoint metadata must not cause execution of checkpoint Python.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "kimi_k3"}))
    path = tmp_path / "processor_config.json"
    config = json.loads(path.read_text())
    config["auto_map"] = {"AutoProcessor": "missing_remote_code.KimiK3Processor"}
    path.write_text(json.dumps(config))
    loaded = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    assert type(loaded) is KimiK3Processor
    assert type(loaded.tokenizer) is TikTokenTokenizer
    assert type(loaded.image_processor) is KimiK3ImageProcessor
    chat = _chat(Image.new("RGB", (56, 28), "red"))
    expected = processor.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    actual = loaded.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_native_packing_and_default_collation_share_expanded_tokens(processor):
    examples = [{"conversation": _chat(Image.new("RGB", (56, 28), "red"))}, {"conversation": _chat()}]
    wrapper = PreTokenizedDatasetWrapper(examples, processor, inject_fake_images=False, max_retries=1)
    first, second = wrapper[0], wrapper[1]
    assert int((first["input_ids"] == processor.image_token_id).sum()) == 2
    assert first["image_grid_thw"].tolist() == [[1, 2, 4]]
    expected_labels = processor.tokenizer.encode("Red.", add_special_tokens=False)
    expected_labels.append(processor.tokenizer.convert_tokens_to_ids("<|close|>"))
    assert first["labels"][first["labels"] != -100].tolist() == expected_labels
    assert torch.all(first["labels"][first["input_ids"] == processor.image_token_id] == -100)
    batch = default_collate_fn(
        examples, processor, max_length=max(len(first["input_ids"]), len(second["input_ids"])) + 2
    )
    for index, sample in enumerate([first, second]):
        n = len(sample["input_ids"]) - 1
        torch.testing.assert_close(batch["input_ids"][index, :n], sample["input_ids"][:-1])
        torch.testing.assert_close(batch["labels"][index, :n], sample["labels"][1:])
    assert batch["n_images_per_sample"].tolist() == [1, 0]


def test_standard_images_text_interface_and_image_truncation(processor):
    image = Image.new("RGB", (56, 28), "red")
    text = processor.apply_chat_template(_chat(image), tokenize=False)
    output = processor(text=[text], images=[image], return_tensors="pt")
    assert int((output["input_ids"] == processor.image_token_id).sum()) == 2
    start = int((output["input_ids"][0] == processor.image_token_id).nonzero()[0])
    # Reject cuts inside the patches and immediately before the closing marker.
    for length in (start + 1, start + 2):
        with pytest.raises(ValueError, match="image token count"):
            processor(text=text, images=image, truncation=True, max_length=length)
    processor.tokenizer.truncation_side = "left"
    begin_id = processor.tokenizer.convert_tokens_to_ids("<|media_begin|>")
    begin = output["input_ids"][0].tolist().index(begin_id)
    with pytest.raises(ValueError, match="image token count"):
        processor(text=text, images=image, truncation=True, max_length=output["input_ids"].shape[1] - begin - 1)
    processor.tokenizer.truncation_side = "right"
    with pytest.raises(ValueError, match="placeholder count"):
        processor(text=text, images=[])
    with pytest.raises(ValueError, match="video"):
        processor(text=text, videos=[image])


def _fake_and_real_batch(processor, path: str, fake_color: str) -> dict[str, torch.Tensor]:
    """Collate an injected-fake text sample and a real-image sample through a native path.

    Args:
        processor: Local K3 processor.
        path: "pretokenized" (wrapper -> pad_collate_fn) or "default_collate".
        fake_color: Fill color of the injected fake image, to perturb its pixels.

    Returns:
        Batch with input_ids/attention_mask [2, sequence], pixel_values
        [patches, 3, 14, 14] and image_grid_thw [2, 3]; row 0 owns the fake image.
    """
    from nemo_automodel.components.datasets.vlm import fake_image

    real = _chat(Image.new("RGB", (56, 28), "red"))
    original = fake_image._FAKE_IMAGE
    fake_image._FAKE_IMAGE = Image.new("RGB", (56, 56), fake_color)
    try:
        if path == "pretokenized":
            # inject_fake_images keeps its default (True), as in native recipes.
            wrapper = PreTokenizedDatasetWrapper(
                [{"conversation": _chat()}, {"conversation": real}], processor, max_retries=1
            )
            return pad_collate_fn([wrapper[0], wrapper[1]], processor)
        examples = [
            {"conversation": inject_fake_image_into_conversation(_chat()), "_injected_fake": True},
            {"conversation": real},
        ]
        return default_collate_fn(examples, processor)
    finally:
        fake_image._FAKE_IMAGE = original


@pytest.mark.parametrize("path", ["pretokenized", "default_collate"])
def test_injected_fake_images_stay_masked_through_model_training(processor, path):
    """Default fake-image injection must train; the masked fake media must not leak into text."""
    from tests.unit_tests.models.kimi_k3.test_multimodal import _tiny_vlm

    torch.manual_seed(26)
    model = _tiny_vlm(
        vocab_size=272,
        media_placeholder_token_id=processor.image_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        patch_size=14,
    )

    def run(batch):
        """Return logits [2, sequence, vocab] for a collated fake/real batch."""
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
        ).logits

    batch = _fake_and_real_batch(processor, path, "white")
    image_tokens = batch["input_ids"] == processor.image_token_id
    assert batch["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 4]]
    assert image_tokens.sum(-1).tolist() == [4, 2]
    # The data contract under test: fake image tokens are masked, real ones are not.
    assert bool((batch["attention_mask"][0][image_tokens[0]] == 0).all())
    assert bool((batch["attention_mask"][1][image_tokens[1]] == 1).all())

    logits = run(batch)
    valid = batch["labels"] != -100
    assert valid[0].any() and valid[1].any()
    loss = torch.nn.functional.cross_entropy(logits[valid], batch["labels"][valid])
    loss.backward()
    assert torch.isfinite(loss)
    vision_grads = [p.grad for name, p in model.named_parameters() if name.startswith("vision_tower.")]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in vision_grads)
    assert any(grad.abs().sum() > 0 for grad in vision_grads)

    with torch.no_grad():
        other_fake = _fake_and_real_batch(processor, path, "black")
        torch.testing.assert_close(other_fake["input_ids"], batch["input_ids"], rtol=0, atol=0)
        assert not torch.equal(other_fake["pixel_values"], batch["pixel_values"])
        perturbed = run(other_fake)
        text = batch["attention_mask"][0] > 0
        # Masked fake media cannot influence any unmasked token of its sample.
        torch.testing.assert_close(perturbed[0][text], logits[0][text].detach(), rtol=0, atol=0)
        torch.testing.assert_close(perturbed[1], logits[1].detach(), rtol=0, atol=0)
        # Control: the same comparison detects a change to the real image.
        real_changed = dict(batch)
        fake_patches = int(batch["image_grid_thw"][0].prod())
        real_changed["pixel_values"] = batch["pixel_values"].clone()
        real_changed["pixel_values"][fake_patches:] = -batch["pixel_values"][fake_patches:]
        assert not torch.allclose(run(real_changed)[1], logits[1].detach())


def _free_port() -> int:
    """Return an unused local TCP port for the process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake_image_cp_rank(rank: int, port: int, batch: dict[str, torch.Tensor], image_token_id: int, pad_id: int) -> None:
    """Compare CP2 against the native unsharded path on a fake-image batch.

    Args:
        rank: CP rank.
        port: Rendezvous port.
        batch: Native fake/real batch; input_ids and attention_mask have shape
            [2, sequence], and row 0 holds a masked fake-image span.
        image_token_id: Media placeholder token ID.
        pad_id: Padding token ID.
    """
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.kimi_k3.multimodal import KimiK3ForConditionalGeneration
    from tests.unit_tests.models.kimi_k3.test_multimodal import _tiny_vlm

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
        config = copy.deepcopy(
            _tiny_vlm(
                vocab_size=272, media_placeholder_token_id=image_token_id, pad_token_id=pad_id, patch_size=14
            ).config
        )
        # One KDA layer followed by one MLA layer; CP flex attention needs head dims of at least 16.
        config.text_config.num_hidden_layers = 2
        config.text_config.qk_nope_head_dim = 8
        config.text_config.qk_rope_head_dim = 8
        config.text_config.v_head_dim = 16
        config.text_config.linear_attn_config = {
            "head_dim": 64,
            "num_heads": 2,
            "short_conv_kernel_size": 4,
            "kda_layers": [1],
            "full_attn_layers": [2],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        }
        config.text_config.kda_use_fused_gate = False
        backend = BackendConfig(
            attn="eager", linear="torch", experts="torch", dispatcher="torch", enable_hf_state_dict_adapter=False
        )
        torch.manual_seed(4511)
        model = KimiK3ForConditionalGeneration(config, backend=backend).cuda()
        model.initialize_weights(torch.device("cuda"), dtype=torch.float32)
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        valid = batch["attention_mask"] > 0
        kda_outputs = []
        hook = model.model.layers["0"].self_attn.register_forward_hook(
            lambda module, inputs, output: kda_outputs.append(output.detach())
        )

        with torch.no_grad():
            cp1_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
            ).logits
            model.cp_mesh = mesh["cp"]
            sharder = ContextParallelSharder(model, mesh, batch, padding_token_id=pad_id, invoke_pre_embed=True)
            context, local = sharder.shard(dict(batch))
            local.pop("labels")
            order = local["kimi_vlm_token_order"]
            with context():
                cp2_logits = model(**local).logits
        hook.remove()

        def source_order(local_tensor: torch.Tensor) -> torch.Tensor:
            """Gather [batch, local, features] shards and restore source order [batch, sequence, features]."""
            parts = [torch.empty_like(local_tensor) for _ in range(2)]
            dist.all_gather(parts, local_tensor.contiguous())
            sharded = torch.cat(parts, dim=1)
            restored = torch.empty_like(sharded).scatter_(1, order.unsqueeze(-1).expand_as(sharded), sharded)
            return restored[:, : valid.shape[1]]

        # Remaining CP differences match a batch row without masked spans (KDA output
        # ~3e-5, logits ~1e-3); an interrupted KDA state shifts them by ~1e-2 and ~1.
        torch.testing.assert_close(source_order(kda_outputs[1])[valid], kda_outputs[0][valid], atol=2e-4, rtol=0)
        torch.testing.assert_close(source_order(cp2_logits)[valid], cp1_logits[valid], atol=1e-2, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="K3 KDA context parallelism needs 2 GPUs")
@pytest.mark.parametrize("path", ["pretokenized", "default_collate"])
def test_injected_fake_image_kda_cp2_matches_unsharded(processor, path):
    """CP must keep KDA state continuous across a masked fake-image span, as the unsharded path does."""
    pytest.importorskip("fla")
    batch = _fake_and_real_batch(processor, path, "white")
    mp.spawn(
        _fake_image_cp_rank,
        args=(_free_port(), batch, processor.image_token_id, processor.tokenizer.pad_token_id),
        nprocs=2,
        join=True,
    )


def test_tool_call_only_assistant_message_matches_native_tokenizer(processor):
    chat = [
        {"role": "user", "content": "Inspect."},
        {
            "role": "assistant",
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "inspect", "arguments": {}}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
        {"role": "assistant", "content": "Done."},
    ]
    expected = processor.tokenizer.apply_chat_template(chat, tokenize=True)
    output = processor.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    assert output["input_ids"][0].tolist() == expected
    assert "pixel_values" not in output or output["pixel_values"].numel() == 0


@pytest.mark.parametrize("side", ["left", "right"])
def test_chat_truncation_honors_truncation_side(processor, side):
    chat = _chat()
    full = processor.tokenizer.apply_chat_template(
        chat, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    ids, mask = list(full["input_ids"]), list(full["assistant_masks"])
    processor.tokenizer.truncation_side = side
    try:
        for limit in (0, 10, len(ids), len(ids) + 3):
            cut = processor.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                truncation=True,
                max_length=limit,
            )
            kept = slice(max(0, len(ids) - limit), None) if side == "left" else slice(0, limit)
            assert list(cut["input_ids"]) == ids[kept]
            assert list(cut["assistant_masks"]) == mask[kept]
        untruncated = processor.tokenizer.apply_chat_template(chat, tokenize=True, truncation=False, max_length=10)
        assert untruncated == ids
        processed = processor.apply_chat_template(chat, tokenize=True, truncation=True, max_length=10)
        assert processed == (ids[-10:] if side == "left" else ids[:10])
    finally:
        processor.tokenizer.truncation_side = "right"


def test_left_chat_truncation_keeps_or_rejects_whole_image_spans(processor):
    image = Image.new("RGB", (56, 28), "red")
    chat = _chat(image)
    full = processor.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
    ids = full["input_ids"][0].tolist()
    begin = ids.index(processor.tokenizer.convert_tokens_to_ids("<|media_begin|>"))
    processor.tokenizer.truncation_side = "left"
    try:
        # Cutting through <|media_begin|> drops part of the image span.
        with pytest.raises(ValueError, match="image token count"):
            processor.apply_chat_template(chat, tokenize=True, truncation=True, max_length=len(ids) - begin - 1)
        # A cut confined to the text before the span keeps the whole image.
        kept = processor.apply_chat_template(
            chat, tokenize=True, return_dict=True, return_tensors="pt", truncation=True, max_length=len(ids) - begin
        )
        assert kept["input_ids"][0].tolist() == ids[begin:]
        torch.testing.assert_close(kept["pixel_values"], full["pixel_values"], rtol=0, atol=0)
        assert kept["image_grid_thw"].tolist() == full["image_grid_thw"].tolist()
    finally:
        processor.tokenizer.truncation_side = "right"


@pytest.mark.parametrize("stage", ["before_resize", "after_resize"])
def test_image_normalization_padding_and_alpha_background(stage):
    processor = KimiK3ImageProcessor(
        media_proc_cfg={
            "transparent_bg_config": {"pattern": "white"},
            "transparent_bg_fill_stage": stage,
        }
    )
    output = processor(Image.new("RGBA", (1, 1), (255, 0, 0, 0)), return_tensors="pt")
    assert output["image_grid_thw"].tolist() == [[1, 2, 2]]
    patches = output["pixel_values"]
    assert patches.shape == (4, 3, 14, 14)
    torch.testing.assert_close(patches[0, :, 0, 0], torch.ones(3))
    assert torch.count_nonzero(patches == 1) == 3
    assert torch.count_nonzero(patches == -1) == patches.numel() - 3
