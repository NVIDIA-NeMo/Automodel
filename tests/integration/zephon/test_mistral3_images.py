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

"""Real offline Pixtral encoding, Zephon packing and native Mistral3 TP/PP."""

import json
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import yaml
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoConfig, Mistral3ForConditionalGeneration

from nemo_automodel._transformers.registry import resolve_image_position_state
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.vlm.datasets import encode_vlm_document
from nemo_automodel.components.datasets.vlm.pp_media import (
    VLM_PP_MEDIA_KEY,
    stage_vlm_media_for_pp,
)
from nemo_automodel.components.datasets.zephon.config import (
    ZephonCheckpointOptions,
    ZephonOptions,
    ZephonPackingOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonSourceOptions,
    ZephonTrainingOptions,
    ZephonWorkerOptions,
)
from nemo_automodel.components.datasets.zephon.vlm_config import (
    ZephonVlmDataloaderConfig,
    ZephonVlmSourceConfig,
)
from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.models.common.packing import (
    PackingCapabilities,
    configure_packing,
)
from nemo_automodel.recipes._typed_config import RecipeConfig
from tests.utils.mistral3_vlm_fixtures import (
    build_model_config,
    build_processor,
    prepare,
)
from tests.utils.zephon_distributed import (
    distributed_environment as distributed_environment,
)
from tests.utils.zephon_distributed import run_ranks
from tests.utils.zephon_positions import assert_joined_position_ranges


@pytest.fixture
def source(tmp_path):
    prepare(tmp_path, count=12)
    return tmp_path


def _model(backend="sdpa"):
    config = build_model_config()
    config._attn_implementation = backend
    return Mistral3ForConditionalGeneration(config).float().train()


def _loader(
    source,
    model,
    *,
    pp=False,
    xdoc=False,
    max_documents=2,
    fake=False,
    distributed=False,
    packing_contract=PackingCapabilities("block_causal"),
):
    config = ZephonVlmDataloaderConfig(
        dataset_config=ZephonVlmSourceConfig(inject_fake_images=fake),
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path=str(Path(source) / "corpus"), fmt="jsonl"),),
            packing=ZephonPackingOptions(
                seq_len=63,
                pack_algorithm="first_fit",
                pack_num_bins=1,
                max_num_documents=max_documents,
                max_num_documents_scope="bin",
                shuffle_after_pack=False,
            ),
            source=ZephonSourceOptions(
                chunk_size=4, shuffle_shards=False, shuffle_within_shard=False, shuffle_block_size=None
            ),
            workers=ZephonWorkerOptions(tokenize_parallelism=1),
            runtime=ZephonRuntimeOptions(runner="inline", mtp_mode=False),
            training=ZephonTrainingOptions(allow_cross_document_attention=xdoc),
            checkpoint=ZephonCheckpointOptions(
                aggregate_dir=str(Path(source) / "aggregate") if distributed else None,
                run_id=f"mistral3-{pp}-{max_documents}-{xdoc}-{fake}" if distributed else None,
            ),
        ),
    )
    return config.build(
        pretrained_model_name_or_path=str(Path(source) / "processor"),
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        position_state_factory=lambda: resolve_image_position_state(model),
        pp_n_microbatches=2 if pp else None,
        packing_contract=packing_contract,
    ).dataloader


def _loss(logits, target):
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum")


def _forward(model, batch):
    configure_packing(model.config._attn_implementation, model=model)
    return model(**{key: value for key, value in batch.items() if key != "labels"}).logits


def test_training_recipe_uses_local_processor_and_preserves_expanded_images(source):
    prepare(source)
    values = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "examples/vlm_finetune/zephon/train.yaml").read_text()
    )
    values["model"]["pretrained_model_name_or_path"] = str(source / "model")
    values["processor"] = {
        "_target_": "transformers.AutoProcessor.from_pretrained",
        "pretrained_model_name_or_path": str(source / "processor"),
    }
    settings = values["dataloader"]["zephon"]
    settings["sources"] = [{"name": "images", "path": str(source / "corpus")}]
    settings["packing"]["seq_len"] = 63
    settings.setdefault("runtime", {}).update(runner="inline", mtp_mode=False)
    settings["checkpoint"].update(aggregate_store=None, run_id=None)
    resolved = RecipeConfig(ConfigNode(values)).vlm_dataloader
    config = AutoConfig.from_pretrained(source / "model", local_files_only=True)
    config._attn_implementation = values["model"]["attn_implementation"]
    model = Mistral3ForConditionalGeneration(config).float().train()
    with closing(
        resolved.build(
            pretrained_model_name_or_path=str(source / "model"),
            dp_rank=0,
            dp_world_size=1,
            batch_size=2,
            position_state_factory=lambda: resolve_image_position_state(model),
        ).dataloader
    ) as loader:
        batch = next(loader)
        assert batch["pixel_values"].shape[-2:] == (8, 12)
        assert batch["image_sizes"].shape[1] == 2
        assert batch["input_ids"].eq(10).sum() == (batch["image_sizes"] // 4).prod(-1).sum()
        assert batch["position_ids"].shape == batch["input_ids"].shape == (2, 63)
        loss = _loss(_forward(model, batch), batch["labels"])
        loss.backward()
        gradient = model.model.vision_tower.patch_conv.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_native_processor_fields_and_packed_attention_policy(source, backend):
    processor = build_processor()
    document = json.loads((source / "corpus/images.jsonl").read_text().splitlines()[2])
    native = processor.apply_chat_template(
        document["conversation"],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
    )
    encoded = encode_vlm_document(document, processor, inject_fake_images=False)
    torch.testing.assert_close(encoded["image_sizes"], native["image_sizes"])
    torch.testing.assert_close(encoded["pixel_values"], native["pixel_values"])
    torch.testing.assert_close(encoded["input_ids"], native["input_ids"][0])
    model = _model(backend)
    off_loader, on_loader = _loader(source, model), _loader(source, model, xdoc=True)
    try:
        off, on = next(off_loader), next(on_loader)
        assert_joined_position_ranges(off["position_ids"], on["position_ids"], off["_packed_seq_ids"])
        for key in off.keys() - {"_packed_seq_ids", "position_ids"}:
            torch.testing.assert_close(off[key], on[key], atol=0, rtol=0)
        later = off["_packed_seq_ids"].gt(1)
        assert later.any()
        baseline_off, baseline_on = _forward(model, off), _forward(model, on)
        altered = off["input_ids"].clone()
        altered[off["_packed_seq_ids"].eq(1) & altered.eq(5)] = 9
        changed_off = {**off, "input_ids": altered}
        changed_on = {**on, "input_ids": altered}
        torch.testing.assert_close(_forward(model, changed_off)[later], baseline_off[later], atol=0, rtol=0)
        assert not torch.allclose(_forward(model, changed_on)[later], baseline_on[later])
        _loss(baseline_off, off["labels"]).backward()
        assert torch.isfinite(model.model.vision_tower.patch_conv.weight.grad).all()
    finally:
        off_loader.close()
        on_loader.close()


def _pipeline_worker(rank, directory, source):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        torch.manual_seed(134)
        reference = _model()
        pp = AutoPipeline(
            world_mesh=DeviceMesh("cpu", [0, 1], mesh_dim_names=("pp",)),
            dp_axis_names=(),
            pp_schedule="gpipe",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).build(reference, loss_fn=_loss)
        for part in pp.parts:
            configure_packing("sdpa", model=part)
        # One document per pack exercises an empty first media chunk; two permit
        # document isolation and xdoc policy in an otherwise identical schedule.
        for max_documents, xdoc, fake in (
            (1, False, False),
            (2, False, False),
            (2, True, False),
            (1, False, True),
            (1, True, True),
        ):
            ordinary = _loader(source, reference, max_documents=max_documents, xdoc=xdoc, fake=fake, distributed=True)
            pipelined = _loader(
                source, reference, max_documents=max_documents, xdoc=xdoc, fake=fake, distributed=True, pp=True
            )
            try:
                for batch_index in range(2):
                    expected_batch, actual_batch = next(ordinary), next(pipelined)
                    if max_documents == 1 and batch_index == 0:
                        assert [chunk.shape[0] for chunk in actual_batch[VLM_PP_MEDIA_KEY]["pixel_values"]] == (
                            [1, 1] if fake else [0, 1]
                        )
                    for key in ("input_ids", "labels", "position_ids", "_packed_seq_ids"):
                        torch.testing.assert_close(expected_batch[key], actual_batch[key], atol=0, rtol=0)
                    assert ("_packed_seq_ids" in expected_batch) == ("_packed_seq_ids" in actual_batch)
                    if "_packed_seq_ids" in expected_batch:
                        torch.testing.assert_close(
                            expected_batch["_packed_seq_ids"],
                            actual_batch["_packed_seq_ids"],
                            atol=0,
                            rtol=0,
                        )
                    reference.zero_grad()
                    for part in pp.parts:
                        part.zero_grad()
                    expected = _forward(reference, expected_batch)
                    expected_loss = _loss(expected, expected_batch["labels"])
                    expected_loss.backward()
                    losses = []
                    ids, targets = actual_batch.pop("input_ids"), actual_batch.pop("labels")
                    with stage_vlm_media_for_pp(pp, pp.parts, actual_batch):
                        actual = pp.step(
                            ids,
                            target=targets if rank == 1 else None,
                            losses=losses if rank == 1 else None,
                            **actual_batch,
                        )
                    if rank == 1:
                        eligible = actual_batch["_packed_seq_ids"] > 0
                        assert targets[~eligible].eq(-100).all()
                        # Per-microbatch causal fast paths may differ only at ignored
                        # padding queries; compare every real token and all gradients.
                        torch.testing.assert_close(actual[eligible], expected[eligible], atol=1e-6, rtol=1e-5)
                        torch.testing.assert_close(sum(losses), expected_loss)
                    for part in pp.parts:
                        for name, parameter in part.named_parameters():
                            torch.testing.assert_close(
                                parameter.grad, reference.get_parameter(name).grad, atol=1e-5, rtol=1e-4, msg=name
                            )
                        assert getattr(part, "_vlm_pixel_values_chunks", None) is None
            finally:
                ordinary.close()
                pipelined.close()
    finally:
        dist.destroy_process_group()


def test_real_loader_mistral3_pipeline_forward_backward(source, tmp_path, distributed_environment):
    run_ranks(_pipeline_worker, (str(tmp_path), str(source)))


def _tensor_worker(rank, directory, source):
    from copy import deepcopy

    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.parallel import parallelize_module

    from nemo_automodel.components.distributed.parallelizer import (
        _get_parallel_plan,
        _update_attention_head_counts_for_tp,
    )

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    loader = None
    try:
        torch.manual_seed(135)
        reference = _model()
        model = deepcopy(reference)
        mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("tp",))
        parallelize_module(model, mesh, _get_parallel_plan(model, tp_size=2))
        _update_attention_head_counts_for_tp(model, 2)
        loader = _loader(source, reference, distributed=True)
        batch = next(loader)
        expected, actual = _forward(reference, batch), _forward(model, batch)
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        _loss(expected, batch["labels"]).backward()
        _loss(actual, batch["labels"]).backward()
        for name, parameter in model.named_parameters():
            gradient = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
            torch.testing.assert_close(gradient, reference.get_parameter(name).grad, atol=1e-5, rtol=1e-4, msg=name)
    finally:
        if loader is not None:
            loader.close()
        dist.destroy_process_group()


def test_real_loader_mistral3_tensor_parallel_forward_backward(source, tmp_path, distributed_environment):
    run_ranks(_tensor_worker, (str(tmp_path), str(source)))
