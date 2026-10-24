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

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset

from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.components.datasets.loader import DataloaderConfig, NeatPackingConfig
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.models.common.packing import configure_packing_for_models
from nemo_automodel.components.models.kimi_linear.model import KimiLinear48BForCausalLM
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.llm.train_ft import _should_pack_validation
from tests.unit_tests.models.kimi_linear.test_model import _backend_config, _tiny_kimi_config


def _assert_recipe_document_isolation(
    *, device: str, use_kda: bool, training: bool, pack_size: int, split: str
) -> dict[str, float]:
    """Exercise the recipe's real loaders and model, returning isolation errors."""
    torch.manual_seed(0)
    recipe_path = Path(__file__).resolve().parents[4] / "examples/llm_finetune/kimi/kimi_linear_48b_a3b_hellaswag.yaml"
    cfg = RecipeConfig(load_yaml_config(recipe_path))
    model_config = _tiny_kimi_config(use_kda=use_kda)
    if use_kda:
        model_config.linear_attn_config["head_dim"] = 32
    model = KimiLinear48BForCausalLM(model_config, backend=_backend_config())
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model.initialize_weights(buffer_device=torch.device(device), dtype=dtype)
    model.to(device=device)
    model.train(training)
    packing_contract = configure_packing_for_models([model])

    assert isinstance(cfg.dataloader.packing, NeatPackingConfig)
    validation_config = cfg.validation_dataloaders["default"]
    assert isinstance(validation_config.packing, NeatPackingConfig)
    assert _should_pack_validation(cfg.dataloader, validation_config, model)
    loader_config = cfg.dataloader if split == "train" else validation_config
    loader_config = replace(
        loader_config,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        packing=replace(loader_config.packing, packed_sequence_size=pack_size),
    )
    documents = [[3, 4, 5, 6, 7], [11, 12, 13], [17, 18, 19, 20], [23, 24]]
    if pack_size == 4096:
        documents = [document * repeats for document, repeats in zip(documents, [420, 550, 500, 900])]
    dataset = Dataset.from_dict({"input_ids": documents, "labels": documents})
    with patch.object(DataloaderConfig, "_build_dataset", return_value=dataset):
        loader = loader_config.build(dp_rank=0, dp_world_size=1, packing_contract=packing_contract)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, pack_size)
    assert "qkv_format" not in batch
    assert {"_packed_seq_ids", "packed_token_indices", "cu_seqlens"} <= batch.keys()
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    sharder = ContextParallelSharder(model, None, batch)
    context, batch = sharder.shard(batch)
    labels = batch.pop("labels")
    first_document = batch["_packed_seq_ids"][0] == 1
    later_document = batch["_packed_seq_ids"][0] == 2
    assert first_document.any() and later_document.any()
    earlier_token_ids = batch["input_ids"][0, first_document].clone()
    changed_batch = dict(batch)
    changed_batch["input_ids"] = batch["input_ids"].clone()
    changed_batch["input_ids"][0, first_document] += 64

    with context():
        baseline = model(**batch).logits
        changed = model(**changed_batch).logits
    assert torch.isfinite(baseline).all() and torch.isfinite(changed).all()
    delta = (baseline[0, later_document] - changed[0, later_document]).abs().max().item()
    torch.testing.assert_close(baseline[0, later_document], changed[0, later_document], rtol=0, atol=0)
    assert (baseline[0, first_document] - changed[0, first_document]).abs().max() > 0

    # A standalone later document is an independent reference for boundary resets.
    standalone_ids = batch["input_ids"][0, later_document].unsqueeze(0)
    with context():
        standalone = model(input_ids=standalone_ids).logits
    tolerance = 2e-2 if device == "cuda" else 1e-5
    standalone_delta = (baseline[0, later_document] - standalone[0]).abs().max().item()
    torch.testing.assert_close(baseline[0, later_document], standalone[0], rtol=tolerance, atol=tolerance)

    supervised = later_document & (labels[0] != -100)
    assert supervised.any()
    loss = F.cross_entropy(baseline[0, supervised].float(), labels[0, supervised])
    loss.backward()
    embedding_grad = model.model.embed_tokens.weight.grad
    assert torch.isfinite(embedding_grad).all()
    leakage = embedding_grad[earlier_token_ids].abs().max().item()
    assert leakage == 0
    assert embedding_grad[standalone_ids.flatten()].abs().max() > 0

    # Removing boundary metadata must make this same perturbation observable.
    with torch.no_grad(), context():
        unmasked = model(input_ids=batch["input_ids"]).logits
        unmasked_changed = model(input_ids=changed_batch["input_ids"]).logits
    control_delta = (unmasked[0, later_document] - unmasked_changed[0, later_document]).abs().max().item()
    assert control_delta > tolerance
    return {
        "packed_logit_delta": delta,
        "standalone_logit_delta": standalone_delta,
        "earlier_embedding_grad": leakage,
        "unmasked_control_delta": control_delta,
    }


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("pack_size", [8, 10])
def test_recipe_neat_mla_document_isolation(split: str, training: bool, pack_size: int) -> None:
    """Later-document logits and gradients must be independent of earlier documents."""
    _assert_recipe_document_isolation(device="cpu", use_kda=False, training=training, pack_size=pack_size, split=split)
