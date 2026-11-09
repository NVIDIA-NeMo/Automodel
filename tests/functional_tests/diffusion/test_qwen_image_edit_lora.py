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

"""Exercise the QIE LoRA recipe on one GPU without downloading model weights.

Failure conditions, specified before adding the recipe: a missing recipe;
unmatched adapters or frozen image/text LoRA branches; changing base weights;
non-finite gradients/loss; missing adapter keys or changed reload predictions;
lost optimizer, LR, data, RNG or step state after a fresh-process resume;
repeated, skipped or changed training inputs across the resume boundary.

The fixture replaces only pretrained weights and cached inputs with tiny local
equivalents. Training, checkpointing and adapter loading use production code.
Four subprocesses isolate CUDA/distributed/RNG state. Their logs, checkpoints,
state snapshots and report.json remain in pytest's temporary output directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils._pytree import tree_map

from nemo_automodel.shared.import_utils import safe_import

_HAS_DIFFUSERS, _ = safe_import("diffusers")

pytestmark = [
    pytest.mark.skipif(not _HAS_DIFFUSERS, reason="diffusers is required for the diffusion recipe"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the recipe requires CUDA"),
]


def _cpu_state(value: object) -> object:
    """Copy a nested state to independent CPU storage.

    Args:
        value: State tree containing tensors of arbitrary shape; parameter and
            optimizer tensors keep their parameter axes, and RNG states are bytes.

    Returns:
        The same tree and tensor layouts, with detached CPU copies.
    """
    return tree_map(
        lambda item: item.detach().cpu().clone() if isinstance(item, torch.Tensor) else item,
        value,
    )


def _assert_equal(actual: object, expected: object) -> None:
    """Check exact nested training-state equality.

    Args:
        actual: State tree containing tensors of arbitrary shape and scalar leaves.
        expected: Reference tree with identical structure and tensor axis layouts.
    """
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for observed, reference in zip(actual, expected, strict=True):
            _assert_equal(observed, reference)
    else:
        assert actual == expected


def _prepare_fixture(output: Path) -> None:
    """Save a two-block Qwen model, three cache samples and one fixed probe."""
    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(123)
    model_dir = output / "base"
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    )
    model.save_pretrained(model_dir / "transformer")
    # The real modular loader can load a denoiser alone; cached training needs
    # neither a VAE nor a text encoder. No loader is mocked or replaced.
    (model_dir / "modular_model_index.json").write_text(
        json.dumps({"transformer": ["diffusers", "QwenImageTransformer2DModel"]})
    )
    cache = output / "cache"
    cache.mkdir()
    signature = {"target": [4, 4, 4], "contexts": [[4, 4, 4]]}
    records = []
    for index in range(3):
        filename = f"sample_{index}.pt"
        torch.save(
            {
                "target_latent": torch.randn(4, 4, 4),
                "context_latents": [torch.randn(4, 4, 4)],
                "prompt_embeddings": torch.randn(5, 12),
                "prompt_attention_mask": torch.tensor([True, True, True, True, False]),
                "conditioning_tensors": {},
                "metadata": {
                    "original_ids": {"row_index": index},
                    "target_spatial_shape": [4, 4],
                    "context_spatial_shapes": [[4, 4]],
                    "compound_bucket_signature": signature,
                    "conditioning_shapes": {},
                },
            },
            cache / filename,
        )
        records.append({"cache_file": filename, "compound_bucket_signature": signature})
    (cache / "metadata_shard_0000.json").write_text(json.dumps(records))
    (cache / "metadata.json").write_text(json.dumps({"shards": ["metadata_shard_0000.json"]}))
    # Two images, four packed tokens each; five text tokens with one masked.
    torch.save(
        {
            "hidden_states": torch.randn(1, 8, 16, dtype=torch.bfloat16),
            "encoder_hidden_states": torch.randn(1, 5, 12, dtype=torch.bfloat16),
            "encoder_hidden_states_mask": torch.tensor([[True, True, True, True, False]]),
            "timestep": torch.tensor([0.25], dtype=torch.bfloat16),
            "img_shapes": [[(1, 2, 2), (1, 2, 2)]],
            "return_dict": False,
        },
        output / "probe.pt",
    )


def _prediction(model: torch.nn.Module, output: Path) -> torch.Tensor:
    """Evaluate the common packed image/text probe on a CUDA transformer.

    Args:
        model: Qwen transformer mapping image [batch, image_tokens, channels]
            and text [batch, text_tokens, hidden] streams to packed velocity.
        output: Directory containing the fixed probe and its image shape metadata.

    Returns:
        CPU velocity tensor of shape [1, 8, 16], with packed image channels last.
    """
    probe = torch.load(output / "probe.pt", map_location="cpu", weights_only=True)
    probe = tree_map(lambda item: item.to("cuda") if isinstance(item, torch.Tensor) else item, probe)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**probe)[0].detach().cpu()


def _run_training(output: Path, phase: str) -> None:
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

    phase_dir = output / phase
    phase_dir.mkdir()
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    cfg["checkpoint"]["checkpoint_dir"] = str(phase_dir / "checkpoints")
    if phase == "resumed":
        previous = json.loads((output / "interrupted" / "report.json").read_text())
        cfg["checkpoint"]["restore_from"] = previous["checkpoint"]
    recipe = TrainDiffusionRecipe(ConfigNode(cfg))
    recipe.setup()
    assert torch.distributed.get_world_size() == 1
    assert recipe.device_mesh is None, "single-rank DDP must not shard the model"
    parameters = dict(recipe.model.named_parameters())
    adapters = {name: parameter for name, parameter in parameters.items() if "lora_" in name}
    base = {name: parameter for name, parameter in parameters.items() if "lora_" not in name}
    assert adapters and all(parameter.requires_grad for parameter in adapters.values())
    assert all(not parameter.requires_grad for parameter in base.values())
    assert all(parameter.is_cuda for parameter in parameters.values())
    before_base = _cpu_state(base)
    before_adapter = _cpu_state(adapters)

    def snapshot() -> dict[str, object]:
        return {
            "adapter": _cpu_state(adapters),
            "optimizer": _cpu_state(recipe.optimizer[0].state_dict()),
            "lr_scheduler": _cpu_state(recipe.lr_scheduler[0].state_dict()),
            "rng": _cpu_state(recipe.rng.state_dict()),
            "dataloader": _cpu_state(recipe.dataloader.state_dict()),
            # state_dict() adds one at checkpoint time; after the loop this
            # counter already denotes the next update, so read it directly.
            "step": int(recipe.step_scheduler.step),
            "epoch": int(recipe.step_scheduler.epoch),
        }

    torch.save(snapshot(), phase_dir / "start.pt")
    seen_gradients = set()
    observed_steps = []
    observed_inputs = []
    original_step = recipe.flow_matching_pipeline.step

    def observe_input(
        model: torch.nn.Module, batch: dict[str, object], **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, object]]:
        """Record each consumed input without changing data or random state.

        Args:
            model: Qwen transformer used by the production flow-matching step.
            batch: Image and context latents [1, 4, 4, 4], text [1, 5, 12],
                attention mask [1, 5], and cache provenance metadata.
            **kwargs: Unchanged production device, dtype, step and logging options.

        Returns:
            Production loss [1, 4, 4, 4], scalar loss, optional mask and metrics.
        """
        observed_inputs.append({"step": int(recipe.step_scheduler.step), "batch": _cpu_state(batch)})
        return original_step(model=model, batch=batch, **kwargs)

    recipe.flow_matching_pipeline.step = observe_input

    def observe_update(
        optimizer: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        """Inspect gradients without changing optimizer or RNG state.

        Args:
            optimizer: AdamW holding LoRA matrices [out, rank] and [rank, in].
            args: Unused optimizer call arguments; the recipe supplies no tensors.
            kwargs: Unused optimizer call keywords; the recipe supplies no tensors.
        """
        for name, parameter in parameters.items():
            if name in base:
                assert parameter.grad is None, name
            elif parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                if parameter.grad.count_nonzero():
                    seen_gradients.add(name)
        observed_steps.append(int(recipe.step_scheduler.step))

    handle = recipe.optimizer[0].register_step_post_hook(observe_update)
    original_save = recipe.save_checkpoint
    saved_checkpoints = []

    def save_and_stop(epoch: int, step: int, train_loss: float) -> None:
        original_save(epoch, step, train_loss)
        assert math.isfinite(train_loss)
        checkpoint = phase_dir / "checkpoints" / f"epoch_{epoch}_step_{step}"
        for relative_path in (
            "model/adapter_model.safetensors",
            "model/adapter_config.json",
            "optim/.metadata",
            "rng/rng_global_rank_0.pt",
            "dataloader/dataloader_dp_rank_0.pt",
            "sampler.pt",
            "step_scheduler.pt",
        ):
            assert (checkpoint / relative_path).is_file(), relative_path
        assert torch.load(checkpoint / "step_scheduler.pt", weights_only=True)["step"] == step + 1
        saved_checkpoints.append(str(checkpoint))
        if phase == "interrupted" and step + 1 == 2:
            recipe.step_scheduler.sigterm_flag = True

    recipe.save_checkpoint = save_and_stop
    try:
        recipe.run_train_validation_loop()
    finally:
        handle.remove()
        recipe.flow_matching_pipeline.step = original_step
    _assert_equal(_cpu_state(base), before_base)
    required = (
        "transformer_blocks.0.attn.to_q.lora_B.weight",
        "transformer_blocks.0.attn.add_q_proj.lora_B.weight",
    )
    for name in required:
        assert name in seen_gradients, f"missing nonzero image/text gradient: {name}"
        assert not torch.equal(before_adapter[name], adapters[name].detach().cpu()), name
    expected_steps = [2, 3] if phase == "resumed" else list(range(2 if phase == "interrupted" else 4))
    assert observed_steps == expected_steps
    torch.save(snapshot(), phase_dir / "final.pt")
    torch.save(observed_inputs, phase_dir / "inputs.pt")
    torch.save(_prediction(recipe.model, output), phase_dir / "prediction.pt")
    (phase_dir / "report.json").write_text(
        json.dumps(
            {
                "steps": observed_steps,
                "input_sample_ids": [
                    [sample["original_ids"] for sample in entry["batch"]["metadata"]["samples"]]
                    for entry in observed_inputs
                ],
                "adapter_tensors": len(adapters),
                "base_tensors": len(base),
                "nonzero_gradient_tensors": len(seen_gradients),
                "checkpoint": saved_checkpoints[-1],
                "checkpoint_steps": [
                    torch.load(Path(path) / "step_scheduler.pt", weights_only=True)["step"]
                    for path in saved_checkpoints
                ],
            },
            indent=2,
        )
    )


def _run_reload(output: Path) -> None:
    from diffusers import QwenImageTransformer2DModel
    from safetensors.torch import load_file

    from examples.diffusion.generate.generate import load_lora_weights_into_pipeline
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        NeMoAutoDiffusionPipeline,
    )
    from nemo_automodel.components.config.loader import ConfigNode

    report = json.loads((output / "continuous" / "report.json").read_text())
    adapter_dir = Path(report["checkpoint"]) / "model"
    transformer = QwenImageTransformer2DModel.from_pretrained(
        output / "base", subfolder="transformer", torch_dtype=torch.bfloat16
    ).cuda()
    transformer.set_attention_backend("native")
    base_prediction = _prediction(transformer, output)
    pipe = NeMoAutoDiffusionPipeline(transformer=transformer)
    load_lora_weights_into_pipeline(pipe, ConfigNode({"model": {"lora_weights": str(adapter_dir)}}))
    expected = torch.load(output / "continuous" / "final.pt", weights_only=True)["adapter"]
    saved = {
        name.removeprefix("base_model.model."): tensor
        for name, tensor in load_file(adapter_dir / "adapter_model.safetensors").items()
    }
    _assert_equal(saved, expected)
    actual = {name: tensor.detach().cpu() for name, tensor in transformer.named_parameters() if "lora_" in name}
    _assert_equal(actual, expected)
    prediction = _prediction(transformer, output)
    _assert_equal(
        prediction,
        torch.load(output / "continuous" / "prediction.pt", weights_only=True),
    )
    assert not torch.equal(base_prediction, prediction), "adapter has no effect on the fixed prediction"
    (output / "reload.json").write_text(json.dumps({"adapter_tensors": len(saved), "prediction_max_abs_error": 0.0}))


def test_qwen_image_edit_lora_recipe_train_reload_resume(tmp_path: Path) -> None:
    """Verify the shipped single-GPU recipe, including an independent resume."""
    repo = Path(__file__).resolve().parents[3]
    recipe_path = repo / "examples/diffusion/finetune/qwen_image_edit_2511_flow_lora.yaml"
    assert recipe_path.is_file(), f"QIE2511 LoRA recipe is missing: {recipe_path}"
    cfg = yaml.safe_load(recipe_path.read_text())
    assert cfg["model"]["model_type"] == "qwen_image_edit" and cfg["peft"]
    assert "ddp" in cfg and "fsdp" not in cfg
    assert cfg["step_scheduler"]["local_batch_size"] == cfg["step_scheduler"]["global_batch_size"] == 1
    _prepare_fixture(tmp_path)
    cfg["model"]["pretrained_model_name_or_path"] = str(tmp_path / "base")
    cfg["optimizer"]["lr"] = 1e-2
    cfg["lr_scheduler"] = {
        "lr_decay_style": "cosine",
        "lr_warmup_steps": 0,
        "min_lr": 1e-3,
    }
    cfg["step_scheduler"].update(
        num_epochs=None,
        max_steps=4,
        ckpt_every_steps=2,
        save_checkpoint_every_epoch=False,
    )
    cfg["data"]["dataloader"].update(cache_dir=str(tmp_path / "cache"), num_workers=0, drop_last=False)
    cfg["checkpoint"].update(is_async=False, save_consolidated=False, restore_from=None)
    cfg["performance"]["check_loss"] = True
    cfg["wandb"] = None
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(repo),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        WORLD_SIZE="1",
        LOCAL_WORLD_SIZE="1",
        RANK="0",
        LOCAL_RANK="0",
    )
    report = {
        "status": "running",
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "commands": [],
    }
    report_path = tmp_path / "report.json"
    print(f"QIE_LORA_REPORT={report_path}", flush=True)
    for phase in ("continuous", "interrupted", "resumed", "reload"):
        command = [sys.executable, str(Path(__file__).resolve()), phase, str(tmp_path)]
        report["commands"].append(command)
        report_path.write_text(json.dumps(report, indent=2))
        with (tmp_path / f"{phase}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        assert result.returncode == 0, (tmp_path / f"{phase}.log").read_text()[-12000:]

    def load(phase: str, filename: str) -> object:
        return torch.load(tmp_path / phase / filename, map_location="cpu", weights_only=True)

    _assert_equal(load("continuous", "start.pt"), load("interrupted", "start.pt"))
    interrupted_state = load("interrupted", "final.pt")
    resumed_state = load("resumed", "start.pt")
    interrupted_data = interrupted_state.pop("dataloader")
    resumed_data = resumed_state.pop("dataloader")
    _assert_equal(interrupted_state, resumed_state)
    # SequentialBucketSampler.load_state_dict stores the cursor in
    # _batches_to_skip; batches_yielded counts output from the current iterator.
    # It is not an eager restore invariant. Check the remaining loader cursor
    # here, then verify every consumed input and the complete final state below.
    for data_state in (interrupted_data, resumed_data):
        data_state["_index_sampler_state"].pop("batches_yielded")
    _assert_equal(interrupted_data, resumed_data)
    _assert_equal(
        load("continuous", "inputs.pt"),
        load("interrupted", "inputs.pt") + load("resumed", "inputs.pt"),
    )
    _assert_equal(load("continuous", "final.pt"), load("resumed", "final.pt"))
    _assert_equal(load("continuous", "prediction.pt"), load("resumed", "prediction.pt"))
    report.update(
        status="passed",
        exact_restore_state=["adapter", "optimizer", "lr_scheduler", "rng", "step", "epoch"],
        exact_final_state=["adapter", "optimizer", "lr_scheduler", "rng", "dataloader", "step", "epoch"],
        consumed_inputs_equal=True,
        comparison_tolerance={"rtol": 0, "atol": 0},
        reload=json.loads((tmp_path / "reload.json").read_text()),
    )
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        if sys.argv[1] == "reload":
            _run_reload(Path(sys.argv[2]))
        else:
            _run_training(Path(sys.argv[2]), sys.argv[1])
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
