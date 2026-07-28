# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.diffusion.collate_fns import TextToImageDataloaderConfig
from nemo_automodel.components.optim.optimizer import LRSchedulerConfig, OptimizerFromFactoryConfig
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.diffusion.train import (
    _build_diffusion_parallel_manager_args,
    _reject_removed_diffusion_keys,
    _resolve_model_dtypes,
    _validate_precision_configuration,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "diffusion"


def test_resolve_model_dtypes_preserves_existing_bf16_defaults():
    cfg = ConfigNode({"model": {}})

    model_dtype, compute_dtype = _resolve_model_dtypes(cfg)

    assert model_dtype is torch.bfloat16
    assert compute_dtype is torch.bfloat16


def test_resolve_model_dtypes_allows_split_storage_and_compute_dtype():
    cfg = ConfigNode({"model": {"torch_dtype": "float32", "compute_dtype": "bfloat16"}})

    model_dtype, compute_dtype = _resolve_model_dtypes(cfg)

    assert model_dtype is torch.float32
    assert compute_dtype is torch.bfloat16


def test_validate_precision_configuration_allows_split_dtype_for_fsdp_full_training():
    _validate_precision_configuration(
        torch.float32,
        torch.bfloat16,
        ddp_cfg=None,
        peft_cfg=None,
    )


@pytest.mark.parametrize(
    ("ddp_cfg", "peft_cfg", "expected_mode"),
    [
        ({"world_size": 1}, None, "DDP"),
        (None, object(), "PEFT/LoRA"),
    ],
)
def test_validate_precision_configuration_rejects_split_dtype_without_fsdp_param_cast(
    ddp_cfg,
    peft_cfg,
    expected_mode,
):
    with pytest.raises(ValueError, match=expected_mode):
        _validate_precision_configuration(
            torch.float32,
            torch.bfloat16,
            ddp_cfg=ddp_cfg,
            peft_cfg=peft_cfg,
        )


def test_validate_precision_configuration_allows_matching_dtype_for_ddp_or_peft():
    _validate_precision_configuration(
        torch.bfloat16,
        torch.bfloat16,
        ddp_cfg={"world_size": 1},
        peft_cfg=object(),
    )


# ---------------------------------------------------------------------------
# Removed legacy diffusion YAML keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legacy_cfg", "expected_replacement"),
    [
        ({"optim": {"learning_rate": 1.0e-5}}, "optimizer.lr"),
        ({"optim": {"optimizer": {"weight_decay": 0.01}}}, "optimizer"),
        ({"optim": {"clip_grad": 2.0}}, "clip_grad_norm.max_norm"),
        ({"step_scheduler": {"log_every": 10}}, "log_remote_every_steps"),
    ],
)
def test_removed_legacy_keys_raise_with_replacement_mapping(legacy_cfg, expected_replacement):
    with pytest.raises(ValueError, match=expected_replacement):
        _reject_removed_diffusion_keys(ConfigNode(legacy_cfg))


def test_standard_keys_pass_removed_key_check():
    cfg = ConfigNode(
        {
            "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-5},
            "clip_grad_norm": {"max_norm": 1.0},
            "step_scheduler": {"log_remote_every_steps": 10, "global_batch_size": 8},
        }
    )

    _reject_removed_diffusion_keys(cfg)


def test_recipe_rejects_legacy_keys_at_construction():
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

    with pytest.raises(ValueError, match="removed diffusion-only keys"):
        TrainDiffusionRecipe(ConfigNode({"optim": {"learning_rate": 1.0e-5}}))


# ---------------------------------------------------------------------------
# Typed optimizer construction through RecipeConfig
# ---------------------------------------------------------------------------


def test_recipe_config_optimizer_builds_adamw_from_diffusion_yaml_block():
    cfg = RecipeConfig(
        ConfigNode(
            {
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 1.0e-5,
                    "weight_decay": 1.0e-4,
                    "betas": [0.9, 0.999],
                    "fused": False,
                }
            }
        )
    )
    model = torch.nn.Linear(2, 2)

    assert isinstance(cfg.optimizer, OptimizerFromFactoryConfig)
    optimizers = cfg.optimizer.build(model, device_mesh=None, is_peft=False)

    assert isinstance(optimizers, list) and len(optimizers) == 1
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert optimizers[0].defaults["lr"] == pytest.approx(1.0e-5)
    assert optimizers[0].defaults["weight_decay"] == pytest.approx(1.0e-4)
    assert optimizers[0].defaults["fused"] is False


def test_optimizer_build_trains_only_lora_params_when_base_is_frozen():
    class LoraLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(4, 4)
            self.lora_A = torch.nn.Parameter(torch.zeros(2, 4))
            self.lora_B = torch.nn.Parameter(torch.zeros(4, 2))

    model = LoraLinear()
    model.base.weight.requires_grad_(False)
    model.base.bias.requires_grad_(False)

    cfg = RecipeConfig(ConfigNode({"optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4}}))
    optimizers = cfg.optimizer.build(model, device_mesh=None, is_peft=True)

    optimizer_params = {id(p) for group in optimizers[0].param_groups for p in group["params"]}
    assert optimizer_params == {id(model.lora_A), id(model.lora_B)}


def test_optimizer_build_raises_when_no_trainable_params():
    model = torch.nn.Linear(2, 2)
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = RecipeConfig(ConfigNode({"optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4}}))

    with pytest.raises(ValueError, match="no trainable parameters"):
        cfg.optimizer.build(model, device_mesh=None, is_peft=False)


# ---------------------------------------------------------------------------
# Typed checkpoint config on a diffusion-shaped YAML
# ---------------------------------------------------------------------------


def test_recipe_config_checkpoint_fills_diffusion_model_fields():
    cfg = RecipeConfig(
        ConfigNode(
            {
                "model": {"pretrained_model_name_or_path": "org/diffusion-model", "cache_dir": "/tmp/hf-cache"},
                "peft": {"dim": 4},
                "checkpoint": {
                    "enabled": True,
                    "checkpoint_dir": "/tmp/ckpts",
                    "model_save_format": "safetensors",
                    "save_consolidated": False,
                    "consolidation_timeout_minutes": 45,
                    "diffusers_compatible": True,
                    "restore_from": "/tmp/ckpts/step-5",
                },
            }
        )
    )

    checkpoint = cfg.checkpoint

    assert checkpoint.model_repo_id == "org/diffusion-model"
    assert str(checkpoint.model_cache_dir) == "/tmp/hf-cache"
    assert checkpoint.is_peft is True
    assert checkpoint.diffusers_compatible is True
    assert checkpoint.consolidation_timeout_minutes == 45
    # restore_from is consumed at load time, never a config field
    assert not hasattr(checkpoint, "restore_from")


# ---------------------------------------------------------------------------
# LR trajectory equivalence with the removed diffusion LR builder
# ---------------------------------------------------------------------------


def _legacy_diffusion_lr(style: str, warmup: int, total: int, base_lr: float, step: int) -> float:
    """LR at ``step`` per the removed diffusion ``build_lr_scheduler`` defaults."""
    init_lr = base_lr if warmup == 0 else base_lr * 0.1
    max_lr = base_lr
    min_lr = base_lr * 0.01
    if warmup > 0 and step <= warmup:
        return init_lr + (max_lr - init_lr) * step / warmup
    if style == "constant":
        return max_lr
    if step > total:
        return min_lr
    ratio = (step - warmup) / (total - warmup)
    coeff = 0.5 * (math.cos(math.pi * ratio) + 1.0)
    return min_lr + (max_lr - min_lr) * coeff


@pytest.mark.parametrize("style", ["constant", "cosine"])
@pytest.mark.parametrize("warmup", [0, 500])
def test_lr_scheduler_config_matches_legacy_diffusion_trajectories(style, warmup):
    base_lr = 1.0e-4
    total = 1000
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    step_scheduler = SimpleNamespace(epoch_len=100, num_epochs=10, max_steps=None)

    schedulers = LRSchedulerConfig(lr_decay_style=style, lr_warmup_steps=warmup).build(
        [optimizer],
        step_scheduler,
    )
    assert len(schedulers) == 1
    scheduler = schedulers[0]

    for step in range(1, total + 1):
        scheduler.step(1)
        if step in (1, warmup or 1, 250, 500, 750, 1000):
            expected = _legacy_diffusion_lr(style, warmup, total, base_lr, step)
            assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6), (
                f"style={style} warmup={warmup} step={step}"
            )


def test_lr_scheduler_config_rejects_fractional_warmup():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0e-4)
    step_scheduler = SimpleNamespace(epoch_len=100, num_epochs=10, max_steps=None)

    with pytest.raises(ValueError, match="Fractional warmup"):
        LRSchedulerConfig(lr_warmup_steps=0.1).build([optimizer], step_scheduler)


# ---------------------------------------------------------------------------
# Example YAMLs stay coercible through the typed recipe boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_path",
    sorted(EXAMPLES_DIR.glob("finetune/*.yaml")) + sorted(EXAMPLES_DIR.glob("pretrain/*.yaml")),
    ids=lambda p: p.name,
)
def test_example_diffusion_yamls_coerce_through_typed_configs(yaml_path):
    raw = yaml.safe_load(yaml_path.read_text())
    optimizer_target = raw["optimizer"].get("_target_")
    assert optimizer_target is not None, "diffusion example YAMLs must declare optimizer._target_"
    if not optimizer_target.startswith("torch."):
        # ConfigNode resolves targets eagerly. Keep this schema sweep CPU-compatible
        # when an example uses an optional optimizer such as Transformer Engine.
        raw["optimizer"]["_target_"] = "torch.optim.AdamW"

    cfg = RecipeConfig(ConfigNode(raw))

    _reject_removed_diffusion_keys(cfg)

    step_scheduler_cfg = cfg.step_scheduler  # raises TypeError on unknown keys
    assert step_scheduler_cfg.global_batch_size > 0

    if "lr_scheduler" in raw:
        assert cfg.lr_scheduler is not None

    if "checkpoint" in raw:
        assert cfg.checkpoint.model_repo_id == raw["model"]["pretrained_model_name_or_path"]

    assert cfg.optimizer is not None


def test_recipe_config_resolves_diffusion_builder_target_to_typed_config():
    config = RecipeConfig(
        ConfigNode(
            {
                "data": {
                    "dataloader": {
                        "_target_": (
                            "nemo_automodel.components.datasets.diffusion."
                            "build_text_to_image_multiresolution_dataloader"
                        ),
                        "cache_dir": "/tmp/cache",
                        "base_resolution": [512, 512],
                        "num_workers": 0,
                    }
                }
            }
        )
    ).diffusion_dataloader

    assert isinstance(config, TextToImageDataloaderConfig)
    assert config.cache_dir == "/tmp/cache"
    assert config.base_resolution == (512, 512)
    assert config.num_workers == 0


def test_recipe_config_rejects_unknown_diffusion_dataloader_field():
    raw = ConfigNode(
        {
            "_target_": "nemo_automodel.components.datasets.diffusion.build_video_multiresolution_dataloader",
            "cache_dir": "/tmp/cache",
            "num_workerz": 2,
        }
    )

    with pytest.raises(TypeError, match="num_workerz"):
        RecipeConfig.resolve_diffusion_dataloader(raw)


def test_manager_args_default_to_pure_ulysses_cp_split():
    args = _build_diffusion_parallel_manager_args(
        fsdp_cfg={"cp_size": 2},
        ddp_cfg=None,
        world_size=8,
        dtype=torch.bfloat16,
        lora_enabled=False,
    )

    assert args["cp_size"] == 2
    assert args["cp_ring_degree"] == 1
    assert args["cp_ulysses_degree"] == 2


def test_manager_args_pass_explicit_ring_ulysses_split_through():
    args = _build_diffusion_parallel_manager_args(
        fsdp_cfg={"cp_size": 4, "cp_ring_degree": 2, "cp_ulysses_degree": 2},
        ddp_cfg=None,
        world_size=8,
        dtype=torch.bfloat16,
        lora_enabled=False,
    )

    # The builder only threads the split through; ring > 1 is rejected later by
    # _enable_context_parallel, where the diffusers version constraint lives.
    assert args["cp_ring_degree"] == 2
    assert args["cp_ulysses_degree"] == 2


def test_manager_args_derive_ulysses_from_ring_when_unset():
    args = _build_diffusion_parallel_manager_args(
        fsdp_cfg={"cp_size": 4, "cp_ring_degree": 2},
        ddp_cfg=None,
        world_size=8,
        dtype=torch.bfloat16,
        lora_enabled=False,
    )

    assert args["cp_ring_degree"] == 2
    assert args["cp_ulysses_degree"] == 2


def test_manager_args_cp_knobs_default_when_cp_disabled():
    args = _build_diffusion_parallel_manager_args(
        fsdp_cfg={},
        ddp_cfg=None,
        world_size=8,
        dtype=torch.bfloat16,
        lora_enabled=False,
    )

    assert args["cp_size"] == 1
    assert args["cp_ring_degree"] == 1
    assert args["cp_ulysses_degree"] == 1
