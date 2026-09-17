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

"""Download-free coverage of model-family LoRA-QAT YAML and config boundaries."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.config.loader import ConfigNode, load_yaml_config
from nemo_automodel.components.quantization.qat import QATConfig, QATRule
from nemo_automodel.components.quantization.weight_qat import WeightQuantizationConfig

EXAMPLES = Path(__file__).resolve().parents[3] / "examples/llm_finetune"
EXAMPLE = EXAMPLES / "qwen/qwen3_0p6b_lora_qat.yaml"
V4_EXAMPLE = EXAMPLES / "deepseek_v4/deepseek_v4_flash_hellaswag_experts_qat.yaml"
GLM_EXAMPLE = EXAMPLES / "glm/glm_5.3_flash_hellaswag_experts_qat.yaml"


@pytest.mark.parametrize("path", [EXAMPLE, V4_EXAMPLE, GLM_EXAMPLE])
def test_example_ci_preserves_supported_smoke_topology(path: Path) -> None:
    """Keep auto-discovered CI jobs within each recipe's rank and step contract."""
    raw = load_yaml_config(path).raw_config
    ci = raw["ci"]
    assert isinstance(ci["recipe_owner"], str) and ci["recipe_owner"].strip()
    hours, minutes, seconds = (int(part) for part in ci["time"].split(":"))
    assert hours >= 0 and 0 <= minutes < 60 and 0 <= seconds < 60
    assert hours * 3600 + minutes * 60 + seconds > 0
    assert type(ci["nodes"]) is int and ci["nodes"] > 0
    assert type(ci["nproc_per_node"]) is int and ci["nproc_per_node"] > 0
    assert ci["node_multiplier"] is False
    world_size = ci["nodes"] * ci["nproc_per_node"]
    assert world_size == raw["distributed"].get("ep_size", 1)
    assert ci["max_steps"] == raw["step_scheduler"]["max_steps"] == 2
    assert ci["local_batch_size"] == raw["step_scheduler"]["local_batch_size"] == 1
    assert world_size * ci["local_batch_size"] == raw["step_scheduler"]["global_batch_size"]
    assert ci["env_vars"]["REQUIRE_FINITE_METRICS"] == "true"


def test_example_instantiates_nested_qat_config() -> None:
    """Exercise real YAML lists without constructing a model, tokenizer, or dataset."""
    cfg = load_yaml_config(EXAMPLE)
    assert isinstance(cfg, ConfigNode)
    raw_qat = deepcopy(cfg.raw_config["qat"]["qat_config"])
    assert set(raw_qat) == {"_target_", "target_modules", "weight"}
    assert raw_qat["_target_"] == "nemo_automodel.components.quantization.qat.QATConfig"
    assert raw_qat["target_modules"] == ["*.q_proj", "*.v_proj"]
    assert raw_qat["weight"] == {"format": "fp8", "block_size": [32, 32]}

    # This is the recipe's actual boundary. Do not manually instantiate list
    # members or convert block_size here: that would hide a broken YAML route.
    qat_config = cfg.qat.qat_config.instantiate()

    assert isinstance(qat_config, QATConfig)
    assert qat_config.rules is None
    assert isinstance(qat_config.target_modules, tuple)
    assert qat_config.target_modules == ("*.q_proj", "*.v_proj")
    assert isinstance(qat_config.weight, WeightQuantizationConfig)
    assert qat_config.weight.format == "fp8"
    assert isinstance(qat_config.weight.block_size, tuple)
    assert qat_config.weight.block_size == (32, 32)
    assert qat_config.weight.scale_format == "e8m0"
    assert cfg.raw_config["qat"]["qat_config"] == raw_qat
    assert cfg.qat.qat_config.instantiate() == qat_config


@pytest.mark.parametrize(
    ("path", "target", "weight_format", "block_size", "scale_format"),
    [
        (V4_EXAMPLE, "model.layers.*.mlp.experts", "mxfp4", (1, 32), "e8m0"),
        (GLM_EXAMPLE, "model.language_model.layers.*.mlp.experts", "fp8", (128, 128), "float32"),
    ],
)
def test_expert_examples_instantiate_plain_rules(
    path: Path, target: str, weight_format: str, block_size: tuple[int, int], scale_format: str
) -> None:
    """Type real recipe rules without model construction or downloads."""
    cfg = load_yaml_config(path)
    raw = deepcopy(cfg.raw_config)
    raw_qat = raw["qat"]["qat_config"]
    assert set(raw_qat) == {"_target_", "rules"}
    assert raw_qat["_target_"] == "nemo_automodel.components.quantization.qat.QATConfig"
    expected_weight = {"format": weight_format, "block_size": list(block_size)}
    expected_weight["scale_format"] = scale_format
    assert cfg.recipe == "TrainFinetuneRecipeForNextTokenPrediction"
    assert raw_qat["rules"] == [{"target_modules": [target], "weight": expected_weight}]
    peft_config = cfg.peft.instantiate()
    assert isinstance(peft_config, PeftConfig)
    assert tuple(peft_config.target_modules) == (target,)
    assert peft_config.dim == 4
    assert peft_config.alpha == 8
    assert peft_config.dropout == 0.0
    assert peft_config.use_dora is False
    assert peft_config.use_triton is False
    assert peft_config.use_memory_efficient_lora is False
    assert cfg.qat.enabled is True
    assert cfg.qat.fake_quant_after_n_steps == 0

    qat_config = cfg.qat.qat_config.instantiate()
    assert isinstance(qat_config, QATConfig)
    assert qat_config.target_modules is None
    assert qat_config.weight is None
    assert isinstance(qat_config.rules, tuple)
    assert len(qat_config.rules) == 1
    rule = qat_config.rules[0]
    assert isinstance(rule, QATRule)
    assert isinstance(rule.target_modules, tuple)
    assert rule.target_modules == (target,)
    assert isinstance(rule.weight, WeightQuantizationConfig)
    assert rule.weight.format == weight_format
    assert rule.weight.block_size == block_size
    assert rule.weight.scale_format == scale_format
    assert isinstance(QAT(qat_config), QAT)
    assert cfg.qat.qat_config.instantiate() == qat_config
    assert cfg.raw_config == raw


def test_legacy_full_parameter_int4_example_remains_unchanged() -> None:
    """Keep the documented legacy config distinct from targeted LoRA QAT."""
    cfg = load_yaml_config(EXAMPLES / "llama3_2/llama3_2_1b_squad_qat.yaml")
    assert "peft" not in cfg.raw_config
    assert cfg.qat.enabled is True
    assert cfg.qat.fake_quant_after_n_steps == 1
    qat_config = cfg.qat.qat_config.instantiate()
    assert isinstance(qat_config, QATConfig)
    assert qat_config.quantizer_type == "int8_dynact_int4weight"
    assert qat_config.quantizer_kwargs == {"groupsize": 256}
    assert qat_config.target_modules is None
    assert qat_config.weight is None
    assert qat_config.rules is None


def test_example_keeps_single_rank_bf16_smoke_test_settings() -> None:
    """Guard the recipe settings without invoking any runtime component targets."""
    cfg = load_yaml_config(EXAMPLE)
    assert cfg.recipe == "TrainFinetuneRecipeForNextTokenPrediction"
    assert cfg.model.pretrained_model_name_or_path == "Qwen/Qwen3-0.6B"
    assert cfg.model.dtype == "bfloat16"
    assert cfg.model.force_hf is True
    assert cfg.model.attn_implementation == "sdpa"
    assert cfg.dataset.tokenizer.pretrained_model_name_or_path == cfg.model.pretrained_model_name_or_path
    assert cfg.dataset.tokenizer.truncation_side == "left"
    assert cfg.dataset.limit_dataset_samples == 64
    assert cfg.dataset.seq_length == 512
    assert cfg.dataset.truncation is True
    assert cfg.step_scheduler.max_steps == 2
    assert cfg.step_scheduler.global_batch_size == cfg.step_scheduler.local_batch_size == 1
    assert cfg.peft.target_modules == ["*.q_proj", "*.v_proj"]
    assert cfg.peft.dim == 4
    assert cfg.peft.alpha == 8
    assert cfg.peft.dropout == 0.0
    assert cfg.peft.use_dora is False
    assert cfg.peft.use_triton is False
    assert cfg.peft.use_memory_efficient_lora is False
    assert cfg.qat.enabled is True
    assert cfg.qat.fake_quant_after_n_steps == 0
    assert cfg.distributed.strategy == "fsdp2"
    assert cfg.distributed.dp_size is None
    assert cfg.distributed.tp_size == cfg.distributed.cp_size == 1
    assert cfg.distributed.get("pp_size", 1) == cfg.distributed.get("ep_size", 1) == 1
    assert cfg.distributed.activation_checkpointing is False
    assert cfg.compile.enabled is False
    assert cfg.packed_sequence.packed_sequence_size == 0
    assert cfg.checkpoint.enabled is False
    assert cfg.get("fp8", None) is None
    assert cfg.get("quantization", None) is None


@pytest.mark.parametrize(
    ("path", "model_id", "config_target", "expert_count", "wrap_outer_model"),
    [
        (
            V4_EXAMPLE,
            "deepseek-ai/DeepSeek-V4-Flash",
            "nemo_automodel.components.models.deepseek_v4.config.DeepseekV4Config.from_pretrained",
            256,
            False,
        ),
        (
            GLM_EXAMPLE,
            "zai-org/GLM-5.3-Flash",
            "nemo_automodel.components.models.glm5_next.config.Glm5NextConfig.from_pretrained",
            288,
            True,
        ),
    ],
)
def test_full_model_recipe_snapshot(
    path: Path, model_id: str, config_target: str, expert_count: int, wrap_outer_model: bool
) -> None:
    """Check main-recipe contracts and parse topology without creating a mesh/model."""
    from nemo_automodel.components.datasets.llm.hellaswag import HellaSwagConfig
    from nemo_automodel.recipes._dist_utils import parse_distributed_section

    cfg = load_yaml_config(path)
    raw = deepcopy(cfg.raw_config)
    assert cfg.recipe == "TrainFinetuneRecipeForNextTokenPrediction"
    assert raw["model"]["_target_"] == "nemo_automodel.NeMoAutoModelForCausalLM.from_config"
    assert raw["model"]["config"]["_target_"] == config_target
    assert cfg.model.config.pretrained_model_name_or_path == cfg.model.config.name_or_path == model_id
    assert cfg.model.load_base_model is True
    assert cfg.model.dtype == "bfloat16"
    assert cfg.model.trust_remote_code is False
    assert cfg.model.use_liger_kernel is False
    assert cfg.model.attn_implementation == "sdpa"
    assert cfg.model.backend.dispatcher == cfg.model.backend.experts == "torch"
    assert cfg.model.backend.attn == "sdpa"
    assert cfg.model.backend.linear == "torch"
    assert cfg.model.backend.rms_norm == "torch_fp32"
    assert cfg.model.backend.rope_fusion is False
    assert cfg.model.backend.enable_hf_state_dict_adapter is True
    assert cfg.model.backend.enable_fsdp_optimizations is True
    backend = cfg.model.backend.instantiate()
    assert backend.dispatcher == backend.experts == "torch"

    # Pure FSDP sharding overlaps EP; it is not 32 * 32 ranks.
    world = cfg.step_scheduler.global_batch_size // cfg.step_scheduler.local_batch_size
    assert world == 32
    assert 2 <= world <= 64
    assert cfg.distributed.ep_size == world
    assert expert_count % world == 0
    assert expert_count // world == (9 if wrap_outer_model else 8)
    assert cfg.distributed.strategy == "fsdp2"
    assert cfg.distributed.dp_replicate_size == 1
    assert cfg.distributed.tp_size == cfg.distributed.cp_size == cfg.distributed.pp_size == 1
    assert cfg.distributed.sequence_parallel is False
    assert cfg.distributed.activation_checkpointing is True
    assert cfg.distributed.moe.reshard_after_forward is False
    assert cfg.distributed.moe.wrap_outer_model is wrap_outer_model
    parsed = parse_distributed_section(raw["distributed"])
    assert parsed["ep_size"] == world
    assert parsed["dp_replicate_size"] == 1
    assert parsed["pipeline_config"] is None
    assert parsed["moe_parallel_config"].wrap_outer_model is wrap_outer_model

    assert cfg.step_scheduler.local_batch_size == 1
    assert cfg.step_scheduler.max_steps == cfg.step_scheduler.num_epochs == 2
    assert cfg.step_scheduler.val_every_steps is None
    assert cfg.step_scheduler.validate_on_checkpoint is False
    assert cfg.step_scheduler.save_checkpoint_every_epoch is False
    assert cfg.checkpoint.enabled is False
    assert cfg.checkpoint.dequantize_base_checkpoint is True
    assert cfg.checkpoint.save_consolidated is False
    assert cfg.compile.enabled is False
    assert cfg.packed_sequence.packed_sequence_size == 0
    assert cfg.dist_env.backend == "nccl"
    assert cfg.dist_env.timeout_minutes == 10
    assert raw["dataset"]["_target_"] == "nemo_automodel.components.datasets.llm.hellaswag.HellaSwag"
    assert raw["dataset"]["tokenizer"]["_target_"] == "transformers.AutoTokenizer.from_pretrained"
    assert cfg.dataset.tokenizer.pretrained_model_name_or_path == model_id
    assert cfg.dataset.tokenizer.trust_remote_code is False
    dataset_kwargs = deepcopy(raw["dataset"])
    dataset_kwargs.pop("_target_")
    dataset_kwargs.pop("tokenizer")
    # Constructor settings only: never build/download HellaSwag. This also rejects
    # invented seq_length/max_length keys (padding is not a truncation bound).
    dataset_config = HellaSwagConfig(**dataset_kwargs)
    assert dataset_config.path_or_dataset == "rowan/hellaswag"
    assert dataset_config.split == "train"
    assert dataset_config.num_samples_limit == 64
    assert dataset_config.pad_to_max_length is False
    assert cfg.dataloader.num_workers == 0
    assert cfg.dataloader.shuffle is True
    assert cfg.dataloader.drop_last is True
    assert cfg.dataloader.collate_fn.pad_seq_len_divisible == 64
    assert raw["optimizer"]["_target_"] == "torch.optim.AdamW"
    assert cfg.optimizer.lr == 1e-5
    assert cfg.optimizer.weight_decay == 0.0
    assert cfg.optimizer.betas == [0.9, 0.95]
    assert cfg.optimizer.eps == 1e-8
    for key in ("fp8", "quantization", "validation_dataset", "restore_from", "autocast"):
        assert key not in raw
    assert "mp_policy" not in raw["distributed"]
    assert "autocast_dtype" not in raw["distributed"]
    assert cfg.raw_config == raw


@pytest.mark.parametrize("path", [V4_EXAMPLE, GLM_EXAMPLE])
def test_full_model_config_preserves_checkpoint_architecture(path: Path, tmp_path: Path) -> None:
    """Resolve model-owned config targets against local JSON, never model weights.

    These checkpoint-style fixtures exercise full dimensions and non-default
    schedules; they are not copied weights or evidence of remote checkpoint parity.
    """
    from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
    from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig

    cfg = load_yaml_config(path)
    original_yaml = deepcopy(cfg.raw_config)
    if path == V4_EXAMPLE:
        snapshot = {
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "num_hidden_layers": 43,
            "n_routed_experts": 256,
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "vocab_size": 129280,
            "num_nextn_predict_layers": 1,
            "use_cache": True,
            # Original V4 schedule includes the trailing MTP layer.
            "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0],
        }
        overrides = original_yaml["model"]["config"]
        assert set(overrides) == {
            "_target_",
            "pretrained_model_name_or_path",
            "name_or_path",
            "num_nextn_predict_layers",
            "use_cache",
        }
        # V4's forward has no cache interface; recent Transformers need not retain
        # use_cache passed through PretrainedConfig. Guard the YAML intent here.
        assert overrides["use_cache"] is False
    else:
        snapshot = {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 45,
                "n_routed_experts": 288,
                "hidden_size": 4096,
                "moe_intermediate_size": 2048,
                "vocab_size": 154880,
                "num_nextn_predict_layers": 1,
                "use_cache": True,
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
                "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"] * 42,
                "indexer_types": ["full"] + ["shared"] * 44,
            },
            "vision_config": {"depth": 24, "hidden_size": 1024, "out_hidden_size": 4096},
        }
        overrides = original_yaml["model"]["config"]
        assert set(overrides) == {"_target_", "pretrained_model_name_or_path", "name_or_path", "text_config"}
        assert overrides["text_config"] == {"num_nextn_predict_layers": 0, "use_cache": False}

    (tmp_path / "config.json").write_text(json.dumps(snapshot), encoding="utf-8")
    model_config = deepcopy(original_yaml["model"]["config"])
    model_config.update(pretrained_model_name_or_path=str(tmp_path), name_or_path=str(tmp_path), local_files_only=True)
    # Call precisely the shipped nested config target. Do NOT instantiate cfg.model.
    resolved = ConfigNode(model_config).instantiate()
    assert resolved.name_or_path == str(tmp_path)
    assert resolved.architectures == snapshot["architectures"]
    if path == V4_EXAMPLE:
        assert isinstance(resolved, DeepseekV4Config)
        text = resolved
        source_text = snapshot
        assert resolved.compress_ratios == snapshot["compress_ratios"]
        assert len(resolved.compress_ratios[: resolved.num_hidden_layers]) == 43
    else:
        assert isinstance(resolved, Glm5NextConfig)
        text = resolved.text_config
        source_text = snapshot["text_config"]
        for field in ("mlp_layer_types", "layer_types", "indexer_types"):
            assert getattr(text, field) == source_text[field]
        assert resolved.vision_config.depth == 24
        assert resolved.vision_config.hidden_size == 1024
        assert resolved.vision_config.out_hidden_size == 4096
        assert text.use_cache is False
    for field in ("num_hidden_layers", "n_routed_experts", "hidden_size", "moe_intermediate_size", "vocab_size"):
        assert getattr(text, field) == source_text[field]
    assert text.num_nextn_predict_layers == 0
    assert cfg.raw_config == original_yaml


@pytest.mark.parametrize("qat_targets", [("*.q_proj", "*.v_proj"), ("*.q_proj",)])
def test_example_peft_selectors_patch_real_nested_qwen_modules(qat_targets: tuple[str, ...]) -> None:
    """Prepare a real tiny Qwen and prove QAT can select a subset of PEFT."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from nemo_automodel.components._peft.lora import LinearLoRA, apply_lora_to_linear_modules
    from nemo_automodel.components.quantization.weight_qat import WeightFakeQuantizer

    cfg = load_yaml_config(EXAMPLE)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            vocab_size=64,
        )
    )
    assert apply_lora_to_linear_modules(model, cfg.peft.instantiate()) == 2
    raw_qat = deepcopy(cfg.raw_config["qat"]["qat_config"])
    raw_qat["target_modules"] = list(qat_targets)
    qat_config = ConfigNode(raw_qat).instantiate()
    assert QAT(qat_config).prepare(model) is model
    for name in ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"):
        module = model.get_submodule(name)
        assert isinstance(module, LinearLoRA)
        if f"*.{name.rsplit('.', 1)[-1]}" in qat_targets:
            assert isinstance(module.weight_fake_quantizer, WeightFakeQuantizer)
            assert module.weight_fake_quantizer.config == qat_config.weight
        else:
            assert module.weight_fake_quantizer is None
    assert not isinstance(model.get_submodule("model.layers.0.self_attn.k_proj"), LinearLoRA)
    assert cfg.peft.target_modules == ["*.q_proj", "*.v_proj"]
