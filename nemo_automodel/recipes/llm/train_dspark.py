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

"""DSpark draft-model training recipe (Qwen3, Gemma4, DeepSeek V4, GLM-5.2, and MiniMax M3 VL targets).

DSpark is a semi-autoregressive parallel drafter: a parallel backbone produces a
block of tokens per anchor in one pass, a serial Markov head injects intra-block
dependency, and a confidence head predicts per-position acceptance. This recipe
mirrors the EAGLE / DFlash scaffolding -- online target hidden-state capture,
gradient accumulation with a trailing-window flush, and the shared checkpointer
plumbing -- and trains the draft with the three-term DSpark objective.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from types import SimpleNamespace

import torch
import torch.distributed as dist
from huggingface_hub import constants as hf_constants
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel._transformers import NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    save_config,
)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.dspark_cache import (
    DTYPE_MAP,
    build_cached_dspark_dataloader,
    read_manifest,
    read_target_weight_modules,
)
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader
from nemo_automodel.components.datasets.vlm.dspark_collate import build_dspark_vlm_dataloader
from nemo_automodel.components.distributed.activation_checkpointing import (
    apply_selective_checkpointing_to_layers,
    apply_submodule_checkpointing,
    is_selective_activation_checkpointing,
)
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.wandb_utils import init_wandb_run, suppress_wandb_log_messages
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.minimax_m3_vl.processing import build_minimax_m3_vl_processor
from nemo_automodel.components.optim.optimizer import build_optimizer
from nemo_automodel.components.speculative.dspark.common import validate_target_layer_ids
from nemo_automodel.components.speculative.dspark.config import (
    build_deepseek_v4_draft_config,
    build_gemma4_draft_config,
    build_glm_5_2_draft_config,
    build_minimax_m3_draft_config,
)
from nemo_automodel.components.speculative.dspark.core import DSparkTrainerModule
from nemo_automodel.components.speculative.dspark.registry import (
    build_target_layer_ids,
    resolve_dspark_draft_spec,
)
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.components.speculative.dspark.target_utils import (
    DEEPSEEK_V4_MODEL_TYPE as _DEEPSEEK_V4_MODEL_TYPE,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    GEMMA4_MODEL_TYPES as _GEMMA4_MODEL_TYPES,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    GLM_5_2_MODEL_TYPE as _GLM_5_2_MODEL_TYPE,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    MINIMAX_M3_MODEL_TYPES as _MINIMAX_M3_MODEL_TYPES,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    apply_target_chat_template as _apply_target_chat_template,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    read_target_model_type as _read_target_model_type,
)
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes.base_recipe import (
    BaseRecipe,
    _find_latest_checkpoint,
    _is_checkpoint_model_config_compatible,
    _resolve_restore_from_to_ckpt_dir,
)
from nemo_automodel.recipes.llm._spec_train_utils import (
    make_warmup_cosine_schedule,
    optim_steps_per_epoch,
)

logger = logging.getLogger(__name__)

_DSPARK_MM_KEYS = tuple(k for k in VLM_INPUT_KEYS if k != "input_ids")


def _extract_mm_kwargs(batch: dict) -> dict:
    """Return only the multimodal keys present in *batch*, for ``generate_batch(**kwargs)``.

    Empty for a text-only batch (Qwen3, Gemma4, or MiniMax M3 without
    ``multimodal: true``), so the ``generate_batch`` call is unchanged in that case.
    """
    return {k: batch[k] for k in _DSPARK_MM_KEYS if k in batch}


class _DraftArgs(dict):
    """Dict with attribute access for the per-architecture draft-config builders."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc


def _gather_full_weight_module(module):
    """Return an object exposing a full (non-DTensor) ``.weight`` tensor.

    The expert-parallel / FSDP-sharded DeepSeek V4 target stores ``embed_tokens`` and
    ``lm_head`` weights as DTensors, while the draft copies them into plain Parameters.
    Gather the sharded weight to a full tensor first (an all-gather, so every rank must
    call this in lockstep); non-sharded targets (Qwen3, Gemma4) pass through unchanged.
    """
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "full_tensor"):
        return SimpleNamespace(weight=weight.full_tensor())
    return module


def _resolve_wandb_kwargs(wandb_cfg: dict) -> dict | None:
    """Convert a ``wandb:`` config block into ``wandb.init`` kwargs, or ``None``.

    ``enable`` is the examples' documentation-only opt-in flag (W&B logging is
    opt-in: example configs ship the block with ``enable: false`` so users start
    logging by flipping it to ``true`` instead of commenting the block in/out);
    it is not a real ``wandb.init`` kwarg, so strip it before forwarding the rest
    -- passing it through raises ``TypeError: init() got an unexpected keyword
    argument 'enable'``. Returns ``None`` when ``enable`` is explicitly ``False``.
    """
    kwargs = dict(wandb_cfg)
    if kwargs.pop("enable", True) is False:
        return None
    return kwargs


def _init_dspark_wandb(*, is_main: bool, wandb_cfg, cfg_dict: dict, default_name: str):
    """Initialize the rank-zero W&B run for a DSpark training job, or return ``None``.

    Centralizes the ``is_main`` / block-presence / ``enable`` gating that
    ``TrainDSparkRecipe.setup`` previously inlined, so it is unit-testable
    without a distributed environment.
    """
    if not is_main or wandb_cfg is None:
        return None
    wandb_kwargs = _resolve_wandb_kwargs(wandb_cfg.to_dict())
    if wandb_kwargs is None:
        return None
    suppress_wandb_log_messages()
    return init_wandb_run(wandb_kwargs, cfg_dict, default_name=default_name)


def _resolve_dspark_optimizer_spec(opt_cfg) -> tuple[str, dict]:
    """Normalize the recipe's ``optimizer:`` config into a ``build_optimizer`` spec.

    Reads an optional ``_target_`` (a registry short name such as ``"fused_adam"``
    or a dotted import path, e.g. ``transformer_engine.pytorch.optimizers.FusedAdam``)
    plus whatever other fields the config carries -- ``lr``/``betas``/``weight_decay``
    and any optimizer-specific kwargs (``master_weights``, ``master_weight_dtype``,
    ``exp_avg_dtype``, ``exp_avg_sq_dtype``, ``store_param_remainders``, ...) -- and
    returns the ``(target, kwargs)`` tuple that ``build_optimizer`` resolves via its
    registry / dotted-import-path / ``OptimizerFromFactoryConfig`` escape hatch.

    Absent an explicit ``_target_``, this defaults to plain ``torch.optim.AdamW``
    with its prior ``betas``/``weight_decay`` defaults (matching the previous
    hardcoded behavior, so existing DSpark configs are unaffected). Those two
    AdamW-shaped defaults are only injected in that no-``_target_`` case: forcing
    them onto an arbitrary explicit ``_target_`` would break optimizers that do
    not accept a ``betas`` kwarg (e.g. plain SGD).
    """
    kwargs = dict(opt_cfg.to_dict())
    # ConfigNode resolves ``_target_`` to the callable in ``to_dict``; recover the original
    # import-path string via ``get_as_string``. Only call it when the key is actually
    # present: ``ConfigNode.get_as_string`` raises ``KeyError`` for an absent key even
    # with an explicit ``None`` default (a ``None`` default is never returned), which
    # crashed every DSpark config whose ``optimizer:`` block omits ``_target_``.
    target = kwargs.pop("_target_", None)
    if target is not None and hasattr(opt_cfg, "get_as_string"):
        target = opt_cfg.get_as_string("_target_")
    kwargs.pop("warmup_ratio", None)
    kwargs.pop("min_lr_ratio", None)
    kwargs["lr"] = float(kwargs["lr"])
    if target is None:
        target = "torch.optim.AdamW"
        kwargs.setdefault("betas", (0.9, 0.95))
        kwargs.setdefault("weight_decay", 0.0)
    return target, kwargs


def _build_dspark_optimizer(trainer_module, opt_cfg, device_mesh=None) -> torch.optim.Optimizer:
    """Build the DSpark trainer's optimizer from its ``optimizer:`` config.

    Thin wrapper around ``build_optimizer`` so ``TrainDSparkRecipe.setup`` has a
    single, unit-testable call site (``build_optimizer`` itself needs no
    distributed environment for a non-pipelined single-part model like the
    DSpark draft, so this is testable with a plain CPU module).
    """
    return build_optimizer(trainer_module, _resolve_dspark_optimizer_spec(opt_cfg), device_mesh=device_mesh)[0]


def _resolve_warmup_steps(warmup_ratio: float, total_optim_steps: int, min_warmup_steps: int = 20) -> int:
    """Return the LR warmup length in optimizer steps.

    ``warmup_ratio * total_optim_steps`` collapses to a handful of steps (or fewer)
    on short / small-dataset runs, dropping a freshly-initialized draft (random
    attention layers, Markov head, confidence head) to near-peak LR within the
    first few optimizer steps -- a reliable way to trigger an early loss spike.
    Floor the ratio-derived step count at ``min_warmup_steps`` unless the caller
    explicitly opts out of warmup with ``warmup_ratio<=0`` (e.g. the smoke config).
    """
    if warmup_ratio <= 0:
        return 1
    return max(min_warmup_steps, int(warmup_ratio * total_optim_steps))


def _resolve_reduced_target_layers(checkpoint_num_layers, requested):
    """Validate the optional ``target_num_hidden_layers`` diagnostic override.

    Returns the reduced layer count (an int in ``[1, checkpoint_num_layers]``) or
    ``None`` when unset. Loading fewer layers lets the full EP / hidden-capture /
    draft path run on one node (the full DeepSeek-V4-Flash target OOMs at load on a
    single 8x80GB box); a draft trained against any reduced target is not usable.
    """
    if requested is None:
        return None
    n = int(requested)
    if n < 1 or n > checkpoint_num_layers:
        raise ValueError(
            f"target_num_hidden_layers={n} must be in [1, {checkpoint_num_layers}] (the checkpoint's depth)."
        )
    return n


def _repair_glm_5_2_qk_rope_head_dim(target_config, raw_config_dict: dict) -> None:
    """Restore a ``qk_rope_head_dim`` clobbered by the HF ``head_dim`` attribute map.

    The published GLM-5.2 config carries ``head_dim: 192`` (the attention-kernel head
    dim) alongside ``qk_rope_head_dim: 64``, and ``GlmMoeDsaConfig``'s
    ``attribute_map = {"head_dim": "qk_rope_head_dim"}`` lets the former clobber the
    latter on load: the loaded config reports ``qk_rope_head_dim=192``, so
    ``kv_a_proj_with_mqa`` builds ``kv_lora_rank + 192 = 704`` wide while the
    checkpoint ships ``512 + 64 = 576`` and shape validation fails. Restore the raw
    checkpoint value (the GLM finetune examples apply the same correction via a
    ``config: head_dim: 64`` override). No-op when the raw config omits the field or
    the loaded value already matches (e.g. a locally repaired checkpoint view).
    """
    raw_qk_rope = raw_config_dict.get("qk_rope_head_dim")
    if raw_qk_rope is None or int(target_config.qk_rope_head_dim) == int(raw_qk_rope):
        return
    logger.warning(
        "GLM-5.2 config head_dim clobbered qk_rope_head_dim (loaded %s) via the HF attribute_map; "
        "restoring qk_rope_head_dim=%s from the raw checkpoint config.",
        target_config.qk_rope_head_dim,
        raw_qk_rope,
    )
    target_config.qk_rope_head_dim = int(raw_qk_rope)


def _apply_draft_activation_checkpointing(draft_model: torch.nn.Module, mode: bool | str) -> None:
    """Apply the recipe's AC mode to the trainable DSpark draft before FSDP."""
    if not mode or (isinstance(mode, str) and mode.lower() == "false"):
        return
    layers = list(getattr(draft_model, "layers", ()))
    if not layers:
        logger.warning("Draft activation checkpointing requested, but the draft exposes no layers.")
        return
    if is_selective_activation_checkpointing(mode):
        apply_selective_checkpointing_to_layers(draft_model, layers, has_kv_sharing=False)
        logger.info("Enabled selective activation checkpointing on %d draft layers", len(layers))
    else:
        # DSpark's native layers are not HF GradientCheckpointingLayer subclasses.
        # Checkpoint their attention/MLP/norm submodules before FSDP indexes params.
        apply_submodule_checkpointing(layers, has_kv_sharing=False)
        logger.info("Enabled full activation checkpointing on %d draft layers", len(layers))


def _validate_cached_dspark_manifest(
    cache_dir: str,
    manifest: dict,
    target_config,
    target_layer_ids: list[int],
    *,
    target_model: str,
    target_model_type: str,
    seq_length: int,
    compute_dtype: torch.dtype,
) -> None:
    """Validate that a DSpark offline cache matches the configured target/draft run."""
    if str(manifest["target_model"]) != str(target_model):
        logger.warning(
            "DSpark cache at %s was built for target_model=%r, but this run configured target_model=%r. "
            "Continuing because raw paths can differ across machines; structural cache fields will still be "
            "validated.",
            cache_dir,
            manifest["target_model"],
            target_model,
        )
    if str(manifest["target_model_type"]) != str(target_model_type):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for target_model_type={manifest['target_model_type']!r}, "
            f"but the configured target has model_type={target_model_type!r}."
        )
    if int(manifest["target_vocab_size"]) != int(target_config.vocab_size):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for target_vocab_size={manifest['target_vocab_size']}, "
            f"but the configured target has {target_config.vocab_size}. The cache does not match this target."
        )
    hidden_size = int(target_config.hidden_size)
    if int(manifest["hidden_size"]) != hidden_size:
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for hidden_size={manifest['hidden_size']}, "
            f"but the configured target has hidden_size={hidden_size}."
        )
    if int(manifest["num_hidden_layers"]) != int(target_config.num_hidden_layers):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for num_hidden_layers={manifest['num_hidden_layers']}, "
            f"but the configured target has num_hidden_layers={target_config.num_hidden_layers}."
        )
    if int(manifest["seq_length"]) != int(seq_length):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for seq_length={manifest['seq_length']}, "
            f"but this run configured seq_length={seq_length}."
        )
    cache_dtype = DTYPE_MAP.get(str(manifest["dtype"]))
    if cache_dtype is None:
        raise ValueError(f"DSpark cache at {cache_dir} has unsupported dtype={manifest['dtype']!r}.")
    if compute_dtype == torch.float32 and cache_dtype != torch.float32:
        raise ValueError(
            f"DSpark cache at {cache_dir} stores dtype={manifest['dtype']}, but CPU cached training "
            "requires fp32 cache tensors. Regenerate with --dtype fp32 or train on CUDA."
        )
    expected_hidden_dim = hidden_size * len(target_layer_ids)
    if int(manifest["target_hidden_dim"]) != expected_hidden_dim:
        raise ValueError(
            f"DSpark cache at {cache_dir} has target_hidden_dim={manifest['target_hidden_dim']}, "
            f"but the configured target/layers need {expected_hidden_dim} "
            f"(hidden_size {hidden_size} x {len(target_layer_ids)} target layers)."
        )
    if int(manifest["target_last_hidden_dim"]) != hidden_size:
        raise ValueError(
            f"DSpark cache at {cache_dir} has target_last_hidden_dim={manifest['target_last_hidden_dim']}, "
            f"but the configured target has hidden_size={hidden_size}."
        )
    recorded_layer_ids = [int(x) for x in manifest["target_layer_ids"]]
    if recorded_layer_ids != list(target_layer_ids):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for target_layer_ids={recorded_layer_ids}, "
            f"but this run requested target_layer_ids={target_layer_ids}."
        )


class TrainDSparkRecipe(BaseRecipe):
    """Recipe for DSpark draft-model training on Qwen3, Gemma4, DeepSeek V4, GLM-5.2, and MiniMax M3 VL targets."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        """Build the target model, DSpark draft, data, optimizer, and trainer module."""
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env", {}).get("backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env", {}).get("timeout_minutes", 30),
        )
        setup_logging()

        recipe_cfg = self.cfg.recipe_args
        self.device = self.dist_env.device or torch.device("cpu")
        # The draft is sharded directly with fully_shard over the default world
        # mesh (no explicit MeshContext), so _dp_allreduce reduces over the world group.
        # A DeepSeek V4 target additionally needs its own expert-parallel / FSDP mesh,
        # kept in self.distributed_setup and used only to load and shard that target.
        self.device_mesh = None
        self.distributed_setup = None

        target_path = recipe_cfg.target_model_name_or_path
        trust_remote_code = bool(recipe_cfg.get("trust_remote_code", False))
        target_model_type = _read_target_model_type(target_path, trust_remote_code)
        is_deepseek_v4_target = target_model_type == _DEEPSEEK_V4_MODEL_TYPE
        is_glm_5_2_target = target_model_type == _GLM_5_2_MODEL_TYPE
        is_gemma4_target = target_model_type in _GEMMA4_MODEL_TYPES
        is_minimax_m3_target = target_model_type in _MINIMAX_M3_MODEL_TYPES
        self.cached_target_path = recipe_cfg.get("cached_target_path", None)
        is_multimodal = bool(recipe_cfg.get("multimodal", False))
        if is_multimodal and not is_minimax_m3_target:
            raise ValueError(
                f"recipe_args.multimodal=true is only supported for a MiniMax M3 VL target "
                f"(model_type in {_MINIMAX_M3_MODEL_TYPES}), got model_type={target_model_type!r}."
            )

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(target_path, trust_remote_code=trust_remote_code)
        chat_template = recipe_cfg.get("chat_template", None)
        # Online DSpark renders 'messages'-format data here and needs the tokenizer's
        # chat template. Offline cached training consumes already-tokenized cache
        # tensors, so a missing template should not block training; still apply an
        # explicit override so saved checkpoints carry the requested tokenizer state.
        if self.cached_target_path is None or chat_template is not None:
            _apply_target_chat_template(self.tokenizer, chat_template)
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        if is_deepseek_v4_target:
            if self.cached_target_path is None:
                # Full V4-Flash target loaded with the same expert-parallel / FSDP and
                # FP8-dequant path as the V4 finetune recipe, so the 256 experts shard
                # across ranks instead of replicating per rank.
                target_config, self.target_model = self._build_deepseek_v4_target(
                    target_path, recipe_cfg, trust_remote_code
                )
            else:
                target_config = DeepseekV4Config.from_pretrained(
                    target_path, name_or_path=target_path, num_nextn_predict_layers=0
                )
                n_reduced = _resolve_reduced_target_layers(
                    target_config.num_hidden_layers, recipe_cfg.get("target_num_hidden_layers", None)
                )
                if n_reduced is not None:
                    target_config.num_hidden_layers = n_reduced
                self.target_model = None
            architectures = list(getattr(target_config, "architectures", None) or ["DeepseekV4ForCausalLM"])
        elif is_minimax_m3_target:
            # MiniMax M3 VL is a ~400B-parameter MoE VLM: load it frozen through the
            # same expert-parallel / FSDP distributed path the VLM finetune recipe
            # uses, sharding the 128 routed experts across ranks instead of
            # replicating per rank. DSpark's forward-hook hidden-state capture needs
            # one non-pipelined `self.model(...)` call, so pp_size must be 1 in the
            # recipe's `distributed:` block; use a larger ep_size instead of PP to
            # shard the parameter memory (see the example yaml for the tradeoff).
            target_config = AutoConfig.from_pretrained(target_path, trust_remote_code=trust_remote_code)
            target_text_overrides = {"num_mtp_modules": 0}
            n_reduced = _resolve_reduced_target_layers(
                target_config.text_config.num_hidden_layers,
                recipe_cfg.get("target_num_hidden_layers", None),
            )
            if n_reduced is not None:
                logger.warning(
                    "Reducing the MiniMax M3 target from %d to %d text layers "
                    "(target_num_hidden_layers): diagnostic/CI only, not a usable drafter.",
                    target_config.text_config.num_hidden_layers,
                    n_reduced,
                )
                target_config.text_config.num_hidden_layers = n_reduced
                target_text_overrides["num_hidden_layers"] = n_reduced
            architectures = list(
                getattr(target_config, "architectures", None) or ["MiniMaxM3SparseForConditionalGeneration"]
            )
            if self.cached_target_path is None:
                self.distributed_setup = create_distributed_setup_from_config(
                    self.cfg,
                    world_size=self.dist_env.world_size,
                )
                backend = BackendConfig(
                    # M3's sparse-attention layers emit an additive float bias from the
                    # DSA indexer that only SDPA's explicit-mask path accepts; TE's
                    # DotProductAttention treats attention_mask as a boolean padding
                    # mask and crashes on the float bias.
                    attn="sdpa",
                    # The target is frozen / forward-only here, so there is no
                    # throughput reason to pay TE's integration complexity, and plain
                    # linears keep embed_tokens/lm_head as plain-shaped weights.
                    linear="torch",
                    rms_norm="torch_fp32",
                    rope_fusion=False,
                    experts=str(recipe_cfg.get("target_experts", "gmm")),
                    dispatcher="hybridep",
                    enable_hf_state_dict_adapter=True,
                    enable_fsdp_optimizations=True,
                )
                self.target_model = NeMoAutoModelForImageTextToText.from_pretrained(
                    target_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=self.compute_dtype,
                    distributed_setup=self.distributed_setup,
                    backend=backend,
                    # The released bf16 checkpoint ships no real MTP weights despite the
                    # config declaring some, and DSpark trains its own separate draft
                    # regardless, so disable the target's native MTP modules.
                    text_config=target_text_overrides,
                )
                # A distributed-setup-loaded model already lands correctly placed
                # (sharded as DTensors); a blanket .to(device) afterward is redundant.
            else:
                self.target_model = None
        elif is_glm_5_2_target:
            if self.cached_target_path is None:
                # GLM-5.2 (GlmMoeDsaForCausalLM) is a ~355B-parameter MLA + DSA MoE LM: load
                # it frozen through the same expert-parallel / FSDP distributed path the GLM
                # finetune recipe uses, sharding the 256 routed experts across ranks instead
                # of replicating per rank.
                target_config, self.target_model = self._build_glm_5_2_target(
                    target_path, recipe_cfg, trust_remote_code
                )
            else:
                target_config = AutoConfig.from_pretrained(target_path, trust_remote_code=trust_remote_code)
                raw_config_dict, _ = PretrainedConfig.get_config_dict(target_path, trust_remote_code=trust_remote_code)
                _repair_glm_5_2_qk_rope_head_dim(target_config, raw_config_dict)
                n_reduced = _resolve_reduced_target_layers(
                    target_config.num_hidden_layers,
                    recipe_cfg.get("target_num_hidden_layers", None),
                )
                if n_reduced is not None:
                    target_config.num_hidden_layers = n_reduced
                self.target_model = None
            architectures = list(getattr(target_config, "architectures", None) or ["GlmMoeDsaForCausalLM"])
        else:
            target_config = AutoConfig.from_pretrained(target_path, trust_remote_code=trust_remote_code)
            architectures = getattr(target_config, "architectures", []) or []
            is_gemma4_target = getattr(target_config, "model_type", "") in _GEMMA4_MODEL_TYPES

            if self.cached_target_path is None:
                target_attn_implementation = recipe_cfg.get("target_attn_implementation", None)
                target_kwargs = {}
                if target_attn_implementation is not None:
                    target_kwargs["attn_implementation"] = target_attn_implementation
                self.target_model = NeMoAutoModelForCausalLM.from_pretrained(
                    target_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=self.compute_dtype,
                    force_hf=bool(recipe_cfg.get("target_force_hf", False)),
                    **target_kwargs,
                )
                self.target_model.to(self.device)
            else:
                self.target_model = None
        if self.target_model is not None:
            self.target_model.requires_grad_(False)

        # Resolve the captured target layers once and share them between the
        # target wrapper (what to capture) and the draft config (the ``fc`` input
        # width) so the two never disagree.
        # Gemma4 and MiniMax M3 VL nest their text fields (layer count, vocab)
        # under text_config.
        target_text_config = target_config.text_config if (is_gemma4_target or is_minimax_m3_target) else target_config
        num_target_layers = int(target_text_config.num_hidden_layers)
        draft_num_hidden_layers = int(recipe_cfg.get("draft_num_hidden_layers", 5))
        target_layer_ids = list(
            recipe_cfg.get("target_layer_ids", None)
            or build_target_layer_ids(num_target_layers, draft_num_hidden_layers)
        )
        target_layer_ids = validate_target_layer_ids(target_layer_ids, num_target_layers)
        # HFDSparkTargetModel validates target_layer_ids against the actual (possibly
        # reduced) layer count via common.validate_target_layer_ids, which also accepts
        # -1 (the embedding output) and enforces strictly-increasing ids.
        self.target_layer_ids = target_layer_ids
        self.target_wrapper = (
            HFDSparkTargetModel(self.target_model, target_layer_ids=target_layer_ids)
            if self.target_model is not None
            else None
        )

        self.block_size = int(recipe_cfg.get("block_size", 7))
        self.num_anchors = int(recipe_cfg.get("num_anchors", 512))
        self.mask_token_id = self._resolve_mask_token_id(recipe_cfg, target_text_config.vocab_size)

        embed_src = None
        head_src = None
        if self.cached_target_path is None:
            if is_multimodal:
                # MiniMax M3's vision_tower is its own FSDP2-sharded unit, so a batch
                # mixing text-only and image-containing samples across DP ranks would
                # desync the FSDP2 all-gather collective and hang training.
                # dspark_vlm_collate_fn injects a masked fake image into any text-only
                # example (mirroring default_collate_fn's own fake-image handling),
                # so mixed corpora are safe here without any dataset curation.
                self.processor = build_minimax_m3_vl_processor(target_path, trust_remote_code=trust_remote_code)
                self.train_dataloader = build_dspark_vlm_dataloader(
                    dataset_cfg=self.cfg.dataset,
                    processor=self.processor,
                    batch_size=recipe_cfg.micro_batch_size,
                    max_length=recipe_cfg.seq_length,
                    shuffle=True,
                    num_workers=recipe_cfg.get("num_workers", 0),
                    distributed=self.dist_env.world_size > 1,
                )
                self.val_dataloader = None
                if self.cfg.get("val_dataset", None) is not None:
                    self.val_dataloader = build_dspark_vlm_dataloader(
                        dataset_cfg=self.cfg.val_dataset,
                        processor=self.processor,
                        batch_size=recipe_cfg.micro_batch_size,
                        max_length=recipe_cfg.seq_length,
                        shuffle=False,
                        num_workers=recipe_cfg.get("num_workers", 0),
                        distributed=self.dist_env.world_size > 1,
                    )
            else:
                self.train_dataloader = build_eagle3_dataloader(
                    data_path=recipe_cfg.train_data_path,
                    tokenizer=self.tokenizer,
                    seq_length=recipe_cfg.seq_length,
                    batch_size=recipe_cfg.micro_batch_size,
                    shuffle=True,
                    num_workers=recipe_cfg.get("num_workers", 0),
                    split=recipe_cfg.get("train_split", None),
                    distributed=self.dist_env.world_size > 1,
                    shuffle_seed=recipe_cfg.get("shuffle_seed", 42),
                    mask_reasoning_content=recipe_cfg.get("mask_reasoning_content", False),
                )
                self.val_dataloader = None
                if recipe_cfg.get("val_data_path", None):
                    self.val_dataloader = build_eagle3_dataloader(
                        data_path=recipe_cfg.val_data_path,
                        tokenizer=self.tokenizer,
                        seq_length=recipe_cfg.seq_length,
                        batch_size=recipe_cfg.micro_batch_size,
                        shuffle=False,
                        num_workers=recipe_cfg.get("num_workers", 0),
                        split=recipe_cfg.get("val_split", None),
                        distributed=self.dist_env.world_size > 1,
                        shuffle_seed=recipe_cfg.get("shuffle_seed", 42),
                        mask_reasoning_content=recipe_cfg.get("mask_reasoning_content", False),
                    )
        else:
            manifest = read_manifest(self.cached_target_path)
            _validate_cached_dspark_manifest(
                self.cached_target_path,
                manifest,
                target_text_config,
                target_layer_ids,
                target_model=target_path,
                target_model_type=target_model_type,
                seq_length=recipe_cfg.seq_length,
                compute_dtype=self.compute_dtype,
            )
            embed_src, head_src = read_target_weight_modules(self.cached_target_path)
            self.train_dataloader = build_cached_dspark_dataloader(
                cache_dir=self.cached_target_path,
                batch_size=recipe_cfg.micro_batch_size,
                shuffle=True,
                num_workers=recipe_cfg.get("num_workers", 0),
                distributed=self.dist_env.world_size > 1,
            )
            self.val_dataloader = None
            if (
                recipe_cfg.get("val_data_path", None) is not None or self.cfg.get("val_dataset", None) is not None
            ) and self.dist_env.is_main:
                logger.warning(
                    "DSpark cached_target_path is set; validation data is ignored because the target model is not loaded."
                )
            if self.dist_env.is_main:
                logger.info(
                    "DSpark OFFLINE cache: streaming %d precomputed samples from %s (target model not loaded).",
                    len(self.train_dataloader.dataset),
                    self.cached_target_path,
                )

        # The Qwen3 / Gemma4 drafts consume a flex_attention BlockMask during training.
        # The DeepSeek V4 and GLM-5.2 drafts instead consume a dense additive mask
        # (the DFlash SDPA path), so they are exempt from the flex_attention requirement.
        attention_backend = recipe_cfg.get("attention_backend", "flex_attention")
        if not (is_deepseek_v4_target or is_glm_5_2_target) and attention_backend != "flex_attention":
            raise ValueError(f"DSpark training requires attention_backend='flex_attention', got {attention_backend!r}.")
        confidence_head_alpha = float(recipe_cfg.get("confidence_head_alpha", 1.0))
        markov_rank = int(recipe_cfg.get("markov_rank", 256))

        if is_deepseek_v4_target or is_glm_5_2_target or is_gemma4_target or is_minimax_m3_target:
            # Gemma4, DeepSeek V4, GLM-5.2, and MiniMax M3 drafts share one typed
            # draft-config builder that takes the same DSpark model-args bundle.
            margs = _DraftArgs(
                num_draft_layers=draft_num_hidden_layers,
                target_layer_ids=target_layer_ids,
                block_size=self.block_size,
                num_anchors=self.num_anchors,
                mask_token_id=self.mask_token_id,
                markov_rank=markov_rank,
                markov_head_type=str(recipe_cfg.get("markov_head_type", "vanilla")),
                confidence_head_alpha=confidence_head_alpha,
                confidence_head_with_markov=bool(recipe_cfg.get("confidence_head_with_markov", True)),
            )
            if is_deepseek_v4_target:
                # The V4 draft is always dense and fixes _attn_implementation to "sdpa"
                # inside the builder, so it is not overridden by attention_backend.
                draft_config_obj = build_deepseek_v4_draft_config(target_config, margs)
            elif is_glm_5_2_target:
                # The GLM draft is always dense and fixes _attn_implementation to "sdpa"
                # inside the builder, so it is not overridden by attention_backend.
                draft_config_obj = build_glm_5_2_draft_config(target_config, margs)
            elif is_minimax_m3_target:
                # MiniMax M3 draft is built from the target's text sub-config (text_config).
                draft_config_obj = build_minimax_m3_draft_config(target_config, margs)
                draft_config_obj._attn_implementation = attention_backend
            else:
                # Gemma4 draft is built from the target's text sub-config (text_config).
                draft_config_obj = build_gemma4_draft_config(target_config, margs)
                draft_config_obj._attn_implementation = attention_backend
        else:
            # Qwen3-style draft: a small non-causal stack reusing the target's
            # architecture defaults plus the DSpark-specific fields.
            draft_config = target_config.to_dict()
            draft_config["architectures"] = ["Qwen3DSparkModel"]
            draft_config["num_hidden_layers"] = draft_num_hidden_layers
            draft_config["layer_types"] = ["full_attention"] * draft_num_hidden_layers
            draft_config["max_window_layers"] = draft_num_hidden_layers
            draft_config["num_target_layers"] = num_target_layers
            draft_config["target_layer_ids"] = target_layer_ids
            draft_config["block_size"] = self.block_size
            draft_config["num_anchors"] = self.num_anchors
            draft_config["mask_token_id"] = self.mask_token_id
            draft_config["markov_rank"] = markov_rank
            if markov_rank > 0:
                draft_config["markov_head_type"] = str(recipe_cfg.get("markov_head_type", "vanilla"))
            draft_config["enable_confidence_head"] = confidence_head_alpha > 0.0
            if confidence_head_alpha > 0.0:
                draft_config["confidence_head_with_markov"] = bool(recipe_cfg.get("confidence_head_with_markov", True))
            # The draft owns an independent (frozen) lm_head seeded from the target.
            draft_config["tie_word_embeddings"] = False
            draft_config_obj = Qwen3Config.from_dict(draft_config)
            draft_config_obj._attn_implementation = attention_backend

        draft_cls = resolve_dspark_draft_spec(architectures).draft_cls
        self.draft_model = draft_cls(draft_config_obj).to(device=self.device, dtype=self.compute_dtype)

        # training only the backbone, fc, Markov head, and confidence head.
        if embed_src is None or head_src is None:
            embed_src = self.target_wrapper.get_input_embeddings()
            head_src = self.target_wrapper.get_output_embeddings()
        if (is_deepseek_v4_target or is_glm_5_2_target or is_minimax_m3_target) and self.cached_target_path is None:
            # The V4 / GLM / MiniMax M3 target's embed_tokens / lm_head are expert-parallel /
            # FSDP-sharded DTensors; gather them to full tensors before the draft copies them.
            embed_src = _gather_full_weight_module(embed_src)
            head_src = _gather_full_weight_module(head_src)
        self.draft_model.initialize_embeddings_and_head(
            embed_tokens=embed_src,
            lm_head=head_src,
            freeze=bool(recipe_cfg.get("freeze_embeddings", True)),
        )

        dist_cfg = self.cfg.get("distributed", None)
        activation_checkpointing = dist_cfg.get("activation_checkpointing", False) if dist_cfg is not None else False
        # The target consumes this setting through its distributed setup, while
        # the separately constructed trainable draft must be wrapped explicitly.
        _apply_draft_activation_checkpointing(self.draft_model, activation_checkpointing)

        trainer_module = DSparkTrainerModule(
            self.draft_model,
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", None),
            ce_loss_alpha=float(recipe_cfg.get("ce_loss_alpha", 0.1)),
            l1_loss_alpha=float(recipe_cfg.get("l1_loss_alpha", 0.9)),
            confidence_head_alpha=confidence_head_alpha,
        ).to(self.device)
        # Multi-GPU strategy: FSDP2 (default) shards the draft per block, or DDP.
        self.parallel_strategy = "ddp"
        if self.dist_env.world_size > 1:
            strategy = dist_cfg.get("strategy", "fsdp2") if dist_cfg is not None else "fsdp2"
            self.parallel_strategy = strategy
            if strategy == "fsdp2":
                from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

                mp_policy = MixedPrecisionPolicy(param_dtype=self.compute_dtype, reduce_dtype=torch.float32)
                for layer in trainer_module.draft_model.layers:
                    fully_shard(layer, mp_policy=mp_policy)
                fully_shard(trainer_module, mp_policy=mp_policy)
            elif strategy == "ddp":
                trainer_module = DistributedDataParallel(
                    trainer_module,
                    device_ids=[self.device.index] if self.device.type == "cuda" else None,
                    output_device=self.device.index if self.device.type == "cuda" else None,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
            else:
                raise ValueError(f"Unsupported distributed.strategy={strategy!r}; use 'fsdp2' or 'ddp'.")
        self.trainer_module = trainer_module

        opt_cfg = self.cfg.optimizer
        self.peak_lr = float(opt_cfg.lr)
        self.optimizer = _build_dspark_optimizer(self.trainer_module, opt_cfg, device_mesh=None)
        logger.info(
            "Optimizer=%s lr=%.3e master_weights=%s master_weight_dtype=%s "
            "store_param_remainders=%s exp_avg_dtype=%s exp_avg_sq_dtype=%s",
            type(self.optimizer).__name__,
            self.peak_lr,
            getattr(self.optimizer, "master_weights", False),
            getattr(self.optimizer, "master_weight_dtype", None),
            getattr(self.optimizer, "store_param_remainders", False),
            getattr(self.optimizer, "exp_avg_dtype", None),
            getattr(self.optimizer, "exp_avg_sq_dtype", None),
        )
        self.grad_accumulation_steps = recipe_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = recipe_cfg.get("max_grad_norm", 1.0)
        self.num_epochs = recipe_cfg.num_epochs
        self.log_every_steps = recipe_cfg.get("log_every_steps", 10)
        self.ckpt_every_steps = recipe_cfg.get("ckpt_every_steps", None)
        self.save_checkpoint_every_epoch = recipe_cfg.get("save_checkpoint_every_epoch", False)
        self.output_dir = pathlib.Path(recipe_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dist_cfg = self.cfg.get("distributed", None)
        self.defer_fsdp_grad_sync = bool(dist_cfg.get("defer_fsdp_grad_sync", True)) if dist_cfg is not None else True
        self.metric_logger = build_metric_logger(str(self.output_dir / "dspark_train_metrics.jsonl"))

        try:
            num_batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            num_batches_per_epoch = 0
        total_optim_steps = max(
            1, self.num_epochs * optim_steps_per_epoch(num_batches_per_epoch, self.grad_accumulation_steps)
        )
        warmup_ratio = float(opt_cfg.get("warmup_ratio", 0.05))
        min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.1))
        warmup_steps = _resolve_warmup_steps(warmup_ratio, total_optim_steps)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_warmup_cosine_schedule(warmup_steps, total_optim_steps, min_lr_ratio)
        )
        self.total_optim_steps = total_optim_steps
        self.runtime = SimpleNamespace(global_step=0)
        self._resume_epoch = 0

        self.rng = StatefulRNG(seed=int(recipe_cfg.get("shuffle_seed", 42)), ranked=self.dist_env.world_size > 1)
        self._build_checkpointer(target_path)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))

        self.wandb_run = _init_dspark_wandb(
            is_main=self.dist_env.is_main,
            wandb_cfg=self.cfg.get("wandb", None),
            cfg_dict=self.cfg.to_dict(),
            default_name="dspark_" + str(target_path).rstrip("/").split("/")[-1],
        )

    @staticmethod
    def _resolve_mask_token_id(recipe_cfg, vocab_size: int) -> int:
        """Resolve and validate the MASK token id filling non-anchor block positions.

        The draft's ``embed_tokens`` row at this id is the learned "predict here"
        signal. It must be a deliberately chosen reserved / unused token id (never a
        silent fallback to ``pad``, which is commonly aliased to ``eos``), and the
        inference runtime must fill block slots with the same id.
        """
        mask_token_id = recipe_cfg.get("mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "DSpark requires recipe_args.mask_token_id to be set explicitly (the token used for "
                "non-anchor block positions). Pick a reserved / rarely-used token id so the mask-slot "
                "embedding does not collide with real content, and use the same id in the inference runtime."
            )
        mask_token_id = int(mask_token_id)
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError(
                f"mask_token_id={mask_token_id} is out of range for the vocab [0, {vocab_size}); "
                "it indexes the draft embed_tokens table."
            )
        return mask_token_id

    def _build_deepseek_v4_target(self, target_path, recipe_cfg, trust_remote_code):
        """Load the full DeepSeek V4 target as a frozen, expert-parallel / FSDP model.

        Mirrors the V4 finetune recipe's model build: an expert-parallel / FSDP
        distributed setup (device_mesh + moe_mesh, derived from the recipe's
        ``distributed`` block) shards the 256 experts across ranks while the
        ``enable_hf_state_dict_adapter`` path dequantizes the FP8 base weights on load.
        The target is MTP-free (``num_nextn_predict_layers=0``) because the draft
        consumes hidden states only, and is returned frozen for inference.

        Returns the resolved ``DeepseekV4Config`` and the sharded target model.
        """
        if self.device.type != "cuda":
            raise RuntimeError(
                "DeepSeek V4 DSpark target training requires CUDA: the target is loaded "
                "with the expert-parallel / FSDP distributed path."
            )
        # Build the device_mesh + moe_mesh from the recipe's `distributed` block
        # (strategy, ep_size, moe, ...). DeepseekV4Config.from_pretrained is used
        # because the custom V4 model_type is not registered with stock AutoConfig.
        self.distributed_setup = create_distributed_setup_from_config(
            self.cfg,
            world_size=self.dist_env.world_size,
        )
        # Pass name_or_path explicitly (as the V4 finetune recipe does) so from_config
        # resolves the base checkpoint to load and dequantize from.
        target_config = DeepseekV4Config.from_pretrained(
            target_path, name_or_path=target_path, num_nextn_predict_layers=0
        )
        # Diagnostic / CI knob: a full 43-layer V4-Flash target dequantizes to
        # ~63 GiB of experts per rank at ep_size=8 and does NOT fit on a single
        # 8x80GB node (it OOMs in the expert dequant before the first forward).
        # Shrinking the layer count loads only the first N layers, so the entire
        # EP / hidden-capture / draft-training path can be exercised end to end on
        # one node. ``target_layer_ids`` must then point at layers that exist in
        # the reduced stack (e.g. [1, 2, 3] for N=4). This is a validation aid:
        # a draft trained against a reduced target is not a usable drafter for the
        # full model. Leave it unset for real training (use multi-node ep_size).
        n_reduced = _resolve_reduced_target_layers(
            target_config.num_hidden_layers, recipe_cfg.get("target_num_hidden_layers", None)
        )
        if n_reduced is not None:
            logger.warning(
                "Reducing the DeepSeek V4 target from %d to %d layers "
                "(target_num_hidden_layers): diagnostic/CI only, not a usable drafter.",
                target_config.num_hidden_layers,
                n_reduced,
            )
            target_config.num_hidden_layers = n_reduced
        backend = self._build_deepseek_v4_backend(recipe_cfg)
        target_model = NeMoAutoModelForCausalLM.from_config(
            config=target_config,
            backend=backend,
            distributed_setup=self.distributed_setup,
            load_base_model=True,
            torch_dtype=self.compute_dtype,
            trust_remote_code=trust_remote_code,
        )
        return target_config, target_model

    @staticmethod
    def _build_deepseek_v4_backend(recipe_cfg) -> BackendConfig:
        """Build the V4 target BackendConfig (TileLang attention, hybrid-EP, FP8 adapter).

        Matches the V4 finetune recipe's backend: dense linears and fp32 RMSNorm on
        torch, the ``torch_mm`` grouped-expert GEMM, the hybrid-EP token dispatcher, and
        the HF state-dict adapter that dequantizes the FP8 base checkpoint on load.
        """
        return BackendConfig(
            attn=str(recipe_cfg.get("target_attn_backend", "tilelang")),
            linear="torch",
            rms_norm="torch_fp32",
            rope_fusion=False,
            dispatcher=str(recipe_cfg.get("target_dispatcher", "hybridep")),
            experts=str(recipe_cfg.get("target_experts", "torch_mm")),
            enable_hf_state_dict_adapter=True,
            enable_fsdp_optimizations=bool(recipe_cfg.get("target_enable_fsdp_optimizations", True)),
        )

    def _build_glm_5_2_target(self, target_path, recipe_cfg, trust_remote_code):
        """Load the full GLM-5.2 target as a frozen, expert-parallel / FSDP model.

        Mirrors ``_build_deepseek_v4_target``: an expert-parallel / FSDP distributed
        setup (derived from the recipe's ``distributed`` block) shards the 256 routed
        experts across ranks. GLM-5.2's ``model_type`` is registered, so ``AutoConfig``
        resolves it directly (unlike DeepSeek V4), but the model must still be built
        with ``from_config`` + ``load_base_model=True``: ``from_pretrained`` re-reads
        the checkpoint's own config and silently rebuilds the full 78-layer target,
        discarding the ``target_num_hidden_layers`` reduction (which OOMs on one node).
        ``from_config`` keeps the (possibly reduced, repaired) config and still loads
        the checkpoint weights, resolved via ``config.name_or_path``.

        DSpark's forward-hook hidden-state capture needs one non-pipelined
        ``self.model(...)`` call, so ``pp_size`` must be 1 in the recipe's
        ``distributed:`` block; use a larger ``ep_size`` instead of PP to shard the
        parameter memory.

        Returns the resolved (repaired) target config and the sharded target model.
        """
        if self.device.type != "cuda":
            raise RuntimeError(
                "GLM-5.2 DSpark target training requires CUDA: the target is loaded "
                "with the expert-parallel / FSDP distributed path."
            )
        target_config = AutoConfig.from_pretrained(target_path, trust_remote_code=trust_remote_code)
        # The published config's head_dim=192 clobbers qk_rope_head_dim on load via the
        # HF attribute_map, breaking checkpoint shape validation (see the helper).
        raw_config_dict, _ = PretrainedConfig.get_config_dict(target_path, trust_remote_code=trust_remote_code)
        _repair_glm_5_2_qk_rope_head_dim(target_config, raw_config_dict)
        n_reduced = _resolve_reduced_target_layers(
            target_config.num_hidden_layers,
            recipe_cfg.get("target_num_hidden_layers", None),
        )
        if n_reduced is not None:
            logger.warning(
                "Reducing the GLM-5.2 target from %d to %d layers "
                "(target_num_hidden_layers): diagnostic/CI only, not a usable drafter.",
                target_config.num_hidden_layers,
                n_reduced,
            )
            target_config.num_hidden_layers = n_reduced
        self.distributed_setup = create_distributed_setup_from_config(
            self.cfg,
            world_size=self.dist_env.world_size,
        )
        backend = self._build_glm_5_2_backend(recipe_cfg)
        target_model = NeMoAutoModelForCausalLM.from_config(
            config=target_config,
            backend=backend,
            distributed_setup=self.distributed_setup,
            load_base_model=True,
            torch_dtype=self.compute_dtype,
            trust_remote_code=trust_remote_code,
        )
        return target_config, target_model

    @staticmethod
    def _build_glm_5_2_backend(recipe_cfg) -> BackendConfig:
        """Build the GLM-5.2 target BackendConfig (mirrors the GLM finetune recipe).

        Dense linears and fp32 RMSNorm on torch, an fp32 router gate (the GLM/DeepSeek-V3
        top-k routing is fp32 for stability), the ``torch_mm`` grouped-expert GEMM, the
        hybrid-EP token dispatcher, and the HF state-dict adapter that maps (and
        dequantizes, when the base checkpoint is FP8) the HF weights on load. The target
        is frozen / forward-only, so SDPA attention is used by default (the DSA indexer
        emits an additive float bias that SDPA's explicit-mask path accepts);
        ``target_attn_backend=tilelang`` switches to the fused sparse kernels when a
        TileLang build is available. ``target_experts`` defaults to ``torch_mm`` (like
        the V4 DSpark backend) because ``gmm`` needs the optional ``grouped_gemm``
        package, which the current AutoModel image does not ship (it fails with
        ``NameError: ops is not defined``).
        """
        return BackendConfig(
            attn=str(recipe_cfg.get("target_attn_backend", "sdpa")),
            # The target is frozen / forward-only, so plain torch linears (which also keep
            # embed_tokens / lm_head as plain-shaped weights) are used, matching the V4 /
            # MiniMax M3 frozen-target backends.
            linear="torch",
            rms_norm="torch_fp32",
            rope_fusion=False,
            gate_precision="float32",
            dispatcher=str(recipe_cfg.get("target_dispatcher", "hybridep")),
            experts=str(recipe_cfg.get("target_experts", "torch_mm")),
            enable_hf_state_dict_adapter=True,
            enable_fsdp_optimizations=bool(recipe_cfg.get("target_enable_fsdp_optimizations", True)),
        )

    def _build_checkpointer(self, target_path: str) -> None:
        """Build the checkpointer using the same plumbing as the EAGLE / DFlash recipes."""
        ckpt_cfg = self.cfg.get("checkpoint", None)
        default_dir = str(self.output_dir / "checkpoints")
        draft_state_dict_keys = list(self.draft_model.state_dict().keys())
        ckpt_kwargs = dict(
            enabled=True,
            checkpoint_dir=default_dir,
            model_save_format="safetensors",
            model_repo_id=str(target_path),
            model_cache_dir=hf_constants.HF_HUB_CACHE,
            save_consolidated=True,
            is_peft=False,
            model_state_dict_keys=draft_state_dict_keys,
        )
        if ckpt_cfg is not None:
            user_cfg = ckpt_cfg.to_dict() if hasattr(ckpt_cfg, "to_dict") else dict(ckpt_cfg)
            user_cfg.pop("restore_from", None)
            ckpt_kwargs.update(user_cfg)
        if ckpt_kwargs.get("model_state_dict_keys") is None:
            ckpt_kwargs["model_state_dict_keys"] = draft_state_dict_keys

        self.checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.checkpointer = Checkpointer(
            config=self.checkpoint_config, dp_rank=dp_rank, tp_rank=0, pp_rank=0, moe_mesh=None
        )

    def _module(self):
        return (
            self.trainer_module.module
            if isinstance(self.trainer_module, DistributedDataParallel)
            else self.trainer_module
        )

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float | None = None,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
        is_final_checkpoint: bool = False,
    ) -> None:
        """Persist the DSpark draft model, optimizer, scheduler, RNG, and meta."""
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        self.checkpointer.async_wait()

        prev_pending = getattr(self, "_last_pending_checkpoint_dir", None)
        prev_best_pending = getattr(self, "_last_pending_best_checkpoint_info", None)

        ckpt_root = self.checkpoint_config.checkpoint_dir
        path = os.path.join(str(ckpt_root), f"epoch_{epoch}_step_{step}")
        is_dist_initialized = dist.is_initialized()
        is_rank_0 = (not is_dist_initialized) or dist.get_rank() == 0
        best_val_metric = (
            val_loss.get(next(iter(val_loss.keys())) if len(val_loss) == 1 else best_metric_key) if val_loss else None
        )

        if prev_pending is not None:
            if is_rank_0:
                self._update_latest_symlink(prev_pending)
            setattr(self, "_last_pending_checkpoint_dir", None)
            if is_dist_initialized:
                dist.barrier()
        if prev_best_pending is not None:
            if is_rank_0 and prev_best_pending.get("val") is not None:
                self._update_best_symlink(prev_best_pending["path"], float(prev_best_pending["val"]))
            setattr(self, "_last_pending_best_checkpoint_info", None)
            if is_dist_initialized:
                dist.barrier()

        if is_rank_0:
            if os.path.exists(path):
                raise FileExistsError(f"Checkpoint directory {path} already exists")
            os.makedirs(path, exist_ok=True)
            loss_dict: dict[str, float] = {}
            if train_loss is not None:
                loss_dict["train_loss"] = float(train_loss)
            if val_loss:
                for k, v in val_loss.items():
                    loss_dict[k] = float(v)
            if loss_dict:
                with open(os.path.join(path, "losses.json"), "w") as f:
                    json.dump(loss_dict, f)
        if is_dist_initialized:
            dist.barrier()

        draft_model = self._module().draft_model
        self.checkpointer.save_model(
            draft_model,
            path,
            tokenizer=self.tokenizer,
            is_final_checkpoint=is_final_checkpoint,
        )
        self.checkpointer.save_optimizer(self.optimizer, draft_model, path, self.lr_scheduler)
        self.checkpointer.save_on_dp_ranks(self.rng, "rng", path)

        if is_rank_0:
            self._save_extra_state(path, epoch=epoch)
            try:
                save_config(self.cfg.raw_config, path)
            except (AttributeError, OSError) as e:
                logger.warning("Failed to save config snapshot: %s", e)
        if is_dist_initialized:
            dist.barrier()

        if getattr(self.checkpointer.config, "is_async", False):
            setattr(self, "_last_pending_checkpoint_dir", path)
            if best_val_metric is not None:
                setattr(self, "_last_pending_best_checkpoint_info", {"path": path, "val": float(best_val_metric)})
        else:
            if is_rank_0:
                self._update_latest_symlink(path)
                if best_val_metric is not None:
                    self._update_best_symlink(path, float(best_val_metric))
            if is_dist_initialized:
                dist.barrier()

    def _save_extra_state(self, path: str, epoch: int) -> None:
        """Persist DSpark meta: global_step, epoch, block_size, mask, and target layers."""
        torch.save(
            {
                "global_step": self.runtime.global_step,
                "epoch": int(epoch),
                "block_size": self.block_size,
                "num_anchors": self.num_anchors,
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": list(self.target_layer_ids),
            },
            os.path.join(path, "dspark_meta.pt"),
        )

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Restore the DSpark draft model, optimizer, scheduler, RNG, and global_step."""
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        is_rank_0 = (not dist.is_initialized()) or dist.get_rank() == 0
        ckpt_root = self.checkpoint_config.checkpoint_dir

        if restore_from:
            ckpt_dir = _resolve_restore_from_to_ckpt_dir(ckpt_root, restore_from)
            if ckpt_dir is None:
                if is_rank_0:
                    logger.warning("restore_from='LATEST' but no checkpoint found in %s", ckpt_root)
                return
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
        else:
            auto = _find_latest_checkpoint(ckpt_root)
            if auto is None:
                return
            ckpt_dir = str(auto)

        ok, reason = _is_checkpoint_model_config_compatible(self.cfg, ckpt_dir)
        if not ok and not restore_from:
            if is_rank_0:
                logger.warning(
                    "Auto-detected checkpoint at %s is incompatible: %s. Skipping restore.", ckpt_dir, reason
                )
            return

        if is_rank_0:
            logger.info("Resuming from checkpoint: %s", ckpt_dir)

        draft_model = self._module().draft_model
        self.checkpointer.load_model(draft_model, os.path.join(ckpt_dir, "model"))
        self.checkpointer.load_optimizer(self.optimizer, draft_model, ckpt_dir, self.lr_scheduler)
        try:
            self.checkpointer.load_on_dp_ranks(self.rng, "rng", ckpt_dir)
        except FileNotFoundError:
            logger.warning("RNG state not found in %s; continuing without restoring RNG.", ckpt_dir)
        self._load_extra_state(ckpt_dir)

    def _load_extra_state(self, ckpt_dir: str) -> None:
        """Restore DSpark meta: global_step and epoch."""
        meta_path = os.path.join(ckpt_dir, "dspark_meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False, map_location="cpu")
            self.runtime.global_step = int(meta.get("global_step", 0))
            self._resume_epoch = int(meta.get("epoch", 0))

    def _log_saved_checkpoint(self, kind: str, epoch: int, step: int) -> None:
        """Log a saved checkpoint on rank 0 when checkpointing is enabled."""
        ckpt_cfg = getattr(self, "checkpoint_config", None)
        if self.dist_env.is_main and ckpt_cfg is not None and ckpt_cfg.enabled:
            logger.info("Saved %s checkpoint to %s/epoch_%d_step_%d", kind, ckpt_cfg.checkpoint_dir, epoch, step)

    def _forward_batch(self, batch):
        """Run one batch through live target capture or the offline cache."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        if self.target_wrapper is None:
            batch["target_hidden_states"] = batch["target_hidden_states"].to(self.compute_dtype)
            batch["target_last_hidden_states"] = batch["target_last_hidden_states"].to(self.compute_dtype)
            return self.trainer_module(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=batch["target_last_hidden_states"],
            )
        target_batch = self.target_wrapper.generate_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            **_extract_mm_kwargs(batch),
        )
        return self.trainer_module(
            input_ids=target_batch.input_ids,
            target_hidden_states=target_batch.target_hidden_states,
            loss_mask=target_batch.loss_mask,
            target_last_hidden_states=target_batch.target_last_hidden_states,
        )

    def _maybe_save_step_checkpoint(self, epoch: int) -> bool:
        """Save a checkpoint mid-epoch when ``ckpt_every_steps`` is configured."""
        every = getattr(self, "ckpt_every_steps", None)
        if every is None or every <= 0 or self.runtime.global_step % every != 0:
            return False
        total_optim_steps = getattr(self, "total_optim_steps", None)
        is_final_checkpoint = total_optim_steps is not None and self.runtime.global_step >= total_optim_steps
        self.save_checkpoint(
            epoch=epoch,
            step=self.runtime.global_step,
            best_metric_key="val_loss",
            is_final_checkpoint=is_final_checkpoint,
        )
        self._log_saved_checkpoint("step", epoch, self.runtime.global_step)
        return True

    def _maybe_save_final_checkpoint(self, completed_epochs: int) -> bool:
        """Always save the fully-trained model at the end, unless a cadence already saved the final step."""
        gs = self.runtime.global_step
        if gs <= 0:
            return False
        every = getattr(self, "ckpt_every_steps", None)
        saved_by_step = bool(every and every > 0 and gs % every == 0)
        saved_by_epoch = bool(getattr(self, "save_checkpoint_every_epoch", False))
        if saved_by_step or saved_by_epoch:
            return False
        self.save_checkpoint(epoch=completed_epochs, step=gs, best_metric_key="val_loss", is_final_checkpoint=True)
        self._log_saved_checkpoint("final", completed_epochs, gs)
        return True

    def _run_eval(self):
        if self.val_dataloader is None:
            return None
        self.trainer_module.eval()
        total_loss = torch.zeros((), device=self.device)
        total_batches = torch.zeros((), device=self.device)
        with torch.no_grad():
            for batch in self.val_dataloader:
                metrics = self._forward_batch(batch)
                total_loss += metrics.loss.detach()
                total_batches += 1
        total_loss = self._dp_allreduce(total_loss)
        total_batches = self._dp_allreduce(total_batches)
        self.trainer_module.train()
        return {"val_loss": (total_loss / total_batches.clamp_min(1)).item()}

    def _wandb_log(self, data: dict, step: int) -> None:
        """Log rank-zero metrics when a W&B run is active."""
        run = getattr(self, "wandb_run", None)
        if run is not None:
            run.log(data, step=step)

    def _finish_wandb(self) -> None:
        run = getattr(self, "wandb_run", None)
        if run is None:
            return
        try:
            run.finish()
        except Exception:
            logger.warning("Failed to finish W&B run cleanly.", exc_info=True)
        finally:
            self.wandb_run = None

    def run_train_validation_loop(self):
        """Run the DSpark training loop."""
        self.trainer_module.train()
        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        if start_epoch >= self.num_epochs:
            if self.dist_env.is_main:
                logger.info("All %d epochs already completed; nothing to do.", self.num_epochs)
            if getattr(self, "metric_logger", None) is not None:
                self.metric_logger.close()
            self._finish_wandb()
            return

        pbar = self._make_progress_bar(total=self.total_optim_steps, initial=self.runtime.global_step)
        try:
            for epoch_idx in range(start_epoch, self.num_epochs):
                if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(epoch_idx)

                running_loss = 0.0
                running_ce = 0.0
                running_l1 = 0.0
                running_conf = 0.0
                running_accept = 0.0
                running_tau = 0.0
                running_conf_abs_err = 0.0
                running_conf_bias = 0.0
                running_conf_cumprod_bias = 0.0
                running_micro = 0
                epoch_loss = 0.0
                micro_step = 0
                pending_micro_batches = 0
                completed_steps = 0
                last_batch_idx = -1
                num_batches = len(self.train_dataloader)
                for batch_idx, batch in enumerate(self.train_dataloader):
                    last_batch_idx = batch_idx
                    is_optim_step = (pending_micro_batches + 1 == self.grad_accumulation_steps) or (
                        batch_idx == num_batches - 1
                    )
                    # get_sync_ctx handles both DDP (no_sync) and FSDP2 (set_requires_gradient_sync).
                    with get_sync_ctx(self.trainer_module, is_optim_step, self.defer_fsdp_grad_sync):
                        metrics = self._forward_batch(batch)
                        loss = metrics.loss / self.grad_accumulation_steps
                        loss.backward()

                    running_loss += metrics.loss.detach().item()
                    running_ce += metrics.ce_loss.detach().item()
                    running_l1 += metrics.l1_loss.detach().item()
                    running_conf += metrics.confidence_loss.detach().item()
                    running_accept += metrics.accept_rate.detach().item()
                    running_tau += metrics.tau.detach().item()
                    running_conf_abs_err += metrics.confidence_abs_error.detach().item()
                    running_conf_bias += metrics.confidence_bias.detach().item()
                    running_conf_cumprod_bias += metrics.confidence_cumprod_bias.detach().item()
                    running_micro += 1
                    epoch_loss += metrics.loss.detach().item()
                    micro_step += 1
                    pending_micro_batches += 1

                    if pending_micro_batches == self.grad_accumulation_steps:
                        torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.lr_scheduler.step()
                        self.runtime.global_step += 1
                        if pbar is not None:
                            pbar.update(1)
                        completed_steps += 1
                        pending_micro_batches = 0
                        self._maybe_save_step_checkpoint(epoch_idx)

                        if self.runtime.global_step % self.log_every_steps == 0:
                            # One collective: window sums of the loss terms and the acceptance
                            # diagnostics plus the micro-batch count, summed across DP ranks,
                            # then divided -> global means.
                            window = self._dp_allreduce(
                                torch.tensor(
                                    [
                                        running_loss,
                                        running_ce,
                                        running_l1,
                                        running_conf,
                                        running_accept,
                                        running_tau,
                                        running_conf_abs_err,
                                        running_conf_bias,
                                        running_conf_cumprod_bias,
                                        float(running_micro),
                                    ],
                                    device=self.device,
                                    dtype=torch.float32,
                                )
                            ).tolist()
                            count = max(1.0, window[-1])
                            avg = {
                                "loss": window[0] / count,
                                "ce_loss": window[1] / count,
                                "l1_loss": window[2] / count,
                                "confidence_loss": window[3] / count,
                                "accept_rate": window[4] / count,
                                "tau": window[5] / count,
                                "confidence_abs_error": window[6] / count,
                                "confidence_bias": window[7] / count,
                                "confidence_cumprod_bias": window[8] / count,
                            }
                            running_loss = running_ce = running_l1 = running_conf = 0.0
                            running_accept = running_tau = 0.0
                            running_conf_abs_err = running_conf_bias = running_conf_cumprod_bias = 0.0
                            running_micro = 0
                            if self.dist_env.is_main:
                                current_lr = self.lr_scheduler.get_last_lr()[0]
                                mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                                self.metric_logger.log(
                                    MetricsSample(
                                        step=self.runtime.global_step,
                                        epoch=epoch_idx,
                                        metrics={**avg, "lr": current_lr, "mem": mem},
                                    )
                                )
                                self._wandb_log(
                                    {
                                        "train/loss": avg["loss"],
                                        "train/ce_loss": avg["ce_loss"],
                                        "train/tv_loss": avg["l1_loss"],
                                        "train/confidence_loss": avg["confidence_loss"],
                                        "train/accept_rate": avg["accept_rate"],
                                        "train/tau": avg["tau"],
                                        "train/confidence_abs_error": avg["confidence_abs_error"],
                                        "train/confidence_bias": avg["confidence_bias"],
                                        "train/confidence_cumprod_bias": avg["confidence_cumprod_bias"],
                                        "train/lr": current_lr,
                                        "train/mem_gib": mem,
                                        "train/epoch": epoch_idx,
                                    },
                                    step=self.runtime.global_step,
                                )
                                if pbar is not None:
                                    pbar.set_postfix(loss=f"{avg['loss']:.4f}", lr=f"{current_lr:.2e}")
                                logger.info(
                                    "step %d | epoch %d | loss %.4f | ce %.4f | tv %.4f | conf %.4f | "
                                    "accept %.3f | tau %.2f | lr %.2e | mem %.2f GiB",
                                    self.runtime.global_step,
                                    epoch_idx,
                                    avg["loss"],
                                    avg["ce_loss"],
                                    avg["l1_loss"],
                                    avg["confidence_loss"],
                                    avg["accept_rate"],
                                    avg["tau"],
                                    current_lr,
                                    mem,
                                )

                # Flush the trailing partial accumulation window (see EAGLE recipes
                # for the rescale rationale).
                if pending_micro_batches > 0:
                    scale = float(self.grad_accumulation_steps) / float(pending_micro_batches)
                    for p in self.trainer_module.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                    torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()
                    self.runtime.global_step += 1
                    if pbar is not None:
                        pbar.update(1)
                    completed_steps += 1
                    pending_micro_batches = 0
                    self._maybe_save_step_checkpoint(epoch_idx)

                eval_metrics = self._run_eval()
                if self.dist_env.is_main:
                    msg = f"Finished epoch {epoch_idx + 1}/{self.num_epochs} completed_steps={completed_steps}"
                    if eval_metrics is not None:
                        msg += f" val_loss={eval_metrics['val_loss']:.4f}"
                        self._wandb_log(
                            {"val/loss": eval_metrics["val_loss"], "val/epoch": epoch_idx},
                            step=self.runtime.global_step,
                        )
                    logger.info(msg)

                if getattr(self, "save_checkpoint_every_epoch", False) and last_batch_idx >= 0:
                    avg_loss = epoch_loss / max(1, micro_step) if micro_step else None
                    self.save_checkpoint(
                        epoch=epoch_idx + 1,
                        step=self.runtime.global_step,
                        train_loss=avg_loss,
                        val_loss=eval_metrics,
                        best_metric_key="val_loss",
                        is_final_checkpoint=epoch_idx + 1 >= self.num_epochs,
                    )
                    self._log_saved_checkpoint("epoch", epoch_idx + 1, self.runtime.global_step)

            self._maybe_save_final_checkpoint(self.num_epochs)
            self._finalize_pending_checkpoint()
        finally:
            if pbar is not None:
                pbar.close()
            if getattr(self, "metric_logger", None) is not None:
                self.metric_logger.close()
            self._finish_wandb()


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDSparkRecipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDSparkRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
