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

"""Fine-tuning recipe for the BAGEL multimodal family.

BAGEL uses packed mixed-modality batches and returns ``dict(ce=..., mse=...)``,
so this recipe subclasses :class:`BaseRecipe` directly instead of the standard
VLM recipe. It supports Stage 1 understanding-only CE and Stage 2 joint
understanding + visual generation with VAE encode and flow-matching MSE.

Key training-step behavior:

  - Per-token CE is reduced via ``ce.sum() * world_size / total_ce_tokens``.
  - Per-token MSE is reduced via
    ``mse.mean(dim=-1).sum() * world_size / total_mse_tokens``.
  - Optimizer: AdamW(lr=2e-5, betas=(0.9, 0.95), eps=1e-15, weight_decay=0).
  - LR schedule: constant with 2000-step warmup.
  - bf16 autocast on the forward pass; FSDP2/HSDP via AM's model
    infrastructure using the ``distributed.strategy: fsdp2`` YAML knob.
  - Global seed = 4396, data seed = 42 (BAGEL defaults).
"""

from __future__ import annotations

import logging
import pathlib
import random
import shutil
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config  # noqa: E402
from nemo_automodel.components.loggers.log_utils import setup_logging  # noqa: E402
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger  # noqa: E402
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages  # noqa: E402
from nemo_automodel.components.models.bagel.backend import resolve_bagel_backend  # noqa: E402
from nemo_automodel.components.models.bagel.hf_backbone_loader import (  # noqa: E402
    build_bagel_from_hf_backbones,
    initialize_bagel_non_backbone_weights,
    load_bagel_hf_backbone_weights,
)
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG  # noqa: E402
from nemo_automodel.components.training.step_scheduler import StepScheduler  # noqa: E402
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config  # noqa: E402
from nemo_automodel.recipes.base_recipe import BaseRecipe  # noqa: E402

try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass


logger = logging.getLogger(__name__)


# Kwargs accepted by BagelForUnifiedMultimodal.forward in Stage 1. The
# packed-batch dict emitted by ``bagel_packed_collate_fn`` carries a few extra
# keys (``batch_data_indexes``, ``ce_loss_weights``, gen-side tensors when a
# t2i/edit sample lands in the pack) that must be filtered out before the call.
_BAGEL_STAGE1_FORWARD_KEYS = {
    "sequence_length",
    "packed_text_ids",
    "packed_text_indexes",
    "sample_lens",
    "packed_position_ids",
    "nested_attention_masks",
    "split_lens",
    "attn_modes",
    "packed_vit_tokens",
    "packed_vit_token_indexes",
    "packed_vit_position_ids",
    "vit_token_seqlens",
    "ce_loss_indexes",
    "packed_label_ids",
    "ce_loss_weights",
}
# Stage 2 adds the gen-side kwargs. ``padded_images`` is converted to
# ``padded_latent`` by the recipe's VAE encode step before forward.
_BAGEL_STAGE2_FORWARD_KEYS = _BAGEL_STAGE1_FORWARD_KEYS | {
    "padded_latent",
    "patchified_vae_latent_shapes",
    "packed_latent_position_ids",
    "packed_vae_token_indexes",
    "packed_timesteps",
    "mse_loss_indexes",
}


def _load_bagel_tokenizer(model_path: str):
    """Load the BAGEL Qwen2 tokenizer and add the four special tokens.

    Returns ``(tokenizer, special_tokens_dict, num_new_tokens)``.
    """
    # Prefer transformers' native Qwen2Tokenizer; the BAGEL checkpoint ships
    # ``tokenizer.json`` + ``vocab.json`` + ``merges.txt`` which Qwen2Tokenizer
    # consumes directly.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    all_special = []
    for v in tokenizer.special_tokens_map.values():
        if isinstance(v, str):
            all_special.append(v)
        elif isinstance(v, list):
            all_special.extend(v)

    new_tokens = []
    for t in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"):
        if t not in all_special:
            new_tokens.append(t)

    num_new = tokenizer.add_tokens(new_tokens)
    special_tokens_dict = dict(
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        start_of_image=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        end_of_image=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    return tokenizer, special_tokens_dict, num_new


def _first_path(*values: Any) -> str | None:
    """Return the first non-empty string-like path from ``values``."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _resolve_bagel_artifact_path(cfg) -> str | None:
    """Resolve the BAGEL artifact source used for config/tokenizer/VAE sidecars.

    Fine-tuning usually uses ``model.pretrained_model_name_or_path``. Pretraining
    can instead use ``model.config.pretrained_model_name_or_path`` with
    ``NeMoAutoModelForMultimodalLM.from_config`` so no base weights are loaded,
    while still resolving tokenizer/config/VAE side artifacts.
    """
    model_config = cfg.get("model.config", None)
    model_config_path = model_config if isinstance(model_config, str) else None
    return _first_path(
        cfg.get("model.pretrained_model_name_or_path", None),
        cfg.get("model.config.pretrained_model_name_or_path", None),
        model_config_path,
        cfg.get("tokenizer.pretrained_model_name_or_path", None),
    )


def _resolve_bagel_tokenizer_path(cfg) -> str:
    """Resolve the tokenizer source for BAGEL training."""
    model_config = cfg.get("model.config", None)
    model_config_path = model_config if isinstance(model_config, str) else None
    tokenizer_path = _first_path(
        cfg.get("tokenizer.pretrained_model_name_or_path", None),
        cfg.get("model.pretrained_model_name_or_path", None),
        cfg.get("model.config.pretrained_model_name_or_path", None),
        model_config_path,
    )
    if tokenizer_path is None:
        raise ValueError(
            "BAGEL training needs a tokenizer source. Set either "
            "'tokenizer.pretrained_model_name_or_path', 'model.pretrained_model_name_or_path', "
            "or 'model.config.pretrained_model_name_or_path'."
        )
    return tokenizer_path


def _maybe_resize_bagel_vocab(model, tokenizer_vocab_size: int, num_new_tokens: int) -> None:
    """Resize BAGEL token embeddings only when the tokenizer is larger.

    The released BAGEL checkpoint intentionally has a padded vocab
    (``152064``) larger than the tokenizer length. Shrinking to the tokenizer
    length removes trainable rows from both ``embed_tokens`` and ``lm_head``,
    which changes the trainable parameter set.
    """
    model_vocab_size = model.model.language_model.config.vocab_size
    if tokenizer_vocab_size > model_vocab_size:
        model.model.language_model.resize_token_embeddings(tokenizer_vocab_size)
        model.model.config.text_config.vocab_size = tokenizer_vocab_size
        model.model.language_model.config.vocab_size = tokenizer_vocab_size
    elif num_new_tokens > 0 and tokenizer_vocab_size < model_vocab_size:
        logger.info(
            "Tokenizer reported %d new token(s), but tokenizer size %d is smaller than "
            "checkpoint vocab size %d; keeping padded BAGEL vocab unchanged.",
            num_new_tokens,
            tokenizer_vocab_size,
            model_vocab_size,
        )


def _load_dataset_info(cfg_ds) -> Dict[str, Any]:
    """Resolve the BAGEL-style dataset-info dict.

    Precedence (highest first):
      1. ``cfg.dataset.dataset_info_path`` — path to a YAML/JSON dataset_info.
      2. ``cfg.dataset.dataset_info`` — inline dict (rare; useful for tests).
    """
    info_path = cfg_ds.get("dataset_info_path", None)
    if info_path is not None:
        # Allow either YAML or JSON.
        p = pathlib.Path(info_path)
        with open(p, "r") as f:
            if p.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            import json

            return json.load(f)

    info_inline = cfg_ds.get("dataset_info", None)
    if info_inline is not None:
        return info_inline.to_dict() if hasattr(info_inline, "to_dict") else dict(info_inline)

    raise ValueError(
        "BAGEL training requires explicit dataset paths. Set either "
        "'dataset.dataset_info_path' or inline 'dataset.dataset_info' in the YAML config."
    )


def _resolve_bagel_vae_path(model_path: str | None, vae_path: str | None) -> str:
    """Resolve ``ae.safetensors`` from a local checkpoint directory or HF repo."""

    if vae_path is not None:
        return vae_path

    if model_path is None:
        raise ValueError(
            "BAGEL Stage 2 needs a VAE source. Set 'model.vae_path' explicitly, "
            "or provide a BAGEL artifact source via 'model.pretrained_model_name_or_path', "
            "'model.config.pretrained_model_name_or_path', or 'tokenizer.pretrained_model_name_or_path'."
        )

    local_model_path = pathlib.Path(model_path)
    local_vae_path = local_model_path / "ae.safetensors"
    if local_vae_path.is_file():
        return str(local_vae_path)

    if local_model_path.exists() or local_model_path.is_absolute() or model_path.startswith("."):
        raise FileNotFoundError(
            "BAGEL Stage 2 needs ae.safetensors next to the local model checkpoint, "
            f"but could not find {local_vae_path}. Set 'model.vae_path' explicitly."
        )

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_path, filename="ae.safetensors")


def _load_grouped_datasets(cfg_ds) -> Dict[str, Any]:
    """Load the grouped-datasets YAML (``data/configs/*.yaml``) into a dict."""
    path = cfg_ds.get("grouped_datasets_path", None)
    if path is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    inline = cfg_ds.get("grouped_datasets", None)
    if inline is None:
        raise ValueError(
            "dataset config must specify either 'grouped_datasets_path' (to a "
            "BAGEL-style data/configs/*.yaml) or an inline 'grouped_datasets' dict."
        )
    return inline.to_dict() if hasattr(inline, "to_dict") else dict(inline)


def _load_bagel_vae(vae_path: str) -> tuple[Any, Any]:
    """Load BAGEL's autoencoder from AM-owned code."""

    from nemo_automodel.components.models.bagel.autoencoder import load_bagel_autoencoder

    return load_bagel_autoencoder(vae_path)


class FinetuneRecipeForMultimodal(BaseRecipe):
    """Fine-tuning recipe for BAGEL packed Stage 1/Stage 2 training.

    Subclasses :class:`BaseRecipe` directly (rather than
    :class:`~nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM`)
    because BAGEL's packed-sequence input schema and
    ``dict(ce=..., mse=...)`` forward-return are incompatible with the
    standard VLM training loop. The VLM recipe was evaluated as a parent
    class; the override surface ended up at ~95% of the class body, so a
    parallel implementation is cleaner.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._throughput_window_start: float | None = None
        self._throughput_token_window = 0.0
        self._throughput_step_window = 0
        self._last_tokens_per_sec = 0.0
        self._last_train_steps_per_sec = 0.0
        self._last_tokens_per_step = 0.0
        self._vae_path: str | None = None
        self.vae_encode_micro_batch_size = int(cfg.get("model.vae_encode_micro_batch_size", 0) or 0)
        if self.vae_encode_micro_batch_size < 0:
            raise ValueError("model.vae_encode_micro_batch_size must be >= 0")

    def _encode_vae_images(self, padded_images: torch.Tensor) -> torch.Tensor:
        """Encode VAE images in chunks to cap frozen-VAE activation peaks."""
        micro_batch = self.vae_encode_micro_batch_size
        if micro_batch <= 0 or micro_batch >= int(padded_images.shape[0]):
            return self.vae_model.encode(padded_images)

        chunks = []
        for start in range(0, int(padded_images.shape[0]), micro_batch):
            end = min(start + micro_batch, int(padded_images.shape[0]))
            chunks.append(self.vae_model.encode(padded_images[start:end]))
        return torch.cat(chunks, dim=0)

    def _build_hf_backbone_bagel_model(
        self,
        *,
        artifact_path: str | None,
        stage: int,
        rank_seed: int,
        freeze_before_infrastructure: bool = False,
    ):
        """Build BAGEL from HF backbones and apply the configured infrastructure."""
        logger.info("Building BAGEL from HF backbones (artifact_source=%s, stage=%d)", artifact_path, stage)
        backend_cfg = resolve_bagel_backend(self.cfg.get("model.backend", None))
        with ScopedRNG(seed=rank_seed, ranked=False):
            model = build_bagel_from_hf_backbones(
                model_cfg=self.cfg.model,
                stage=stage,
                vae_config=self.cfg.get("model.vae_config", None),
                meta_init=True,
                load_backbone_weights=False,
                backend=backend_cfg,
            )
            if freeze_before_infrastructure:
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)

            from nemo_automodel._transformers.infrastructure import (
                apply_model_infrastructure,
                instantiate_infrastructure,
            )
            from nemo_automodel.components.quantization.fp8 import build_fp8_config
            from nemo_automodel.components.utils.compile_utils import build_compile_config

            model_wrapper, autopipeline, parallelize_fn, qat_quantizer = instantiate_infrastructure(
                distributed_config=self.distributed_config,
                pipeline_config=self.pipeline_config,
                moe_parallel_config=self.moe_parallel_config,
                activation_checkpointing=self.activation_checkpointing,
                device=self.dist_env.device,
                mesh=self.mesh_context,
            )
            fp8_config = build_fp8_config(self.cfg.fp8) if self.cfg.get("fp8", None) is not None else None
            compile_config = (
                build_compile_config(self.cfg.compile) if self.cfg.get("compile", None) is not None else None
            )
            freeze_cfg = self.cfg.get("freeze_config", None)
            model = apply_model_infrastructure(
                model=model,
                pretrained_model_name_or_path="",
                mesh=self.mesh_context,
                peft_config=self.cfg.get("peft", None),
                fp8_config=fp8_config,
                qat_quantizer=qat_quantizer,
                compile_config=compile_config,
                parallelize_fn=parallelize_fn,
                model_wrapper=model_wrapper,
                is_meta_device=True,
                device=self.dist_env.device,
                load_base_model=False,
                freeze_config=freeze_cfg.to_dict() if freeze_cfg is not None else None,
            )
            initialize_bagel_non_backbone_weights(model)
            load_bagel_hf_backbone_weights(model, self.cfg.model)
        return model

    def _use_sharded_model_ema(self, *, ema_impl: str, model_init_mode: str) -> bool:
        """Resolve the EMA implementation choice."""
        if ema_impl == "shadow":
            return False
        if ema_impl == "sharded_model":
            return True
        if ema_impl != "auto":
            raise ValueError(
                f"Unsupported ema.implementation={ema_impl!r}; expected 'auto', 'sharded_model', or 'shadow'."
            )
        return model_init_mode == "hf_backbones" and self.distributed_config.__class__.__name__ == "FSDP2Config"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self):
        """Build distributed, model, tokenizer, data, optimizer, scheduler."""
        torch.cuda.reset_peak_memory_stats()

        # -- distributed bringup ------------------------------------------
        from nemo_automodel.components.distributed.init_utils import initialize_distributed

        cfg_dist_env = self.cfg.get("dist_env", {})
        backend = cfg_dist_env.get("backend", "nccl")
        timeout = cfg_dist_env.get("timeout_minutes", 10)
        self.dist_env = initialize_distributed(backend=backend, timeout_minutes=timeout)
        setup_logging()

        (
            self.distributed_setup,
            self.mesh_context,
            self.distributed_config,
            self.device_mesh,
            self.moe_mesh,
            self.pp_enabled,
            self.pipeline_config,
            self.moe_parallel_config,
            self.activation_checkpointing,
        ) = self._distributed_setup_attributes(
            create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)
        )

        if self.pp_enabled:
            raise NotImplementedError("Pipeline parallelism is not supported for FinetuneRecipeForMultimodal.")

        # -- RNG ----------------------------------------------------------
        # BAGEL uses a ranked seed: seed = global_seed * world_size + rank.
        # PackedDataset applies its own worker reseed at iter-time.
        self.global_seed = int(self.cfg.get("seed", 4396))
        self.data_seed = int(self.cfg.get("dataset.data_seed", 42))
        rank_seed = self.global_seed * self.dist_env.world_size + self.dist_env.rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)
        self.rng = StatefulRNG(seed=self.global_seed, ranked=True)

        # -- wandb --------------------------------------------------------
        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = self._build_wandb()
            logging.info("Running run at %s", run.url)

        self._log_experiment_details()
        self._log_library_versions()

        # -- tokenizer ----------------------------------------------------
        artifact_path = _resolve_bagel_artifact_path(self.cfg)
        tokenizer_path = _resolve_bagel_tokenizer_path(self.cfg)
        self.tokenizer, self.special_tokens, self.num_new_tokens = _load_bagel_tokenizer(tokenizer_path)

        # -- model --------------------------------------------------------
        stage = int(self.cfg.get("model.stage", 1))
        self.stage = stage

        # Stage 2: load the VAE first so its z_channels/downsample can flow
        # into BagelConfig.vae_config (the model's gen-side __init__ needs
        # them). VAE stays SEPARATE from BagelForUnifiedMultimodal —
        # frozen, inference-only, owned by this recipe; encode runs in the
        # training step before model.forward.
        self.vae_model = None
        self.vae_params = None
        if stage == 2:
            vae_path = _resolve_bagel_vae_path(artifact_path, self.cfg.get("model.vae_path", None))
            logger.info("Loading VAE from %s ...", vae_path)
            vae_model, vae_params = _load_bagel_vae(vae_path)
            self._vae_path = vae_path
            for p in vae_model.parameters():
                p.requires_grad = False
            self.vae_model = vae_model.to(self.dist_env.device).eval()
            self.untrack_state("vae_model")
            self.vae_params = vae_params
            self.cfg.set_by_dotted(
                "model.vae_config",
                {
                    "z_channels": int(vae_params.z_channels),
                    "downsample": int(vae_params.downsample),
                },
            )
            # BAGEL-7B-MoT latent_pos_embed is sized for 64x64
            # (4096 entries). If the artifact config omits that value, use
            # the checkpoint-compatible default.
            if self.cfg.get("model.max_latent_size", None) is None:
                self.cfg.set_by_dotted("model.max_latent_size", 64)

        model_init_mode = self.cfg.model.get("init_mode", "auto")
        if model_init_mode != "hf_backbones" and self.cfg.model.get("_target_", None) is None:
            raise ValueError(
                "BAGEL model config must use an AutoModel target, for example "
                "'model._target_: nemo_automodel.NeMoAutoModelForMultimodalLM.from_pretrained'."
            )

        if model_init_mode == "hf_backbones":
            model = self._build_hf_backbone_bagel_model(
                artifact_path=artifact_path,
                stage=stage,
                rank_seed=rank_seed,
            )
        elif model_init_mode == "auto":
            from nemo_automodel.recipes.vlm.finetune import build_model as build_vlm_model

            logger.info("Loading BAGEL through AutoModel (artifact_source=%s, stage=%d)", artifact_path, stage)
            model = build_vlm_model(
                cfg_model=self.cfg.model,
                cfg_freeze=self.cfg.get("freeze_config", None),
                cfg_peft=self.cfg.get("peft", None),
                seed=self.global_seed,
                cfg_fp8=self.cfg.get("fp8", None),
                cfg_compile=self.cfg.get("compile", None),
                distributed_setup=self.distributed_setup,
            )
        else:
            raise ValueError(f"Unsupported model.init_mode={model_init_mode!r}; expected 'auto' or 'hf_backbones'.")

        _maybe_resize_bagel_vocab(model, tokenizer_vocab_size=len(self.tokenizer), num_new_tokens=self.num_new_tokens)

        self.model = model
        self.model_parts = [model]
        self.pp = None

        # -- EMA tracking (optional) -------------------------------------
        # Optional exponential moving average for diffusion-style checkpointing.
        # Set ``ema.decay: null`` to disable.
        ema_cfg = self.cfg.get("ema", None)
        ema_decay = None if ema_cfg is None else ema_cfg.get("decay", 0.9999)
        if ema_decay is not None:
            ema_impl = ema_cfg.get("implementation", "auto")
            if self._use_sharded_model_ema(ema_impl=str(ema_impl), model_init_mode=model_init_mode):
                if model_init_mode != "hf_backbones":
                    raise ValueError(
                        "ema.implementation='sharded_model' currently requires model.init_mode='hf_backbones'."
                    )
                from nemo_automodel.components.training.ema import ShardedModelEMAManager

                ema_model = self._build_hf_backbone_bagel_model(
                    artifact_path=artifact_path,
                    stage=stage,
                    rank_seed=rank_seed,
                    freeze_before_infrastructure=True,
                )
                _maybe_resize_bagel_vocab(
                    ema_model,
                    tokenizer_vocab_size=len(self.tokenizer),
                    num_new_tokens=self.num_new_tokens,
                )
                self.ema = ShardedModelEMAManager(ema_model=ema_model, train_model=self.model, decay=float(ema_decay))
                logger.info(
                    "EMA enabled with sharded_model implementation, decay=%s, %d params tracked",
                    ema_decay,
                    len(self.ema),
                )
            else:
                from nemo_automodel.components.training.ema import EMAManager

                self.ema = EMAManager(self.model, decay=float(ema_decay))
                logger.info(
                    "EMA enabled with shadow implementation, decay=%s, %d params tracked",
                    ema_decay,
                    len(self.ema),
                )
            if self.dist_env.is_main:
                logger.info("EMA implementation ready.")
        else:
            self.ema = None

        # -- optimizer ----------------------------------------------------
        opt_cfg = self.cfg.optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = opt_cfg.instantiate(params=trainable_params)
        self.optimizer = [optimizer]

        # -- LR scheduler (constant with warmup; BAGEL-style) ------------
        lr_cfg = self.cfg.get("lr_scheduler", None)
        if lr_cfg is not None:
            warmup_steps = int(lr_cfg.get("warmup_steps", 2000))
            schedule = lr_cfg.get("schedule", "constant")
            if schedule != "constant":
                raise NotImplementedError(
                    f"FinetuneRecipeForMultimodal only supports constant schedule with warmup; got {schedule!r}"
                )
            self.warmup_steps = warmup_steps
            self.base_lr = float(optimizer.param_groups[0]["lr"])
            self._use_lr_warmup = True
        else:
            self._use_lr_warmup = False
            self.warmup_steps = 0
            self.base_lr = float(optimizer.param_groups[0]["lr"])

        # -- dataset + dataloader ----------------------------------------
        self.dataloader = self._build_bagel_dataloader()
        self.val_dataloader = None

        # -- step scheduler ----------------------------------------------
        cfg_ss = self.cfg.get("step_scheduler", None)
        local_bs = cfg_ss.get("local_batch_size", 1) if cfg_ss is not None else 1
        ss_kwargs = dict(
            num_epochs=1,
            global_batch_size=8,
            local_batch_size=local_bs,
            dp_size=self._get_dp_group_size(),
            ckpt_every_steps=500,
            log_remote_every_steps=10,
            dataloader=self.dataloader,
        )
        if cfg_ss is not None:
            ss_kwargs.update(cfg_ss.to_dict())
        self.step_scheduler = StepScheduler(**ss_kwargs)
        self._setup_garbage_collection(self.step_scheduler)

        # -- checkpoint + metric loggers ---------------------------------
        self.peft_config = None
        ckpt_cfg = self.cfg.get("checkpoint", None)
        checkpoint_artifact_path = artifact_path or ""
        ckpt_dir = (
            ckpt_cfg.get("checkpoint_dir", "multimodal_checkpoints/bagel_understanding/")
            if ckpt_cfg is not None
            else "multimodal_checkpoints/bagel_understanding/"
        )
        self._ckpt_dir = ckpt_dir
        ckpt_enabled = bool(ckpt_cfg.get("enabled", False)) if ckpt_cfg is not None else False

        # Import lazily because checkpointing is optional and only needed when
        # enabled in the YAML.
        self.checkpointer = None
        if ckpt_enabled:
            from nemo_automodel.components.checkpoint.checkpointing import (
                Checkpointer,
                CheckpointingConfig,
            )

            ckpt_kwargs = dict(
                enabled=True,
                checkpoint_dir=ckpt_dir,
                model_save_format="torch_save",
                model_cache_dir=checkpoint_artifact_path,
                model_repo_id=checkpoint_artifact_path,
                save_consolidated=False,
                is_peft=False,
            )
            if ckpt_cfg is not None:
                cfg_ckpt = ckpt_cfg.to_dict()
                cfg_ckpt.pop("restore_from", None)
                ckpt_kwargs |= cfg_ckpt
            checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
            self.checkpointer = Checkpointer(
                config=checkpoint_config,
                dp_rank=self._get_dp_rank(include_cp=True),
                tp_rank=self._get_tp_rank(),
                pp_rank=self._get_pp_rank(),
                moe_mesh=self.moe_mesh,
            )
            self.checkpointer.validate_model_checkpointing(self.model)

        self.best_metric_key = "default"
        self.max_grad_norm = float(self.cfg.get("clip_grad_norm.max_norm", 1.0))
        self.lr_scheduler = None

        pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.metric_logger_train = build_metric_logger(pathlib.Path(ckpt_dir) / "training.jsonl")
        self.metric_logger_valid = build_metric_logger(pathlib.Path(ckpt_dir) / "validation.jsonl")

        if self.checkpointer is not None:
            restore_from = self.cfg.get("checkpoint.restore_from", None)
            self.load_checkpoint(restore_from)
            torch.cuda.empty_cache()

        # -- loss_fn is inlined in _forward_backward_step; the attr is kept
        # for BaseRecipe state-dict compatibility.
        self.loss_fn = None

        self._log_step_scheduler_details(self.step_scheduler)

    def _build_bagel_dataloader(self) -> StatefulDataLoader:
        """Build the BAGEL packed-sequence DataLoader.

        The ``PackedDataset`` is an ``IterableDataset`` so we pass no sampler;
        per-rank shard selection happens inside the dataset.
        """
        from nemo_automodel.components.datasets.multimodal.collate_fns import bagel_packed_collate_fn
        from nemo_automodel.components.datasets.multimodal.datasets import make_bagel_multimodal_dataset

        cfg_ds = self.cfg.dataset
        cfg_dl = self.cfg.dataloader

        grouped_datasets = _load_grouped_datasets(cfg_ds)
        dataset_info = _load_dataset_info(cfg_ds)

        train_dataset = make_bagel_multimodal_dataset(
            tokenizer=self.tokenizer,
            special_tokens=self.special_tokens,
            grouped_datasets=grouped_datasets,
            local_rank=self.dist_env.rank,
            world_size=self.dist_env.world_size,
            num_workers=int(cfg_dl.get("num_workers", 1)),
            expected_num_tokens=int(cfg_ds.get("expected_num_tokens", 32768)),
            max_num_tokens_per_sample=int(cfg_ds.get("max_num_tokens_per_sample", 16384)),
            max_num_tokens=int(cfg_ds.get("max_num_tokens", 36864)),
            prefer_buffer_before=int(cfg_ds.get("prefer_buffer_before", 16384)),
            max_buffer_size=int(cfg_ds.get("max_buffer_size", 50)),
            interpolate_pos=bool(cfg_ds.get("interpolate_pos", False)),
            use_flex=bool(cfg_ds.get("use_flex", False)),
            data_status=None,
            data_seed=self.data_seed,
            text_cond_dropout_prob=float(cfg_ds.get("text_cond_dropout_prob", 0.1)),
            vit_cond_dropout_prob=float(cfg_ds.get("vit_cond_dropout_prob", 0.4)),
            vae_cond_dropout_prob=float(cfg_ds.get("vae_cond_dropout_prob", 0.1)),
            vae_image_downsample=int(cfg_ds.get("vae_image_downsample", 16)),
            max_latent_size=int(cfg_ds.get("max_latent_size", 32)),
            vit_patch_size=int(cfg_ds.get("vit_patch_size", 14)),
            max_num_patch_per_side=int(cfg_ds.get("max_num_patch_per_side", 70)),
            dataset_info=dataset_info,
        )
        # Keep the data-file shuffle seed and packed-sample RNG seed separate.
        #
        # AM's ``PackedDataset.__iter__`` reseeds at worker-start with
        #   ``rank_seed = _global_seed * world_size + local_rank``
        #     (times ``num_workers`` + ``worker_id`` when >1 workers).
        #
        # ``set_epoch(data_seed)`` controls each group's data-file ordering.
        # The packed-sample RNG follows the ranked training seed because it
        # drives group selection, conditioning dropout, and timestep sampling
        # during packing.
        train_dataset._global_seed = int(self.global_seed)
        train_dataset.set_epoch(self.data_seed)

        self._train_dataset = train_dataset
        self.untrack_state("_train_dataset")

        nw = int(cfg_dl.get("num_workers", 1))
        dl_kwargs = dict(
            batch_size=1,  # one packed dict per step.
            num_workers=nw,
            pin_memory=bool(cfg_dl.get("pin_memory", True)),
            collate_fn=bagel_packed_collate_fn,
            drop_last=True,
        )
        if nw > 0:
            dl_kwargs["prefetch_factor"] = int(cfg_dl.get("prefetch_factor", 2))

        # DataLoader workers fork lazily on first iteration. Reset the parent
        # process RNG immediately before returning the DataLoader so workers
        # inherit a BAGEL-compatible state even before PackedDataset.__iter__
        # applies its explicit per-worker reseed.
        rank_seed = self.global_seed * self.dist_env.world_size + self.dist_env.rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)

        return StatefulDataLoader(train_dataset, **dl_kwargs)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _apply_warmup(self, step: int) -> None:
        """BAGEL uses constant LR with linear warmup from 0 to base_lr.
        Matches HF get_constant_schedule_with_warmup: scale = step/warmup_steps
        (so the first optimizer step at step=0 runs with lr=0)."""
        if not self._use_lr_warmup or self.warmup_steps <= 0:
            return
        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            scale = 1.0
        for opt in self.optimizer:
            for pg in opt.param_groups:
                pg["lr"] = self.base_lr * scale

    def _prepare_batch(self, batch) -> Dict[str, Any]:
        """Move the SimpleCustomBatch packed dict onto the current CUDA device."""
        device = self.dist_env.device
        batch = batch.cuda(device)
        return batch.to_dict()

    def _forward_backward_step(
        self,
        idx: int,
        batch,
        *,
        loss_buffer: List[torch.Tensor],
        num_ce_tokens_global: int,
        num_mse_tokens_global: int = 0,
        num_batches: int,
        is_train: bool = True,
    ):
        """One packed-sample forward + backward.

        Replaces the parent VLM step because BAGEL's forward takes a packed-
        sequence kwarg dict and returns ``{"ce": Tensor|None, "mse": Tensor|
        None}`` instead of the HF ``ModelOutput`` the VLM recipe expects.

        Stage 2: VAE-encodes ``padded_images`` into ``padded_latent`` before
        the model forward, then composes ``ce + mse`` reductions into one
        microbatch loss.
        """
        data = self._prepare_batch(batch)

        # Stage 2: VAE encode raw images -> latents BEFORE model forward.
        # VAE stays in fp32 + .eval(); encode runs under bf16 autocast, no_grad.
        if self.vae_model is not None and "padded_images" in data:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                with torch.no_grad():
                    padded_images = data.pop("padded_images")
                    data["padded_latent"] = self._encode_vae_images(padded_images)
                    del padded_images

        # Filter dict to what BAGEL forward accepts. Stage 1 keeps only the
        # CE-side kwargs (gen-side tensors absent from the pack); Stage 2
        # additionally accepts the gen-side kwargs the model needs.
        allowed = _BAGEL_STAGE2_FORWARD_KEYS if self.stage == 2 else _BAGEL_STAGE1_FORWARD_KEYS
        forward_kwargs = {k: v for k, v in data.items() if k in allowed}

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = self.model(**forward_kwargs)

        ce = out.get("ce", None)
        mse = out.get("mse", None)

        # BAGEL-style reductions, one per loss type:
        #   ce  : per-token CE  -> ce.sum()           * world_size / num_ce_tokens_global
        #   mse : per-token MSE (mean over patch dim) -> mse.mean(-1).sum() * world_size / num_mse_tokens_global
        # For gradient accumulation, the caller passes token counts summed over
        # the whole optimizer step so unequal packed microbatches stay
        # token-weighted instead of each contributing an independent mean.
        # After FSDP's bf16 mean-grad all-reduce, the effective gradient is
        # the per-token-averaged loss gradient on each side.
        ws = self.dist_env.world_size
        microbatch_loss = torch.zeros((), device=self.dist_env.device, dtype=torch.float32)
        ce_logged: Optional[torch.Tensor] = None
        mse_logged: Optional[torch.Tensor] = None
        if ce is not None and num_ce_tokens_global > 0:
            ce_term = ce.sum() * ws / max(num_ce_tokens_global, 1)
            microbatch_loss = microbatch_loss + ce_term
            ce_logged = ce_term.detach()
        if mse is not None and num_mse_tokens_global > 0:
            mse_term = mse.mean(dim=-1).sum() * ws / max(num_mse_tokens_global, 1)
            microbatch_loss = microbatch_loss + mse_term
            mse_logged = mse_term.detach()

        if ce is None and mse is None:
            raise RuntimeError(
                "BAGEL forward returned both ce=None and mse=None; the pack "
                "carried no loss-bearing tokens. Check the data pipeline."
            )

        loss_buffer.append(
            {
                "loss": microbatch_loss.detach().clone(),
                "ce": ce_logged.clone() if ce_logged is not None else None,
                "mse": mse_logged.clone() if mse_logged is not None else None,
            }
        )

        if is_train:
            microbatch_loss.backward()

        return data

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a training step; supports grad accumulation trivially.

        BAGEL packs variable token counts per microbatch. For gradient
        accumulation, normalize each CE/MSE contribution by the token count
        over the whole optimizer step, not by each microbatch independently.
        """
        device = self.dist_env.device
        loss_buffer: List[Dict[str, Optional[torch.Tensor]]] = []
        total_tokens = 0
        total_ce_tokens = 0
        total_mse_tokens = 0

        num_batches = len(batches)
        for batch in batches:
            # Pre-allreduce token counts for this micro-batch.
            raw = batch.to_dict()
            num_ce_local = int(raw["ce_loss_indexes"].numel()) if "ce_loss_indexes" in raw else 0
            num_mse_local = int(raw["mse_loss_indexes"].numel()) if "mse_loss_indexes" in raw else 0
            seq_len_local = int(raw["sequence_length"])
            counts = torch.tensor([num_ce_local, num_mse_local, seq_len_local], dtype=torch.long, device=device)
            if dist.is_initialized():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            num_ce_global = int(counts[0].item())
            num_mse_global = int(counts[1].item())
            seq_len_global = int(counts[2].item())
            total_ce_tokens += num_ce_global
            total_mse_tokens += num_mse_global
            total_tokens += seq_len_global

        for i, batch in enumerate(batches):
            self._forward_backward_step(
                i,
                batch,
                loss_buffer=loss_buffer,
                num_ce_tokens_global=total_ce_tokens,
                num_mse_tokens_global=total_mse_tokens,
                num_batches=num_batches,
            )

        # Grad clip + step.
        # FSDP2 sharded parameters expose ``clip_grad_norm_`` via the manager,
        # but the simplest cross-wrapper approach is torch.nn.utils.clip_grad_norm_
        # on trainable params. FSDP2 DTensor params play nice with it in newer
        # torch versions; for older ones we fall back to the raw compute.
        if max_grad_norm is not None and max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=max_grad_norm,
                norm_type=2.0,
                foreach=True,
            )
        else:
            grad_norm = torch.tensor(0.0, device=device)

        # LR warmup (stateless; called before each optimizer.step).
        self._apply_warmup(self.step_scheduler.step)

        for opt in self.optimizer:
            opt.step()
            opt.zero_grad(set_to_none=True)

        # EMA update: apply right after optimizer.step so the shadow weights
        # average the post-step train weights.
        if self.ema is not None:
            self.ema.update(self.model)

        # Reporting loss: aggregate per-microbatch entries (each is a dict
        # with {"loss", "ce", "mse"}). Sum across micro-batches, then mean
        # across the DP group via allreduce.
        def _agg(key):
            terms = [b[key] for b in loss_buffer if b.get(key) is not None]
            if not terms:
                return None
            s = torch.stack(terms).sum()
            s = self._dp_allreduce(s, include_cp=True)
            if dist.is_initialized():
                s = s / self._get_dp_group_size(include_cp=True)
            return s.item()

        reporting_loss_val = _agg("loss") or 0.0
        ce_val = _agg("ce")
        mse_val = _agg("mse")

        self._throughput_token_window += float(total_tokens)
        self._throughput_step_window += 1
        if self.step_scheduler.is_remote_logging_step:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t = time.perf_counter()
            if self._throughput_window_start is None:
                self._throughput_window_start = t
            time_delta = max(t - self._throughput_window_start, 1e-6)
            tps = self._throughput_token_window / time_delta
            train_steps_per_sec = self._throughput_step_window / time_delta
            tokens_per_step = self._throughput_token_window / max(self._throughput_step_window, 1)
            self._last_tokens_per_sec = tps
            self._last_train_steps_per_sec = train_steps_per_sec
            self._last_tokens_per_step = tokens_per_step
            self._throughput_window_start = t
            self._throughput_token_window = 0.0
            self._throughput_step_window = 0
        else:
            tps = self._last_tokens_per_sec
            train_steps_per_sec = self._last_train_steps_per_sec
            tokens_per_step = self._last_tokens_per_step

        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss_val,
                "ce": ce_val if ce_val is not None else 0.0,
                "mse": mse_val if mse_val is not None else 0.0,
                "grad_norm": grad_norm_val,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / max(self._get_dp_group_size(), 1),
                "tokens_per_sec": tps,
                "tokens_per_step": tokens_per_step,
                "train_steps_per_sec": train_steps_per_sec,
                "num_tokens_per_step": total_tokens,
                "num_ce_tokens": total_ce_tokens,
                "num_mse_tokens": total_mse_tokens,
                "num_label_tokens": total_ce_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Main loop. Full-model checkpointing is controlled by ``ckpt_every_steps``
    # in the YAML.
    # ------------------------------------------------------------------
    def _copy_vae_sidecar_to_checkpoint(self, checkpoint_path: pathlib.Path) -> None:
        if int(getattr(self, "stage", 1)) != 2 or self._vae_path is None:
            return

        is_dist_initialized = dist.is_initialized()
        is_rank_0 = not is_dist_initialized or dist.get_rank() == 0
        if not is_rank_0:
            return

        src = pathlib.Path(self._vae_path)
        dst = checkpoint_path / "ae.safetensors"
        try:
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
            logger.info("Copied BAGEL VAE sidecar to %s", dst)
        except OSError:
            logger.warning(
                "Failed to copy BAGEL VAE sidecar from %s to %s; the checkpoint still requires "
                "the configured VAE source or HF repo to restore Stage 2 generation.",
                src,
                dst,
                exc_info=True,
            )

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: Optional[Dict[str, float]] = None,
        best_metric_key: str = "default",
    ) -> None:
        """Save BAGEL state and include the frozen VAE as a checkpoint sidecar."""
        if self.checkpointer is None or not self.checkpointer.config.enabled:
            return

        super().save_checkpoint(epoch, step, train_loss, val_loss, best_metric_key)
        checkpoint_path = pathlib.Path(self.checkpointer.config.checkpoint_dir) / f"epoch_{epoch}_step_{step}"
        self._copy_vae_sidecar_to_checkpoint(checkpoint_path)
        if dist.is_initialized():
            dist.barrier()

    def run_train_validation_loop(self):
        """BAGEL training loop — no validation, optional periodic checkpoint."""
        self.model.train()
        self._throughput_window_start = time.perf_counter()
        self._throughput_token_window = 0.0
        self._throughput_step_window = 0

        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            for batch_idx, batches in enumerate(self.step_scheduler):
                log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                self.log_train_metrics(log_data)

                if self.step_scheduler.is_ckpt_step and self.checkpointer is not None:
                    self.save_checkpoint(
                        epoch,
                        self.step_scheduler.step,
                        log_data.metrics["loss"],
                        None,
                        best_metric_key=self.best_metric_key,
                    )
                self._maybe_collect_garbage()

        self.metric_logger_train.close()
        self.metric_logger_valid.close()
        if self.checkpointer is not None:
            self.checkpointer.close()

    # ------------------------------------------------------------------
    # Logging helpers shared with FinetuneRecipeForVLM so BAGEL can keep its
    # dedicated packed-sequence training loop without inheriting VLM behavior.
    # ------------------------------------------------------------------
    def _build_wandb(self):
        assert self.cfg.get("wandb", None) is not None
        from wandb import Settings

        kwargs = self.cfg.wandb.to_dict()
        if kwargs.get("name", "") == "":
            # default name: model basename.
            mp = _resolve_bagel_artifact_path(self.cfg) or "bagel"
            kwargs["name"] = "_".join(str(mp).split("/")[-2:])
        run = wandb.init(
            **kwargs,
            config=self.cfg.to_dict(),
            settings=Settings(silent=True),
        )
        return run

    def log_train_metrics(self, log_data) -> None:
        if not self.dist_env.is_main:
            return
        if not self.step_scheduler.is_remote_logging_step:
            self.metric_logger_train.log(log_data)
            return
        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=self.step_scheduler.step)
        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | ce {:.4f} | mse {:.4f} | grad_norm {:.4f} | "
            "lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | "
            "num_ce_tokens {} | num_mse_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["ce"],
                log_data.metrics.get("mse", 0.0),
                log_data.metrics["grad_norm"],
                log_data.metrics["lr"],
                log_data.metrics["mem"],
                log_data.metrics["tps"],
                log_data.metrics["tps_per_gpu"],
                log_data.metrics["num_ce_tokens"],
                log_data.metrics.get("num_mse_tokens", 0),
            )
        )
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Entry point (legacy; CLI goes through cli/app.py)
# ---------------------------------------------------------------------------


def main(config_path: Optional[str] = None) -> None:
    """Run the BAGEL multimodal training recipe from a YAML config path."""
    if config_path is None:
        config_path = "examples/multimodal_finetune/bagel/bagel_sft.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = FinetuneRecipeForMultimodal(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
