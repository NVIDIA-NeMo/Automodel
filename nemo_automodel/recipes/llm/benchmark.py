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

import hashlib
import json
import logging
import pathlib
from dataclasses import asdict, dataclass

import torch
from torch.distributed.tensor import DTensor

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.tp_replicas import synchronize_tp_replica_gradients
from nemo_automodel.components.loss.utils import _count_label_tokens, _get_loss_ignore_index
from nemo_automodel.components.training.timers import Timers
from nemo_automodel.components.training.utils import (
    get_expert_tp_replication_factor,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.flops_utils import (
    calculate_mfu,
    get_batch_flops_counter,
    get_flops_formula_for_hf_config,
)
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.shared.flops import BatchFlopsCounter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IterationMetrics:
    """One global optimizer window, timed by its slowest rank."""

    step: int
    seconds: float
    tflops: float
    all_token_head_tflops: float
    num_tokens: int
    num_supervised_tokens: int


@dataclass(frozen=True)
class _CudaMemoryPeak:
    """One rank's allocator peaks in bytes and process-lifetime pressure counters."""

    rank: int
    max_allocated_bytes: int
    max_reserved_bytes: int
    num_alloc_retries: int
    num_ooms: int


def _infer_vocab_size(model_cfg):
    """Infer vocab_size from a model config, handling custom config classes and VL composite configs.

    Args:
        model_cfg: The model config section (cfg.model) containing _target_, config, etc.

    Returns:
        The vocab_size integer, or raises AttributeError if not found.
    """
    from transformers import AutoConfig

    config_section = model_cfg.config
    # Recipes may set trust_remote_code either at the model level (nemo_automodel
    # convention) or nested under `config` -- accept either.
    trust_remote_code = getattr(model_cfg, "trust_remote_code", False) or getattr(
        config_section, "trust_remote_code", False
    )

    # Use the config's _target_ if it's a custom config class (e.g. DeepseekV32Config)
    config_target = getattr(config_section, "_target_", None)
    model_config = None

    if config_target is not None and callable(config_target):
        target_name = getattr(config_target, "__qualname__", "")
        if "AutoConfig" not in target_name:
            if hasattr(config_target, "from_pretrained"):
                # `_target_` resolved to a config class; call its classmethod.
                model_config = config_target.from_pretrained(config_section.pretrained_model_name_or_path)
            else:
                # `_target_` already resolved to the factory itself
                # (e.g. `DeepseekV32Config.from_pretrained`); invoke it directly.
                model_config = config_target(config_section.pretrained_model_name_or_path)
    elif isinstance(config_target, str) and "AutoConfig" not in config_target:
        import importlib

        module_path, class_name = config_target.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        config_cls = getattr(mod, class_name)
        model_config = config_cls.from_pretrained(config_section.pretrained_model_name_or_path)

    if model_config is None:
        try:
            model_config = AutoConfig.from_pretrained(
                config_section.pretrained_model_name_or_path, trust_remote_code=trust_remote_code
            )
        except Exception as exc:
            # Step-3.5 publishes MTP layer metadata in ``layer_types`` in addition
            # to its transformer-layer count.  Newer Transformers validates those
            # lengths while loading a config even though benchmark setup only needs
            # ``vocab_size``.  Reuse the tokenizer compatibility patch, then retry.
            message = str(exc)
            if "num_hidden_layers" not in message or "layer_types" not in message:
                raise
            from nemo_automodel._transformers.v4_patches.layer_types import relax_layer_types_validator

            relax_layer_types_validator()
            model_config = AutoConfig.from_pretrained(
                config_section.pretrained_model_name_or_path, trust_remote_code=trust_remote_code
            )

    if hasattr(model_config, "vocab_size"):
        return model_config.vocab_size
    elif hasattr(model_config, "text_config") and hasattr(model_config.text_config, "vocab_size"):
        return model_config.text_config.vocab_size
    else:
        raise AttributeError(f"Could not find vocab_size on {type(model_config).__name__} or its text_config")


class BenchmarkingRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction):
    """Benchmarking recipe for next-token prediction.

    This class extends TrainFinetuneRecipeForNextTokenPrediction to provide
    a simplified benchmarking-focused training loop with timers and profiling support.
    It reuses the setup() and _forward_backward_step() methods from the parent class.

    benchmark.flops_scope defaults to "model". Explicit "text" scope counts
    only the text backbone over the complete measured iteration time; vision
    encoder and projector FLOPs are excluded.
    """

    def __init__(self, cfg):
        """Initialize the benchmarking recipe.

        Args:
            cfg: Configuration dictionary/object for benchmarking.
        """
        # Store benchmarking-specific parameters from benchmark section
        bench_cfg = cfg.benchmark
        self._bench_warmup_steps = bench_cfg.warmup_steps
        self._bench_peak_tflops = bench_cfg.peak_tflops
        self._bench_nsys_start = bench_cfg.nsys_start
        self._bench_nsys_end = bench_cfg.nsys_end
        self._bench_nsys_ranks = bench_cfg.nsys_ranks
        self._bench_json_output_path = getattr(bench_cfg, "json_output_path", None)
        self._bench_flops_scope = getattr(bench_cfg, "flops_scope", "model")
        if self._bench_flops_scope not in ("model", "text"):
            raise ValueError(f"benchmark.flops_scope must be 'model' or 'text', got {self._bench_flops_scope!r}")
        self._wandb_enabled = cfg.get("wandb", None) is not None

        # Infer max_steps from step_scheduler
        self._bench_steps = cfg.step_scheduler.max_steps

        # Prepacked dataset factories do not accept synthetic-data constructor keys.
        self._bench_seq_len = cfg.get("packed_sequence.packed_sequence_size", cfg.get("dataset.seq_len"))
        if self._bench_seq_len is None:
            raise ValueError("Benchmark requires dataset.seq_len or packed_sequence.packed_sequence_size")
        self._batch_flops_counter: BatchFlopsCounter | None = None
        self._bench_iterations: list[_IterationMetrics] = []
        self._bench_all_token_head_tflops = 0.0
        self._bench_step_tokens = 0
        self._bench_step_supervised_tokens = 0
        self._bench_batch_metadata_path = cfg.get("benchmark.batch_metadata_path")
        self._bench_loaded_model_audit_path = cfg.get("benchmark.loaded_model_audit_path")

        # Infer vocab_size from model config and inject it into dataset config
        if cfg.get("dataset.vocab_size") is not None and hasattr(cfg, "model"):
            if hasattr(cfg.model, "config") and hasattr(cfg.model.config, "pretrained_model_name_or_path"):
                vocab_size = _infer_vocab_size(cfg.model)
                cfg.dataset.vocab_size = vocab_size
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"Inferred vocab_size={vocab_size} from model config")

        # Inject batch_size from step_scheduler into dataset config
        if cfg.get("dataset.batch_size") is not None:
            local_batch_size = cfg.get("step_scheduler.local_batch_size", 1)
            cfg.dataset.batch_size = local_batch_size
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Using batch_size={local_batch_size} from step_scheduler.local_batch_size")

        super().__init__(cfg)
        self.timers = Timers(log_level=2, log_option="minmax")

    def setup(self):
        """Setup the benchmarking environment.

        This method calls the parent's setup() but adapts it for benchmarking purposes.
        It skips validation dataloader, checkpointing, and other training-specific features.
        """
        with self.timers("setup", log_level=1):
            # Call parent setup
            super().setup()

        self._write_loaded_model_audit()

        # Store wandb run object if initialized by parent
        import wandb

        self.wandb_run = wandb.run

        # Clear validation dataloader (not needed for benchmarking)
        self.val_dataloader = None

        # Get step_scheduler config
        seq_len = self._bench_seq_len
        global_batch_size = self.cfg.step_scheduler.global_batch_size

        # Calculate FLOPs
        flops_config = self.model_parts[0].config
        if self._bench_flops_scope == "text":
            get_text_config = getattr(flops_config, "get_text_config", None)
            if callable(get_text_config):
                flops_config = get_text_config()
        self._batch_flops_counter = get_batch_flops_counter(flops_config)
        if self._batch_flops_counter is not None and self.cfg.get("peft") is not None:
            raise ValueError("Model-owned batch FLOPs currently require full-parameter training, without PEFT")
        flops_formula = get_flops_formula_for_hf_config(flops_config)
        if flops_formula is None:
            raise ValueError(
                f"No FLOPs formula for {type(flops_config).__name__} with "
                f"benchmark.flops_scope={self._bench_flops_scope!r}. "
                "Set benchmark.flops_scope='text' to explicitly count only the text backbone."
            )
        flops = flops_formula(flops_config, gbs=global_batch_size, seq_len=seq_len)
        self.tflops = flops / (10**12)

        # Add Multi-Token-Prediction (MTP) head FLOPs (e.g. Nemotron-3 Ultra). The backbone
        # FLOPs formula omits the MTP head, and the HF config retains only the physical depth
        # count, so read the effective settings (depths run, repeated-layer flag, per-depth
        # block types) from the built model.
        backbone_tflops = self.tflops
        mtp_tflops = self._mtp_tflops(global_batch_size, seq_len)
        self.tflops += mtp_tflops
        if self.dist_env.is_main:
            logger.info(f"MTP TFLOPs/GPU: {mtp_tflops:.6f} (backbone {backbone_tflops:.6f}, total {self.tflops:.6f})")

        if hasattr(self.cfg, "peft"):
            # Calculate trainable vs non-trainable parameters
            # Need to get these before autopipeline splits the model across PP ranks
            if self.pp is not None:
                # PP enabled - use pre-sharding values stored on autopipeline
                lora_params = self.pp.trainable_params_before_pp
                total_params = self.pp.total_params_before_pp
            else:
                # No PP - model is not sharded, calculate directly
                from nemo_automodel.components.utils.model_utils import print_trainable_parameters

                lora_params, total_params = print_trainable_parameters(self.model_parts[0])
            frozen_params = total_params - lora_params

            # Adjust TFLOPS for PEFT: training has 3 computational phases:
            # 1) Forward pass: processes all parameters (frozen + LoRA)
            # 2) Backward to inputs: processes all parameters
            # 3) Backward to parameters: linear in size of trainable weights (lora params)
            #    - For full fine-tuning: phases 1, 2, 3 are roughly equal (1/3 each)
            #    - For LoRA: phases 1, 2 are the same, but phase 3 scales with lora/frozen ratio
            # Since base TFLOPS assumes all params trainable, we scale by:
            # (2 * frozen_flops + lora_flops) / (3 * frozen_flops) = (2 + lora/frozen) / 3
            param_ratio = lora_params / frozen_params
            tflops_multiplier = (2 + param_ratio) / 3
            self.tflops *= tflops_multiplier
            if self.dist_env.is_main:
                logger.info(
                    f"PEFT params - lora_params: {lora_params:,}, frozen_params: {frozen_params:,}, param_ratio (lora/frozen): {param_ratio:.4f}"
                )
                logger.info(f"TFLOPS multiplier for PEFT: (2 + {param_ratio:.4f}) / 3 = {tflops_multiplier:.4f}")

        self._bench_all_token_head_tflops = self.tflops
        if self.dist_env.is_main:
            logger.info(
                f"Shape-estimated global step TFLOPs: {self.tflops:.6f} | FLOPs scope: {self._bench_flops_scope}"
            )
            if self._batch_flops_counter is not None:
                logger.info("Per-step MFU uses CPU document boundaries and supervised LM-head rows.")

        # Log setup time to wandb
        if self._wandb_enabled:
            self.timers.write_to_wandb(
                names=["setup"],
                writer=self.wandb_run,
                iteration=0,
                normalizer=1.0,
                reset=False,
                barrier=True,
            )

        self.timers.log(
            names=["setup"],
            rank=0,
            normalizer=1000.0,  # Convert to seconds
            reset=True,
            barrier=True,
        )

    def _mtp_tflops(self, global_batch_size, seq_len):
        """TFLOPs added by a Multi-Token-Prediction (MTP) head, if the model has one.

        The backbone FLOPs formula omits the MTP head, and the HF config retains only the
        physical depth count (and no per-depth block pattern). So read the EFFECTIVE settings
        from the built model: ``mtp_config.num_layers`` (depths actually run),
        ``mtp_config.use_repeated_layer``, and the per-sublayer ``block_type`` from
        ``mtp.layers``. Returns 0.0 when the model has no enabled MTP head.
        """
        from nemo_automodel.components.utils.flops_utils import _nemotronh_mtp_flops

        for mp in self.model_parts:
            mtp_cfg = getattr(mp, "mtp_config", None)
            mtp = getattr(mp, "mtp", None)
            layers = getattr(mtp, "layers", None) if mtp is not None else None
            if mtp_cfg is None or not getattr(mtp_cfg, "enabled", False) or layers is None:
                continue
            block_types = [getattr(s, "block_type", "moe") for s in layers]
            try:
                flops = _nemotronh_mtp_flops(
                    mp.config,
                    global_batch_size,
                    seq_len,
                    mtp_cfg.num_layers,
                    block_types,
                    mtp_cfg.use_repeated_layer,
                )
            except AttributeError as exc:
                if self.dist_env.is_main:
                    logger.warning(
                        "Skipping MTP head FLOPs for %s because the Nemotron-H FLOPs formula "
                        "does not support this config: %s",
                        type(mp.config).__name__,
                        exc,
                    )
                continue
            if self.dist_env.is_main:
                logger.info(
                    f"MTP head FLOPs: N={mtp_cfg.num_layers} repeated={mtp_cfg.use_repeated_layer} "
                    f"blocks={block_types} -> +{flops / (10**12):.6f} TFLOPs"
                )
            return flops / (10**12)
        return 0.0

    def run_benchmark(self):
        """Run the benchmarking loop.

        This method implements a simplified training loop focused on benchmarking
        with timers and profiling support, similar to the original benchmarking script.
        """
        rank = self.dist_env.rank

        # Get benchmarking config
        steps = self._bench_steps
        warmup_steps = self._bench_warmup_steps
        local_batch_size = self.cfg.get("step_scheduler.local_batch_size")
        global_batch_size = self.cfg.step_scheduler.global_batch_size

        nsys_start = self._bench_nsys_start
        nsys_end = self._bench_nsys_end
        nsys_ranks = self._bench_nsys_ranks

        peak_tflops = self._bench_peak_tflops

        # Set models to training mode
        for mp in self.model_parts:
            mp.train()

        # Calculate gradient accumulation steps
        dp_size = self._get_dp_group_size()
        ga_steps = global_batch_size // (local_batch_size * dp_size)
        assert ga_steps > 0, "Global batch size must be divisible by local batch size * dp_size"

        if rank == 0:
            logger.info(f"Running {steps} iterations with {warmup_steps} warmup steps")
            logger.info(
                f"GA steps: {ga_steps}, DP size: {dp_size}, Local batch size: {local_batch_size}, Global batch size: {global_batch_size}"
            )

        # Create dataloader iterator
        dataloader_iter = iter(self.dataloader)

        # Main benchmarking loop
        for i in range(steps):
            self.step_scheduler.step = i
            # Start nsys profiling if configured
            if i == nsys_start and rank in nsys_ranks:
                logger.info(f"Rank {rank} | Starting nsys profiling")
                torch.cuda.cudart().cudaProfilerStart()
                # Per-microbatch NVTX ranges below already delimit the work.
                # Entering emit_nvtx() without closing it leaks RecordFunction
                # callbacks into later DTensor/FSDP operations.

            if rank == 0:
                logger.info(f"Rank {rank} | Iteration {i}")

            # Zero gradients
            for opt in self.optimizer:
                opt.zero_grad()

            # Time the iteration
            iter_timer = "iteration_warmup" if i < warmup_steps else "iteration"
            with self.timers(iter_timer, log_level=1):
                # Normalize every microbatch by the whole window's global supervision.
                batches = [next(dataloader_iter) for _ in range(ga_steps)]
                ignore_index = _get_loss_ignore_index(self.loss_fn)
                num_label_tokens = torch.tensor(
                    sum(_count_label_tokens(batch["labels"], ignore_index) for batch in batches),
                    dtype=torch.long,
                )
                num_label_tokens = self._dp_allreduce(num_label_tokens).item()
                self._set_moe_aux_loss_backward_scale(num_batches=ga_steps, num_label_tokens=num_label_tokens)
                loss_buffer = []
                prepare_for_grad_accumulation(self.model_parts, pp_enabled=self.pp_enabled)

                for ga_step_idx, batch in enumerate(batches):
                    if ga_step_idx == ga_steps - 1:
                        prepare_for_final_backward(self.model_parts, pp_enabled=self.pp_enabled)

                    torch.cuda.nvtx.range_push(f"iteration_{i}_ga_step_{ga_step_idx}")

                    with self.timers(f"forward_backward_{ga_step_idx}", log_level=2):
                        self._forward_backward_step(
                            ga_step_idx,
                            batch,
                            loss_buffer=loss_buffer,
                            num_label_tokens=num_label_tokens,
                            num_batches=ga_steps,
                            is_train=True,
                        )

                    torch.cuda.nvtx.range_pop()

                    if ga_step_idx == 0:
                        prepare_after_first_microbatch()

                # Optimizer step
                with self.timers("optimizer", log_level=2):
                    synchronize_tp_replica_gradients(self.model_parts, self.device_mesh)
                    scale_grads_and_clip_grad_norm(
                        self.max_grad_norm,
                        self.model_parts,
                        norm_type=2.0,
                        pp_enabled=self.pp_enabled,
                        device_mesh=self.device_mesh,
                        moe_mesh=self.moe_mesh,
                        ep_axis_name="ep"
                        if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names
                        else None,
                        pp_axis_name="pp" if self.pp_enabled else None,
                        foreach=True,
                        num_label_tokens=num_label_tokens,
                        dp_group_size=self._get_dp_group_size(include_cp=True),
                        expert_tp_replication_factor=get_expert_tp_replication_factor(
                            self.model_parts, self.device_mesh
                        ),
                        grad_norm_backend=self.cfg.get("clip_grad_norm.backend", "triton"),
                    )
                    for opt in self.optimizer:
                        opt.step()
                    logger.debug("Optimizer step")

            # Match the training-loop lifecycle: record one complete eager
            # optimizer step, then capture outside the measured iteration.
            if self.partial_cuda_graph_manager is not None and self._partial_cuda_graph_capture_pending:
                self.partial_cuda_graph_manager.capture()
                self._partial_cuda_graph_capture_pending = False

            # Non-pipeline losses were normalized before backward; PP returns sums.
            reporting_loss = torch.sum(torch.stack(loss_buffer))
            reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
            reporting_loss = reporting_loss.to(torch.float32)
            if self.pp_enabled and num_label_tokens > 0:
                reporting_loss = reporting_loss / num_label_tokens

            if self.pp_enabled:
                reporting_loss = reporting_loss.to(self.dist_env.device)
                # Send loss to first rank if pp group rank is 0
                src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
                if self.dist_env.rank == src_rank:
                    torch.distributed.send(reporting_loss, dst=0)
                elif self.dist_env.is_main:
                    torch.distributed.recv(reporting_loss, src=src_rank)

            reporting_loss = reporting_loss.cpu().item()

            if rank == 0:
                print(f"num_label_tokens={num_label_tokens} | loss={reporting_loss:.4f}")
                logger.info(
                    f"Rank {rank} | Iteration {i} | num_label_tokens={num_label_tokens} | "
                    f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB | "
                    f"loss={reporting_loss:.4f}"
                )
                if self._wandb_enabled and self.wandb_run is not None:
                    self.wandb_run.log({"loss": reporting_loss}, step=i)

            # Collect and log MoE load balance metrics (inherited from train_ft)
            self._collect_moe_load_balance()
            if self._wandb_enabled and self.wandb_run is not None:
                self._log_moe_metrics(i, self.wandb_run.log)

            # Count metadata after timing: batches are still global, pre-CP CPU tensors.
            self._record_batch_flops(batches, num_label_tokens=num_label_tokens)
            self._record_batch_metadata(batches, step=i)

            # Calculate and log MFU
            self._log_iteration_metrics(iter_timer, ga_steps, peak_tflops, rank, i)

            # Stop nsys profiling if configured
            if i == nsys_end and rank in nsys_ranks:
                logger.info(f"Rank {rank} | Stopping nsys profiling")
                torch.cuda.cudart().cudaProfilerStop()

            self._maybe_collect_garbage()

        if self.partial_cuda_graph_manager is not None:
            self.partial_cuda_graph_manager.log_stats("benchmark")

        # Final summary
        self._log_benchmark_summary(steps, warmup_steps, peak_tflops, rank)

    def _mesh_coordinates(self) -> dict[str, int]:
        """Describe this rank's logical mesh without any communication."""
        if self.device_mesh is None:
            return {}
        return {name: self.device_mesh[name].get_local_rank() for name in self.device_mesh.mesh_dim_names}

    def _record_batch_metadata(self, batches: list[dict[str, torch.Tensor]], *, step: int) -> None:
        """Optionally hash actual CPU [batch, global_sequence] rows outside timing.

        Each rank writes its own JSONL file, preserving CP/PP duplicates as
        evidence instead of adding an all-gather to the benchmark. The hash
        concatenates int64 bytes in input/label/position/mask order, matching
        frozen packed-dataset provenance.
        """
        if self._bench_batch_metadata_path is None:
            return
        directory = pathlib.Path(self._bench_batch_metadata_path)
        directory.mkdir(parents=True, exist_ok=True)
        ignore_index = _get_loss_ignore_index(self.loss_fn)
        records = []
        for microbatch, batch in enumerate(batches):
            keys = [key for key in ("input_ids", "labels", "position_ids", "attention_mask") if key in batch]
            for key in keys:
                if batch[key].device.type != "cpu" or batch[key].shape != batch["input_ids"].shape:
                    raise ValueError("Batch metadata requires matching CPU tensors before context-parallel sharding")
            for row in range(batch["input_ids"].shape[0]):
                digest = hashlib.sha256()
                hashes = {}
                for key in keys:
                    raw = batch[key][row].to(torch.int64).contiguous().numpy().tobytes()
                    digest.update(raw)
                    hashes[key] = hashlib.sha256(raw).hexdigest()
                records.append(
                    {
                        "microbatch": microbatch,
                        "row": row,
                        "sha256": digest.hexdigest(),
                        "tensor_sha256": hashes,
                        "sequence_length": batch["input_ids"].shape[1],
                        "supervised_tokens": _count_label_tokens(batch["labels"][row], ignore_index),
                    }
                )
        payload = {
            "step": step,
            "rank": self.dist_env.rank,
            "world_size": self.dist_env.world_size,
            "mesh": self._mesh_coordinates(),
            "dp_rank": self._get_dp_rank(),
            "sampler": type(self.dataloader.sampler).__qualname__,
            "shuffle": self.cfg.get("dataloader.shuffle"),
            "seed": self.cfg.get("seed", 42),
            "rows": records,
        }
        path = directory / f"rank-{self.dist_env.rank}.jsonl"
        with path.open("x" if step == 0 else "a") as stream:
            stream.write(json.dumps(payload) + "\n")

    def _write_loaded_model_audit(self) -> None:
        """Optionally inspect parameters and buffers after native distributed setup.

        No tensor is unsharded, cast or repaired. Large tensors use 16 evenly
        spaced local elements, avoiding a flatten that could copy a noncontiguous
        expert tensor. Small tensors include an exact raw-byte hash. This audit
        records loaded state, but does not itself compare against checkpoint
        source values.
        """
        if self._bench_loaded_model_audit_path is None:
            return
        directory = pathlib.Path(self._bench_loaded_model_audit_path)
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        with torch.no_grad():
            for part_index, model in enumerate(self.model_parts):
                for kind, tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
                    for name, tensor in tensors:
                        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
                        local = local.detach()
                        if local.numel() <= 8192:
                            sampled = local.cpu().contiguous()
                            full_hash = hashlib.sha256(
                                sampled.reshape(-1).view(torch.uint8).numpy().tobytes()
                            ).hexdigest()
                            values = sampled.reshape(-1)[:16].tolist()
                            count = local.numel()
                        else:
                            indices = torch.linspace(0, local.numel() - 1, 16, dtype=torch.long, device=local.device)
                            coordinates = torch.unravel_index(indices, local.shape)
                            sampled = local[coordinates].cpu().contiguous()
                            full_hash = None
                            values = sampled.tolist()
                            count = sampled.numel()
                        records.append(
                            {
                                "part": part_index,
                                "kind": kind,
                                "name": name,
                                "global_shape": list(tensor.shape),
                                "local_shape": list(local.shape),
                                "dtype": str(local.dtype),
                                "device": str(local.device),
                                "requires_grad": tensor.requires_grad,
                                "placements": [str(item) for item in tensor.placements]
                                if isinstance(tensor, DTensor)
                                else [],
                                "sample_values": values,
                                "hashed_elements": count,
                                "sample_sha256": hashlib.sha256(
                                    sampled.reshape(-1).view(torch.uint8).numpy().tobytes()
                                ).hexdigest(),
                                "full_local_sha256": full_hash,
                            }
                        )
        payload = {
            "scope": "after_native_distributed_setup_before_first_forward",
            "rank": self.dist_env.rank,
            "world_size": self.dist_env.world_size,
            "mesh": self._mesh_coordinates(),
            "dp_rank": self._get_dp_rank(),
            "checkpoint": self.cfg.get("model.pretrained_model_name_or_path"),
            "tensors": records,
        }
        path = directory / f"rank-{self.dist_env.rank}.json"
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2)

    def _record_batch_flops(self, batches: list[dict[str, torch.Tensor]], *, num_label_tokens: int) -> None:
        """Record one accumulation window without duplicating CP or PP work.

        Args:
            batches: Original CPU input/label/position/mask tensors, each with
                shape [local_batch, global_sequence], before CP sharding.
            num_label_tokens: Supervised-token count already reduced over DP.
        """
        self._bench_step_supervised_tokens = num_label_tokens
        if self._batch_flops_counter is None:
            self._bench_all_token_head_tflops = self.tflops
            self._bench_step_tokens = self.cfg.step_scheduler.global_batch_size * self._bench_seq_len
            return
        ignore_index = _get_loss_ignore_index(self.loss_fn)
        counts = [self._batch_flops_counter(batch, ignore_index=ignore_index) for batch in batches]
        totals = torch.tensor(
            [
                sum(count.flops for count in counts) / 1e12,
                sum(count.all_token_head_flops for count in counts) / 1e12,
                sum(count.num_tokens for count in counts),
                sum(count.num_supervised_tokens for count in counts),
            ],
            dtype=torch.float64,
            device=self.dist_env.device,
        )
        # Each CP rank and PP stage sees the same logical batch. Only DP owns
        # additional independent examples; the numerator must count them once.
        totals = self._dp_allreduce(totals, include_cp=False).tolist()
        self.tflops, self._bench_all_token_head_tflops = totals[:2]
        self._bench_step_tokens = int(totals[2])
        if int(totals[3]) != num_label_tokens:
            raise ValueError("Batch FLOPs supervision count disagrees with the native global loss count")

    def _log_iteration_metrics(self, iter_timer, ga_steps, peak_tflops, rank, iteration):
        max_iter_time = self.timers._get_global_min_max_time([iter_timer], reset=False, barrier=False, normalizer=1.0)[
            iter_timer
        ][1]

        self._bench_iterations.append(
            _IterationMetrics(
                iteration,
                max_iter_time,
                self.tflops,
                self._bench_all_token_head_tflops,
                self._bench_step_tokens,
                self._bench_step_supervised_tokens,
            )
        )
        if rank == 0:
            mfu = calculate_mfu(
                self.tflops,
                self.dist_env.world_size,
                max_iter_time,
                reference_mfu=peak_tflops,
            )
            logger.info(f"Max iter time: {max_iter_time:.6f} seconds")
            logger.info(f"MFU: {mfu:.6f}% | FLOPs scope: {self._bench_flops_scope}")
            if self._wandb_enabled and self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "benchmark/mfu_percent": mfu,
                        "benchmark/iteration_seconds": max_iter_time,
                        "benchmark/tflops_per_step": self.tflops,
                        "benchmark/all_token_head_mfu_percent": calculate_mfu(
                            self._bench_all_token_head_tflops,
                            self.dist_env.world_size,
                            max_iter_time,
                            reference_mfu=peak_tflops,
                        ),
                        "benchmark/valid_tokens": self._bench_step_tokens,
                        "benchmark/supervised_tokens": self._bench_step_supervised_tokens,
                    },
                    step=iteration,
                )

        # Log detailed timers
        timer_names = [iter_timer, "optimizer"] + [f"forward_backward_{ga_step_idx}" for ga_step_idx in range(ga_steps)]
        # Log timers to wandb
        if self._wandb_enabled:
            self.timers.write_to_wandb(
                names=timer_names,
                writer=self.wandb_run,
                iteration=iteration,
                normalizer=1.0,
                reset=False,
                barrier=False,
            )

        self.timers.log(
            names=timer_names,
            rank=0,
            normalizer=1000.0,  # Convert to seconds
            reset=True,
            barrier=True,
        )

    def _collect_cuda_memory_peaks(self) -> list[_CudaMemoryPeak]:
        """Collect allocator peaks after timing, before allocating collective buffers.

        Parent setup resets the peak statistics before constructing the model.
        Peaks include setup, warmup and measured iterations, with no benchmark reset.
        CUDA allocations outside the PyTorch allocator are not represented.
        Retry/OOM counters cover the process lifetime; resetting peak memory
        statistics does not reset these accumulated counters.
        """
        allocated = torch.cuda.max_memory_allocated(self.dist_env.device)
        reserved = torch.cuda.max_memory_reserved(self.dist_env.device)
        stats = torch.cuda.memory_stats(self.dist_env.device)
        local = torch.tensor(
            [allocated, reserved, stats["num_alloc_retries"], stats["num_ooms"]],
            dtype=torch.int64,
            device=self.dist_env.device,
        )
        gathered = torch.empty(4 * self.dist_env.world_size, dtype=torch.int64, device=self.dist_env.device)
        torch.distributed.all_gather_into_tensor(gathered, local)
        return [_CudaMemoryPeak(rank, *peaks) for rank, peaks in enumerate(gathered.view(-1, 4).cpu().tolist())]

    def _log_benchmark_summary(self, steps, warmup_steps, peak_tflops, rank):
        memory_peaks = self._collect_cuda_memory_peaks()
        torch.distributed.barrier()
        if rank == 0:
            logger.info(f"{'=' * 60}")
            logger.info("Benchmarking Summary")
            logger.info(f"{'=' * 60}")

        # Get active times for summary
        setup_time = self.timers._timers["setup"].active_time() if "setup" in self.timers._timers else 0
        measured = [item for item in self._bench_iterations if item.step >= warmup_steps]
        if not measured:
            raise ValueError("Benchmark needs at least one measured iteration after warmup")
        iter_time = sum(item.seconds for item in measured)
        warmup_time = sum(item.seconds for item in self._bench_iterations if item.step < warmup_steps)
        measured_tflops = sum(item.tflops for item in measured)
        all_token_head_tflops = sum(item.all_token_head_tflops for item in measured)
        mean_step_tflops = measured_tflops / len(measured)

        if rank == 0:
            logger.info(f"Total setup time: {setup_time:.2f} seconds")
            logger.info(f"Total warmup time ({warmup_steps} steps): {warmup_time:.2f} seconds")
            logger.info(f"Total iteration time ({steps - warmup_steps} steps): {iter_time:.2f} seconds")

            # Aggregate useful work over the sum of per-step slowest-rank times.
            avg_iter_time = iter_time / len(measured)

            logger.info(
                f"Average iteration time: {avg_iter_time:.3f} seconds"
                + (
                    f" (excluding first {warmup_steps} warmup iterations)"
                    if steps > warmup_steps
                    else f" (all {steps} iterations)"
                )
            )

            mfu = calculate_mfu(measured_tflops, self.dist_env.world_size, iter_time, reference_mfu=peak_tflops)
            logger.info(
                f"Average MFU: {mfu:.6f}% | FLOPs scope: {self._bench_flops_scope}"
                + (
                    f" (excluding first {warmup_steps} warmup iterations)"
                    if steps > warmup_steps
                    else f" (all {steps} iterations)"
                )
            )
            logger.info(f"{'=' * 60}\n")

            # Prepare summary data
            summary_data = {
                "total_steps": steps,
                "warmup_steps": warmup_steps,
                "training_steps": steps - warmup_steps,
                "setup_time_seconds": setup_time,
                "warmup_time_seconds": warmup_time,
                "training_time_seconds": iter_time,
                "avg_iter_time_seconds": avg_iter_time,
                "avg_mfu_percent": mfu,
                "flops_scope": self._bench_flops_scope,
                "tflops_per_gpu": mean_step_tflops / self.dist_env.world_size,
                "mean_tflops_per_step": mean_step_tflops,
                "all_token_head_mfu_percent": calculate_mfu(
                    all_token_head_tflops,
                    self.dist_env.world_size,
                    iter_time,
                    reference_mfu=peak_tflops,
                ),
                "flops_accounting": "actual_documents_supervised_head"
                if self._batch_flops_counter is not None
                else "shape_formula",
                "timing_accounting": "sum_of_per_step_max_rank_seconds",
                "iterations": [asdict(item) for item in self._bench_iterations],
                "cuda_memory_peak_scope": "setup_warmup_and_measured_iterations_before_summary_collectives",
                "cuda_memory_peaks": [asdict(item) for item in memory_peaks],
                "cuda_allocator_counter_scope": "process_lifetime_before_summary_collectives",
                "cuda_max_allocated_bytes": max(item.max_allocated_bytes for item in memory_peaks),
                "cuda_max_reserved_bytes": max(item.max_reserved_bytes for item in memory_peaks),
                "peak_tflops": peak_tflops,
                "world_size": self.dist_env.world_size,
                "global_batch_size": self.cfg.step_scheduler.global_batch_size,
                "local_batch_size": self.cfg.get("step_scheduler.local_batch_size"),
                "seq_len": self._bench_seq_len,
            }

            # Log to wandb as table
            if self.wandb_run is not None:
                import wandb

                # Create a table with the summary data (use raw numeric values)
                summary_table = wandb.Table(
                    columns=["Metric", "Value"],
                    data=[
                        ["Total Steps", steps],
                        ["Warmup Steps", warmup_steps],
                        ["Training Steps", steps - warmup_steps],
                        ["Setup Time (s)", setup_time],
                        ["Warmup Time (s)", warmup_time],
                        ["Training Time (s)", iter_time],
                        ["Avg Iteration Time (s)", avg_iter_time],
                        ["Avg MFU (%)" if self._bench_flops_scope == "model" else "Avg text-backbone MFU (%)", mfu],
                        ["TFLOPs/GPU/s", peak_tflops * mfu / 100],
                        ["Peak TFLOPs", peak_tflops],
                        ["World Size", self.dist_env.world_size],
                        ["Global Batch Size", self.cfg.step_scheduler.global_batch_size],
                        ["Local Batch Size", self.cfg.get("step_scheduler.local_batch_size")],
                        ["Sequence Length", self._bench_seq_len],
                    ],
                )
                wandb.log({"benchmark_summary": summary_table})

                # Also log as scalar metrics for easy filtering
                wandb.log(
                    {
                        "summary/avg_iter_time_seconds": avg_iter_time,
                        "summary/avg_mfu_percent": mfu,
                        "summary/flops_scope": self._bench_flops_scope,
                        "summary/training_time_seconds": iter_time,
                        "summary/tflops_per_gpu": mean_step_tflops / self.dist_env.world_size,
                        "summary/mean_tflops_per_step": mean_step_tflops,
                        "summary/cuda_max_allocated_bytes": summary_data["cuda_max_allocated_bytes"],
                        "summary/cuda_max_reserved_bytes": summary_data["cuda_max_reserved_bytes"],
                    }
                )

            # Save summary to JSON file if output path is provided
            if self._bench_json_output_path is not None:
                summary_file = pathlib.Path(self._bench_json_output_path)
                # Create parent directory if it doesn't exist
                summary_file.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_file, "w") as f:
                    json.dump(summary_data, f, indent=2)
                logger.info(f"Benchmark summary saved to {summary_file.absolute()}")

        # Finish wandb run
        if self.wandb_run is not None and rank == 0:
            import wandb

            wandb.finish()


def main(config_path=None):
    """Main entry point for the benchmarking recipe.

    Loads the configuration, sets up the recipe, and runs the benchmark.
    """
    if config_path is None:
        # Default to moonlight_16b_torch.yaml in examples/llm_benchmark/moonlight
        config_path = (
            pathlib.Path(__file__).parent.parent.parent.resolve()
            / "examples"
            / "benchmark"
            / "configs"
            / "moonlight_16b_torch.yaml"
        )

    cfg = parse_args_and_load_config(config_path)
    recipe = BenchmarkingRecipeForNextTokenPrediction(cfg)
    try:
        recipe.setup()
        recipe.run_benchmark()
    finally:
        if recipe.partial_cuda_graph_manager is not None:
            recipe.partial_cuda_graph_manager.close()
            recipe.partial_cuda_graph_manager = None


if __name__ == "__main__":
    main()
