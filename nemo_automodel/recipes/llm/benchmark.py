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
from typing import Optional
import torch
from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

(flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn), process_flash_kwargs_fn = lazy_import_flash_attention(implementation=None)

from transformers.modeling_flash_attention_utils import fa_peft_integration_check

def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    implementation: Optional[str] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    (Optional) kwargs are described further in `_process_flash_attention_kwargs` and `FlashAttentionKwargs`.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
    """
    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    # Extract the flash attention kwargs that have been requested (and are supported by the implementation)
    flash_kwargs = process_flash_kwargs_fn(
        query_length=query_length,
        key_length=key_states.size(1),
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        use_top_left_mask=use_top_left_mask,
        softcap=softcap,
        deterministic=deterministic,
        **kwargs,
    )

    # We will use `flash_varlen_fn` to prevent cross-example attention and also allow padding free approach under two cases:
    # Case 1. If position ids is provided and the position ids indicate packed sequences, see `_is_packed_sequence`.
    # Case 2. Some models pass directly pre-computed `cu_seqlens` so we don't need to infer it from position ids. It is safe to
    # use `flash_varlen_fn` knowing we already have all necessary the kwargs.
    #
    # NOTE: it is user's responsibility to take care of flattenning `position_ids` if that's needed by the model.
    # See #39121 for more information.
    # with torch.profiler.record_function("is_fa_with_position_ids"):
    #     is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))
    # with torch.profiler.record_function("is_fa_with_varlen_kwargs"):
    #     is_fa_with_varlen_kwargs = all(
    #         kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    #     )
    is_fa_with_position_ids = False
    is_fa_with_varlen_kwargs = False
    
    # Contains at least one padding token in the sequence
    # if attention_mask is not None:
    #     with torch.profiler.record_function("flash_attention_with_padding"):
    #         q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
    #             query_states, key_states, value_states, attention_mask, query_length, unpad_fn
    #         )

    #         # TODO for now this is required to work with
    #         # https://huggingface.co/kernels-community/metal-flash-sdpa/blob/main/torch-ext/metal_flash_sdpa/__init__.py
    #         if "mps" in str(q.device):
    #             cu_seq_lens_k = cu_seq_lens_k.clone()

    #         out_unpad = flash_varlen_fn(
    #             q,
    #             k,
    #             v,
    #             cu_seqlens_q=cu_seq_lens_q,
    #             cu_seqlens_k=cu_seq_lens_k,
    #             max_seqlen_q=max_length_q,
    #             max_seqlen_k=max_length_k,
    #             **flash_kwargs,
    #         )
    #         if isinstance(out_unpad, tuple):
    #             out_unpad = out_unpad[0]

    #         out = pad_fn(out_unpad, indices_q, query_states.size(0), query_length)

    # Padding free, i.e. sequences flattened into one total sequence
    # elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
    #     with torch.profiler.record_function("flash_attention_varlen"):
    #         if cu_seq_lens_q is None or cu_seq_lens_k is None:
    #             q, k, v, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _prepare_from_posids(
    #                 query_states, key_states, value_states, position_ids, query_length=query_length
    #             )
    #         else:
    #             q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
    #             k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
    #             v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

    #         # TODO for now this is required to work with
    #         # https://huggingface.co/kernels-community/metal-flash-sdpa/blob/main/torch-ext/metal_flash_sdpa/__init__.py
    #         if "mps" in str(q.device):
    #             cu_seq_lens_k = cu_seq_lens_k.clone()

    #         out = flash_varlen_fn(
    #             q,
    #             k,
    #             v,
    #             cu_seqlens_q=cu_seq_lens_q,
    #             cu_seqlens_k=cu_seq_lens_k,
    #             max_seqlen_q=max_length_q,
    #             max_seqlen_k=max_length_k,
    #             **flash_kwargs,
    #         )
    #         if isinstance(out, tuple):
    #             out = out[0]

    #         out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))

    # # No padding
    # else:
    out = flash_fn(query_states, key_states, value_states, **flash_kwargs)
    if isinstance(out, tuple):
        out = out[0]

    # Print debug info only once
    if not hasattr(_flash_attention_forward, '_debug_printed'):
        print("[debug] overloaded flash attention", flush=True)
        _flash_attention_forward._debug_printed = True
    # if is_fa_with_position_ids:
    #     print(f"is_fa_with_position_ids={is_fa_with_position_ids}, is_fa_with_varlen_kwargs={is_fa_with_varlen_kwargs}, attention_mask={attention_mask} position_ids={position_ids}", flush=True)

    return out

# patch the original flash attention forward
import transformers.modeling_flash_attention_utils as fa_utils
fa_utils._flash_attention_forward = _flash_attention_forward



import logging
import pathlib

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.training.timers import Timers
from nemo_automodel.components.training.utils import (
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from nemo_automodel.components.utils.flops_utils import calculate_mfu, get_flops_formula_for_hf_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

logger = logging.getLogger(__name__)


class BenchmarkingRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction):
    """Benchmarking recipe for next-token prediction.

    This class extends TrainFinetuneRecipeForNextTokenPrediction to provide
    a simplified benchmarking-focused training loop with timers and profiling support.
    It reuses the setup() and _forward_backward_step() methods from the parent class.
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

        # Infer max_steps from step_scheduler
        self._bench_steps = cfg.step_scheduler.max_steps

        # Get seq_len from dataset config
        self._bench_seq_len = cfg.dataset.seq_len

        # Infer vocab_size from model config and inject it into dataset config
        if hasattr(cfg, "dataset") and hasattr(cfg, "model"):
            # Get vocab_size from model config
            if hasattr(cfg.model, "config") and hasattr(cfg.model.config, "pretrained_model_name_or_path"):
                from transformers import AutoConfig

                model_config = AutoConfig.from_pretrained(cfg.model.config.pretrained_model_name_or_path)
                vocab_size = model_config.vocab_size
                # Inject vocab_size into dataset config
                cfg.dataset.vocab_size = vocab_size
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"Inferred vocab_size={vocab_size} from model config")

        # Inject batch_size from step_scheduler into dataset config
        if hasattr(cfg, "dataset") and hasattr(cfg, "step_scheduler"):
            local_batch_size = getattr(cfg.step_scheduler, "local_batch_size", 1)
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

        # Clear validation dataloader (not needed for benchmarking)
        self.val_dataloader = None

        # Get step_scheduler config
        seq_len = self._bench_seq_len
        global_batch_size = self.cfg.step_scheduler.global_batch_size

        # Calculate FLOPs
        flops_formula = get_flops_formula_for_hf_config(self.model_parts[0].config)
        flops = flops_formula(self.model_parts[0].config, gbs=global_batch_size, seq_len=seq_len)
        self.tflops = flops / (10**12)

        if self.dist_env.is_main:
            logger.info(f"TFLOPs/GPU: {self.tflops:.6f}")

        self.timers.log(
            names=["setup"],
            rank=0,
            normalizer=1000.0,  # Convert to seconds
            reset=True,
            barrier=True,
        )

    def run_benchmark(self):
        """Run the benchmarking loop.

        This method implements a simplified training loop focused on benchmarking
        with timers and profiling support, similar to the original benchmarking script.
        """
        rank = self.dist_env.rank
        device = self.dist_env.device

        # Get benchmarking config
        steps = self._bench_steps
        warmup_steps = self._bench_warmup_steps
        local_batch_size = self.cfg.step_scheduler.local_batch_size
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
            # Start nsys profiling if configured
            if i == nsys_start and rank in nsys_ranks:
                logger.info(f"Rank {rank} | Starting nsys profiling")
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

            if rank == 0:
                logger.info(f"Rank {rank} | Iteration {i}")

            # Zero gradients
            for opt in self.optimizer:
                opt.zero_grad()

            # Time the iteration
            iter_timer = "iteration_warmup" if i < warmup_steps else "iteration"
            with self.timers(iter_timer, log_level=1):
                # Gradient accumulation loop
                num_label_tokens = 0
                loss_buffer = []
                prepare_for_grad_accumulation(self.model_parts, pp_enabled=self.pp_enabled)

                for ga_step_idx in range(ga_steps):
                    if ga_step_idx == ga_steps - 1:
                        prepare_for_final_backward(self.model_parts, pp_enabled=self.pp_enabled)

                    # Get batch from dataloader
                    batch = next(dataloader_iter)
                    torch.cuda.nvtx.range_push(f"iteration_{i}_ga_step_{ga_step_idx}")

                    num_label_tokens += (batch["labels"] != -100).sum().item()

                    with self.timers(f"forward_backward_{ga_step_idx}", log_level=2):
                        self._forward_backward_step(
                            ga_step_idx,
                            batch,
                            loss_buffer=loss_buffer,
                            num_label_tokens=None,
                            num_batches=ga_steps,
                            is_train=True,
                        )

                loss = (
                    torch.sum(torch.stack(loss_buffer)).to(device)
                    if loss_buffer
                    else torch.tensor([-1.0], device=device)
                )
                if self.pp_enabled:
                    src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
                    if self.dist_env.rank == src_rank:
                        torch.distributed.send(loss, dst=0)
                    elif self.dist_env.is_main:
                        torch.distributed.recv(loss, src=src_rank)

                if rank == 0:
                    print(f"num_label_tokens={num_label_tokens} | loss={loss.detach().item():.4f}")
                    loss = loss / num_label_tokens
                    logger.info(
                        f"Rank {rank} | Iteration {i} | num_label_tokens={num_label_tokens} | "
                        f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB | "
                        f"loss={loss.detach().item():.4f}"
                    )

                    torch.cuda.nvtx.range_pop()

                # Optimizer step
                with self.timers("optimizer", log_level=2):
                    for opt in self.optimizer:
                        opt.step()
                    logger.debug("Optimizer step")

            # Calculate and log MFU
            self._log_iteration_metrics(iter_timer, ga_steps, peak_tflops, rank)

            # Stop nsys profiling if configured
            if i == nsys_end and rank in nsys_ranks:
                logger.info(f"Rank {rank} | Stopping nsys profiling")
                torch.cuda.cudart().cudaProfilerStop()

        # Final summary
        self._log_benchmark_summary(steps, warmup_steps, peak_tflops, rank)

    def _log_iteration_metrics(self, iter_timer, ga_steps, peak_tflops, rank):
        max_iter_time = self.timers._get_global_min_max_time([iter_timer], reset=False, barrier=False, normalizer=1.0)[
            iter_timer
        ][1]

        if rank == 0:
            mfu = calculate_mfu(
                self.tflops,
                self.dist_env.world_size,
                max_iter_time,
                reference_mfu=peak_tflops,
            )
            logger.info(f"Max iter time: {max_iter_time:.6f} seconds")
            logger.info(f"MFU: {mfu:.6f}%")

        # Log detailed timers
        self.timers.log(
            names=[iter_timer, "optimizer"] + [f"forward_backward_{ga_step_idx}" for ga_step_idx in range(ga_steps)],
            rank=0,
            normalizer=1000.0,  # Convert to seconds
            reset=True,
            barrier=True,
        )

    def _log_benchmark_summary(self, steps, warmup_steps, peak_tflops, rank):
        torch.distributed.barrier()
        if rank == 0:
            logger.info(f"{'=' * 60}")
            logger.info("Benchmarking Summary")
            logger.info(f"{'=' * 60}")

        # Get active times for summary
        setup_time = self.timers._timers["setup"].active_time() if "setup" in self.timers._timers else 0
        iter_time = self.timers._timers["iteration"].active_time() if "iteration" in self.timers._timers else 0
        warmup_time = (
            self.timers._timers["iteration_warmup"].active_time() if "iteration_warmup" in self.timers._timers else 0
        )

        if rank == 0:
            logger.info(f"Total setup time: {setup_time:.2f} seconds")
            logger.info(f"Total warmup time ({warmup_steps} steps): {warmup_time:.2f} seconds")
            logger.info(f"Total iteration time ({steps - warmup_steps} steps): {iter_time:.2f} seconds")

            # Calculate average iteration time
            if steps > warmup_steps:
                avg_iter_time = iter_time / (steps - warmup_steps)
            else:
                avg_iter_time = iter_time / steps

            logger.info(
                f"Average iteration time: {avg_iter_time:.3f} seconds"
                + (
                    f" (excluding first {warmup_steps} warmup iterations)"
                    if steps > warmup_steps
                    else f" (all {steps} iterations)"
                )
            )

            mfu = calculate_mfu(self.tflops, self.dist_env.world_size, avg_iter_time, reference_mfu=peak_tflops)
            logger.info(
                f"Average MFU: {mfu:.6f}%"
                + (
                    f" (excluding first {warmup_steps} warmup iterations)"
                    if steps > warmup_steps
                    else f" (all {steps} iterations)"
                )
            )
            logger.info(f"{'=' * 60}\n")


def main(config_path=None):
    """Main entry point for the benchmarking recipe.

    Loads the configuration, sets up the recipe, and runs the benchmark.
    """
    if config_path is None:
        # Default to moonlight_16b_torch.yaml in examples/benchmark/configs
        config_path = (
            pathlib.Path(__file__).parent.parent.parent.resolve()
            / "examples"
            / "benchmark"
            / "configs"
            / "moonlight_16b_torch.yaml"
        )

    cfg = parse_args_and_load_config(config_path)
    recipe = BenchmarkingRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_benchmark()


if __name__ == "__main__":
    main()
