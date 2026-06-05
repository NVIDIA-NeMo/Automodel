# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""SGLang ModelRunner forward for the EAGLE-3 target (server-side, GPU only).

This module owns every SGLang-internal touch point so the rest of the
speculative stack stays SGLang-agnostic and importable without SGLang. It is
imported lazily (only from :meth:`SGLangEagle3TargetModel.from_pretrained`).

Mechanism (mirrors SpecForge's SGLang backend, which is the only path that
returns the supervision tensors directly without a Mooncake transfer layer):

1. Build a SGLang ``ModelRunner`` with ``enable_return_hidden_states=True``.
2. Wrap the model's ``LogitsProcessor`` so a single extend forward returns
   *all-position* full-vocab logits (stock SGLang only keeps the last position)
   alongside the three concatenated EAGLE-3 auxiliary hidden states.
3. Run one extend per request and stack the per-row results into batched
   ``[batch, seq, *]`` tensors for :class:`SGLangEagle3TargetModel`.

Unlike SpecForge (which embeds the target inside the training job and reuses the
trainer's TP process group), this runs in a *standalone* server process, so it
performs SGLang's own single-process distributed init rather than reusing an
external group, and it drops SpecForge's ``shard_returns`` / VLM paths that only
matter inside the training loop.

SGLang's private ``LogitsProcessor`` helpers are version-coupled: the calls here
track ``sglang==0.5.9`` (SpecForge's pin). This forward path requires a GPU and
SGLang, so it is validated on the training server, not in CPU unit tests; the
CPU tests exercise the contract layer in
:mod:`nemo_automodel.components.speculative.eagle.sglang_target` against a fake
runner instead.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)


def _wrap_logits_processors_for_eagle3(model) -> None:  # pragma: no cover - requires GPU + SGLang
    """Replace every SGLang ``LogitsProcessor`` in ``model`` with an EAGLE-3 wrapper.

    The wrapper makes one extend forward return all-position full-vocab logits
    plus the concatenated auxiliary hidden states, instead of stock SGLang's
    last-position-only logits. Ported (simplified, no tensor-parallel sharding)
    from SpecForge ``sglang_backend/utils.py`` for ``sglang==0.5.9``.
    """
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    class _LogitsProcessorForEagle3(torch.nn.Module):
        def __init__(self, inner: LogitsProcessor):
            super().__init__()
            self.logits_processor = inner

        def forward(
            self,
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states: Optional[list] = None,
            hidden_states_before_norm: Optional[torch.Tensor] = None,
        ):
            lp = self.logits_processor
            # Force the extend forward down the decode branch so the patched
            # path below is taken for every position, matching SpecForge.
            logits_metadata.forward_mode = ForwardMode.DECODE
            if isinstance(logits_metadata, ForwardBatch):
                logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

            pruned_states, pruned_states_before_norm, aux_pruned_states, sample_indices, *_ = lp._get_pruned_states(
                hidden_states,
                hidden_states_before_norm,
                aux_hidden_states,
                logits_metadata,
            )
            logits = self._compute_full_logits(lp, pruned_states, lm_head, logits_metadata)
            aux = lp._get_hidden_states_to_store(
                hidden_states,
                hidden_states_before_norm,
                aux_hidden_states,
                pruned_states,
                pruned_states_before_norm,
                aux_pruned_states,
                sample_indices,
                logits_metadata,
            )
            return _Eagle3LogitsOutput(logits=logits, aux_hidden_states=aux)

        @staticmethod
        def _compute_full_logits(lp, hidden_states, lm_head, logits_metadata):
            from sglang.srt.distributed import tensor_model_parallel_all_gather
            from sglang.srt.utils.common import is_npu

            hidden_states, local_hidden_states = lp._gather_dp_attn_hidden_states(hidden_states, logits_metadata)
            logits = lp._compute_lm_head(hidden_states, lm_head, None)
            if lp.logit_scale is not None:
                logits.mul_(lp.logit_scale)
            if lp.do_tensor_parallel_all_gather:
                if lp.use_attn_tp_group:
                    logits = lp._gather_attn_tp_logits(logits)
                else:
                    logits = tensor_model_parallel_all_gather(logits)
            logits = lp._scatter_dp_attn_logits(logits, local_hidden_states, logits_metadata)
            logits = lp._copy_logits_to_buffer(logits, logits_metadata)
            if lp.final_logit_softcapping:
                if not is_npu():
                    from sglang.srt.layers.logits_processor import fused_softcap

                    fused_softcap(logits, lp.final_logit_softcapping)
                else:
                    cap = lp.final_logit_softcapping
                    logits = cap * torch.tanh(logits / cap)
            return logits

    for name, submodule in list(model.named_modules()):
        if isinstance(submodule, LogitsProcessor):
            setattr(model, name, _LogitsProcessorForEagle3(submodule))
            logger.info("wrapped %s with EAGLE-3 logits processor", name)


class _Eagle3LogitsOutput:  # pragma: no cover - only built inside the GPU-only wrapper
    """Carries the all-position logits + aux hidden states out of the wrapper."""

    def __init__(self, logits: torch.Tensor, aux_hidden_states: torch.Tensor):
        self.logits = logits
        self.aux_hidden_states = aux_hidden_states


class SGLangTargetRunner:
    """Standalone SGLang ModelRunner that returns EAGLE-3 supervision tensors.

    Built via :meth:`build`; consumed through the
    :class:`~nemo_automodel.components.speculative.eagle.sglang_target.SGLangRunnerProtocol`
    surface (``model`` / ``set_aux_layers`` / ``forward_eagle3`` /
    ``input_embedding_weight``).
    """

    def __init__(self, model_runner):
        self._model_runner = model_runner

    @property
    def model(self):
        """The loaded nn.Module (exposes ``.config`` and ``.parameters()``)."""
        return self._model_runner.model

    @classmethod
    def build(  # pragma: no cover - requires GPU + SGLang
        cls,
        model_path: str,
        *,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        trust_remote_code: bool = False,
        **sglang_kwargs,
    ) -> "SGLangTargetRunner":
        """Construct the SGLang ModelRunner for a standalone target server.

        ``sglang_kwargs`` are forwarded to ``ServerArgs`` (e.g. ``page_size``,
        ``mem_fraction_static``, ``attention_backend``). The constructor mirrors
        SpecForge's ``sglang==0.5.9`` usage and is GPU/SGLang-only.
        """
        import torch.distributed as dist
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.server_args import ServerArgs

        if not torch.cuda.is_available():
            raise RuntimeError("SGLangTargetRunner requires CUDA; run it on a GPU server, not the editing host.")

        server_args = ServerArgs(
            model_path=model_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype if dtype is not None else "auto",
            enable_return_hidden_states=True,
            disable_cuda_graph=True,  # extend-only forward; CUDA graphs add no benefit here
            disable_radix_cache=True,
            tp_size=tp_size,
            pp_size=1,
            **sglang_kwargs,
        )

        gpu_id = torch.cuda.current_device()
        # Standalone server: SGLang manages its own single-rank world rather than
        # reusing a trainer process group (the SpecForge embedded case).
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(int(server_args.port or 8000) + 200))
            init_distributed_environment(
                backend="nccl",
                world_size=tp_size,
                rank=0,
                local_rank=gpu_id,
                distributed_init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            )
            initialize_model_parallel(tensor_model_parallel_size=tp_size)

        model_config = ModelConfig.from_server_args(server_args)
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=0,
            tp_size=tp_size,
            pp_rank=0,
            pp_size=1,
            nccl_port=None,
            server_args=server_args,
        )
        _wrap_logits_processors_for_eagle3(model_runner.model)
        return cls(model_runner)

    def set_aux_layers(self, aux_layer_ids: Sequence[int]) -> None:
        """Tell the SGLang model which 3 decoder layers to capture."""
        self._model_runner.model.set_eagle3_layers_to_capture(list(aux_layer_ids))

    def input_embedding_weight(self) -> torch.Tensor:
        """Return the target input-embedding weight ``[vocab, hidden]``."""
        return self._model_runner.model.get_input_embeddings().weight

    @torch.no_grad()
    def forward_eagle3(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one extend per row and stack the per-position logits and aux states.

        Returns ``(logits[batch, seq, vocab], aux[batch, seq, 3 * hidden])``,
        both unshifted; the contract layer applies the EAGLE-3 shift. Sequences
        must share a length (training batches are padded), so the per-row
        results stack cleanly.
        """
        del attention_mask  # each row is a full causal sequence in SGLang
        logits_list, aux_list = self._extend(input_ids)
        logits = torch.stack(logits_list, dim=0)
        aux = torch.stack(aux_list, dim=0)
        return logits, aux

    def _extend(self, input_ids: torch.Tensor) -> tuple[list, list]:  # pragma: no cover - requires GPU + SGLang
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.radix_cache import RadixCache
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        runner = self._model_runner
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        rows = torch.split(input_ids, 1, dim=0)
        reqs, input_lens = [], []
        for idx, row in enumerate(rows):
            token_ids = row.view(-1).tolist()
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=token_ids,
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)
            input_lens.append(len(token_ids))

        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            page_size=runner.server_args.page_size,
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=RadixCache(cache_params),
            model_config=runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        output = runner.forward(forward_batch).logits_output

        logits_list = list(torch.split(output.logits, input_lens, dim=0))
        aux_list = list(torch.split(output.aux_hidden_states, input_lens, dim=0))
        runner.req_to_token_pool.clear()
        runner.token_to_kv_pool_allocator.clear()
        return logits_list, aux_list

    def close(self) -> None:
        """Release the SGLang model runner (best effort)."""
        runner, self._model_runner = self._model_runner, None
        if runner is None:
            return
        for name in ("token_to_kv_pool_allocator", "req_to_token_pool"):
            pool = getattr(runner, name, None)
            clear = getattr(pool, "clear", None)
            if clear is not None:
                clear()
