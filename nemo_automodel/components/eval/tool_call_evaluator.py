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
"""Generation-based evaluator for tool-call accuracy during agent SFT.

The loss-only validation that ships with the training recipe cannot
distinguish "loss going down because the model learned the format" from
"loss going down because the model is overfitting the response style
while emitting wrong tool names". This evaluator closes that gap by
running ``model.generate()`` on held-out prompts that terminate right
before an assistant tool-call turn, parsing the generated text with
:mod:`nemo_automodel.components.eval.tool_call_parser`, and comparing
against the ground-truth tool calls extracted from the dataset.

The evaluator is intentionally framework-agnostic: it operates on any
HuggingFace-style model with a ``.generate()`` method and a tokenizer
that supports ``apply_chat_template(..., tools=...)``. Distributed
sharding and all-reduce of metrics are left to the caller (the training
recipe), which already has the dist environment in hand.
"""

import gc
import logging
from typing import Any, Dict, List, Optional, Union

import torch

from nemo_automodel.components.datasets.llm.agent_chat import (
    make_agent_chat_eval_samples,
)
from nemo_automodel.components.eval.tool_call_parser import (
    compute_sample_metrics,
    parse_tool_calls,
)

logger = logging.getLogger(__name__)


_METRIC_KEYS = (
    "has_call",
    "name_correct",
    "args_json_valid",
    "args_field_recall",
    "args_field_precision",
    "args_exact_match",
)


class ToolCallAccuracyEvaluator:
    """Generation-based tool-call accuracy evaluator for agent SFT.

    The evaluator lazily loads a list of eval samples (one per assistant
    tool-call position in the source dataset). On each call to
    :meth:`evaluate` it renders each sample's ``prompt_messages`` and
    ``tools`` through the tokenizer's chat template, generates a
    continuation, parses any tool calls out of the generated text, and
    aggregates per-sample metrics into a corpus-level dict.

    Constructor args (all keyword-only):
        dataset_name: HF Hub dataset id to load eval samples from.
            Mutually exclusive with ``path``.
        path: Local JSON/JSONL file (or list of files) to load eval
            samples from. Mutually exclusive with ``dataset_name``.
        split: Dataset split (only used with ``dataset_name``).
        limit_dataset_samples: Cap on dialogues read before expansion.
        max_eval_samples: Cap on total expanded eval samples.
        max_new_tokens: Generation budget per sample.
        max_prompt_tokens: If set, prompts longer than this many tokens
            are skipped (logged once). Prevents OOM on degenerate samples.
        do_sample: Generation sampling toggle. Default greedy for
            reproducibility across validation checkpoints.
        metric_prefix: Prefix applied to all returned metric keys.
        sample_shard: Optional ``(rank, world_size)`` tuple. When set,
            only every ``world_size``-th sample starting at ``rank`` is
            processed; the caller is responsible for all-reducing the
            returned ``_count`` and weighted-summed metrics.
    """

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        path: Optional[Union[str, List[str]]] = None,
        split: str = "train",
        limit_dataset_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
        max_new_tokens: int = 256,
        max_prompt_tokens: Optional[int] = None,
        do_sample: bool = False,
        metric_prefix: str = "tool_call",
        sample_shard: Optional[tuple] = None,
        raise_on_cuda_oom: bool = True,
        run_on_fsdp2: bool = False,
    ) -> None:
        if (dataset_name is None) == (path is None):
            raise ValueError("Exactly one of `dataset_name` or `path` must be provided")

        self._dataset_name = dataset_name
        self._path = path
        self._split = split
        self._limit_dataset_samples = limit_dataset_samples
        self._max_eval_samples = max_eval_samples
        self._max_new_tokens = max_new_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._do_sample = do_sample
        self.metric_prefix = metric_prefix.rstrip("/")
        self._sample_shard = sample_shard
        self._raise_on_cuda_oom = raise_on_cuda_oom
        self.run_on_fsdp2 = run_on_fsdp2

        self._samples_cache: Optional[List[Dict[str, Any]]] = None

    def _cleanup_cuda(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_samples(self) -> List[Dict[str, Any]]:
        if self._samples_cache is None:
            self._samples_cache = make_agent_chat_eval_samples(
                dataset_name=self._dataset_name,
                path=self._path,
                split=self._split,
                limit_dataset_samples=self._limit_dataset_samples,
                max_eval_samples=self._max_eval_samples,
            )
            logger.info(
                "ToolCallAccuracyEvaluator: loaded %d eval samples",
                len(self._samples_cache),
            )
        return self._samples_cache

    def _iter_my_samples(self) -> List[Dict[str, Any]]:
        all_samples = self._load_samples()
        if self._sample_shard is None:
            return all_samples
        rank, world_size = self._sample_shard
        if world_size <= 1:
            return all_samples
        return [s for i, s in enumerate(all_samples) if i % world_size == rank]

    def _render_prompt_ids(
        self,
        tokenizer,
        sample: Dict[str, Any],
        skip_reasons: Optional[Dict[str, int]] = None,
    ) -> Optional[List[int]]:
        """Render one eval sample's prompt through ``apply_chat_template``.

        We deliberately split the chat-template render (``tokenize=False``)
        from the tokenization step: some templates / transformers versions
        return a list of token *strings* under ``tokenize=True``, which
        then crashes ``torch.tensor(..., dtype=long)`` downstream. Going
        through text first sidesteps that and matches the canonical HF
        usage shown in the model cards.

        Returns ``None`` if the template raises (e.g. doesn't accept the
        ``tools`` kwarg) or if the prompt exceeds ``max_prompt_tokens``.
        """
        kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if sample.get("tools") is not None:
            kwargs["tools"] = sample["tools"]
        try:
            text = tokenizer.apply_chat_template(sample["prompt_messages"], **kwargs)
        except Exception as exc:
            if skip_reasons is not None:
                skip_reasons["chat_template_raised"] = skip_reasons.get("chat_template_raised", 0) + 1
            logger.warning(
                "apply_chat_template failed on sample id=%s turn=%s: %s",
                sample.get("example_id"),
                sample.get("turn_index"),
                exc,
            )
            return None

        if not isinstance(text, str):
            if skip_reasons is not None:
                key = f"chat_template_returned_{type(text).__name__}"
                skip_reasons[key] = skip_reasons.get(key, 0) + 1
            logger.warning(
                "apply_chat_template returned %s, expected str (sample id=%s)",
                type(text).__name__,
                sample.get("example_id"),
            )
            return None

        try:
            encoded = tokenizer(text, add_special_tokens=False)
            ids = encoded["input_ids"]
        except Exception as exc:
            if skip_reasons is not None:
                skip_reasons["encode_raised"] = skip_reasons.get("encode_raised", 0) + 1
            logger.warning(
                "tokenizer encode failed on sample id=%s turn=%s: %s",
                sample.get("example_id"),
                sample.get("turn_index"),
                exc,
            )
            return None

        if self._max_prompt_tokens is not None and len(ids) > self._max_prompt_tokens:
            if skip_reasons is not None:
                skip_reasons["prompt_too_long"] = skip_reasons.get("prompt_too_long", 0) + 1
            logger.warning(
                "skipping eval sample id=%s turn=%s: prompt length %d > max_prompt_tokens=%d",
                sample.get("example_id"),
                sample.get("turn_index"),
                len(ids),
                self._max_prompt_tokens,
            )
            return None
        return list(ids)

    @torch.inference_mode()
    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        """Run generation-based tool-call evaluation against ``model``.

        Caller is expected to have placed the model in eval mode and on
        the appropriate device. The evaluator infers the device from the
        first model parameter so it works with FSDP, DDP, or single-GPU
        layouts without explicit configuration.

        Args:
            model: a HuggingFace causal-LM with a ``.generate()`` method.
            tokenizer: tokenizer paired with ``model``; must have a chat
                template that supports the ``tools`` kwarg.

        Returns:
            Dict of metric name -> float. All metric values are means in
            ``[0.0, 1.0]`` except ``_count`` which is the number of
            samples actually scored on this rank.
        """
        samples = self._iter_my_samples()
        sums = {k: 0.0 for k in _METRIC_KEYS}
        n_scored = 0
        skip_reasons: Dict[str, int] = {}

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        logger.info(
            "tool_call_evaluator: starting eval on %d samples, tokenizer=%s",
            len(samples),
            type(tokenizer).__name__,
        )

        for sample in samples:
            prompt_ids = self._render_prompt_ids(tokenizer, sample, skip_reasons=skip_reasons)
            if prompt_ids is None:
                continue

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": self._max_new_tokens,
                "do_sample": self._do_sample,
                "pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            }
            try:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            except torch.OutOfMemoryError:
                skip_reasons["generate_cuda_oom"] = skip_reasons.get("generate_cuda_oom", 0) + 1
                self._cleanup_cuda()
                if self._raise_on_cuda_oom:
                    raise
                continue
            except Exception as exc:
                if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
                    skip_reasons["generate_cuda_oom"] = skip_reasons.get("generate_cuda_oom", 0) + 1
                    self._cleanup_cuda()
                    if self._raise_on_cuda_oom:
                        raise
                    continue
                skip_reasons["generate_raised"] = skip_reasons.get("generate_raised", 0) + 1
                logger.warning(
                    "model.generate failed on sample id=%s turn=%s: %s",
                    sample.get("example_id"),
                    sample.get("turn_index"),
                    exc,
                )
                continue

            new_tokens = output[0, input_ids.shape[1] :].tolist()
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
            pred_calls = parse_tool_calls(decoded)
            metrics = compute_sample_metrics(pred_calls, sample["gt_tool_calls"])
            for k in _METRIC_KEYS:
                sums[k] += metrics[k]
            n_scored += 1
            del input_ids, attention_mask, output

        n_considered = len(samples)
        n_skipped = n_considered - n_scored
        if n_skipped > 0:
            logger.warning(
                "tool_call_evaluator scored %d/%d samples; skipped %d (reasons: %s)",
                n_scored,
                n_considered,
                n_skipped,
                skip_reasons if skip_reasons else "unknown",
            )
        else:
            logger.info(
                "tool_call_evaluator scored %d/%d samples", n_scored, n_considered
            )

        result: Dict[str, float] = {}
        if n_scored > 0:
            for k in _METRIC_KEYS:
                result[f"{self.metric_prefix}/{k}"] = sums[k] / n_scored
        else:
            for k in _METRIC_KEYS:
                result[f"{self.metric_prefix}/{k}"] = 0.0
        result[f"{self.metric_prefix}/_count"] = float(n_scored)
        result[f"{self.metric_prefix}/_skipped"] = float(n_skipped)
        for reason, count in skip_reasons.items():
            result[f"{self.metric_prefix}/_skip_{reason}"] = float(count)

        # generate() under FSDP unshards parameters and caches large
        # intermediate buffers; release them so the next training step
        # does not OOM on its own logits.
        self._cleanup_cuda()
        return result
