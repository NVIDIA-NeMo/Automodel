# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a normal HY4 training recipe while recording topology-parity logits.

The runner does not replace any training component. It builds the regular
``TrainFinetuneRecipeForNextTokenPrediction`` and attaches observation-only
hooks after PP/EP/FSDP setup. The first PP stage records a hash of each packed
input, and the final stage records a fixed sample of full-vocabulary logits.
The hash lets a comparator join samples even when changing DP size changes the
rank/local-microbatch assignment.

Set ``HY4_PARITY_ARTIFACT_DIR`` to a distinct output directory for each run.
``HY4_PARITY_LOGIT_POSITIONS`` controls how many evenly spaced token positions
are retained from each packed sequence (default: 128).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

_ARTIFACT_DIR_ENV = "HY4_PARITY_ARTIFACT_DIR"
_POSITIONS_ENV = "HY4_PARITY_LOGIT_POSITIONS"


def _packed_input_hash(input_ids: torch.Tensor) -> str:
    """Hash packed token IDs shaped ``[microbatch, tokens]`` by exact bytes."""
    cpu_ids = input_ids.detach().to(device="cpu").contiguous()
    return hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest()


def _sample_positions(sequence_length: int, count: int, device: torch.device) -> torch.Tensor:
    """Return unique, evenly spaced indices shaped ``[min(count, tokens)]``."""
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
    count = min(max(int(count), 1), sequence_length)
    if count == sequence_length:
        return torch.arange(sequence_length, device=device, dtype=torch.long)
    positions = torch.linspace(0, sequence_length - 1, steps=count, device=device, dtype=torch.float64)
    return positions.round().to(torch.long).unique(sorted=True)


def _extract_logits(output: Any) -> torch.Tensor | None:
    """Extract main logits shaped ``[microbatch, tokens, vocab]`` from a stage output."""
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor) and output[0].dim() == 3:
        return output[0]
    if isinstance(output, torch.Tensor) and output.dim() == 3:
        return output
    return None


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Atomically save a tensor artifact whose tensor layouts are recorded in ``payload``."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class HyV4ParallelParityRecorder:
    """Observation hooks for packed input hashes and sampled full-vocabulary logits."""

    def __init__(self, recipe: TrainFinetuneRecipeForNextTokenPrediction, artifact_dir: Path, positions: int):
        self.recipe = recipe
        self.artifact_dir = artifact_dir
        self.positions = positions
        self.rank = torch.distributed.get_rank()
        self.dp_rank = int(recipe._get_dp_rank(include_cp=True))
        self.pp_rank = int(recipe._get_pp_rank())
        self.vocab_size = int(recipe.model_parts[0].config.vocab_size)
        self.input_call = 0
        self.output_call = 0
        self.local_input_hashes: dict[int, str] = {}
        self.handles: list[Any] = []
        artifact_dir.mkdir(parents=True, exist_ok=True)

    def record_batch_inputs(self, input_ids: torch.Tensor | None) -> None:
        """Record rows of local token IDs shaped ``[local_batch, packed_tokens]``."""
        if self.pp_rank != 0 or not isinstance(input_ids, torch.Tensor):
            return
        if input_ids.dim() != 2:
            raise ValueError(
                f"HY4 parity expects input IDs [local_batch, packed_tokens], got {tuple(input_ids.shape)}."
            )

        for row in input_ids.unbind(dim=0):
            microbatch_ids = row.unsqueeze(0)
            call = self.input_call
            self.input_call += 1
            digest = _packed_input_hash(microbatch_ids)
            self.local_input_hashes[call] = digest
            metadata = {
                "rank": self.rank,
                "dp_rank": self.dp_rank,
                "pp_rank": self.pp_rank,
                "call": call,
                "input_sha256": digest,
                "input_shape": list(microbatch_ids.shape),
            }
            path = self.artifact_dir / f"input-rank{self.rank:03d}-call{call:04d}.json"
            path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")

    def _record_output(self, _module, _args, _kwargs, output: Any) -> None:
        """Record sampled logits shaped ``[positions, vocab]`` from a final-stage output."""
        logits = _extract_logits(output)
        if logits is None or logits.shape[-1] != self.vocab_size:
            return
        if logits.shape[0] != 1:
            raise ValueError(f"HY4 parity recording expects PP microbatch size 1, got logits {tuple(logits.shape)}.")

        call = self.output_call
        self.output_call += 1
        positions = _sample_positions(logits.shape[1], self.positions, logits.device)
        sampled_logits = logits.detach()[0].index_select(0, positions).to(dtype=torch.float32, device="cpu")
        if not torch.isfinite(sampled_logits).all():
            raise FloatingPointError(f"Non-finite HY4 logits at rank={self.rank}, call={call}.")
        payload = {
            "rank": self.rank,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "call": call,
            "input_sha256": self.local_input_hashes.get(call),
            "full_logits_shape": list(logits.shape),
            "positions": positions.to(device="cpu"),
            "logits": sampled_logits,
        }
        path = self.artifact_dir / f"logits-rank{self.rank:03d}-call{call:04d}.pt"
        _atomic_torch_save(payload, path)
        print(
            f"HY4_PARITY_ARTIFACT rank={self.rank} dp_rank={self.dp_rank} "
            f"pp_rank={self.pp_rank} call={call} path={path}",
            flush=True,
        )

    def install(self) -> None:
        """Attach hooks to each local pipeline part without changing its forward contract."""
        for model_part in self.recipe.model_parts:
            self.handles.append(model_part.register_forward_hook(self._record_output, with_kwargs=True))

    def remove(self) -> None:
        """Remove every observation hook installed by :meth:`install`."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class HyV4ParallelParityRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Regular training recipe with one observation at its pre-forward batch boundary."""

    parity_recorder: HyV4ParallelParityRecorder | None = None

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_label_tokens,
        num_batches,
        is_train: bool = True,
    ):
        """Hash ``[local_batch, packed_tokens]`` IDs, then run the unchanged train step."""
        if self.parity_recorder is not None:
            self.parity_recorder.record_batch_inputs(batch.get("input_ids"))
        return super()._forward_backward_step(
            idx,
            batch,
            loss_buffer=loss_buffer,
            num_label_tokens=num_label_tokens,
            num_batches=num_batches,
            is_train=is_train,
        )


def main() -> None:
    """Build the configured production recipe, record parity artifacts, and train."""
    artifact_dir_value = os.environ.get(_ARTIFACT_DIR_ENV)
    if not artifact_dir_value:
        raise ValueError(f"{_ARTIFACT_DIR_ENV} must name an output directory.")
    positions = int(os.environ.get(_POSITIONS_ENV, "128"))

    cfg = parse_args_and_load_config()
    recipe = HyV4ParallelParityRecipe(cfg)
    recipe.setup()
    recorder = HyV4ParallelParityRecorder(recipe, Path(artifact_dir_value), positions)
    recipe.parity_recorder = recorder
    recorder.install()
    try:
        recipe.run_train_validation_loop()
    finally:
        recorder.remove()


if __name__ == "__main__":
    main()
