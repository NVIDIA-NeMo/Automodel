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

"""Check the ``training.jsonl`` log of the 2-GPU ERNIE 5.5 expert-parallel training run.

The run trains a randomly initialized proxy on one repeated batch of two documents, so a model that trains lowers
the loss. The two loss bounds were measured on two GPUs with seed 1234: the loss starts at 8.349 and ends at 5.017
with eager DSA and at 5.027 with cuDNN DSA, 0.60 of its first value.

Called by ``L2_Ernie5_5_MoE_EP2.sh`` after the torchrun invocation. ``run_ernie5_5_lora_training.py`` imports
``check_training_log`` and passes its own bound.

Usage:
    python check_ernie5_5_training_log.py training.jsonl --expected-steps 8 --vocab-size 4096
"""

import argparse
import json
import math


def check_training_log(path: str, expected_steps: int, vocab_size: int, max_final_fraction: float = 0.7) -> None:
    """Assert that the logged steps show a randomly initialized model that trained.

    Args:
        path: Path to the ``training.jsonl`` written by the recipe.
        expected_steps: Number of optimizer steps the run was asked for.
        vocab_size: Vocabulary size of the proxy configuration.
        max_final_fraction: The loss has to end below this fraction of its first value. The full fine-tuning run
            passes the default two steps before its end; the LoRA run passes 0.85 at the seventh of its sixteen
            updates.

    Raises:
        AssertionError: If a step is missing, a loss or a gradient norm is not finite, a gradient norm is zero,
            the first loss is not that of a randomly initialized model, or the loss fell too little.
    """
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [record for record in records if "step" in record and "loss" in record]
    losses = [float(record["loss"]) for record in records]
    grad_norms = [float(record["grad_norm"]) for record in records if record.get("grad_norm") is not None]
    print(f"steps={len(records)} losses={losses} grad_norms={grad_norms}")
    assert len(records) == expected_steps, f"Expected {expected_steps} logged steps in {path}, found {len(records)}"
    assert all(math.isfinite(loss) for loss in losses), f"Non-finite training loss: {losses}"
    assert len(grad_norms) == expected_steps, "The recipe did not log grad_norm for every step"
    assert all(math.isfinite(norm) and norm > 0.0 for norm in grad_norms), (
        f"Gradient norms must be finite and non-zero; a zero norm means the model never trained: {grad_norms}"
    )
    # A randomly initialized model spreads its probability over the vocabulary, so the first loss is close to
    # ln(vocab_size) = 8.318. A loss that sums both ranks but divides by the tokens of one, or the reverse, is off
    # by a factor of two (16.6 or 4.2).
    uniform_loss = math.log(vocab_size)
    assert abs(losses[0] - uniform_loss) < 0.25, (
        f"The first loss {losses[0]:.6f} is not that of a randomly initialized model, "
        f"ln({vocab_size}) = {uniform_loss:.6f}"
    )
    assert losses[-1] < max_final_fraction * losses[0], (
        f"Training loss fell too little on a repeated batch: {losses[0]:.6f} -> {losses[-1]:.6f}, "
        f"expected less than {max_final_fraction * losses[0]:.6f}"
    )
    print(f"ERNIE 5.5 EP2 training OK: loss {losses[0]:.6f} -> {losses[-1]:.6f} over {expected_steps} steps")


def main() -> None:
    """Parse the command line and check one training log."""
    parser = argparse.ArgumentParser(description="Check the training log of the ERNIE 5.5 EP2 functional test.")
    parser.add_argument("training_log", help="Path to the training.jsonl written by the recipe.")
    parser.add_argument("--expected-steps", type=int, required=True, help="Number of optimizer steps of the run.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size of the proxy configuration.")
    args = parser.parse_args()
    check_training_log(args.training_log, args.expected_steps, args.vocab_size)


if __name__ == "__main__":
    main()
