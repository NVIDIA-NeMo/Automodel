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

"""Compare the ``training.jsonl`` logs of the ERNIE 5.5 packed runs with ``cp_size`` 1 and 2.

Both runs train the same randomly initialized proxy on the same packed rows, so every update supervises the same
tokens. The losses agree up to BF16 rounding and the expert and index top-k choices that it changes at near-ties:
on two GB200 GPUs the losses of the two runs differ by at most 2.4e-3 at the first update and by at most 1.3e-2
at any of the four updates, with either attention backend. This checks the training path of the recipe;
``test_ernie5_5_context_parallel.py`` compares the shards with one process exactly.

Called by ``L2_Ernie5_5_MoE_Packed_CP2.sh`` after the two torchrun invocations.

Usage:
    python check_ernie5_5_packed_cp_log.py cp1/training.jsonl cp2/training.jsonl --expected-steps 4 --vocab-size 4096
"""

import argparse
import json
import math

FIRST_LOSS_ATOL, LOSS_ATOL = 1e-2, 5e-2


def _steps(path: str) -> list[dict[str, float]]:
    """Return the logged optimizer steps of one ``training.jsonl``."""
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [record for record in records if "step" in record and "loss" in record]


def check_packed_cp_logs(cp1_path: str, cp2_path: str, expected_steps: int, vocab_size: int) -> None:
    """Assert that the runs with ``cp_size`` 1 and 2 supervised the same tokens and logged the same losses.

    Args:
        cp1_path: Path to the ``training.jsonl`` of the run with ``cp_size`` 1.
        cp2_path: Path to the ``training.jsonl`` of the run with ``cp_size`` 2.
        expected_steps: Number of optimizer steps each run was asked for.
        vocab_size: Vocabulary size of the proxy configuration.

    Raises:
        AssertionError: If a step is missing, a loss is not finite, the first loss is not that of a randomly
            initialized model, the two runs supervised different numbers of tokens in an update, or their losses
            differ by more than the bounds.
    """
    cp1, cp2 = _steps(cp1_path), _steps(cp2_path)
    assert len(cp1) == len(cp2) == expected_steps, f"Expected {expected_steps} steps, found {len(cp1)} and {len(cp2)}"
    losses = [(float(a["loss"]), float(b["loss"])) for a, b in zip(cp1, cp2)]
    tokens = [(a["num_label_tokens"], b["num_label_tokens"]) for a, b in zip(cp1, cp2)]
    print(f"losses (cp1, cp2)={losses} num_label_tokens (cp1, cp2)={tokens}")
    assert all(math.isfinite(loss) for pair in losses for loss in pair), f"Non-finite training loss: {losses}"
    assert all(a == b and a > 0 for a, b in tokens), f"The runs supervised different tokens: {tokens}"
    uniform_loss = math.log(vocab_size)
    assert abs(losses[0][0] - uniform_loss) < 0.25, (
        f"The first loss {losses[0][0]:.6f} is not that of a randomly initialized model, "
        f"ln({vocab_size}) = {uniform_loss:.6f}"
    )
    differences = [abs(a - b) for a, b in losses]
    assert differences[0] < FIRST_LOSS_ATOL, f"First losses differ by {differences[0]:.6f}: {losses[0]}"
    assert max(differences) < LOSS_ATOL, f"Losses differ by up to {max(differences):.6f}: {losses}"
    print(f"ERNIE 5.5 packed CP2 matches CP1: loss differences {differences}")


def main() -> None:
    """Parse the command line and compare the two training logs."""
    parser = argparse.ArgumentParser(description="Compare the packed ERNIE 5.5 runs with cp_size 1 and 2.")
    parser.add_argument("cp1_log", help="Path to the training.jsonl of the run with cp_size 1.")
    parser.add_argument("cp2_log", help="Path to the training.jsonl of the run with cp_size 2.")
    parser.add_argument("--expected-steps", type=int, required=True, help="Number of optimizer steps of each run.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size of the proxy configuration.")
    args = parser.parse_args()
    check_packed_cp_logs(args.cp1_log, args.cp2_log, args.expected_steps, args.vocab_size)


if __name__ == "__main__":
    main()
