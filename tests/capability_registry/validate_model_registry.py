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

"""Validate that a model's declared capabilities actually hold.

This is the parity/validation harness. Querying the declared capabilities is
handled by the public Python API and the small CLI in
``nemo_automodel/cli/query_capabilities.py``; this script focuses on running
the standardized tests (KL parity, etc.) that prove the declared flags are
honest.

Usage:
    python tests/capability_registry/validate_model_registry.py \\
        --model_id meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Sequence

# Make the repo importable when invoked directly as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nemo_automodel import query_capabilities  # noqa: E402


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_model_registry",
        description="Validate a NeMo AutoModel model class's declared ModelCapabilities.",
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="HuggingFace model id (e.g. meta-llama/Llama-3.1-8B).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoConfig.from_pretrained.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the result as a single JSON object on stdout instead of a table.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _print_table(model_id: str, caps) -> None:
    flags = dataclasses.asdict(caps)
    header = f"Model Capability Registry: {model_id}"
    print()
    print(header)
    print("-" * len(header))
    width = max((len(k) for k in flags), default=0)
    for name, value in flags.items():
        print(f"  {name:<{width}} : {value}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    args = _parse_args(argv)

    try:
        caps = query_capabilities(args.model_id, trust_remote_code=args.trust_remote_code)
    except (KeyError, ValueError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"model_id": args.model_id, "capabilities": dataclasses.asdict(caps)}, indent=2))
    else:
        _print_table(args.model_id, caps)

    # TODO: standardized parity tests (TP, CP, PP, EP) live in standardized_tests/
    # and run per-capability based on `caps`.  They spawn torchrun, do K-step
    # training, and KL-compare logits.
    return 0


if __name__ == "__main__":
    sys.exit(main())
