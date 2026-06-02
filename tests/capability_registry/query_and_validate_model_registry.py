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

"""CLI wrapper around :func:`query_capabilities`.

Usage:
    python tests/capability_registry/query_and_validate_model_registry.py \\
        --model_id google/gemma-4-26B-A4B-it

Looks up the NeMo custom model class registered for ``model_id``'s
architecture, reads its nested ``ModelCapabilities`` dataclass, and prints
the declared flags. Validation that the declared values actually hold
(parity tests, etc.) is out of scope here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Make the repo importable when invoked directly as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.capability_registry._capability_query import query_capabilities  # noqa: E402


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="query_and_validate_model_registry",
        description="Query a NeMo AutoModel custom model class's declared ModelCapabilities.",
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="HuggingFace model id (e.g. google/gemma-4-26B-A4B-it).",
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


def _print_table(model_id: str, capabilities: dict[str, bool]) -> None:
    """Pretty-print the capability flags as a small table."""
    header = f"Model Capability Registry: {model_id}"
    print()
    print(header)
    print("-" * len(header))
    width = max((len(k) for k in capabilities), default=0)
    for name, value in capabilities.items():
        print(f"  {name:<{width}} : {value}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    args = _parse_args(argv)

    try:
        capabilities = query_capabilities(args.model_id, trust_remote_code=args.trust_remote_code)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"model_id": args.model_id, "capabilities": capabilities}, indent=2))
    else:
        _print_table(args.model_id, capabilities)
    return 0


if __name__ == "__main__":
    sys.exit(main())
