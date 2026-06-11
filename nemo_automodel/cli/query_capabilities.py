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

"""Thin CLI wrapper around :func:`nemo_automodel.query_capabilities`.

Usage::

    python -m nemo_automodel.cli.query_capabilities --model-id meta-llama/Llama-3.1-8B
    python -m nemo_automodel.cli.query_capabilities --model-id <id> --json
    python -m nemo_automodel.cli.query_capabilities --model-id <id> --trust-remote-code
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from nemo_automodel._transformers.model_capabilities import (
    ModelCapabilities,
    query_capabilities,
)


def _print_table(model_id: str, caps: ModelCapabilities) -> None:
    flags = dataclasses.asdict(caps)
    width = max(len(k) for k in flags)
    print(f"Model Capability Registry: {model_id}")
    print("-" * (28 + len(model_id)))
    for name, value in flags.items():
        print(f"  {name.ljust(width)} : {value}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m nemo_automodel.cli.query_capabilities",
        description="Query NeMo AutoModel's per-architecture capability registry.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model id (e.g. meta-llama/Llama-3.1-8B).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forwarded to AutoConfig.from_pretrained when loading the config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    args = _parse_args(argv)
    caps = query_capabilities(args.model_id, trust_remote_code=args.trust_remote_code)
    if args.json:
        print(json.dumps(dataclasses.asdict(caps), indent=2))
    else:
        _print_table(args.model_id, caps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
