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

"""Standalone CLI: query a model's ``supports_*`` flags and validate them.

Usage:
    python tests/capability_registry/query_and_validate_model_registry.py \\
        --model_id meta-llama/Llama-3.1-8B

For each capability the model claims to support, this script auto-spawns
``torchrun`` to run a standardized parity test (reference vs variant with that
parallelism applied) and reports whether the resulting per-token KL divergence
stays below ``--kl_threshold``. If it doesn't, the registry lied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

# Make the repo importable when invoked directly as a script (not via uv run).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="query_and_validate_model_registry",
        description="Query a NeMo AutoModel model's capability flags and validate them.",
    )
    parser.add_argument("--model_id", required=True, help="HuggingFace model id (e.g. meta-llama/Llama-3.1-8B).")
    parser.add_argument(
        "--capabilities",
        nargs="+",
        default=None,
        help="Which capabilities to validate. Default: every implemented capability the model supports.",
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        default=1e-2,
        help="Max per-token KL divergence above which a capability test fails. Default: 1e-2.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Model dtype. Default: bfloat16.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Per-capability nproc_per_node for torchrun. Default: 2.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="Number of training batches (K-1 train + 1 forward-capture). Default: 4.",
    )
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=1,
        help="Per-rank batch size. Default: 1.",
    )
    parser.add_argument(
        "--query_only",
        action="store_true",
        help="Print the registry and exit without validating any capability.",
    )
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Optional path to write a structured JSON report.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HuggingFace loaders.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _dtype_from_str(s: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def _print_registry(model_id: str, registry: dict[str, bool]) -> None:
    print(f"\nModel Capability Registry: {model_id}")
    print("-" * (28 + len(model_id)))
    width = max((len(k) for k in registry), default=2)
    for cap, val in registry.items():
        print(f"  supports_{cap:<{width}} : {val}")
    print()


def _print_summary(results: list) -> None:
    print("Capability Validation")
    print("---------------------")
    n_pass = n_fail = n_skip = 0
    for r in results:
        if r.skipped:
            tag = "SKIP"
            n_skip += 1
        elif r.passed:
            tag = "PASS"
            n_pass += 1
        else:
            tag = "FAIL"
            n_fail += 1
        kl_str = (
            f"  max_kl={r.max_kl:.3e}  threshold={r.threshold:.2e}"
            if r.max_kl is not None
            else ""
        )
        reason = f"  ({r.variant_label})"
        extra = f"  [{r.error}]" if r.error else ""
        print(f"  [{tag}]  {r.capability:<4}{reason}{kl_str}{extra}")
    verdict = (
        "registry verified."
        if n_fail == 0
        else "the registry lied."
    )
    print(f"\nResult: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP — {verdict}")


def _worker_main(args: argparse.Namespace) -> int:
    """Entry point inside a torchrun-spawned child process."""
    import torch.distributed as dist

    from tests.capability_registry._distributed_utils import (
        init_distributed,
        write_result_json,
    )
    from tests.capability_registry._runner import CAPABILITY_TESTS
    from tests.capability_registry.standardized_tests._base import CapabilityTestResult

    cap = os.environ.get("NAR_CAPABILITY_UNDER_TEST")
    result_path = os.environ.get("NAR_RESULT_PATH")
    if cap is None or result_path is None:
        print("ERROR: torchrun worker missing NAR_CAPABILITY_UNDER_TEST or NAR_RESULT_PATH", file=sys.stderr)
        return 2

    test = CAPABILITY_TESTS[cap]
    init_distributed()

    try:
        result = test.run(
            model_id=args.model_id,
            dtype=_dtype_from_str(args.dtype),
            kl_threshold=args.kl_threshold,
            num_steps=args.num_steps,
            local_batch_size=args.local_batch_size,
        )
    except Exception as exc:  # noqa: BLE001 - report errors as structured failures
        import traceback

        tb = traceback.format_exc()
        print(f"[{cap}:rank?] EXCEPTION:\n{tb}", file=sys.stderr, flush=True)
        result = CapabilityTestResult(
            capability=cap,
            passed=False,
            skipped=False,
            max_kl=None,
            threshold=args.kl_threshold,
            variant_label=f"{cap.upper()}={args.world_size}",
            error=f"{type(exc).__name__}: {exc}",
        )

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank == 0:
        write_result_json(result_path, result.to_dict())
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0 if (result.passed or result.skipped) else 1


def _parent_main(args: argparse.Namespace) -> int:
    """Driver entry point — does the query, then orchestrates validation runs."""
    from tests.capability_registry._capability_query import CAPABILITIES, query_capabilities
    from tests.capability_registry._distributed_utils import (
        check_sufficient_gpus,
        spawn_torchrun_for_capability,
    )
    from tests.capability_registry._runner import CAPABILITY_TESTS
    from tests.capability_registry.standardized_tests._base import CapabilityTestResult

    registry = query_capabilities(args.model_id, trust_remote_code=args.trust_remote_code)
    _print_registry(args.model_id, registry)

    if args.query_only:
        if args.report_json is not None:
            with open(args.report_json, "w") as f:
                json.dump({"model_id": args.model_id, "registry": registry, "results": []}, f, indent=2)
        return 0

    requested_caps = args.capabilities if args.capabilities is not None else list(CAPABILITIES)

    results: list[CapabilityTestResult] = []
    for cap in requested_caps:
        if cap not in CAPABILITY_TESTS:
            print(f"  [SKIP]  {cap:<4}  (unknown capability)")
            continue

        test = CAPABILITY_TESTS[cap]
        if not registry.get(cap, False):
            results.append(
                CapabilityTestResult(
                    capability=cap,
                    passed=True,
                    skipped=True,
                    max_kl=None,
                    threshold=args.kl_threshold,
                    variant_label=f"model.supports_{cap}=False",
                )
            )
            continue
        if not test.implemented:
            results.append(
                CapabilityTestResult(
                    capability=cap,
                    passed=True,
                    skipped=True,
                    max_kl=None,
                    threshold=args.kl_threshold,
                    variant_label="not implemented",
                )
            )
            continue

        # Pre-flight GPU check — clean SKIP rather than NCCL crash.
        msg = check_sufficient_gpus(test.world_size)
        if msg is not None:
            results.append(
                CapabilityTestResult(
                    capability=cap,
                    passed=True,
                    skipped=True,
                    max_kl=None,
                    threshold=args.kl_threshold,
                    variant_label=f"{cap.upper()}={test.world_size}",
                    error=msg,
                )
            )
            continue

        # Replay our own CLI args to the child, minus --capabilities (each child
        # validates a single capability passed via the NAR_CAPABILITY_UNDER_TEST env var).
        child_argv = _argv_for_child(args, world_size=test.world_size)
        result_dict = spawn_torchrun_for_capability(
            capability=cap,
            world_size=test.world_size,
            parent_argv=child_argv,
            script_path=str(Path(__file__).resolve()),
        )
        results.append(CapabilityTestResult.from_dict(result_dict))

    _print_summary(results)

    if args.report_json is not None:
        with open(args.report_json, "w") as f:
            json.dump(
                {
                    "model_id": args.model_id,
                    "registry": registry,
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
            )

    return 0 if all(r.passed or r.skipped for r in results) else 1


def _argv_for_child(args: argparse.Namespace, *, world_size: int) -> list[str]:
    """Build a flat CLI argv list to pass to the torchrun child.

    We deliberately strip ``--capabilities`` (the child reads the single
    capability under test from ``NAR_CAPABILITY_UNDER_TEST``) and ``--query_only``
    (the child never queries).
    """
    out = [
        "--model_id",
        args.model_id,
        "--kl_threshold",
        str(args.kl_threshold),
        "--dtype",
        args.dtype,
        "--world_size",
        str(world_size),
        "--num_steps",
        str(args.num_steps),
        "--local_batch_size",
        str(args.local_batch_size),
    ]
    if args.trust_remote_code:
        out.append("--trust_remote_code")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point — dispatches to worker or parent mode based on env vars."""
    args = _parse_args(argv)

    from tests.capability_registry._distributed_utils import is_torchrun_worker

    if is_torchrun_worker():
        return _worker_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
