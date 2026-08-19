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

"""Pre-warm the shared enroot image cache for a SWE-bench subset.

Imports every instance's `docker://swebench/sweb.eval.x86_64.<iid>` image ONCE into the shared
lustre cache (MSWEA_ENROOT_SQSH_DIR), with LOW concurrency + retries/backoff. This must run
BEFORE the agent runs: importing 300 images at the agent's worker concurrency (16x, x2 for
base+SFT) triggers Docker Hub burst-throttling and enroot blob-fetch contention (single imports
succeed fine — the failures are purely concurrency). After pre-warming, the agent runs and the
grader just read the cache (no concurrent pulls), so imports never fail mid-run and the served
GPUs stay fed.

Idempotent: already-cached images are skipped, so re-running resumes where it left off.
"""

import argparse
import concurrent.futures
import os
import subprocess
import time
import uuid
from pathlib import Path

from datasets import load_dataset

DATASET_MAP = {
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "full": "princeton-nlp/SWE-Bench",
}
ENROOT = os.getenv("MSWEA_ENROOT_EXECUTABLE", "enroot")
SQSH_DIR = Path(os.getenv("MSWEA_ENROOT_SQSH_DIR", "/tmp/grade_sqsh"))


def _ref(iid: str) -> str:
    return f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_').lower()}:latest"


def import_one(iid: str, retries: int, import_timeout: int) -> tuple:
    """Import one SWE-bench instance image into the shared enroot cache, with retries."""
    ref = _ref(iid)
    sqsh = SQSH_DIR / f"{ref.replace('/', '+').replace(':', '+')}.sqsh"
    if sqsh.exists() and sqsh.stat().st_size > 0:
        return iid, "cached"
    env = dict(os.environ, ENROOT_MOUNT_HOME="n", ENROOT_MAX_PROCESSORS=os.getenv("ENROOT_MAX_PROCESSORS", "8"))
    last = ""
    for attempt in range(retries):
        tmp = sqsh.with_name(f"{sqsh.name}.tmp.{uuid.uuid4().hex[:8]}")
        try:
            p = subprocess.run(
                [ENROOT, "import", "-o", str(tmp), f"docker://{ref}"],
                capture_output=True,
                text=True,
                timeout=import_timeout,
                env=env,
            )
            if p.returncode == 0:
                os.replace(tmp, sqsh)
                return iid, "imported"
            last = (p.stderr or p.stdout or "")[-300:]
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        time.sleep(min(60, 5 * (2**attempt)))  # exponential backoff on throttling
    return iid, f"FAILED: {last}"


def main():
    """CLI entry point: pre-import all SWE-bench instance images into the shared enroot cache."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=6, help="LOW concurrency to avoid registry throttling")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--import-timeout", type=int, default=1200)
    args = ap.parse_args()

    SQSH_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted({r["instance_id"] for r in load_dataset(DATASET_MAP.get(args.subset, args.subset), split=args.split)})
    print(f"Pre-warming {len(ids)} images into {SQSH_DIR} with {args.workers} workers...", flush=True)

    counts = {"cached": 0, "imported": 0, "failed": 0}
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(import_one, i, args.retries, args.import_timeout): i for i in ids}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            iid, status = fut.result()
            done += 1
            if status == "cached":
                counts["cached"] += 1
            elif status == "imported":
                counts["imported"] += 1
                print(f"  [{done}/{len(ids)}] imported {iid}", flush=True)
            else:
                counts["failed"] += 1
                failed.append(iid)
                print(f"  [{done}/{len(ids)}] {iid} {status}", flush=True)

    print(f"DONE: cached={counts['cached']} imported={counts['imported']} failed={counts['failed']}", flush=True)
    if failed:
        print("FAILED_IDS: " + ",".join(failed), flush=True)


if __name__ == "__main__":
    main()
