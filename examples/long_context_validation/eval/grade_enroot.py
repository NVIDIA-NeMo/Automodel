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

"""Local SWE-bench grading via enroot (Docker-less, no cloud).

Reuses the OFFICIAL swebench grading logic (`make_test_spec` + `get_eval_report`) so the
resolved/unresolved verdict matches the benchmark exactly — only the execution backend is
swapped from Docker to enroot, mirroring `enroot_env.py`. This exists because this cluster
has no Docker for the standard local harness, so grading runs in enroot instead.

Per instance:
  1. enroot container from docker://swebench/sweb.eval.x86_64.<iid> (import cached, ENROOT_MOUNT_HOME=n).
  2. `git apply` the candidate model_patch in /testbed (empty/failed-apply -> unresolved, no test run).
  3. run swebench's `eval_script` (reinstalls pkg, applies gold test_patch, runs FAIL_TO_PASS/
     PASS_TO_PASS between the `>>>>> Start/End Test Output` markers) -> capture full log.
  4. `get_eval_report(test_spec, prediction, log, include_tests_status=True)` -> resolved bool.

Output: <output-dir>/report.json  (per-instance reports + summary), and <output-dir>/<iid>/test_output.log.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import uuid
from pathlib import Path

from datasets import load_dataset
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

DATASET_MAP = {
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "full": "princeton-nlp/SWE-Bench",
}
ENROOT = os.getenv("MSWEA_ENROOT_EXECUTABLE", "enroot")
SQSH_DIR = Path(os.getenv("MSWEA_ENROOT_SQSH_DIR", "/tmp/grade_sqsh"))


def _env():
    # ENROOT_MOUNT_HOME=n so the image's own /root/.bashrc (conda activate testbed) runs
    # instead of the host bashrc (see enroot_env.py for the full rationale).
    return dict(os.environ, ENROOT_MOUNT_HOME="n")


def _image_sqsh(iid: str, import_timeout: int) -> Path:
    repo_id = iid.replace("__", "_1776_").lower()
    ref = f"swebench/sweb.eval.x86_64.{repo_id}:latest"
    SQSH_DIR.mkdir(parents=True, exist_ok=True)
    sqsh = SQSH_DIR / f"{ref.replace('/', '+').replace(':', '+')}.sqsh"
    if not (sqsh.exists() and sqsh.stat().st_size > 0):
        # Atomic import into a shared cache (see enroot_env.py) — safe under concurrent runs.
        tmp = sqsh.with_name(f"{sqsh.name}.tmp.{uuid.uuid4().hex[:8]}")
        subprocess.run(
            [ENROOT, "import", "-o", str(tmp), f"docker://{ref}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=import_timeout,
            env=dict(_env(), ENROOT_MAX_PROCESSORS=os.getenv("ENROOT_MAX_PROCESSORS", "8")),
        )
        os.replace(tmp, sqsh)
    return sqsh


def grade_one(inst: dict, pred: dict, output_dir: Path, run_timeout: int, import_timeout: int) -> tuple:
    """Grade one prediction: apply the model patch, run FAIL_TO_PASS/PASS_TO_PASS in enroot, return the report."""
    iid = inst["instance_id"]
    idir = output_dir / iid
    idir.mkdir(parents=True, exist_ok=True)
    log_path = idir / "test_output.log"
    ts = make_test_spec(inst)
    patch = (pred.get("model_patch") or "").strip()

    # Empty patch: let get_eval_report short-circuit (patch_is_None) — no container needed.
    if not patch:
        log_path.write_text("")
        return iid, get_eval_report(ts, pred, str(log_path), include_tests_status=True)

    name = f"grade-{uuid.uuid4().hex[:8]}"
    sqsh = _image_sqsh(iid, import_timeout)
    subprocess.run(
        [ENROOT, "create", "--name", name, str(sqsh)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=_env(),
    )
    try:
        # Stage the candidate patch + the official eval script for mounting into the container.
        (idir / "model.patch").write_text(pred.get("model_patch") or "")
        (idir / "eval.sh").write_text(ts.eval_script)
        mounts = ["-m", f"{idir / 'model.patch'}:/tmp/model.patch", "-m", f"{idir / 'eval.sh'}:/tmp/eval.sh"]
        # Apply the candidate patch, then run the eval script; capture everything to the log.
        # The `APPLY_PATCH_*` markers let a human see apply failures in the log; get_eval_report
        # keys off the test-output markers emitted by eval.sh.
        script = (
            "cd /testbed && "
            "(git apply -v /tmp/model.patch && echo APPLY_PATCH_PASS || "
            " (git apply --3way -v /tmp/model.patch && echo APPLY_PATCH_PASS || echo APPLY_PATCH_FAIL)) && "
            "bash /tmp/eval.sh"
        )
        with open(log_path, "w") as lf:
            subprocess.run(
                [ENROOT, "start", "--rw", *mounts, name, "bash", "-c", script],
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=run_timeout,
                env=_env(),
            )
    finally:
        subprocess.Popen(f"{ENROOT} remove -f {name} >/dev/null 2>&1 &", shell=True)

    return iid, get_eval_report(ts, pred, str(log_path), include_tests_status=True)


def main():
    """CLI entry point: grade a preds.json against SWE-bench specs in local enroot containers."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", required=True, help="predictions json (mini-swe-agent preds.json)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-timeout", type=int, default=1800)
    ap.add_argument("--import-timeout", type=int, default=900)
    ap.add_argument("--instance-ids", default="", help="comma-separated subset of instance ids to grade")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = json.loads(Path(args.preds).read_text())
    ds = {r["instance_id"]: r for r in load_dataset(DATASET_MAP.get(args.subset, args.subset), split=args.split)}

    ids = [i.strip() for i in args.instance_ids.split(",") if i.strip()] or list(preds.keys())
    ids = [i for i in ids if i in ds]
    print(f"Grading {len(ids)} instances with {args.workers} workers...", flush=True)

    reports = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(grade_one, ds[i], preds[i], output_dir, args.run_timeout, args.import_timeout): i for i in ids
        }
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                iid, rep = fut.result()
                reports.update(rep)
                r = rep.get(iid, {})
                print(
                    f"  {iid:40s} resolved={r.get('resolved')} applied={r.get('patch_successfully_applied')}",
                    flush=True,
                )
            except Exception as e:
                reports[i] = {"resolved": False, "error": f"{type(e).__name__}: {e}"}
                print(f"  {i:40s} ERROR {type(e).__name__}: {e}", flush=True)

    resolved = [i for i, r in reports.items() if r.get("resolved")]
    summary = {
        "subset": args.subset,
        "split": args.split,
        "total_graded": len(ids),
        "resolved": len(resolved),
        "resolve_rate": (len(resolved) / len(ids)) if ids else 0.0,
        "resolved_ids": sorted(resolved),
    }
    (output_dir / "report.json").write_text(json.dumps({"summary": summary, "reports": reports}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
