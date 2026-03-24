#!/usr/bin/env python3
"""Fix benchmark num_gpus / num_nodes to match distributed parallelism config.

num_gpus must be a multiple of tp_size * cp_size * pp_size * dp_replicate_size * ep_size.
num_nodes must equal num_gpus // 8.

Usage:
  python fix_benchmark_gpus.py           # dry-run
  python fix_benchmark_gpus.py --execute # apply changes
"""

import re
import sys
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).parent


def get_int(val, default=1):
    """Return int value, treating None/'none'/missing as default."""
    if val is None or str(val).lower() == "none":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def compute_min_gpus(config: dict) -> int:
    dist = config.get("distributed", {})
    if not isinstance(dist, dict):
        return 1
    tp = get_int(dist.get("tp_size"), 1)
    cp = get_int(dist.get("cp_size"), 1)
    pp = get_int(dist.get("pp_size"), 1)
    # Some configs use dp_size, others use dp_replicate_size — take whichever is larger
    dp_rep = max(get_int(dist.get("dp_replicate_size"), 1), get_int(dist.get("dp_size"), 1))
    ep = get_int(dist.get("ep_size"), 1)
    return tp * cp * pp * dp_rep * ep


def smallest_multiple(value: int, factor: int) -> int:
    """Smallest multiple of factor that is >= value."""
    if factor == 0:
        return value
    import math
    return math.ceil(value / factor) * factor


def update_benchmark_field(content: str, field: str, new_value: int) -> str:
    """Replace a field value inside the benchmark: section only."""
    # Match the benchmark section start
    bm_match = re.search(r'^benchmark\s*:', content, re.MULTILINE)
    if not bm_match:
        return content

    bm_start = bm_match.start()

    # Find end of benchmark section: next top-level key (non-indented, non-comment, non-blank)
    after = content[bm_match.end():]
    next_top = re.search(r'\n(?=[^\s#\n])', after)
    bm_end = bm_match.end() + (next_top.start() + 1 if next_top else len(after))

    bm_section = content[bm_start:bm_end]
    new_bm_section = re.sub(
        r'^(\s+' + re.escape(field) + r'\s*:\s*)\d+',
        lambda m: m.group(1) + str(new_value),
        bm_section,
        flags=re.MULTILINE,
    )

    if new_bm_section == bm_section:
        return content  # no change

    return content[:bm_start] + new_bm_section + content[bm_end:]


def process_file(yaml_path: Path, execute: bool) -> bool:
    """Process one file. Returns True if changes were made (or would be made)."""
    try:
        content = yaml_path.read_text(errors="replace")
        config = yaml.safe_load(content) or {}
    except Exception as e:
        print(f"  ERROR {yaml_path}: {e}")
        return False

    if not isinstance(config, dict):
        return False

    bm = config.get("benchmark")
    if not isinstance(bm, dict):
        return False

    min_gpus = compute_min_gpus(config)

    cur_num_gpus = get_int(bm.get("num_gpus"), 8)
    cur_num_nodes = get_int(bm.get("num_nodes"), 1)

    # num_gpus must be a multiple of min_gpus; use smallest multiple >= cur_num_gpus
    # but at minimum min_gpus itself
    if cur_num_gpus < min_gpus or cur_num_gpus % min_gpus != 0:
        new_num_gpus = smallest_multiple(max(cur_num_gpus, min_gpus), min_gpus)
    else:
        new_num_gpus = cur_num_gpus

    new_num_nodes = max(1, new_num_gpus // 8)

    gpus_changed = new_num_gpus != cur_num_gpus
    nodes_changed = new_num_nodes != cur_num_nodes

    if not gpus_changed and not nodes_changed:
        return False

    rel = yaml_path.relative_to(BASE_DIR)
    print(f"\n  {rel}")
    if gpus_changed:
        print(f"     num_gpus:  {cur_num_gpus} → {new_num_gpus}  (parallelism product={min_gpus})")
    if nodes_changed:
        print(f"     num_nodes: {cur_num_nodes} → {new_num_nodes}")

    if execute:
        new_content = content
        if gpus_changed:
            new_content = update_benchmark_field(new_content, "num_gpus", new_num_gpus)
        if nodes_changed:
            new_content = update_benchmark_field(new_content, "num_nodes", new_num_nodes)
        yaml_path.write_text(new_content, encoding="utf-8")
        print(f"     ✓ updated")

    return True


def main():
    execute = "--execute" in sys.argv
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"\n{'#'*60}")
    print(f"# Fix benchmark num_gpus/num_nodes  [{mode}]")
    print(f"{'#'*60}")
    if not execute:
        print("\nNo changes will be made. Pass --execute to apply.\n")

    yaml_files = sorted(BASE_DIR.rglob("*.yaml"))
    changed = 0
    for yf in yaml_files:
        if process_file(yf, execute):
            changed += 1

    print(f"\n\nDone. {changed} file(s) {'updated' if execute else 'need updating'}.")
    if not execute and changed:
        print("Re-run with --execute to apply changes.")


if __name__ == "__main__":
    main()
