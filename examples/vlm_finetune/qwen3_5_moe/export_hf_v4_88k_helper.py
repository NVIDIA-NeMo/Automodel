# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Helper for ``export_hf_v4_88k.sh``: dtype remap, base-metadata overlay, and verification.

Two subcommands:

* ``remap-dtypes <fqn_to_dtype_mapping.json> <base_snapshot>``: for every tensor whose dtype
  in the base model differs from the mapping, rewrite the mapping to the base's dtype so the
  offline consolidation's ``--cast-dtype`` applies to it. Keeps ``*.orig``.
* ``overlay-and-verify <export_dir> <base_snapshot>``: copy the base's non-weight files over the
  export, recompute ``model.safetensors.index.json``'s ``total_size``, then assert that every
  tensor name, shape and dtype matches the base. Exits non-zero on any mismatch.
"""

import glob
import json
import os
import shutil
import struct
import sys

METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _headers(files: list[str]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Read ``{name: (dtype, shape)}`` from safetensors headers without loading data."""
    out = {}
    for path in files:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for key, value in header.items():
            if key != "__metadata__":
                out[key] = (value["dtype"], tuple(value["shape"]))
    return out


def _base_headers(base_snapshot: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    files = sorted(glob.glob(os.path.join(base_snapshot, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors in base snapshot {base_snapshot}")
    return _headers(files)


def remap_dtypes(mapping_path: str, base_snapshot: str) -> None:
    """Rewrite the checkpoint's dtype mapping so every tensor is marked with the base model's dtype."""
    base = _base_headers(base_snapshot)
    with open(mapping_path) as fh:
        mapping = json.load(fh)
    changed = {k: (v, base[k][0]) for k, v in mapping.items() if k in base and base[k][0] != v}
    if not changed:
        print("remap-dtypes: mapping already matches the base; nothing to do")
        return
    if not os.path.exists(mapping_path + ".orig"):
        shutil.copy2(mapping_path, mapping_path + ".orig")
    for key, (_, base_dtype) in changed.items():
        mapping[key] = base_dtype
    with open(mapping_path, "w") as fh:
        json.dump(mapping, fh, indent=2)
    kinds = sorted({f"{old}->{new}" for old, new in changed.values()})
    print(f"remap-dtypes: remapped {len(changed)} tensors ({', '.join(kinds)}); original kept at {mapping_path}.orig")


def overlay_and_verify(export_dir: str, base_snapshot: str) -> None:
    """Copy the base's metadata files over the export, fix the index size, and verify tensor parity."""
    copied = []
    for name in METADATA_FILES:
        src = os.path.join(base_snapshot, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(export_dir, name))
            copied.append(name)
    print(f"overlay: copied from base: {', '.join(copied)}")

    export_files = sorted(glob.glob(os.path.join(export_dir, "*.safetensors")))
    index_path = os.path.join(export_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        index.setdefault("metadata", {})["total_size"] = sum(os.path.getsize(f) for f in export_files)
        with open(index_path, "w") as fh:
            json.dump(index, fh, indent=2)

    base = _base_headers(base_snapshot)
    export = _headers(export_files)
    problems = []
    missing = sorted(set(base) - set(export))
    extra = sorted(set(export) - set(base))
    if missing:
        problems.append(f"{len(missing)} tensors missing from export, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} extra tensors in export, e.g. {extra[:3]}")
    dtype_bad = [k for k in base if k in export and base[k][0] != export[k][0]]
    shape_bad = [k for k in base if k in export and base[k][1] != export[k][1]]
    if dtype_bad:
        problems.append(
            f"{len(dtype_bad)} dtype mismatches, e.g. {[(k, base[k][0], export[k][0]) for k in dtype_bad[:3]]}"
        )
    if shape_bad:
        problems.append(f"{len(shape_bad)} shape mismatches, e.g. {shape_bad[:3]}")
    base_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(base_snapshot, "*.safetensors")))
    export_bytes = sum(os.path.getsize(f) for f in export_files)
    print(
        f"verify: base {len(base)} tensors / {base_bytes:,} bytes; export {len(export)} tensors / {export_bytes:,} bytes "
        f"in {len(export_files)} shards"
    )
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        same = open(os.path.join(export_dir, name), "rb").read() == open(os.path.join(base_snapshot, name), "rb").read()
        print(f"verify: {name} identical to base: {same}")
        if not same:
            problems.append(f"{name} differs from base")
    if problems:
        raise SystemExit("VERIFY FAILED: " + "; ".join(problems))
    print("verify: all tensor names, shapes and dtypes match the base model")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("remap-dtypes", "overlay-and-verify"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "remap-dtypes":
        remap_dtypes(sys.argv[2], sys.argv[3])
    else:
        overlay_and_verify(sys.argv[2], sys.argv[3])
