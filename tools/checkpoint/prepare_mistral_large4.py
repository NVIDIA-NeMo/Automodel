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

"""Native metadata conversion and a zero-copy local checkpoint view."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config


def prepare_checkpoint(source: Path, destination: Path, *, multimodal: bool = False) -> Path:
    """Create a new metadata view and links; never overwrite an existing directory.

    The source FP8 payload remains unchanged. Use the adapter-owned load parts
    with the existing dequantize_base_checkpoint=True option at initialization.
    Native resume does not use this source-only option.
    """
    source = source.resolve(strict=True)
    config = MistralLarge4Config.from_native(source)
    if multimodal:
        if not config.vision_config:
            raise ValueError("Multimodal preparation requires vision_encoder metadata")
        config.architectures = ["MistralLarge4ForConditionalGeneration"]
    index_path = source / "consolidated.safetensors.index.json"
    index = json.loads(index_path.read_text())
    files = set(index["weight_map"].values()) | {"tekken.json", "params.json", "chat_template.jinja"}
    for name in files:
        if Path(name).name != name or not (source / name).is_file():
            raise ValueError(f"Missing or unsafe native checkpoint filename: {name}")
    # Validate everything before creating output. Existing files are never reused.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        config.save_pretrained(destination)
        for name in sorted(files):
            (destination / name).symlink_to(source / name)
        (destination / "model.safetensors.index.json").symlink_to(index_path)
        (destination / "preparation.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "params_sha256": hashlib.sha256((source / "params.json").read_bytes()).hexdigest(),
                    "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                    "weight_bytes_modified": False,
                    "dequantize_base_checkpoint": True,
                    "executable_model_available": True,
                    "architecture": config.architectures[0],
                    "stage": "floating training; disable fp8_indexer and fp8_kv_cache",
                },
                indent=2,
            )
            + "\n"
        )
    except Exception:
        # This directory was created by this call, so partial preparation is safe to remove.
        shutil.rmtree(destination)
        raise
    return destination


def main() -> None:
    """Prepare the native checkpoint view for later AutoModel construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--multimodal", action="store_true", help="Select image/text training architecture")
    args = parser.parse_args()
    prepare_checkpoint(args.source, args.destination, multimodal=args.multimodal)


if __name__ == "__main__":
    main()
