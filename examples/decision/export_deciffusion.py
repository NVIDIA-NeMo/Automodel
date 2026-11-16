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

"""Export a validation-selected Deciffusion decision stage as a complete, servable checkpoint.

  uv run python examples/decision/export_deciffusion.py --base /path/to/deciffusion-checkpoint \
      --stage runs/support/lora/LOWEST_VAL --output exports/support_lora

LoRA stages are merged into the base weights (W += alpha / r * B @ A, in fp32); full fine-tuning stages are written
from their consolidated weights in the base dtypes. The shared-context flag is removed from the exported config
because these runs train one question per read. The stage must have been trained on this exact base.
"""

import argparse
import json
from pathlib import Path

from nemo_automodel.components.checkpoint.utils import read_checkpoint_pointer
from nemo_automodel.components.models.deciffusion.export import (
    checkpoint_fingerprint,
    export_full_checkpoint,
    merge_lora_checkpoint,
)


def main() -> None:
    """Resolve the stage, verify its base, and export it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="the local checkpoint the stage was trained from")
    parser.add_argument("--stage", required=True, help="stage directory, or its LOWEST_VAL / LATEST pointer")
    parser.add_argument("--output", required=True, help="new directory for the complete checkpoint")
    args = parser.parse_args()
    stage = Path(args.stage)
    if stage.name in {"LOWEST_VAL", "LATEST"}:
        resolved = read_checkpoint_pointer(stage.parent, stage.name)
        if resolved is None:
            raise SystemExit(f"{stage} has no published target")
        stage = Path(resolved)
    manifest = json.loads((stage / "stage.json").read_text())
    expected = manifest.get("base_fingerprint")
    if expected is None or checkpoint_fingerprint(args.base) != expected:
        raise SystemExit(f"{args.base} is not the base this stage was trained from (fingerprint mismatch)")
    if manifest["method"] == "lora":
        receipt = merge_lora_checkpoint(args.base, stage / "model", args.output)
    else:
        receipt = export_full_checkpoint(args.base, stage / "model" / "consolidated", args.output)
    print(json.dumps({"stage": str(stage), **receipt}, indent=2))


if __name__ == "__main__":
    main()
