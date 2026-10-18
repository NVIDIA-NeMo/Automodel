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

"""Merge a post-trained Clef stage (LoRA + joint head) into a release directory like ``Cloudflare/clef``.

The output loads with Cloudflare's ``joint_schema_model.load_release_model`` / ``systemone`` unchanged::

    python examples/decision/export_clef_release.py --base Cloudflare/clef \\
        --stage runs/clef/rl/best --out exports/clef-soc-triage

LoRA tensors saved by ``TrainClefDecisionRecipe`` are keyed by the Hugging Face module names
(``<module>.lora_A.weight`` / ``<module>.lora_B.weight``); each pair is merged as
``W += (alpha / dim) * B @ A`` into ``<module>.weight`` of the base checkpoint.
"""

import argparse
import logging
from pathlib import Path

from nemo_automodel.components.models.clef.export import export_release


def main() -> None:
    """Merge a stage directory into a full release directory."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="Cloudflare/clef", help="base release (HF repo id or local directory)")
    ap.add_argument(
        "--revision", default="ed3eed331870db2eff4b0db01237128ede8a00ce", help="immutable base model revision"
    )
    ap.add_argument("--stage", required=True, help="stage directory with adapter.safetensors and joint_head.*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    export_release(args.base, revision=args.revision, stage=Path(args.stage), out=Path(args.out))


if __name__ == "__main__":
    main()
