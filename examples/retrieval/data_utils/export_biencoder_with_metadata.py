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

"""Re-save a bi-encoder export with explicit retrieval wrapper metadata."""

import argparse
import logging
from pathlib import Path

from nemo_automodel import NeMoAutoModelBiEncoder, NeMoAutoTokenizer

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the metadata export CLI."""
    parser = argparse.ArgumentParser(
        description="Load a HF-loadable bi-encoder export and re-save it with explicit retrieval metadata"
    )
    parser.add_argument("input_model", type=str, help="Path or Hugging Face ID for the source bi-encoder export")
    parser.add_argument("output_dir", type=str, help="Directory to write the metadata-bearing export")
    parser.add_argument(
        "--pooling",
        type=str,
        choices=("avg", "weighted_avg", "cls", "last", "colbert"),
        default=None,
        help="Pooling strategy to persist. Omit to keep saved metadata or the AutoModel default.",
    )
    l2_group = parser.add_mutually_exclusive_group()
    l2_group.add_argument(
        "--l2-normalize",
        dest="l2_normalize",
        action="store_true",
        help="Persist l2_normalize=true",
    )
    l2_group.add_argument(
        "--no-l2-normalize",
        dest="l2_normalize",
        action="store_false",
        help="Persist l2_normalize=false",
    )
    parser.set_defaults(l2_normalize=None)
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=None,
        help="Tokenizer path to save with the export. Defaults to input_model.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Forward trust_remote_code=True to loaders")
    parser.add_argument("--torch-dtype", type=str, default="auto", help="Torch dtype forwarded to model loading")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_kwargs = {
        "use_liger_kernel": False,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": args.torch_dtype,
    }
    if args.pooling is not None:
        model_kwargs["pooling"] = args.pooling
    if args.l2_normalize is not None:
        model_kwargs["l2_normalize"] = args.l2_normalize

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = NeMoAutoModelBiEncoder.from_pretrained(args.input_model, **model_kwargs)
    model.save_pretrained(str(output_dir))

    tokenizer_path = args.tokenizer_name_or_path or args.input_model
    tokenizer = NeMoAutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Wrote bi-encoder export with retrieval metadata to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
