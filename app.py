#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Convenience wrapper for running the NeMo AutoModel CLI from a source checkout.

Usage::

    python app.py <config.yaml> [--nproc-per-node N] [--key.subkey=override ...]

This is equivalent to running::

    automodel <config.yaml> ...

For production use, prefer the installed ``automodel`` (or ``am``) entry-point.
"""

import logging
import os
import sys
import warnings

# Pydantic v2 emits UnsupportedFieldAttributeWarning for Field(repr=...) /
# Field(frozen=...) used inside 3.12-style `type` aliases in third-party libs.
# There is nothing actionable for us here, so silence them globally.
# Must run before any import that triggers pydantic schema generation.
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass

from nemo_automodel.cli.app import main

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # When launched via external torchrun, each worker runs this file.
    # Silence non-rank-0 workers before any logging to avoid duplicate output.
    if int(os.environ.get("RANK", "0")) > 0:
        logging.disable(logging.CRITICAL)
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Running from source checkout (app.py). "
        "For production use, install the package and run `automodel` or `am` instead."
    )
    sys.exit(main() or 0)
