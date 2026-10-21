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

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("source", ["numpy", "pil", "path"])
def test_trackio_image_logging_preserves_pixels(tmp_path: Path, source: str) -> None:
    """Exercise the real optional logger, including Pillow encoding, without network access."""
    pytest.importorskip("trackio")
    # Trackio reads TRACKIO_DIR at import time. Isolate it from other tests' imports.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
from pathlib import Path

import numpy as np
import trackio
from PIL import Image

from nemo_automodel.components.loggers.trackio_utils import TrackioLogger

root = Path(os.environ["TRACKIO_DIR"])
pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
image = Image.fromarray(pixels)
source = os.environ["IMAGE_SOURCE"]
if source == "numpy":
    value = pixels
elif source == "pil":
    value = image
else:
    value = root.parent / "source.png"
    image.save(value)

run = trackio.init(
    project="pillow-compat", name="image", embed=False,
    auto_log_gpu=False, auto_log_cpu=False,
)
try:
    TrackioLogger(run).log_metrics({"loss": 0.5}, step=1)
    run.log({"image": trackio.Image(value, caption="pixels")}, step=2)
finally:
    run.finish()

saved = list(root.rglob("*.png"))
assert len(saved) == 1, saved
with Image.open(saved[0]) as decoded:
    np.testing.assert_array_equal(np.asarray(decoded.convert("RGB")), pixels)

try:
    trackio.Image(pixels.astype(np.float32))
except ValueError:
    pass
else:
    raise AssertionError("Trackio must reject non-uint8 image arrays")
""",
        ],
        env={**os.environ, "TRACKIO_DIR": str(tmp_path / "trackio"), "IMAGE_SOURCE": source},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
