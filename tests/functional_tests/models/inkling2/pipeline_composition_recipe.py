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

"""Four-GPU scheduled public-recipe qualification with exact fresh-process resume."""

import json
import sys
from pathlib import Path

from tests.functional_tests.models.inkling2.test_pipeline_recipe import run_recipe_resume

if __name__ == "__main__":
    schedule, directory = sys.argv[1:3]
    attention = sys.argv[3] if len(sys.argv) > 3 else "eager"
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    run_recipe_resume(schedule, output, dp_size=2, ep_size=2, attention=attention)
    (output / "comparison.json").write_text(
        json.dumps(
            {"schedule": schedule, "ranks": 4, "resume": "exact", "pp_size": 2, "ep_size": 2, "attention": attention}
        )
        + "\n"
    )
