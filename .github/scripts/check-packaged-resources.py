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

"""Check installed tools and examples without importing from the checkout.

Run with the installation's interpreter and ``-I`` to test package discovery.
"""

from importlib.metadata import distribution
from importlib.resources import files
from importlib.util import find_spec


def main() -> None:
    """Check module discovery and representative bundled resources."""
    for name in (
        "nemo_automodel.tools.diffusion.preprocessing_multiprocess",
        "nemo_automodel.tools.retrieval.prepare_normalized_vl_retrieval_data",
        "nemo_automodel.examples.diffusion.finetune.finetune",
        "nemo_automodel.examples.llm_finetune.finetune",
    ):
        if find_spec(name) is None:
            raise RuntimeError(f"Installed module is missing: {name}")

    examples = files("nemo_automodel.examples")
    for name in (
        "diffusion/finetune/wan2_1_t2v_flow.yaml",
        "diffusion/generate/configs/generate_wan.yaml",
        "llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml",
        "llm_finetune/deepseek_v41/tulu3_chat_template.jinja",
        "convergence/tulu3/models/gpt-oss-20b/chat_template.jinja",
    ):
        if not examples.joinpath(name).read_bytes():
            raise RuntimeError(f"Installed resource is empty: {name}")

    top_level = distribution("nemo-automodel").read_text("top_level.txt")
    if top_level is None or set(top_level.split()) != {"nemo_automodel"}:
        raise RuntimeError(f"Unexpected top-level packages: {top_level!r}")


if __name__ == "__main__":
    main()
