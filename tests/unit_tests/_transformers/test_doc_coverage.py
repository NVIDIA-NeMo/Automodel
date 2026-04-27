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

"""Guard against the gemma4-style regression where a new model architecture
lands in ``MODEL_ARCH_MAPPING`` without any corresponding page under
``docs/model-coverage/``.

A missing model card emits a GitHub Actions ``::warning::`` annotation on the
PR (visible in the checks UI) for the first ``_DOC_GRACE_PERIOD_DAYS`` days
after the arch was registered, instead of failing the test. This supports the
day-0 partner-release workflow: model lands now (often required by the
marketing announcement schedule), docs follow in a separate PR. Once the
grace window expires the test hard-fails so undocumented arches do not
accumulate silently.
"""

import pathlib
import subprocess
import time

# Architectures documented under a different literal name in
# docs/model-coverage/. Each value must appear verbatim in at least one .md
# file under docs/model-coverage/.
#
# Add an entry here ONLY when the documentation legitimately uses a different
# name than the registry / HF class name (e.g., HF class name differs from
# registry alias, case differences, or a variant is grouped on a shared family
# page). New entries should include a short inline comment explaining the
# mismatch.
#
# Shared by ``test_doc_coverage.py`` (registry archs) and
# ``test_recipe_doc_coverage.py`` (arches resolved from example YAMLs).
_DOC_ARCH_ALIASES = {
    # HF ships the class as ``BaiChuanForCausalLM`` (CamelCase) — registry
    # uses ``BaichuanForCausalLM``. Documented on the Baichuan page.
    "BaichuanForCausalLM": "BaiChuanForCausalLM",
    # HF upstream renamed ``Gemma3nForConditionalGeneration`` between releases;
    # the "Gemma 3n" variant is covered on the Gemma 3 VL page.
    "Gemma3nForConditionalGeneration": "Gemma 3n",
    # Checkpoint-facing alias of ``KimiK25VLForConditionalGeneration``, covered
    # by the Kimi-VL page.
    "KimiK25ForConditionalGeneration": "Kimi-K25-VL",
    "KimiK25VLForConditionalGeneration": "Kimi-K25-VL",
    # Retrieval/bi-encoder variants of Llama, covered on the GritLM page.
    "LlamaBidirectionalForSequenceClassification": "GritLM",
    "LlamaBidirectionalModel": "GritLM",
    # HF ships ``LlavaOnevisionForConditionalGeneration`` (lowercase "n");
    # registry uses ``LlavaOneVisionForConditionalGeneration`` (the NVIDIA
    # re-impl for LLaVA-OneVision-1.5 with RICE ViT).
    "LlavaOneVisionForConditionalGeneration": "LlavaOnevisionForConditionalGeneration",
    # Registry also exposes the NVIDIA LLaVA-OneVision-1.5 re-impl under the
    # class name ``LLaVAOneVision1_5_ForConditionalGeneration`` (all-caps
    # "LLaVA" + explicit "1_5_" infix). The same model is documented on the
    # lmms-lab/llava-onevision page under ``LlavaOneVisionForConditionalGeneration``.
    "LLaVAOneVision1_5_ForConditionalGeneration": "LlavaOneVisionForConditionalGeneration",
    # Ministral3 text model; covered on the Ministral3 / Ministral3-VL pages
    # that list the VLM arch ``Mistral3ForConditionalGeneration``.
    "Ministral3ForCausalLM": "Mistral3ForConditionalGeneration",
    # Bi-encoder variant of Ministral3, covered on the same Ministral3 / Ministral3-VL pages.
    "Ministral3BidirectionalModel": "Mistral3ForConditionalGeneration",
    # Mistral4 text model is the backbone of Mistral-Small-4 VLM; documented
    # on the Mistral-Small-4 page via the recipe path ``mistral4``.
    "Mistral4ForCausalLM": "mistral4",
    # OLMo2 page uses the vendor-branded spelling ``OLMo2`` (all caps "OLM");
    # HF normalized the class name to ``Olmo2``.
    "Olmo2ForCausalLM": "OLMo2ForCausalLM",
    # HF upstream added an extra underscore between "5" and "VL"
    # (``Qwen2_5_VLForConditionalGeneration``); the Qwen2.5-VL page still uses
    # the pre-rename spelling.
    "Qwen2_5_VLForConditionalGeneration": "Qwen2_5VLForConditionalGeneration",
    # Qwen3-Omni, Qwen3-VL and Qwen3.5-VL are documented with the VL-facing
    # arch name; the registry wires their MoE backbones under these keys.
    "Qwen3OmniMoeForConditionalGeneration": "Qwen3OmniForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration": "Qwen3VLForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeVLForConditionalGeneration",
    # Dense Qwen3.5 text/VL backbone; grouped with the VL variants on the
    # Qwen3.5-VL page.
    "Qwen3_5ForConditionalGeneration": "Qwen3.5",
    # HF split Seed-OSS into its own arch; the Seed page (``seed.md``) covers
    # both Seed-Coder and Seed-OSS under the "Seed-OSS" name.
    "SeedOssForCausalLM": "Seed-OSS",
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


# Days to wait after an arch is registered before a missing model card becomes
# a hard test failure. During the grace window the test still surfaces the
# missing-doc finding as a GitHub Actions ``::warning::`` annotation on the PR
# but does not block the merge. See module docstring for the rationale.
_DOC_GRACE_PERIOD_DAYS = 2


def _arch_registration_age_days(arch_name: str, repo_root: pathlib.Path) -> float | None:
    """Return days since ``arch_name`` was last added to ``registry.py``.

    Uses ``git log -S "<arch_name>"`` on ``registry.py`` to find the most
    recent commit that changed the count of occurrences of the arch's
    quoted dict-key string — i.e., the commit that registered it. Returns
    ``None`` if git history is unavailable (not a git repo, shallow clone
    that pruned the commit, ``git`` not on PATH, etc.). Callers treat
    ``None`` as "fresh" so missing-doc cases default to a warning rather
    than blocking on infra issues; CI's build-container is configured with
    ``fetch-depth: 0`` so the lookup succeeds in the test container.
    """
    registry_path = repo_root / "nemo_automodel" / "_transformers" / "registry.py"
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                "-1",
                "--format=%ct",
                "-S",
                f'"{arch_name}"',
                "--",
                str(registry_path),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    if not out:
        return None
    try:
        commit_time = int(out)
    except ValueError:
        return None
    return (time.time() - commit_time) / 86400.0


def test_every_registered_arch_has_model_coverage_doc():
    """Every architecture in ``MODEL_ARCH_MAPPING`` must be mentioned in at
    least one ``docs/model-coverage/*.md`` file, either by its own name or by
    a mapped alias in ``_DOC_ARCH_ALIASES``.

    Missing-doc entries are graced for ``_DOC_GRACE_PERIOD_DAYS`` days from
    the commit that registered the arch — during the window the test prints
    a ``::warning::`` annotation (rendered on the PR's checks page) but
    still passes, so a day-0 partner release can land with the docs PR
    following separately. After the window the test hard-fails.

    This still guards against the gemma4-style regression where a new arch
    is registered but never gets a corresponding model card.
    """
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    repo_root = _repo_root()
    docs_dir = repo_root / "docs" / "model-coverage"
    assert docs_dir.is_dir(), f"docs/model-coverage/ not found at {docs_dir}"

    md_contents = [p.read_text(encoding="utf-8") for p in docs_dir.rglob("*.md")]
    assert md_contents, "No .md files found under docs/model-coverage/"

    pending: list[tuple[str, str, float | None]] = []
    expired: list[tuple[str, str, float | None]] = []
    for arch_name in MODEL_ARCH_MAPPING:
        needle = _DOC_ARCH_ALIASES.get(arch_name, arch_name)
        if any(needle in content for content in md_contents):
            continue
        age_days = _arch_registration_age_days(arch_name, repo_root)
        # Treat unknown ages (git unavailable / shallow clone) as fresh so an
        # infra problem can't unfairly block a PR; CI guarantees full history.
        if age_days is not None and age_days >= _DOC_GRACE_PERIOD_DAYS:
            expired.append((arch_name, needle, age_days))
        else:
            pending.append((arch_name, needle, age_days))

    for arch, needle, age in pending:
        if age is None:
            remaining_msg = f"{_DOC_GRACE_PERIOD_DAYS} day(s) (registration date unknown — assuming fresh)"
        else:
            remaining_msg = f"{max(0.0, _DOC_GRACE_PERIOD_DAYS - age):.1f} day(s)"
        # GitHub Actions parses ``::warning file=…::…`` lines from stdout into
        # PR-level annotations. Pytest is run with ``-s`` (no capture) in CI
        # so the print reaches the workflow log unmodified.
        print(
            f"::warning file=nemo_automodel/_transformers/registry.py::"
            f"Architecture '{arch}' has no model card under docs/model-coverage/ "
            f"(looked for {needle!r}). {remaining_msg} remain in the "
            f"{_DOC_GRACE_PERIOD_DAYS}-day grace window before this becomes a "
            f"hard failure — please open a follow-up docs PR or add the arch "
            f"name to an existing .md."
        )

    if expired:
        details = "\n".join(
            f"  - {arch} (looked for {needle!r}, registered {age:.1f} days ago)" for arch, needle, age in expired
        )
        raise AssertionError(
            f"The following registered architectures have been missing a model "
            f"card in docs/model-coverage/ for more than "
            f"{_DOC_GRACE_PERIOD_DAYS} days:\n"
            f"{details}\n\n"
            "Fix by either:\n"
            "  1. Adding a new .md file under docs/model-coverage/ (preferred for "
            "new architectures — e.g., docs/model-coverage/vlm/google/gemma4.md), or\n"
            "  2. Updating an existing .md file to mention the arch name, or\n"
            "  3. Adding an entry to _DOC_ARCH_ALIASES in this test file with a "
            "comment explaining the mismatch."
        )


def test_doc_arch_aliases_target_strings_appear_in_docs():
    """Every value in ``_DOC_ARCH_ALIASES`` must literally appear in some
    ``docs/model-coverage/*.md`` file.

    Prevents aliases from pointing at strings that never existed or got
    removed — if the target string is missing, the aliased arch is silently
    undocumented and the doc-coverage check becomes a no-op for that entry.
    """
    docs_dir = _repo_root() / "docs" / "model-coverage"
    md_contents = [p.read_text(encoding="utf-8") for p in docs_dir.rglob("*.md")]

    bad = []
    for arch, needle in _DOC_ARCH_ALIASES.items():
        if not any(needle in content for content in md_contents):
            bad.append((arch, needle))
    assert not bad, "_DOC_ARCH_ALIASES entries pointing at strings absent from the docs:\n" + "\n".join(
        f"  - {arch} -> {needle!r}" for arch, needle in bad
    )
