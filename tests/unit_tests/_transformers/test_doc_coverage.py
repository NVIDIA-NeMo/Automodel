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
"""

import pathlib

# Architectures documented under a different literal name in
# docs/model-coverage/. Each value must appear verbatim in at least one .mdx
# file under docs/model-coverage/.
#
# Add an entry here ONLY when the documentation legitimately uses a different
# name than the registry / HF class name (e.g., HF class name differs from
# registry alias, case differences, or a variant is grouped on a shared family
# page). New entries should include a short inline comment explaining the
# mismatch.
#
# Shared by ``test_doc_coverage.py`` (registry archs), the generated
# ready-to-run architecture table, and ``test_recipe_doc_coverage.py`` (arches
# resolved from example YAMLs).
_DOC_ARCH_ALIASES = {
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
    # Multimodal retrieval variant covered by the existing NVIDIA Llama Nemotron embedding page.
    "LlamaNemotronVLModel": "llama-nemotron-embed-1b-v2",
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
    # Qwen3-Omni and Qwen3-VL are documented with the VL-facing arch name; the
    # registry wires their MoE backbones under these keys.
    "Qwen3OmniMoeForConditionalGeneration": "Qwen3OmniForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration": "Qwen3VLForConditionalGeneration",
    # Qwen3.5-MoE text and VL entry points share the unified Qwen3.5 page.
    "Qwen3_5MoeForCausalLM": "Qwen3.5-MoE",
    "Qwen3_5MoeForConditionalGeneration": "Qwen3.5-MoE",
    # Dense Qwen3.5 text/VL backbone; grouped with the VL variants on the
    # Qwen3.5-VL page.
    "Qwen3_5ForCausalLM": "Qwen3.5",
    "Qwen3_5ForConditionalGeneration": "Qwen3.5",
    # Legacy checkpoint architecture name for the renamed Qwen3.8-Flash-Next
    # implementation, documented under its current architecture name.
    "Qwen4ExpForConditionalGeneration": "Qwen3_8_FlashNextForConditionalGeneration",
    # HF split Seed-OSS into its own arch; the Seed page (``seed.mdx``) covers
    # both Seed-Coder and Seed-OSS under the "Seed-OSS" name.
    "SeedOssForCausalLM": "Seed-OSS",
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _model_card_documents() -> list[tuple[pathlib.Path, str]]:
    """Return rendered model-card documents, excluding section landing pages."""
    docs_dir = _repo_root() / "docs" / "model-coverage"
    excluded_names = {"index.mdx", "latest-models.mdx", "overview.mdx"}
    documents = []
    for path in sorted(docs_dir.rglob("*.mdx")):
        if path.name in excluded_names:
            continue
        content = path.read_text(encoding="utf-8")
        frontmatter = content.split("---", maxsplit=2)
        if len(frontmatter) == 3 and "slug: model-coverage/" in frontmatter[1]:
            documents.append((path, content))
    return documents


def _without_generated_registry_table(content: str) -> str:
    start_marker = "{/* BEGIN GENERATED MODEL ARCHITECTURES */}"
    end_marker = "{/* END GENERATED MODEL ARCHITECTURES */}"
    start = content.find(start_marker)
    end = content.find(end_marker)
    if start == -1 or end == -1:
        return content
    return content[:start] + content[end + len(end_marker) :]


def test_every_model_card_has_architecture_reference_table():
    """Every rendered model card must use the shared architecture-section contract."""
    documents = _model_card_documents()
    assert documents, "No rendered model cards found under docs/model-coverage/"

    invalid = []
    for path, content in documents:
        reasons = []
        if "## Model Context" not in content:
            reasons.append("missing '## Model Context'")
        else:
            architecture_section = content.split("## Model Context", maxsplit=1)[1]
            architecture_section = architecture_section.split("\n## ", maxsplit=1)[0]
            if "| Property | Value |" not in architecture_section:
                reasons.append("missing architecture Property/Value table")
        if "## Model Reference" in content or "### Model Architecture" in content:
            reasons.append("uses legacy nested Model Reference headings")
        if "### Model Summary" in content:
            reasons.append("uses legacy '### Model Summary' heading")
        if "### Available Checkpoints" in content:
            reasons.append("uses legacy '### Available Checkpoints' heading")
        if not content.isascii():
            reasons.append("contains non-ASCII characters")
        if reasons:
            invalid.append(f"  - {path.relative_to(_repo_root())}: {', '.join(reasons)}")

    assert not invalid, "Model cards that violate the shared page contract:\n" + "\n".join(invalid)


def test_every_registered_arch_has_model_coverage_doc():
    """Every architecture in ``MODEL_ARCH_MAPPING`` must be mentioned in at
    least one ``docs/model-coverage/*.mdx`` file, either by its own name or by
    a mapped alias in ``_DOC_ARCH_ALIASES``.

    This guards against the regression where a new arch (e.g. gemma4) is
    registered but never gets a corresponding model card in the docs.
    """
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    docs_dir = _repo_root() / "docs" / "model-coverage"
    assert docs_dir.is_dir(), f"docs/model-coverage/ not found at {docs_dir}"

    md_contents = [_without_generated_registry_table(p.read_text(encoding="utf-8")) for p in docs_dir.rglob("*.mdx")]
    assert md_contents, "No .mdx files found under docs/model-coverage/"

    missing = []
    for arch_name in MODEL_ARCH_MAPPING:
        needle = _DOC_ARCH_ALIASES.get(arch_name, arch_name)
        if not any(needle in content for content in md_contents):
            missing.append((arch_name, needle))

    if missing:
        details = "\n".join(f"  - {arch} (looked for {repr(needle)})" for arch, needle in missing)
        raise AssertionError(
            "The following registered architectures have no model card in "
            "docs/model-coverage/:\n"
            f"{details}\n\n"
            "Fix by either:\n"
            "  1. Adding a new .mdx file under docs/model-coverage/ (preferred for "
            "new architectures — e.g., docs/model-coverage/vlm/google/gemma-4-31B-it.mdx), or\n"
            "  2. Updating an existing .mdx file to mention the arch name, or\n"
            "  3. Adding an entry to _DOC_ARCH_ALIASES in this test file with a "
            "comment explaining the mismatch."
        )


def test_doc_arch_aliases_target_strings_appear_in_docs():
    """Every value in ``_DOC_ARCH_ALIASES`` must literally appear in some
    ``docs/model-coverage/*.mdx`` file.

    Prevents aliases from pointing at strings that never existed or got
    removed — if the target string is missing, the aliased arch is silently
    undocumented and the doc-coverage check becomes a no-op for that entry.
    """
    docs_dir = _repo_root() / "docs" / "model-coverage"
    md_contents = [_without_generated_registry_table(p.read_text(encoding="utf-8")) for p in docs_dir.rglob("*.mdx")]

    bad = []
    for arch, needle in _DOC_ARCH_ALIASES.items():
        if not any(needle in content for content in md_contents):
            bad.append((arch, needle))
    assert not bad, "_DOC_ARCH_ALIASES entries pointing at strings absent from the docs:\n" + "\n".join(
        f"  - {arch} -> {needle!r}" for arch, needle in bad
    )
