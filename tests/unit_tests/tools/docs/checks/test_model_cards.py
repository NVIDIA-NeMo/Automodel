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


# Over the default 5s budget on purpose: this module drives git through subprocesses over throwaway repositories.
# Shrink the work or the process count before raising this further.

from pathlib import Path

import pytest

from tools.docs.checks import model_cards as checks
from tools.docs.checks.model_cards import (
    _is_local_checkpoint_override,
    _model_card_command_violations,
    _validate_card_aliases,
    _validate_changed_card_identity,
    _validate_commands,
    _validate_descriptions,
    _validate_model_card_recipe_identity,
)

pytestmark = pytest.mark.timeout(70)


def test_model_card_path_aliases_preserve_published_paths():
    _validate_card_aliases(Path(__file__).resolve().parents[5])


def test_changed_model_cards_match_linked_recipe_identity():
    _validate_changed_card_identity(Path(__file__).resolve().parents[5])


@pytest.mark.parametrize(
    ("title", "checkpoint", "message"),
    [
        ("Step-5-Preview", "Step-5-Preview-BF16", "title .* must match a linked recipe model ID"),
        ("Step-5-Preview-BF16", "Step-5-Preview", "link the exact Hugging Face checkpoint"),
    ],
)
def test_model_card_identity_rejects_recipe_mismatch(tmp_path: Path, title: str, checkpoint: str, message: str):
    recipe = tmp_path / "examples/llm_finetune/step5.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("model:\n  pretrained_model_name_or_path: stepfun-ai/Step-5-Preview-BF16\n")
    card = tmp_path / "step-5-preview.mdx"
    card.write_text(
        f'---\ntitle: "{title}"\nslug: model-coverage/large-language-models/stepfun-ai/{title}\n---\n'
        f"[Checkpoint](https://huggingface.co/stepfun-ai/{checkpoint})\n"
        "[SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/step5.yaml)\n"
    )
    with pytest.raises(ValueError, match=message):
        _validate_model_card_recipe_identity(tmp_path, card)


@pytest.mark.parametrize("title,valid", [("Step-5-Preview", True), ("Step-5-Preview-BF16", False)])
def test_reference_card_identifies_fp8_checkpoint_without_bf16_recipe(tmp_path: Path, title: str, valid: bool):
    card = tmp_path / "step-5-preview.mdx"
    card.write_text(
        f'---\ntitle: "{title}"\nslug: model-coverage/large-language-models/stepfun-ai/{title}\n---\n'
        "[FP8](https://huggingface.co/stepfun-ai/Step-5-Preview)\n"
        "[BF16](https://huggingface.co/stepfun-ai/Step-5-Preview-BF16)\n"
    )
    if valid:
        _validate_model_card_recipe_identity(tmp_path, card)
    else:
        with pytest.raises(ValueError, match="primary HF checkpoint"):
            _validate_model_card_recipe_identity(tmp_path, card)


@pytest.mark.parametrize("recipe_backed", [True, False])
def test_released_checkpoint_identity_accepts_a_stable_preview_filename(tmp_path: Path, recipe_backed: bool):
    card = tmp_path / "docs/model-coverage/llm/org/preview-name.mdx"
    card.parent.mkdir(parents=True)
    document = (
        '---\ntitle: "Released-Model-7B"\n'
        "slug: model-coverage/large-language-models/org/Released-Model-7B\n---\n"
        "[Checkpoint](https://huggingface.co/org/Released-Model-7B)\n"
    )
    if recipe_backed:
        recipe = tmp_path / "examples/model.yaml"
        recipe.parent.mkdir()
        recipe.write_text("model:\n  pretrained_model_name_or_path: org/Released-Model-7B\n")
        document += "[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/model.yaml)\n"
    card.write_text(document)

    _validate_model_card_recipe_identity(tmp_path, card)

    assert card.read_text() == document
    assert not card.with_name("released-model-7b.mdx").exists()


@pytest.mark.parametrize("slug", [None, "", " ", "/", 123, False])
def test_stable_model_card_filename_requires_an_explicit_slug(tmp_path: Path, slug):
    import yaml

    card = tmp_path / "preview-name.mdx"
    metadata = {"title": "Released-Model-7B"}
    if slug is not None:
        metadata["slug"] = slug
    card.write_text(
        "---\n" + yaml.safe_dump(metadata) + "---\n[Checkpoint](https://huggingface.co/org/Released-Model-7B)\n"
    )

    with pytest.raises(ValueError, match="add an explicit frontmatter slug.*keep the existing MDX filename"):
        _validate_model_card_recipe_identity(tmp_path, card)


def test_model_card_descriptions_meet_seo_contract():
    _validate_descriptions(Path(__file__).resolve().parents[5])


@pytest.mark.parametrize(
    "value,document,expected",
    [
        ("/data/model", "", True),
        ('"/data/model"', "", True),
        ('"${MODEL}"', "export MODEL=/data/model", True),
        ('"${MODEL}"', "", False),
        ('"${MODEL}"', "export MODEL=other/model", False),
        ("other/model", "", False),
    ],
)
def test_local_checkpoint_override(value, document, expected):
    assert _is_local_checkpoint_override(value, document) is expected


def test_model_card_commands_use_checked_in_recipes():
    _validate_commands(Path(__file__).resolve().parents[5])


def test_recipe_link_errors_include_every_link_card_and_compatibility_copy(tmp_path):
    from tools.docs.build import coverage_tables as coverage

    models = []
    source_paths = ["docs/model-coverage/llm/first.mdx", "docs/model-coverage/llm/second.mdx"]
    alias = "docs/model-coverage/legacy/old-name.mdx"
    expected = []
    for index, source_path in enumerate([*source_paths, alias]):
        path = tmp_path / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        links = []
        for suffix in ("a", "b"):
            recipe_path = f"examples/model_{index}_{suffix}.yaml"
            recipe = tmp_path / recipe_path
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text("model: {}\n")
            links.append(f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{'a' * 40}/{recipe_path})")
            expected.append(recipe_path)
        path.write_text("---\nslug: model-coverage/large-language-models/org/Model\n---\n" + "\n".join(links))
        if source_path in source_paths:
            models.append(coverage._ModelDoc("LLM", f"/models/{index}", (), "Model", source_path))

    with pytest.raises(ValueError) as caught:
        coverage._collect_documented_recipe_urls(tmp_path, models)

    message = str(caught.value)
    for source_path in [*source_paths, alias]:
        assert source_path in message
    for recipe_path in expected:
        assert f"replace its commit permalink with {coverage.REPOSITORY_URL}/{recipe_path}" in message


def test_command_validation_collects_link_and_yaml_errors_across_cards(tmp_path):
    recipe_sources = {"pinned": "model: {}\n", "syntax": "[", "mapping": "[]"}
    for name, source in recipe_sources.items():
        recipe = tmp_path / "examples" / f"{name}.yaml"
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe.write_text(source)
    for name in [*recipe_sources, "missing"]:
        card = tmp_path / "docs" / "model-coverage" / "llm" / "test" / f"{name}.mdx"
        card.parent.mkdir(parents=True, exist_ok=True)
        revision = "a" * 40 if name == "pinned" else "main"
        card.write_text(
            f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{revision}/examples/{name}.yaml)\n"
            f"```bash\nuv run automodel -c examples/{name}.yaml\n```\n"
        )

    offenders = _model_card_command_violations(tmp_path)

    assert len(offenders) == 4
    for name in [*recipe_sources, "missing"]:
        assert any(f"{name}.mdx: cannot read recipe examples/{name}.yaml:" in item for item in offenders)


@pytest.mark.parametrize("synchronized", [True, False])
def test_flux_compatibility_copy_must_match_its_canonical_card(tmp_path: Path, synchronized: bool):
    alias = "docs/model-coverage/diffusion/black-forest-labs/flux3-image-klein.mdx"
    canonical = "docs/model-coverage/diffusion/black-forest-labs/flux-3-image-klein-base.mdx"
    content = (
        '---\ntitle: "flux-3-image-klein-base"\n'
        "slug: model-coverage/diffusion/black-forest-labs/flux-3-image-klein-base\n---\nBody\n"
    )
    (tmp_path / canonical).parent.mkdir(parents=True)
    (tmp_path / canonical).write_text(content)
    (tmp_path / alias).write_text(content if synchronized else content + "Compatibility-only edit\n")

    navigation = tmp_path / "docs/fern/versions/nightly.yml"
    navigation.parent.mkdir(parents=True)
    navigation.write_text(f"navigation:\n  - page: Model\n    path: ../../{canonical.removeprefix('docs/')}\n")

    if synchronized:
        checks._validate_card_aliases(tmp_path)
    else:
        with pytest.raises(ValueError, match="Compatibility card must stay synchronized"):
            checks._validate_card_aliases(tmp_path)


@pytest.mark.parametrize(
    ("description", "message"),
    [
        ("", "length 0"),
        ("A" * 180, "outside 120-170"),
        ("A" * 130, "does not contain the exact title"),
        ("Model-1B NeMo AutoModel " + "a" * 100 + "é", "non-ASCII"),
    ],
)
def test_description_validator_rejects_invalid_metadata(tmp_path: Path, description: str, message: str):
    card = tmp_path / "docs/model-coverage/llm/org/model-1b.mdx"
    card.parent.mkdir(parents=True)
    card.write_text(f'---\ntitle: "Model-1B"\ndescription: "{description}"\n---\n')

    with pytest.raises(ValueError, match=message):
        checks._validate_descriptions(tmp_path)


def test_description_validator_accepts_a_valid_model_card(tmp_path: Path):
    card = tmp_path / "docs/model-coverage/llm/org/model-1b.mdx"
    card.parent.mkdir(parents=True)
    description = (
        "Fine-tune Model-1B with NeMo AutoModel using repository-backed training recipes, "
        "checkpoint loading, and distributed configuration."
    )
    card.write_text(f'---\ntitle: "Model-1B"\ndescription: "{description}"\n---\n')

    checks._validate_descriptions(tmp_path)
