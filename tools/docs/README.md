# Documentation tools

Run the single documentation check from the repository root:

```bash
uv run --script tools/docs/check_docs.py
```

`check_docs.py` owns orchestration for contributors, agents, and CI. Publishing
uses the same command with `--keep-generated` to retain validated tables.

| Location | Responsibility |
| --- | --- |
| `catalog.py` | Shared architecture aliases, provider mappings, and generation markers. Keep aliases literal so trusted preview tooling can read PR metadata without executing PR code. |
| `cards.py` | Model-card discovery by slug and canonical-file selection from nightly navigation. |
| `checks/sources.py` | Coordinates source contracts before generation. |
| `checks/model_cards.py` | Card identity, descriptions, commands, and compatibility copies. |
| `checks/navigation.py` | Navigation, provider hierarchy, routes, redirects, and links. |
| `checks/architecture.py` | Architecture reference sections and registry documentation coverage. |
| `checks/recipes.py` | Recipe model discovery and publisher coverage; optional HF architecture resolution. |
| `checks/mdx.mjs` | MDX syntax validation. |
| `build/coverage_tables.py` | Derived support, release, and architecture tables. |
| `build/code_styles.py` | Rendering styles and their cleanup. |
| `assets/provider_icons.py` | Provider logo maintenance. |

Model-card source filenames stay stable across checkpoint releases. Use an
explicit frontmatter `slug` and the released checkpoint `title` to name the public
page; keep navigation `path` pointing to the existing MDX. Update descriptions, checkpoint links,
sidebar labels, and provider indexes together, adding redirects for old published
site URLs. Filename differences are allowed; checkpoint and route mismatches are
still rejected.

A card may stay in its original source directory, including a different provider
or category directory or another location under `docs/`. Its full frontmatter
slug determines the public model category, provider, and checkpoint route. Place
its navigation entry under that category and provider, with `path` pointing to
the original file. This preserves previously shared GitHub source URLs without
requiring a move or a new copy. For example:

```yaml
# docs/guides/preview-announcement.mdx frontmatter
title: "Released-Model-7B"
slug: model-coverage/vision-language-models/org/Released-Model-7B
```

Existing compatibility copies are discovered by their shared slug. Exactly one
file for that slug must appear in nightly navigation; that file is canonical for
generated tables. Other copies stay on disk and must remain byte-identical.
Missing or ambiguous canonical selection and divergent copies fail validation.
No per-file alias registration in `catalog.py` is needed.

Regression tests mirror this layout under `tests/unit_tests/tools/docs/`.
Rules belong in the tools; tests exercise valid inputs, invalid inputs, and
failure/cleanup behavior. CI collects the entire test directory separately
from the documentation check:

```bash
uv run --no-project --isolated --python 3.12 \
  --with pytest --with pytest-timeout --with pyyaml --with pillow \
  python -m pytest -q --noconftest -o 'norecursedirs=.* node_modules __pycache__' \
    tests/unit_tests/tools/docs
```

The recursion override includes the source `build/` test directory, which
pytest otherwise skips by default. The regression suite requires Node.js 20
and npm for the MDX CLI fixtures.
HF config resolution remains opt-in through `NEMO_DOC_COVERAGE_RESOLVE_ARCHS=1`
and requires the training environment and accessible or cached HF configs.
