# Validation, tooling, preview, and publish

## Validate

The only docs-check entry point for contributors, agents, and CI is
`tools/docs/check_docs.py`. Run it from the repository root before committing or
opening a PR:

```bash
uv run --script tools/docs/check_docs.py
```

This validates documentation sources before generation, then runs the MDX and
Fern checks with the pinned Fern CLI. It installs only the isolated docs
dependencies (PyYAML), requires Node.js 20 and `make`, and cleans generated tables
and styles after validation, including on failure.

Add validation steps to the script so local checks and CI stay in sync.
Keep source validation implementations in `tools/docs/`; pytest provides
regression tests for the tools instead of owning documentation rules.
Mirror the `checks/`, `build/`, and `assets/` layout under
`tests/unit_tests/tools/docs/`; shared metadata belongs in `catalog.py`.
The Makefile's `docs-check` and `docs-mdx-check` targets forward to the script
for compatibility; `_docs-check` is internal to the script.

Publishing invokes the same script with `--keep-generated` to retain validated
model-coverage tables for deployment. The script owns table generation; the
publish workflow must not invoke the generator separately. Failures always
clean up generated tables.

The shared check must pass before commit. It runs a no-secret MDX parser before `fern check`, so raw HTML must be valid JSX (for example, use `<img ... />`, not `<img ...>`). The dev server's broken-link warnings for version-prefixed routes (e.g. `/latest/get-started/installation` from MDX that uses `/get-started/installation`) are **false positives** — Fern's strict validator doesn't resolve version-agnostic links. The published site renders them correctly. The URLMap-based `validate_fern_internal_links.py` (under the convert-to-fern toolkit) is authoritative.

To regenerate the autodoc library reference (gitignored under `docs/fern/product-docs/`):

```bash
make docs                # runs `fern docs md generate` then `fern docs dev`
```

`fern docs md generate` populates `docs/fern/product-docs/` from the `nemo_automodel` package source declared in `docs.yml` `libraries:` block. Without this step, a cold `fern docs dev` fails with `Folder not found: ./product-docs/...`.

`check_docs.py` requires `uv`, Node.js 20 with `npm`/`npx`, Git, `make`, and
full repository history; follow the setup in `CONTRIBUTING.md`. Report the
command's result when handing off documentation changes. Group validators in
`checks/`, rendering helpers in `build/`, and asset tools in `assets/`; see
`tools/docs/README.md` for the file map.

## Preview and publish

| Goal | Command |
|---|---|
| Local preview at `http://localhost:3002` | `make docs` |
| Validate before pushing (no server; run at repo root) | `uv run --script tools/docs/check_docs.py` |
| Shared preview URL on `*.docs.buildwithfern.com` (needs `DOCS_FERN_TOKEN`) | `make docs-preview` |
| Trigger production publish workflow on `origin/main` | `make docs-publish` |

PRs that touch `docs/**` get an automatic Fern preview URL posted as a 🌿 comment by `fern-docs-preview-comment.yml`. No manual step.

Check and publish workflows use the shared script's Makefile target to restore
archived pages; previews restore archives in their own trusted build steps.
Frozen backward-version trees live on `docs-archive`, not on `main`.

```
                    ┌─ fern-docs-ci.yml                  → tools/docs/check_docs.py (push to pull-request/<n>)
PR (touches docs/) ─┼─ fern-docs-preview-build.yml       → stitch → upload docs/ artifact (no secrets)
                    └─ fern-docs-preview-comment.yml     → 🌿 preview URL comment (consumes artifact)

Push to main (touches docs/) → publish-fern-docs.yml → stitch → docs.nvidia.com/nemo/automodel
Tag push (docs/v*)           → publish-fern-docs.yml → stitch → docs.nvidia.com/nemo/automodel
Manual dispatch              → publish-fern-docs.yml → stitch → docs.nvidia.com/nemo/automodel
```

The preview-comment + publish jobs require the `DOCS_FERN_TOKEN` org secret (already wired for `build-docs.yml`).
