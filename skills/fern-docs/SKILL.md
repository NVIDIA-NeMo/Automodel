---
name: fern-docs
description: Maintain the NeMo AutoModel Fern docs site under fern/ — add, update, move, or remove pages; manage redirects, slugs, navigation, and version aliases; run validation and previews.
when_to_use: Editing or adding documentation pages, fixing broken links, renaming a slug, updating the sidebar, adding a redirect, regenerating the API library reference, debugging fern check / broken-link errors, cutting a new version train, 'edit docs', 'add doc page', 'fern check failing', 'preview fails locally'.
---

# Fern Docs Maintenance — NeMo AutoModel

Unified skill for adding, updating, moving, and removing pages on the NeMo AutoModel Fern documentation site at `docs.nvidia.com/nemo/automodel`.

## Scope rule

**ALL docs edits happen under `fern/`.** The legacy Sphinx tree at `docs/` is read-only reference; do not add new pages there. New pages, release notes, migration guides — everything belongs under `fern/versions/v0.4/pages/`.

**Single canonical content tree.** Unlike NeMo Gym (which historically maintained two parallel version trees), AutoModel keeps one canonical content tree at `fern/versions/v0.4/pages/`. Three nav YAMLs (`v0.4.yml`, `latest.yml`, `nightly.yml`) all point at the same MDX files, so a single edit flows to all three URL slugs (`/v0.4/...`, `/latest/...`, `/nightly/...`). **Do not duplicate pages across version folders.**

**Sidebar fidelity rule.** Section captions, page titles, and Model Coverage child ordering must match the **published v0.4.0 sidebar at docs.nvidia.com/nemo/automodel/latest** verbatim. Don't silently shorten a title or reorder siblings — the docs PM and content engineers diff against the published site and any drift is treated as a regression. If you want a shorter sidebar label, change the toctree-derived display name in the source — never just retitle in the MDX.

## Layout at a glance

```
fern/
├── fern.config.json              # Org slug + Fern CLI pin (4.62.4+)
├── docs.yml                      # Site config: instances, versions, redirects, libraries, theme
├── main.css                      # NVIDIA-green theme overrides
├── assets/                       # Logos and shared SVGs (NVIDIA_dark/light/symbol)
├── components/                   # BadgeLinks.tsx, Tag.tsx, CustomFooter.tsx
├── versions/
│   ├── v0.4.yml                  # CANONICAL nav for the v0.4 train
│   ├── latest.yml                # GA alias — copy of v0.4.yml; sync after every nav edit
│   ├── nightly.yml               # Bleeding-edge alias — currently mirrors v0.4
│   └── v0.4/pages/               # MDX content (130+ pages)
└── product-docs/                 # GENERATED Python API reference (gitignored)
```

```
File                                                      URL
─────────────────────────────────────────────────────────  ────────────────────────────────────────────
fern/versions/v0.4/pages/get-started/installation.mdx     /latest/get-started/installation
                                                          /v0.4/get-started/installation
                                                          /nightly/get-started/installation
```

## Operations

### Add a page

1. Gather: title, target section, filename (kebab-case `.mdx`), subdirectory under `fern/versions/v0.4/pages/`.
2. Create the MDX with frontmatter:

   ```mdx
   ---
   title: "<Page Title>"
   description: "One-line SEO description (or empty string)"
   position: 4
   ---

   <body — typically no leading `# H1`; Fern renders the title automatically>
   ```

3. Add a `- page:` entry to `fern/versions/v0.4.yml` under the right `section:`, with an explicit `slug:` if the desired URL differs from the slugified title:

   ```yaml
   - page: "<Page Title>"
     path: ./v0.4/pages/<subdir>/<filename>.mdx
     slug: <short-url-segment>
   ```

4. **Sync the aliases:** `cp fern/versions/v0.4.yml fern/versions/latest.yml && cp fern/versions/v0.4.yml fern/versions/nightly.yml`.
5. `make docs-check` (runs `fern check`) and verify URL resolves on `make docs` preview.

### Update a page

1. Locate by path, title, or keyword: `grep -rn "<keyword>" fern/versions/v0.4/pages/ --include="*.mdx"`.
2. **Content only** — edit the single MDX file. There is no mirror to maintain (latest/nightly serve the same content).
3. **Title change** — update the frontmatter `title:` and (if the page is in `versions/v0.4.yml`) update the `- page:` entry's display label. Re-sync `latest.yml` and `nightly.yml`.
4. **Section move** — `git mv` the file, update `path:` in `versions/v0.4.yml`, fix incoming links, re-sync aliases.
5. **Slug change** — change `slug:` in the YAML (or rename the file and let the default slug update). Add a `redirects:` entry in `docs.yml` so the old URL keeps working.

### Remove a page

1. Find incoming links: `grep -rn "<filename>" fern/versions/v0.4/pages/ --include="*.mdx"`.
2. `git rm fern/versions/v0.4/pages/<path>.mdx`.
3. Remove the `- page:` block from `versions/v0.4.yml` (and re-sync `latest.yml` / `nightly.yml`).
4. Fix or delete incoming links.
5. Add a redirect in `docs.yml` if the URL was public.

### Worked example: add a guide

Request: *"Add a fine-tuning guide for Qwen3.6 under Recipes & E2E Examples."*

1. Create `fern/versions/v0.4/pages/guides/llm/qwen3-6-finetune.mdx`:

   ```mdx
   ---
   title: "Fine-Tune Qwen3.6"
   description: "End-to-end SFT and PEFT recipes for Qwen3.6 on NeMo AutoModel"
   ---

   This guide walks through fine-tuning Qwen3.6 with NeMo AutoModel...
   ```

2. Add to `fern/versions/v0.4.yml` under the `Recipes & E2E Examples` section, slotted in publication-order with the other fine-tune entries:

   ```yaml
   - page: "Fine-Tune Qwen3.6"
     path: ./v0.4/pages/guides/llm/qwen3-6-finetune.mdx
     slug: qwen3-6-finetune
   ```

3. `cp fern/versions/v0.4.yml fern/versions/latest.yml && cp fern/versions/v0.4.yml fern/versions/nightly.yml`.
4. `make docs-check` then `make docs` to preview at `http://localhost:3002/latest/recipes-e2e-examples/qwen3-6-finetune`.

### Worked example: rename a slug with a redirect

Request: *"Rename `/recipes-e2e-examples/sft-peft` to `/recipes-e2e-examples/fine-tuning`."*

1. Edit `versions/v0.4.yml`, change the `slug:` on the SFT & PEFT entry from `sft-peft` to `fine-tuning`.
2. Add a redirect to `fern/docs.yml`:

   ```yaml
   redirects:
     - source: "/:version/recipes-e2e-examples/sft-peft"
       destination: "/:version/recipes-e2e-examples/fine-tuning"
   ```

3. `grep -rn "/recipes-e2e-examples/sft-peft" fern/versions/v0.4/pages/` and update incoming body links.
4. Re-sync `latest.yml` and `nightly.yml`.

## Content guidelines

NeMo AutoModel uses **Fern-native MDX components**. Don't use GitHub `> [!NOTE]` syntax — it doesn't render in MDX.

| Purpose | Component |
|---|---|
| Neutral aside | `<Note>...</Note>` |
| Helpful tip | `<Tip>...</Tip>` |
| Informational callout | `<Info>...</Info>` |
| Warning | `<Warning>...</Warning>` |
| Error / danger | `<Error>...</Error>` |
| Card grid on landing pages | `<Cards>` with `<Card title="..." href="...">` children |
| Card chips ("start here", "5 min") | `<Tag variant="primary">label</Tag>` — sphinx-design `{bdg-*}` mapping |
| Header badge rows (PyPI, license, GitHub) | `<BadgeLinks badges={[{href, src, alt}, ...]} />` |

Required imports when using `<Tag>` or `<BadgeLinks>` (`landing_badges.py` adds these in the post stage; in hand-written pages add them yourself):

```mdx
import { Tag } from "@/components/Tag";
import { BadgeLinks } from "@/components/BadgeLinks";
```

`<Tag variant="...">` accepts: `primary`, `secondary`, `success`, `warning`, `danger`, `info`, `light`, `dark` (1:1 with sphinx-design `{bdg-*}` variants).

Images live in `fern/assets/` (shared across all pages) or alongside the MDX file (page-scoped). Reference page-scoped images with relative paths (`./image.png`), not absolute (`/image.png`) — Fern's path resolver doesn't normalize root-relative image paths the same way as link targets.

## Frontmatter

```yaml
---
title: "<Page Title>"        # required — Fern renders this as the page H1
description: ""              # required (may be "") — SEO meta description
position: 1                  # optional — orders auto-discovered pages within a folder
---
```

**Don't repeat the title as a leading `# H1` in the body.** Fern already renders `title:` at the top of the page, and a duplicate creates a double heading. The post-stage `remove_duplicate_h1.py` strips them when title and H1 match exactly, but it can't catch near-duplicates (e.g. `title: "About"` vs `# About NeMo AutoModel`) — keep the body H1-free, or promote the descriptive form to `subtitle:` if you want both visible.

## Internal links

Use **version-agnostic** paths — no `/latest/`, `/v0.4/`, or `/nightly/` prefix:

```mdx
[Install NeMo AutoModel](/get-started/installation)
[Llama coverage](/model-coverage/large-language-models/llama)
```

The same MDX backs every version slug; a hard-coded prefix sends readers across versions unintentionally. URL slugs come from explicit `slug:` overrides in the version YAML (set during the migration so URLs stay short while sidebar titles match the verbose published H1s) — so `Install NeMo AutoModel` is at `/get-started/installation`, not `/get-started/install-nemo-automodel`.

For cross-repo references (yaml configs, Python source), use absolute GitHub URLs:

```mdx
[mistral4_medpix.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/mistral4/mistral4_medpix.yaml)
```

## Validate

```bash
make docs-check          # `fern check` — config + MDX validation
```

`fern check` must pass before commit. The dev server's broken-link warnings for version-prefixed routes (e.g. `/latest/get-started/installation` from MDX that uses `/get-started/installation`) are **false positives** — Fern's strict validator doesn't resolve version-agnostic links. The published site renders them correctly. The URLMap-based `validate_fern_internal_links.py` (under the convert-to-fern toolkit) is authoritative.

To regenerate the autodoc library reference (gitignored under `product-docs/`):

```bash
make docs                # runs `fern docs md generate` then `fern docs dev`
```

`fern docs md generate` populates `product-docs/` from the `nemo_automodel` package source declared in `docs.yml` `libraries:` block. Without this step, a cold `fern docs dev` fails with `Folder not found: ./product-docs/...`.

## Preview and publish

| Goal | Command |
|---|---|
| Local preview at `http://localhost:3002` | `make docs` |
| Validation only (no server) | `make docs-check` |
| Shared preview URL on `*.docs.buildwithfern.com` (needs `DOCS_FERN_TOKEN`) | `make docs-preview` |
| Trigger production publish workflow on `origin/main` | `make docs-publish` |

PRs that touch `fern/**` get an automatic Fern preview URL posted as a 🌿 comment by `fern-docs-preview-comment.yml`. No manual step.

```
                    ┌─ fern-docs-ci.yml                  → fern check (push to pull-request/<n>)
PR (touches fern/) ─┼─ fern-docs-preview-build.yml       → upload fern/ artifact (no secrets)
                    └─ fern-docs-preview-comment.yml     → 🌿 preview URL comment

Push to main (touches fern/) → publish-fern-docs.yml → docs.nvidia.com/nemo/automodel
Tag push (docs/v*)           → publish-fern-docs.yml → docs.nvidia.com/nemo/automodel
Manual dispatch              → publish-fern-docs.yml → docs.nvidia.com/nemo/automodel
```

The preview-comment + publish jobs require the `DOCS_FERN_TOKEN` org secret (already wired for `build-docs.yml`).

## Cutting a new version train

When NeMo AutoModel ships a new GA (e.g. `v0.5`):

1. `cp -r fern/versions/v0.4 fern/versions/v0.5` — frozen snapshot of the previous "latest" content.
2. `cp fern/versions/v0.4.yml fern/versions/v0.5.yml` and rewrite `./v0.4/` path prefixes to `./v0.5/`.
3. Update `fern/versions/latest.yml` to point at the new train: `cp fern/versions/v0.5.yml fern/versions/latest.yml`. (`latest` is the auto-bumping GA alias.)
4. In `fern/docs.yml` `versions:`, add a new frozen-pin entry (`display-name: "0.5.0 · 26.07"`, `slug: v0.5`, `availability: stable`) and keep the previous pin (`v0.4`) for permalink stability.
5. `fern/versions/v0.4/pages/` continues to back the v0.4 frozen-pin URL slug; new edits land in `fern/versions/v0.5/pages/` (re-sync `latest.yml` after each).
6. Promote `nightly` to `availability: stable` if and when its content tree gets cut over.
7. Tag `docs/v0.5.0` and push to publish.

## Commits and DCO

Every commit needs a `Signed-off-by:` trailer:

```bash
git commit -s -m "docs: add fine-tuning guide for Qwen3.6"
```

If sign-off is missing on a recent commit, amend with `git commit --amend -s`. PR titles follow Conventional Commits: `docs(fern): <short summary>`. See [`AGENTS.md`](../../AGENTS.md) for the full repo commit convention.

## Debugging

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML; `slug:` must not collide with siblings |
| Page 404 in preview | Missing `slug:` override (default slugifies the long display title) or `position:` collision in an auto-discovered folder |
| `Folder not found: ./product-docs/...` on `fern docs dev` | Run `make docs` once to populate the library reference |
| `[ERR_PNPM_IGNORED_BUILDS]` on first `fern docs dev` | pnpm 10+ blocks esbuild's postinstall — `pnpm config set onlyBuiltDependencies '["esbuild"]' --location global`, then `rm -rf ~/.fern/app-preview` and retry |
| Broken-link warning on version-agnostic path | `fern docs broken-links` false-positives; URLMap-based validator is authoritative |
| `JSX expressions must have one parent element` | Wrap multi-element JSX in `<>...</>` or a `<div>` |
| Old Sphinx URL breaks | Add a `redirects:` entry in `fern/docs.yml`; the redirect generator already handles `/index.html` and `.html` legacy forms |
| Image not rendering | Use relative path (`./image.png`) for page-scoped images, not root-relative (`/image.png`) |
| Sidebar caption looks shortened vs published site | Compare against `docs.nvidia.com/nemo/automodel/latest` and restore the verbatim title in `versions/v0.4.yml` |
| `latest.yml` and `v0.4.yml` drift | Re-sync: `cp fern/versions/v0.4.yml fern/versions/latest.yml` (and same for `nightly.yml`) |

## Key references

| File | Purpose |
|---|---|
| `fern/docs.yml` | Site config — `instances`, `versions`, `redirects`, `libraries`, theme |
| `fern/versions/v0.4.yml` | Canonical nav tree |
| `fern/versions/{latest,nightly}.yml` | Aliases — content copies of `v0.4.yml` |
| `fern/versions/v0.4/pages/` | MDX content (130+ pages) |
| `fern/components/` | `BadgeLinks.tsx`, `Tag.tsx`, `CustomFooter.tsx` |
| `fern/main.css` | Theme overrides — NVIDIA green, badge spacing |
| `fern/README.md` | Human-facing orientation |
| `Makefile` | `make docs / docs-check / docs-preview / docs-publish` |
| `.github/workflows/fern-docs-*.yml` | CI: check, preview build, preview comment |
| `.github/workflows/publish-fern-docs.yml` | CI: publish to docs.nvidia.com/nemo/automodel |
| `docs/` | **Legacy** Sphinx source — read-only reference for fidelity checks |
