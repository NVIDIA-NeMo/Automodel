---
name: fern-docs
description: Maintain the NeMo AutoModel Fern docs site under docs/ (MDX content) + docs/fern/ (infra) — add, update, move, or remove pages; manage redirects, slugs, navigation, and version aliases; run validation and previews. Use when editing or adding documentation pages, fixing broken links, renaming a slug, updating the sidebar, adding a redirect, regenerating the API library reference, debugging fern check / broken-link errors, cutting a new version train, 'edit docs', 'add doc page', 'fern check failing', 'preview fails locally'.
---

# Fern Docs Maintenance — NeMo AutoModel

Unified skill for adding, updating, moving, and removing pages on the NeMo AutoModel Fern documentation site at `docs.nvidia.com/nemo/automodel`.

## Scope Rule

**Nightly MDX content lives at the top level of `docs/`** (e.g. `docs/index.mdx`, `docs/guides/llm/finetune.mdx`). **`docs/fern/` holds only Fern build infrastructure** — config, theme, and components. New pages, release notes, migration guides → add as a top-level `.mdx` under `docs/`.

**Only the nightly tree is kept on `main`.** Frozen backward-version snapshots live on the `docs-archive` branch and are restored at build time — see *Archived Backward Versions* below.

**Four content trees, plus a GA alias YAML.**

- `docs/` — bleeding-edge (nightly) tree. Every PR lands here. Mounted at the `nightly` URL slug via `docs/fern/versions/nightly.yml` (paths reach back up via `../../<rel>.mdx`).
- `docs/fern/versions/v0.4/pages/` — frozen 0.4.0 GA snapshot. **Not on `main`**: it lives on the `docs-archive` branch and is restored under this path by `make -C docs/fern docs-stitch` (local) or the `stitch-fern-versions` CI action (build). Mounted at the `v0.4` URL slug via `v0.4.yml`. Only changes via deliberate back-port (on `docs-archive`).
- `docs/fern/versions/v0.5/pages/` — frozen 0.5.0 GA snapshot, also restored from `docs-archive`. Mounted at the `v0.5` URL slug through `v0.5.yml`.
- `docs/fern/versions/v0.7/pages/` — frozen 0.7.0 GA snapshot, also restored from `docs-archive`. Mounted at the `v0.7` URL slug through `v0.7.yml`.
- `docs/fern/versions/latest.yml` — GA alias. Its `path:` lines mount the current GA's content at `./v0.7/pages/...`. Repointed at the next GA's tree when one is cut.

Nightly continues to change after each release; the v0.4, v0.5, and v0.7 snapshots stay frozen. **Default editing target is `docs/` top-level.** Back-ports to a frozen version happen on the `docs-archive` branch, not here — call out the divergence in the PR description.

### Archived Backward Versions

Fern has no native way to source a version train's prose from another git ref: `fern generate --docs` reads the single local working tree, and publish is a full-site snapshot (a train missing from the tree is *unpublished*). So frozen GA pages are kept off `main` on the **`docs-archive` branch** and restored before every Fern build.

- **The registry** for checking and publishing lives in the Makefile's `docs-stitch` target, invoked by `tools/docs/check_docs.py`. Archived preview builds declare their archive inputs in their workflow. The `<git-ref>` is opaque: a **branch** (default — all frozen versions in one place, easy back-ports) or a **tag** like `docs/v0.4.0` (immutable snapshot). `latest` is an alias of an existing pages tree and needs **no** entry.
- **The mechanism** is the `.github/actions/stitch-fern-versions` composite action in archived preview builds, or the equivalent Makefile `docs-stitch` target in local checks, CI checks, and publishing: `git fetch --depth=1 origin <ref>` then `git restore --source=FETCH_HEAD -- docs/fern/versions/<vdir>/pages`. `docs.yml` + the nav YAMLs always come from the live checkout, never the archive ref, so frozen prose can't drift into the wrong train.
- **Locally**, `make docs` / `docs-preview` and the shared checker use `make docs-stitch`, which does the same restore. Override the check's archive ref with `ARCHIVE_REF=docs/v0.4.0 uv run --script tools/docs/check_docs.py`.
- The restored path is gitignored on `main`, so a local stitch won't show the pages as untracked.

**Sidebar fidelity rule.** Section captions, page titles, and Model Coverage child ordering must match the **published v0.4.0 sidebar at docs.nvidia.com/nemo/automodel/v0.4** verbatim. Don't silently shorten a title or reorder siblings — the docs PM and content engineers diff against the published site and any drift is treated as a regression. If you want a shorter sidebar label, change the toctree-derived display name in the source — never just retitle in the MDX.

## Layout at a Glance

```text
docs/                                ← nightly MDX (top level)
├── index.mdx, breaking-changes.mdx, release-notes.mdx, ...
├── about/, guides/, model-coverage/, dataset-coverage/, launcher/, api-reference/
├── *.png / *.jpg                    ← page-scoped images
└── fern/                            ← infra only
    ├── fern.config.json             # Org slug + Fern CLI pin (5.139.0)
    ├── docs.yml                     # Site config + global-theme: nvidia (inherits
    │                                #   logos / footer / theme CSS / fonts / OneTrust JS
    │                                #   from NVIDIA/fern-components)
    ├── components/                  # BadgeLinks.tsx, Tag.tsx
    │                                #   (repo-specific; NVIDIA footer ships in global theme)
    ├── versions/
    │   ├── nightly.yml              # Nav for nightly — paths → ../../<rel>.mdx (up into docs/)
    │   ├── v0.4.yml                 # Nav for frozen 0.4.0 — paths → ./v0.4/pages/
    │   ├── v0.4/pages/              # Frozen 0.4.0 MDX (back-ports only)
    │   ├── v0.5.yml                 # Nav for frozen 0.5.0 — paths → ./v0.5/pages/
    │   ├── v0.5/pages/              # Frozen 0.5.0 MDX (back-ports only)
    │   ├── v0.7.yml                 # Nav for frozen 0.7.0 — paths → ./v0.7/pages/
    │   ├── v0.7/pages/              # Frozen 0.7.0 MDX (back-ports only)
    │   └── latest.yml               # GA alias — paths → ./v0.7/pages/; repointed at next GA cut
    └── product-docs/                # GENERATED Python API reference (gitignored)
```

```text
File                                                     URL (after /nemo/automodel)
docs/guides/installation.mdx                              /nightly/get-started/installation
docs/fern/versions/v0.7/pages/guides/installation.mdx       /latest/get-started/installation
                                                         /v0.7/get-started/installation
docs/fern/versions/v0.5/pages/guides/installation.mdx       /v0.5/get-started/installation
docs/fern/versions/v0.4/pages/guides/installation.mdx       /v0.4/get-started/installation
```

## Prerequisites

Install Git, Make, Node.js 20 with npm, and `uv`. Use the Fern CLI version pinned in `docs/fern/fern.config.json` (currently 5.139.0). The shared docs check, `uv run --script tools/docs/check_docs.py`, provisions Python 3.12 and PyYAML separately from the training environment.

Run the commands in this guide from the repository root. They use `make -C docs/fern <target>`; if you are already in `docs/fern/`, use `make <target>` instead. Before API library generation, run `make -C docs/fern docs-login` to provision your Fern dashboard account and complete CLI login.

## Operations

### Add a Page

1. Gather: title, target section, filename (kebab-case `.mdx` for ordinary pages), subdirectory under `docs/`. Preserve canonical Hub-ID-based paths and case for dataset cards.
2. Create the MDX at `docs/<subdir>/<filename>.mdx` with frontmatter:

   ```mdx
   ---
   title: "<Page Title>"
   description: "Concise summary of the page"
   position: 4
   ---

   <body — typically no leading `# H1`; Fern renders the title automatically>
   ```

3. Add a `- page:` entry to `docs/fern/versions/nightly.yml` under the right `section:`, with `path:` reaching up into `docs/` via `../../`:

   ```yaml
   - page: "<Page Title>"
     path: ../../<subdir>/<filename>.mdx
     slug: <short-url-segment>
   ```

4. Run `uv run --script tools/docs/check_docs.py` from the repository root, then verify the URL in the `make -C docs/fern docs` preview. `latest.yml` mounts the frozen v0.7 tree and is not synchronized with nightly.

### Update a Page

1. Locate by path, title, or keyword: `rg -n "<keyword>" docs/ -g "*.mdx" -g "!docs/fern/**"`.
2. **Content only** — edit the single MDX file at `docs/<...>.mdx`.
3. **Title change** — update the frontmatter `title:` and update the `- page:` entry's display label in `docs/fern/versions/nightly.yml`.
4. **Section move** — `git mv` the file within `docs/`, update `path:` in `nightly.yml`, fix incoming links.
5. **Slug change** — set an explicit `slug:` while keeping the source filename stable. For model cards, set the full route in MDX frontmatter; keep the navigation `path:` pointing to that file. Add a `redirects:` entry in `docs/fern/docs.yml` so the old published URL keeps working.

### Model repository name becomes public

Keep the existing model-card filename and directory when the released checkpoint
name differs from the original MDX name. Update its frontmatter `title`, `description`, and
explicit `slug`, Hugging Face links, and linked recipe checkpoint IDs to the
released identity. Update the sidebar label and provider-index link, leaving the
navigation `path` pointed at the original MDX file. For example, an existing
`preview-name.mdx` can use:

```yaml
title: "Released-Model-7B"
slug: model-coverage/large-language-models/org/Released-Model-7B
```

The source filename does not determine the public model name or route. Do not
rename, move, or duplicate the card to match the release name. Add redirects for
previously published docs-site routes, and preserve existing compatibility copies.
Run the shared docs check to validate the metadata, navigation, and recipe identity.

### Redirect Quirks

Four things to watch when editing `redirects:` in `docs/fern/docs.yml`:

1. **`:path*` does NOT match the empty-path case.** `/<basepath>/v0.4/:path*/index.html` will *not* match `/<basepath>/v0.4/index.html` (where `:path*` would have to be empty). Each version-root `index.html` needs its own explicit rule. NeMo Curator (NVIDIA-NeMo/Curator#1938) discovered this when their version-root URLs 404'd. AutoModel ships explicit rules for `latest`, `v0.7`, `v0.5`, `v0.4`, `nightly`, and the legacy `0.7`, `0.5`, and `0.4` forms — when you add a new version slug, add four new explicit rules: `<slug>/index.html`, `<slug>/index`, plus the same two for any legacy form (e.g. `0.5` → `v0.5`).
2. **Older un-migrated versions need a fallback.** Whatever versions the published Sphinx site exposed (check the version-switcher dropdown on `docs.nvidia.com/nemo/<product>/latest/`) but you didn't migrate into Fern still need to resolve. The pattern: redirect each old slug's URLs to the equivalent path under `/latest/` so external bookmarks and search results land on the closest current page instead of 404ing. Five rules per old version: `<slug>/index.html`, `<slug>/index`, `<slug>/:path*/index.html`, `<slug>/:path*`, `<slug>/:path*.html` — all destinations `/latest/...`. AutoModel ships these for `0.3.0`, `0.2.0`, `0.1.0`.
3. **Order matters.** Specific rules must come before catch-alls — Fern uses first-match. Slot new rules *before* the `:path*/index.html` and `:path*.html` catch-alls.
4. **Don't ship `redirects: []`** then re-run the redirect generator on top — it replaces the whole `redirects:` block. Edit by hand or back up the existing rules first.

### Remove a Page

1. Find incoming links: `rg -n -F "<filename>" docs/ -g "*.mdx" -g "!docs/fern/**"`.
2. `git rm docs/<...>.mdx`.
3. Remove the `- page:` block from `docs/fern/versions/nightly.yml`.
4. Fix or delete incoming links.
5. Add a redirect in `docs/fern/docs.yml` if the URL was public.

### Add a Guide: Worked Example

Request: *"Add a fine-tuning guide for Qwen3.6 under Recipes & E2E Examples."*

1. Create `docs/guides/llm/qwen3-6-finetune.mdx`:

   ```mdx
   ---
   title: "Fine-Tune Qwen3.6"
   description: "End-to-end SFT and PEFT recipes for Qwen3.6 on NeMo AutoModel"
   ---

   This guide walks through fine-tuning Qwen3.6 with NeMo AutoModel...
   ```

2. Add to `docs/fern/versions/nightly.yml` under the `Recipes & E2E Examples` section, slotted in publication-order with the other fine-tune entries:

   ```yaml
   - page: "Fine-Tune Qwen3.6"
     path: ../../guides/llm/qwen3-6-finetune.mdx
     slug: qwen3-6-finetune
   ```

3. Run `uv run --script tools/docs/check_docs.py` from the repository root, then `make -C docs/fern docs`, and open the new page through the nightly sidebar.

### Rename a Slug With a Redirect: Worked Example

Request: *"Rename `/recipes-e2e-examples/sft-peft` to `/recipes-e2e-examples/fine-tuning`."*

1. Edit `docs/fern/versions/nightly.yml`, change the `slug:` on the SFT & PEFT entry from `sft-peft` to `fine-tuning`.
2. Add a redirect to `docs/fern/docs.yml` for the nightly route being renamed:

   ```yaml
   redirects:
     - source: "/nemo/automodel/nightly/recipes-e2e-examples/sft-peft"
       destination: "/nemo/automodel/nightly/recipes-e2e-examples/fine-tuning"
   ```

3. `rg -n -F "/recipes-e2e-examples/sft-peft" docs/ -g "*.mdx" -g "!docs/fern/**"` and update incoming body links.

## Content Guidelines

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

Add these imports when using `<Tag>` or `<BadgeLinks>`:

```mdx
import { Tag } from "@/components/Tag";
import { BadgeLinks } from "@/components/BadgeLinks";
```

`<Tag variant="...">` accepts: `primary`, `secondary`, `success`, `warning`, `danger`, `info`, `light`, `dark` (1:1 with sphinx-design `{bdg-*}` variants).

Page-scoped images live alongside the MDX file (e.g. `docs/guides/audio/qwen_omni_asr.png`). Reference them with relative paths (`./image.png`), not absolute (`/image.png`) — Fern's path resolver doesn't normalize root-relative image paths the same way as link targets. The NVIDIA logos and favicon come from the `nvidia` global theme; do not add them locally.

## Frontmatter

```yaml
---
title: "<Page Title>"        # required — Fern renders this as the page H1
description: "<Concise page summary>"  # describe the page for search results
position: 1                  # optional — orders auto-discovered pages within a folder
---
```

**Don't repeat the title as a leading `# H1` in the body.** Fern already renders `title:` at the top of the page, and a duplicate creates a double heading. No build step removes duplicate headings, so keep the body H1-free.

## Internal Links

Use **version-agnostic** paths — no `/latest/`, `/v0.7/`, `/v0.5/`, `/v0.4/`, or `/nightly/` prefix:

```mdx
[Install NeMo AutoModel](/get-started/installation)
[LLM model list](/model-coverage/large-language-models/overview)
```

Version-agnostic links keep readers in their current documentation version; a hard-coded prefix can send them to another version. URL slugs come from explicit `slug:` overrides in the version YAML (set during the migration so URLs stay short while sidebar titles match the verbose published H1s) — so `Install NeMo AutoModel` is at `/get-started/installation`, not `/get-started/install-nemo-automodel`.

For cross-repo references (yaml configs, Python source), use absolute GitHub URLs:

```mdx
[mistral4_medpix.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/mistral4/mistral4_medpix.yaml)
```

## Validate

The only docs-check entry point for contributors, agents, and CI is
`tools/docs/check_docs.py`. Run it from the repository root before committing or
opening a PR:

```bash
uv run --script tools/docs/check_docs.py
```

This validates documentation sources before generation, then runs the MDX and
Fern checks with the pinned Fern CLI. Source validation includes the dataset
catalog, its cards, and the Hub dataset IDs referenced by example recipes. The
script installs only the isolated docs dependencies (PyYAML), requires Node.js 20
and `make`, and cleans generated tables and styles after validation, including on
failure.

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

The shared check must pass before commit. It runs a no-secret MDX parser before `fern check`, so raw HTML must be valid JSX (for example, use `<img ... />`, not `<img ...>`).

Inspect link warnings against the version navigation and preview before treating them as broken routes. Version-agnostic links should preserve the current version; verify that the destination exists in that version's navigation.

To regenerate the autodoc library reference (gitignored under `docs/fern/product-docs/`) and start a local preview:

```bash
make -C docs/fern docs
```

This target runs `fern docs md generate`, which populates `docs/fern/product-docs/` from the `nemo_automodel` package source declared in the `docs.yml` `libraries:` block, then starts `fern docs dev`. Without generation, a cold `fern docs dev` can fail with `Folder not found: ./product-docs/...`.

## Preview and Publish

| Goal | Command From the Repository Root |
|---|---|
| Local preview at `http://localhost:3002` | `make -C docs/fern docs` |
| Validation only (no server) | `uv run --script tools/docs/check_docs.py` |
| Shared preview URL on `*.docs.buildwithfern.com` (needs `DOCS_FERN_TOKEN`) | `make -C docs/fern docs-preview` |
| Trigger production publish workflow on `origin/main` | `make -C docs/fern docs-publish` |

Approved upstream PR mirrors that change docs or preview inputs receive an automatic preview URL comment from `fern-docs-preview.yml`. Fork-origin PRs must first pass the mirror approval process; direct fork pushes do not publish previews.

| Workflow | Trigger | Behavior |
|---|---|---|
| `fern-docs-ci.yml` | Push to `pull-request/[0-9]+` | Run `tools/docs/check_docs.py` (source validation, including dataset cards; model tables; archives; MDX syntax; `fern check`), then the documentation-tool regression tests |
| `fern-docs-preview.yml` | Push to an approved `pull-request/[0-9]+` mirror with docs or preview changes | Restore archives, stage PR pages and validated PR nightly navigation with trusted configuration and tooling from `main`, publish the preview, and update its comment |
| `publish-fern-docs.yml` | Push to `main` affecting docs inputs, `docs/v*` tag, or manual dispatch | Run `tools/docs/check_docs.py --keep-generated`, then publish to `docs.nvidia.com/nemo/automodel` |

Check and publish workflows use the shared script's Makefile target to restore
archived pages; previews restore archives in their own trusted build steps.
Frozen backward-version trees live on `docs-archive`, not on `main`.

The preview and production publication steps use the `DOCS_FERN_TOKEN` organization secret. The preview validates the PR's nightly navigation and uses it in place of the trusted navigation, so new pages, including dataset cards, appear in the preview sidebar. Navigation entries are limited to pages, sections, folders already present in trusted navigation, and trusted or embedded WebP provider icons; page paths must reference existing Markdown or MDX files in the documentation content tree.

## Cut a New Version Train

When NeMo AutoModel ships the next GA (for example, `v0.8`), run release commands from the repository root:

1. Create a frozen snapshot of nightly: `mkdir -p docs/fern/versions/v0.8/pages && rsync -a --exclude='fern' docs/ docs/fern/versions/v0.8/pages/`.
2. Copy the navigation with `cp docs/fern/versions/nightly.yml docs/fern/versions/v0.8.yml`, then rewrite `../../` path prefixes to `./v0.8/pages/` in the new file.
3. Repoint the GA alias: `cp docs/fern/versions/v0.8.yml docs/fern/versions/latest.yml`.
4. Add a frozen entry to `docs/fern/docs.yml` `versions:` (`display-name: "0.8.0"`, `slug: v0.8`, `availability: stable`). Keep the previous v0.4, v0.5, and v0.7 entries for permalink stability; keep nightly at `availability: beta`.
5. Add redirects before the global catch-alls. Map `/nemo/automodel/v0.8/index.html` and `/nemo/automodel/v0.8/index` to `/nemo/automodel/v0.8`. Map legacy `/nemo/automodel/0.8`, `/nemo/automodel/0.8/index.html`, and `/nemo/automodel/0.8/index` to `/nemo/automodel/v0.8`; map `/nemo/automodel/0.8/:path*/index.html`, `/nemo/automodel/0.8/:path*.html`, and `/nemo/automodel/0.8/:path*` to `/nemo/automodel/v0.8/:path*`.
6. Commit `docs/fern/versions/v0.8/pages/` on the `docs-archive` branch and push it. Remove the temporary pages subtree from the `main` checkout and add it to `.gitignore`; keep `v0.8.yml`, `latest.yml`, and `docs.yml` on `main`.
7. Add `v0.8` to the `docs-stitch` target in `docs/fern/Makefile`, which the shared check uses in CI, publication, and local builds, and to the `Restore trusted archived pages` loop in `fern-docs-preview.yml`.
8. Keep `docs/` moving forward as nightly. The archived release snapshots change only through deliberate back-ports on `docs-archive`.
9. Once the release configuration and archived pages are available, tag and push `docs/v0.8.0` to publish.

## Commits and DCO

Every commit needs a `Signed-off-by:` trailer:

```bash
git commit -s -m "docs: add fine-tuning guide for Qwen3.6"
```

If sign-off is missing on a recent commit, amend with `git commit --amend -s`. PR titles follow Conventional Commits: `docs(fern): <short summary>`. See [`AGENTS.md`](../../../AGENTS.md) for the full repo commit convention.

## Debugging

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to `nightly.yml`'s location (so nightly entries reach back up via `../../`); `slug:` must not collide with siblings |
| Page 404 in preview | Missing `slug:` override (default slugifies the long display title) or `position:` collision in an auto-discovered folder |
| `Folder not found: ./product-docs/...` on `fern docs dev` | Run `make -C docs/fern docs` once to populate the library reference |
| `[ERR_PNPM_IGNORED_BUILDS]` on first `fern docs dev` | pnpm 10+ blocks esbuild's postinstall — `pnpm config set onlyBuiltDependencies '["esbuild"]' --location global`, then `rm -rf ~/.fern/app-preview` and retry |
| Broken-link warning on version-agnostic path | Check the destination against the current version navigation and preview |
| `JSX expressions must have one parent element` | Wrap multi-element JSX in `<>...</>` or a `<div>` |
| Old Sphinx URL breaks | Add a `redirects:` entry in `docs/fern/docs.yml`; check both `/index.html` and `.html` legacy forms |
| Image not rendering | Use relative path (`./image.png`) for page-scoped images, not root-relative (`/image.png`) |
| Sidebar caption looks shortened vs published site | Compare original migrated captions against `docs.nvidia.com/nemo/automodel/v0.4` and restore the verbatim title in `docs/fern/versions/nightly.yml` |
| `path: ../../foo.mdx` doesn't resolve | Confirm the MDX file is at `docs/foo.mdx` (top level), not still under `docs/fern/versions/nightly/pages/` — that legacy tree no longer exists |
| `fern check` fails on missing archived page paths | Run `make -C docs/fern docs-stitch`, or run `uv run --script tools/docs/check_docs.py` from the repository root, which restores the frozen trees from `docs-archive` before validation |
| `archive ref '...' does not contain '...'` in CI | The `stitch-fern-versions` action couldn't find the version's pages on its registry ref. Confirm the `docs-archive` branch (or the configured tag) still holds `docs/fern/versions/<vdir>/pages` |

## Key References

| File | Purpose |
|---|---|
| `docs/fern/docs.yml` | Site config — `instances`, `versions`, `redirects`, `libraries`, theme |
| `docs/fern/versions/nightly.yml` | Canonical nav tree — paths reach up into `docs/` via `../../` |
| `docs/fern/versions/{latest,v0.4,v0.5,v0.7}.yml` | Frozen GA navigation; latest mounts `./v0.7/pages/...` |
| `docs/` (top-level *.mdx) | Nightly MDX content and page-scoped images |
| `docs/dataset-coverage/catalog.json` | Dataset-card registry; authoring rules live in `docs/templates/dataset-card-guide.md` |
| `docs/fern/versions/{v0.4,v0.5,v0.7}/pages/` | Frozen release snapshots — **on the `docs-archive` branch, not `main`**; stitched in at build time |
| `docs-archive` branch | Holds all frozen backward-version `pages/` trees; restored by `stitch-fern-versions` / `make -C docs/fern docs-stitch` |
| `.github/actions/stitch-fern-versions/` | Composite action that restores archives for archived preview builds; the shared check uses `make -C docs/fern docs-stitch` |
| `docs/fern/components/` | `BadgeLinks.tsx`, `Tag.tsx` (repo-specific; NVIDIA footer ships via `global-theme: nvidia`) |
| `docs/fern/README.md` | Human-facing orientation |
| `tools/docs/check_docs.py` | Shared docs check for contributors, agents, and CI; tool layout in `tools/docs/README.md` |
| `docs/fern/Makefile` | Local validation, preview, archive restore, and publication targets; invoke from the repository root with `make -C docs/fern <target>` |
| `.github/workflows/fern-docs-*.yml` | CI validation and approved-mirror preview publication |
| `.github/workflows/publish-fern-docs.yml` | CI: publish to docs.nvidia.com/nemo/automodel |
