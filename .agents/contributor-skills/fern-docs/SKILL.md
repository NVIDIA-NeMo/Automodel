---
name: fern-docs
description: Maintain the NeMo AutoModel Fern docs site under docs/ (MDX content) + docs/fern/ (infra) — add, update, move, or remove pages; manage redirects, slugs, navigation, and version aliases; run validation and previews.
when_to_use: Editing or adding documentation pages, fixing broken links, renaming a slug, updating the sidebar, adding a redirect, regenerating the API library reference, debugging fern check / broken-link errors, cutting a new version train, 'edit docs', 'add doc page', 'fern check failing', 'preview fails locally'.
---

# Fern Docs Maintenance — NeMo AutoModel

Unified skill for adding, updating, moving, and removing pages on the NeMo AutoModel Fern documentation site at `docs.nvidia.com/nemo/automodel`.

## Scope Rule

**Nightly MDX content lives at the top level of `docs/`** (e.g. `docs/index.mdx`, `docs/guides/llm/finetune.mdx`). **`docs/fern/` holds only Fern build infrastructure** — config, theme, and components. New pages, release notes, migration guides → add as a top-level `.mdx` under `docs/`.

**Only the nightly tree is kept on `main`.** Frozen backward-version snapshots live on the `docs-archive` branch and are restored at build time — see *Archived Backward Versions* below.

**Three content trees, plus a GA alias YAML.**

- `docs/` — bleeding-edge (nightly) tree. Every PR lands here. Mounted at the `nightly` URL slug via `docs/fern/versions/nightly.yml` (paths reach back up via `../../<rel>.mdx`).
- `docs/fern/versions/v0.4/pages/` — frozen 0.4.0 GA snapshot. **Not on `main`**: it lives on the `docs-archive` branch and is restored under this path by `make -C docs/fern docs-stitch` (local) or the `stitch-fern-versions` CI action (build). Mounted at the `v0.4` URL slug via `v0.4.yml`. Only changes via deliberate back-port (on `docs-archive`).
- `docs/fern/versions/v0.5/pages/` — frozen 0.5.0 GA snapshot, also restored from `docs-archive`. Mounted at the `v0.5` URL slug through `v0.5.yml`.
- `docs/fern/versions/latest.yml` — GA alias. Its `path:` lines mount the current GA's content at `./v0.5/pages/...`. Repointed at the next GA's tree when one is cut.

Nightly continues to change after each release; the v0.4 and v0.5 snapshots stay frozen. **Default editing target is `docs/` top-level.** Back-ports to a frozen version happen on the `docs-archive` branch, not here — call out the divergence in the PR description.

### Archived Backward Versions

Fern has no native way to source a version train's prose from another git ref: `fern generate --docs` reads the single local working tree, and publish is a full-site snapshot (a train missing from the tree is *unpublished*). So frozen GA pages are kept off `main` on the **`docs-archive` branch** and restored before every Fern build.

- **The registry** lives in `publish-fern-docs.yml` and `fern-docs-ci.yml` as `archived-versions: |` lines: `v0.4=docs-archive` and `v0.5=docs-archive`. Each value names the git ref that contains that version's pages. `latest` aliases an existing pages tree and needs no entry.
- **The mechanism** is the `.github/actions/stitch-fern-versions` composite action, which fetches each ref and restores `docs/fern/versions/<vdir>/pages`. The preview workflow, `fern-docs-preview.yml`, separately checks out `docs-archive` and restores both trees in its `Restore trusted archived pages` step. Configuration and navigation come from the live checkout for validation and publication; previews use trusted configuration from `main`.
- **Locally**, `make -C docs/fern docs`, `docs-check`, and `docs-preview` depend on `docs-stitch`, which restores both frozen trees. Set `ARCHIVE_REF=<ref>` to use another ref; that ref must contain both `v0.4/pages/` and `v0.5/pages/`.
- Both restored paths are gitignored on `main`, so a local stitch does not show the pages as untracked.

**Sidebar fidelity rule.** Section captions, page titles, and Model Coverage child ordering must match the **published v0.4.0 sidebar at docs.nvidia.com/nemo/automodel/v0.4** verbatim. Don't silently shorten a title or reorder siblings — the docs PM and content engineers diff against the published site and any drift is treated as a regression. If you want a shorter sidebar label, change the toctree-derived display name in the source — never just retitle in the MDX.

## Layout at a Glance

```text
docs/                                ← nightly MDX (top level)
├── index.mdx, breaking-changes.mdx, release-notes.mdx, ...
├── about/, guides/, model-coverage/, dataset-cards/, launcher/, api-reference/
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
    │   └── latest.yml               # GA alias — paths → ./v0.5/pages/; repointed at next GA cut
    └── product-docs/                # GENERATED Python API reference (gitignored)
```

```text
File                                                     URL (after /nemo/automodel)
docs/guides/installation.mdx                              /nightly/get-started/installation
docs/fern/versions/v0.5/pages/guides/installation.mdx       /latest/get-started/installation
                                                         /v0.5/get-started/installation
docs/fern/versions/v0.4/pages/guides/installation.mdx       /v0.4/get-started/installation
```

## Prerequisites

Install Git, Make, Node.js with npm, Python 3.10 or later, and `uv`. Use the Fern CLI version pinned in `docs/fern/fern.config.json` (currently 5.139.0). Dataset-card validation and model-table generation use `uv run --no-project --with 'PyYAML==6.0.3'` to provision their docs dependency separately from the training environment.

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

4. Run `make -C docs/fern docs-check` to validate dataset cards, regenerate model tables, restore archived pages, validate MDX syntax, and run `fern check`. Verify the URL in the `make -C docs/fern docs` preview. `latest.yml` mounts the frozen v0.5 tree and is not synchronized with nightly.

### Update a Page

1. Locate by path, title, or keyword: `rg -n "<keyword>" docs/ -g "*.mdx" -g "!docs/fern/**"`.
2. **Content only** — edit the single MDX file at `docs/<...>.mdx`.
3. **Title change** — update the frontmatter `title:` and update the `- page:` entry's display label in `docs/fern/versions/nightly.yml`.
4. **Section move** — `git mv` the file within `docs/`, update `path:` in `nightly.yml`, fix incoming links.
5. **Slug change** — change `slug:` in the YAML (or rename the file and let the default slug update). Add a `redirects:` entry in `docs/fern/docs.yml` so the old URL keeps working.

### Redirect Quirks

Four things to watch when editing `redirects:` in `docs/fern/docs.yml`:

1. **`:path*` does NOT match the empty-path case.** `/<basepath>/v0.4/:path*/index.html` will *not* match `/<basepath>/v0.4/index.html` (where `:path*` would have to be empty). Each version-root `index.html` needs its own explicit rule. NeMo Curator (NVIDIA-NeMo/Curator#1938) discovered this when their version-root URLs 404'd. AutoModel ships explicit rules for `latest`, `v0.5`, `v0.4`, `nightly`, and the legacy `0.5` and `0.4` forms — when you add a new version slug, add four new explicit rules: `<slug>/index.html`, `<slug>/index`, plus the same two for any legacy form (e.g. `0.5` → `v0.5`).
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

3. Run `make -C docs/fern docs-check`, then `make -C docs/fern docs`, and open the new page through the nightly sidebar.

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

Use **version-agnostic** paths — no `/latest/`, `/v0.5/`, `/v0.4/`, or `/nightly/` prefix:

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

```bash
make -C docs/fern docs-check          # Dataset cards, model tables, archived pages, MDX syntax, Fern config
make -C docs/fern docs-dataset-cards  # Focused dataset-card tests and example coverage validation
```

`docs-check` must pass before commit. It runs the dataset-card unit tests and validator, regenerates model-coverage tables, restores both archived page trees, validates MDX syntax, and runs `fern check`. Raw HTML in MDX must be valid JSX (for example, use `<img ... />`, not `<img ...>`).

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
| Validation only (no server) | `make -C docs/fern docs-check` |
| Dataset-card tests and validation | `make -C docs/fern docs-dataset-cards` |
| Shared preview URL on `*.docs.buildwithfern.com` (needs `DOCS_FERN_TOKEN`) | `make -C docs/fern docs-preview` |
| Trigger production publish workflow on `origin/main` | `make -C docs/fern docs-publish` |

Approved upstream PR mirrors that change docs or preview inputs receive an automatic preview URL comment from `fern-docs-preview.yml`. Fork-origin PRs must first pass the mirror approval process; direct fork pushes do not publish previews.

| Workflow | Trigger | Behavior |
|---|---|---|
| `fern-docs-ci.yml` | Push to `pull-request/[0-9]+` | Restore archives, validate dataset cards and MDX syntax, run `fern check` |
| `fern-docs-preview.yml` | Push to an approved `pull-request/[0-9]+` mirror with docs or preview changes | Restore archives, stage PR pages with trusted configuration and tooling from `main`, publish the preview, and update its comment |
| `publish-fern-docs.yml` | Push to `main` affecting docs inputs, `docs/v*` tag, or manual dispatch | Restore archives and publish to `docs.nvidia.com/nemo/automodel` |

The preview and production publication steps use the `DOCS_FERN_TOKEN` organization secret. Preview navigation comes from `main`, so a PR-only navigation change may not appear in the automatic preview. Use the local preview to inspect navigation changes.

## Cut a New Version Train

When NeMo AutoModel ships the next GA (for example, `v0.6`), run release commands from the repository root:

1. Create a frozen snapshot of nightly: `mkdir -p docs/fern/versions/v0.6/pages && rsync -a --exclude='fern' docs/ docs/fern/versions/v0.6/pages/`.
2. Copy the navigation with `cp docs/fern/versions/nightly.yml docs/fern/versions/v0.6.yml`, then rewrite `../../` path prefixes to `./v0.6/pages/` in the new file.
3. Repoint the GA alias: `cp docs/fern/versions/v0.6.yml docs/fern/versions/latest.yml`.
4. Add a frozen entry to `docs/fern/docs.yml` `versions:` (`display-name: "0.6.0"`, `slug: v0.6`, `availability: stable`). Keep the previous v0.4 and v0.5 entries for permalink stability; keep nightly at `availability: beta`.
5. Add redirects before the global catch-alls. Map `/nemo/automodel/v0.6/index.html` and `/nemo/automodel/v0.6/index` to `/nemo/automodel/v0.6`. Map legacy `/nemo/automodel/0.6`, `/nemo/automodel/0.6/index.html`, and `/nemo/automodel/0.6/index` to `/nemo/automodel/v0.6`; map `/nemo/automodel/0.6/:path*/index.html`, `/nemo/automodel/0.6/:path*.html`, and `/nemo/automodel/0.6/:path*` to `/nemo/automodel/v0.6/:path*`.
6. Commit `docs/fern/versions/v0.6/pages/` on the `docs-archive` branch and push it. Remove the temporary pages subtree from the `main` checkout and add it to `.gitignore`; keep `v0.6.yml`, `latest.yml`, and `docs.yml` on `main`.
7. Add `v0.6=docs-archive` to `archived-versions:` in `publish-fern-docs.yml` and `fern-docs-ci.yml`. Add `v0.6` to the `Restore trusted archived pages` loop in `fern-docs-preview.yml` and update `docs-stitch` in `docs/fern/Makefile` to restore the new tree locally.
8. Keep `docs/` moving forward as nightly. The archived release snapshots change only through deliberate back-ports on `docs-archive`.
9. Once the release configuration and archived pages are available, tag and push `docs/v0.6.0` to publish.

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
| `fern check` fails on missing `./v0.4/pages/...` or `./v0.5/pages/...` paths | Run `make -C docs/fern docs-stitch` (or `make -C docs/fern docs-check`, which depends on it) to restore both frozen trees from `docs-archive` |
| `archive ref '...' does not contain '...'` in CI | The `stitch-fern-versions` action couldn't find the version's pages on its registry ref. Confirm the `docs-archive` branch (or the configured tag) still holds `docs/fern/versions/<vdir>/pages` |

## Key References

| File | Purpose |
|---|---|
| `docs/fern/docs.yml` | Site config — `instances`, `versions`, `redirects`, `libraries`, theme |
| `docs/fern/versions/nightly.yml` | Canonical nav tree — paths reach up into `docs/` via `../../` |
| `docs/fern/versions/{latest,v0.4,v0.5}.yml` | Frozen GA navigation; latest mounts `./v0.5/pages/...` |
| `docs/` (top-level *.mdx) | Nightly MDX content and page-scoped images |
| `docs/fern/versions/{v0.4,v0.5}/pages/` | Frozen release snapshots — **on the `docs-archive` branch, not `main`**; stitched in at build time |
| `docs-archive` branch | Holds all frozen backward-version `pages/` trees; restored by `stitch-fern-versions` / `make -C docs/fern docs-stitch` |
| `.github/actions/stitch-fern-versions/` | Composite action that restores archives for CI validation and production publication |
| `docs/fern/components/` | `BadgeLinks.tsx`, `Tag.tsx` (repo-specific; NVIDIA footer ships via `global-theme: nvidia`) |
| `docs/fern/README.md` | Human-facing orientation |
| `docs/fern/Makefile` | Local validation, preview, archive restore, and publication targets; invoke from the repository root with `make -C docs/fern <target>` |
| `.github/workflows/fern-docs-*.yml` | CI validation and approved-mirror preview publication |
| `.github/workflows/publish-fern-docs.yml` | CI: publish to docs.nvidia.com/nemo/automodel |
