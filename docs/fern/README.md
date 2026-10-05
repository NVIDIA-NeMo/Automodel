# NeMo AutoModel — Fern Docs

This directory holds the Fern build infrastructure (config, repo-specific components, frozen version snapshots) for the NeMo AutoModel documentation site at **[docs.nvidia.com/nemo/automodel](https://docs.nvidia.com/nemo/automodel)**.

**The MDX content lives one level up, in `docs/` itself** — every nightly page is a top-level sibling of this `docs/fern/` directory (e.g. `docs/index.mdx`, `docs/guides/llm/finetune.mdx`). Fern reads those files via relative `path: ../../<...>.mdx` entries in `docs/fern/versions/nightly.yml`.

NVIDIA branding (logos, favicon, footer, fonts, NVIDIA-green CSS, OneTrust JS) comes from the central control repo at **[NVIDIA/fern-components](https://github.com/NVIDIA/fern-components)** via `global-theme: nvidia` in `docs.yml` — no logos or theme CSS are vendored locally.

## Quick Links

| What | Where |
|---|---|
| Published site | https://docs.nvidia.com/nemo/automodel |
| Fern dashboard | https://dashboard.buildwithfern.com (NVIDIA org) |
| Skill for agents | [`../../.agents/contributor-skills/fern-docs/SKILL.md`](../../.agents/contributor-skills/fern-docs/SKILL.md) |
| CI workflows | [`../../.github/workflows/fern-docs-*.yml`](../../.github/workflows/) |
| Make targets | [`./Makefile`](./Makefile) |

## Quickstart

Install Git, Make, Node.js with npm, Python 3.10 or later, and `uv` before running the targets below. The Makefile uses `uv run --no-project --with 'PyYAML==6.0.3'` to provide the pinned docs dependency for dataset-card validation and model-table generation, separately from the training environment.

First time on this machine:

```bash
# All Make targets live in docs/fern/Makefile — run them from this directory
# (`cd docs/fern && make <target>`), or from the repository root with
# `make -C docs/fern <target>`.

# 1. Install the Fern CLI globally (one-time)
npm install -g fern-api
# or use it ad-hoc via:  npx -y fern-api@latest <subcommand>

# 2. Provision your Fern account + CLI auth (one-time per machine).
#    Walks you through the dashboard sign-in step before running `fern login`.
cd docs/fern && make docs-login

# 3. Build the API library reference and start the local dev server
make docs           # http://localhost:3002

# 4. Validate dataset cards, MDX syntax, and Fern configuration before committing
make docs-check
```

**`make docs-login` is load-bearing.** Skip it and `fern docs md generate` returns `HTTP 403: User does not belong to organization` — the CLI's `fern login` flow alone is *not* enough; Fern requires that you sign in to the dashboard first so your account record exists in Fern's user DB. See [NeMo Gym #1185](https://github.com/NVIDIA-NeMo/Gym/issues/1185) for the ugly version of that bug.

### Fern CLI and Docs Reference

| Resource | Link |
|---|---|
| Fern docs (overview, writing, configuration) | https://buildwithfern.com/learn/docs |
| Fern CLI reference | https://buildwithfern.com/learn/cli-api-reference/cli-reference/commands |
| MDX components (Cards, Callouts, Tabs, …) | https://buildwithfern.com/learn/docs/writing-content/components |
| Frontmatter fields | https://buildwithfern.com/learn/docs/configuration/page-level-settings |
| Versioning | https://buildwithfern.com/learn/docs/building-your-docs/versioning |
| Redirects | https://buildwithfern.com/learn/docs/configuration/site-level-settings#redirects-configuration |
| `libraries:` (Python autodoc) | https://buildwithfern.com/learn/docs/api-references/library-reference |
| Fern Slack (NVIDIA) | `#fern` |

## Layout

```text
docs/                            ← nightly MDX lives here (sibling of fern/)
├── index.mdx, breaking-changes.mdx, release-notes.mdx, ...
├── about/, guides/, model-coverage/, launcher/, api-reference/
├── *.png / *.jpg                ← page-scoped images
└── fern/                        ← THIS DIRECTORY
    ├── fern.config.json         # Fern CLI pin (5.139.0) and org slug
    ├── docs.yml                 # Site config: instances, versions, redirects, libraries, global-theme: nvidia
    ├── components/              # BadgeLinks.tsx, Tag.tsx (repo-specific only;
    │                            #   NVIDIA-branded footer/logo/CSS ship via global-theme)
    ├── versions/
    │   ├── nightly.yml          # Nav for nightly — paths point at ../../<path>.mdx (up into docs/)
    │   ├── v0.4.yml             # Nav for the frozen 0.4.0 GA snapshot — paths at ./v0.4/pages/
    │   ├── v0.4/pages/          # Frozen 0.4.0 content — lives on the `docs-archive` branch, NOT main;
    │   │                        #   restored here by `make docs-stitch` / CI (gitignored on main)
    │   ├── v0.5.yml             # Nav for the frozen 0.5.0 GA snapshot — paths at ./v0.5/pages/
    │   ├── v0.5/pages/          # Frozen 0.5.0 content — also restored from docs-archive
    │   └── latest.yml           # GA alias — paths at ./v0.5/pages/; repointed at the next GA cut
    └── product-docs/            # GENERATED Python API reference (gitignored — `make docs` regenerates)
```

```text
File path                                                  Published URL
─────────────────────────────────────────────────────────  ─────────────────────────────────────────────────
docs/guides/installation.mdx                               docs.nvidia.com/nemo/automodel/nightly/get-started/installation
docs/fern/versions/v0.5/pages/guides/installation.mdx       docs.nvidia.com/nemo/automodel/v0.5/get-started/installation
                                                           docs.nvidia.com/nemo/automodel/latest/get-started/installation  (latest mounts v0.5 content)
```

The **`docs/` top-level tree is the nightly tree** — every PR lands there. The **`docs/fern/versions/v0.4/pages/` and `v0.5/pages/` trees are frozen release snapshots**, changed only through deliberate back-ports. `latest.yml` mounts `./v0.5/pages/` so `/latest/...` URLs serve the current GA — at the next GA cut, `latest.yml` repoints at the new train.

**Only the nightly tree is on `main`.** The frozen `v0.4/pages/` and `v0.5/pages/` snapshots live on the **`docs-archive` branch** and are restored into the working copy before Fern builds. Locally, `make docs-stitch` restores them; CI restores them before validation, preview, and publication. Fern reads one local tree and publishes a full-site snapshot, so the archived pages must be physically present at build time; they can't be sourced from another branch natively. The restore paths are gitignored on `main`. To use another archive ref, set `ARCHIVE_REF` when running Make; that ref must contain both archived page trees.

## Local Development

Run these commands from this directory (`cd docs/fern` first), or use `make -C docs/fern <target>` from the repository root:

```bash
make docs           # docs-stitch + `fern docs md generate` + `fern docs dev` → http://localhost:3002
make docs-stitch    # restore frozen backward-version pages from the docs-archive branch
make docs-check     # dataset cards, model tables, archived pages, MDX syntax, and `fern check`
make docs-dataset-cards # validate Hub dataset cards and example coverage
make docs-model-tables  # regenerate model-coverage tables
make docs-preview   # docs-stitch + shared preview URL on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)
make docs-publish   # trigger the `Publish Fern Docs` workflow on origin/main
```

For first-time-on-this-machine setup, see the [Quickstart](#quickstart) above — `make docs-login` walks through dashboard provisioning + `fern login` together.

`fern docs md generate` (run by `make docs`) populates `docs/fern/product-docs/` from the `nemo_automodel` package source declared in the `libraries:` block of `docs.yml`. Without it, a cold `fern docs dev` will fail with `Folder not found: ./product-docs/...`. Re-run only when the upstream Python source changes — for prose-only iteration, `cd docs/fern && fern docs dev` alone is enough.

## Sidebar Fidelity Rule

**The published v0.4.0 sidebar at docs.nvidia.com/nemo/automodel/v0.4 is the reference for the original section captions, page titles, and Model Coverage child ordering.** Don't silently shorten "Install NeMo AutoModel" to "Installation" or rename a section caption — engineers and the docs PM diff this site against the published one and any drift looks like a content regression.

If you want a shorter or different sidebar label, change the toctree-derived display name in the source — never just retitle in the converted MDX.

## Authoring Conventions

### Frontmatter

```yaml
---
title: "<Page Title>"        # required — used by Fern as the page title and breadcrumb
description: "<Concise page summary>"  # describe the page for search results
position: 1                  # optional — orders auto-discovered folders
---
```

The MDX body should generally **not** repeat the title as a leading `# H1` — Fern renders the frontmatter title at the top of the page automatically, and a duplicate H1 doubles up the heading visually. No build step removes duplicate headings, so keep the body H1-free.

### Components

Use the bundled custom components in `components/`:

| Component | Purpose | Import |
|---|---|---|
| `<BadgeLinks ... />` | Header badge rows on landing pages (PyPI, license, GitHub, …) | `import { BadgeLinks } from "@/components/BadgeLinks";` |
| `<Tag variant="...">label</Tag>` | Card chips ("start here", "5 min", etc.) | `import { Tag } from "@/components/Tag";` |

The shared NVIDIA `<CustomFooter />` (privacy / Do Not Sell / etc.) ships from the `nvidia` global theme — wired automatically, **not** authored in this repo.

Standard Fern components are also available — `<Note>`, `<Tip>`, `<Info>`, `<Warning>`, `<Cards>` / `<Card>`, etc. Don't use GitHub `> [!NOTE]` syntax — it does not render in MDX.

### Internal Links

Use **version-agnostic paths** (no `/latest/`, `/v0.4/`, or `/nightly/` prefix):

```mdx
[Install NeMo AutoModel](/get-started/installation)
[LLM model list](/model-coverage/large-language-models/overview)
```

Version-agnostic links keep readers in their current documentation version; a hard-coded prefix can send them to another version. Page slugs come from explicit `slug:` overrides in the version YAML, not from the (often verbose) display title — so `Install NeMo AutoModel` is at `/get-started/installation`, not `/get-started/install-nemo-automodel`.

### Cross-Repo References (YAML Configs, Source Files)

Repository source paths like `examples/llm_finetune/foo.yaml` or `nemo_automodel/components/...` are not part of the docs site. Link to them as **absolute GitHub URLs**:

```mdx
[foo.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/foo.yaml)
```

## Versioning

`docs.yml` `versions:` lists four entries:

| display-name | slug | availability | path |
|---|---|---|---|
| `Nightly` | `nightly` | `beta` | `./versions/nightly.yml` |
| `Latest` | `latest` | `stable` | `./versions/latest.yml` |
| `0.5.0 · 26.06` | `v0.5` | `stable` | `./versions/v0.5.yml` |
| `0.4.0 · 26.04` | `v0.4` | `stable` | `./versions/v0.4.yml` |

**`nightly` reads the MDX directly from `docs/`** (through `path: ../../<...>.mdx` in `nightly.yml`). Changes to `docs/` on `main` trigger publication. **`v0.4` and `v0.5` are frozen GA snapshots** under their respective `docs/fern/versions/<version>/pages/` directories; they change only through deliberate back-ports. `latest.yml` mounts the current GA's content at `./v0.5/pages/...`.

When the next GA cuts (e.g. `v0.6`):

1. From the repository root, create an exclusion-aware snapshot of nightly: `mkdir -p docs/fern/versions/v0.6/pages && rsync -a --exclude='fern' docs/ docs/fern/versions/v0.6/pages/`.
2. Copy the nightly navigation with `cp docs/fern/versions/nightly.yml docs/fern/versions/v0.6.yml`, then replace each `../../` source prefix with `./v0.6/pages/` in the new file.
3. Repoint the GA alias with `cp docs/fern/versions/v0.6.yml docs/fern/versions/latest.yml`.
4. Add the new frozen-pin entry to `docs/fern/docs.yml` `versions:` (`display-name: "0.6.0"`, `slug: v0.6`, `availability: stable`); keep existing version entries per support policy.
5. Add redirects to `docs/fern/docs.yml` before the global catch-alls: map explicit `/nemo/automodel/v0.6/index.html` and `/nemo/automodel/v0.6/index` sources to `/nemo/automodel/v0.6`; map legacy `/nemo/automodel/0.6`, `/nemo/automodel/0.6/index.html`, and `/nemo/automodel/0.6/index` sources to `/nemo/automodel/v0.6`; and map `/nemo/automodel/0.6/:path*/index.html`, `/nemo/automodel/0.6/:path*.html`, and `/nemo/automodel/0.6/:path*` sources to `/nemo/automodel/v0.6/:path*`. Keep the root index rules explicit because `:path*` does not match an empty path.
6. Commit `docs/fern/versions/v0.6/pages/` on the `docs-archive` branch and push that branch. On `main`, remove the pages subtree and add `docs/fern/versions/v0.6/pages/` to `.gitignore`; keep `v0.6.yml`, `latest.yml`, and `docs.yml` on `main`.
7. Add `v0.6=docs-archive` to `archived-versions:` in `publish-fern-docs.yml` and `fern-docs-ci.yml`. Add the version to the `Restore trusted archived pages` loop in `fern-docs-preview.yml`. Update the `docs-stitch` target in `docs/fern/Makefile` to restore the new pages subtree for local builds as well.
8. Keep `docs/` moving forward as nightly. Existing archived page trees and the new `v0.6/pages/` tree are frozen and change only through deliberate back-ports on `docs-archive`.
9. After the release configuration and archived pages are available, tag and push `docs/v0.6.0` to publish the version train.

## CI and Publishing

| Workflow | Trigger | Purpose |
|---|---|---|
| `fern-docs-ci.yml` | `push: pull-request/[0-9]+` (FW-CI mirror) | Dataset-card validation, MDX syntax validation, and `fern check` on PRs |
| `fern-docs-preview.yml` | `push: pull-request/[0-9]+` (approved mirror) | Stage docs with trusted tooling, publish a preview and update the comment |
| `publish-fern-docs.yml` | push to `main` (`docs/**`), `docs/v*` tag, or manual | Publish to docs.nvidia.com/nemo/automodel |

Required org secret: **`DOCS_FERN_TOKEN`**, scoped to the fixed Fern CLI library-generation and preview-publication steps.

Approved upstream PR mirrors that change docs or preview inputs get a preview URL posted as a 🌿 comment. Fork-origin PRs must first pass the mirror approval process; direct fork pushes do not publish previews. The workflow verifies the current PR head before publishing its comment and removes stale preview comments when docs changes are reverted.

Preview configuration, components, tooling, and navigation outside the Data section come from `main`; PR page content overlays that trusted tree. The preview imports the PR's Data navigation after validating its fields and local page paths, so new dataset cards appear under **Data → Dataset Catalog**. Newer main-only pages remain available, while explicit PR page deletions are applied relative to the merge base. Deleting a page still referenced by trusted navigation outside Data can fail the build until that navigation is updated on `main`. Archived v0.4/v0.5 pages come from `docs-archive`.

The preview workflow pins Fern 5.139.0, sets `FERN_NO_VERSION_REDIRECTION=true`, and verifies the exact expected preview host before commenting. It does not execute PR package scripts or make authenticated page-link requests.

## Commits

DCO sign-off is required:

```bash
git commit -s -m "docs: <add|update|remove> <page-title>"
```

PR titles follow Conventional Commits (e.g., `docs(fern): add gemma4 fine-tuning guide`) — see [`AGENTS.md`](../../AGENTS.md) for the full convention.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML (so nightly paths reach back up via `../../`) |
| Page 404 in preview | `slug:` collision in the same section, or missing `slug:` override (default slugifies the long display title) |
| `Folder not found: ./product-docs/...` in `fern docs dev` | Run `make docs` once; library generation populates `product-docs/` |
| `Unexpected closing tag`, especially after raw HTML such as `<img>` | Use valid MDX/JSX syntax, for example self-close void elements as `<img ... />`; `make docs-check` catches this before publish |
| `[ERR_PNPM_IGNORED_BUILDS]` on first `fern docs dev` | pnpm 10+ blocks esbuild's postinstall — `pnpm config set onlyBuiltDependencies '["esbuild"]' --location global`, then `rm -rf ~/.fern/app-preview` and retry |
| Broken-link warning for version-agnostic path | Check the destination against the current version navigation and verify it in the preview before treating the warning as a broken route |
| `JSX expressions must have one parent element` | Wrap multi-element JSX in `<>...</>` or a `<div>` |
| Card badges have no spacing | Use `<Tag>` (NeMo AutoModel landing pattern), not raw HTML; spacing comes from the `nvidia` global theme's CSS |
| Old Sphinx URL breaks | Add a `redirects:` entry in `docs.yml` |
| `<basepath>/<version>/index.html` 404s but deep paths work | `:path*` does not match the empty-path case ([NVIDIA-NeMo/Curator#1938](https://github.com/NVIDIA-NeMo/Curator/pull/1938)). Each version-root `index.html` needs its own explicit redirect rule — slot before the `:path*/index.html` catch-all |

## Reference

- [Fern docs (upstream)](https://buildwithfern.com/docs)
- [convert-to-fern toolkit](https://gitlab-master.nvidia.com/fern/documentation-scripts) — the migration pipeline used to scaffold this site
- [NeMo Gym Fern docs](https://github.com/NVIDIA-NeMo/Gym/tree/main/fern) — sister site with the same theme + CI pattern
