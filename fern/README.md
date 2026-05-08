# NeMo AutoModel — Fern Docs

This directory holds the Fern MDX source for the NeMo AutoModel documentation site at **[docs.nvidia.com/nemo/automodel](https://docs.nvidia.com/nemo/automodel)**.

The legacy Sphinx tree under `../docs/` remains in place for reference until the Fern site is fully validated; new pages and edits should land here.

## Quick links

| What | Where |
|---|---|
| Published site | https://docs.nvidia.com/nemo/automodel |
| Latest preview (`/latest/`) | https://nemo-automodel.docs.buildwithfern.com/nemo/automodel/latest |
| Fern dashboard | https://app.buildwithfern.com (NVIDIA org) |
| Skill for agents | [`../skills/fern-docs/SKILL.md`](../skills/fern-docs/SKILL.md) |
| CI workflows | [`../.github/workflows/fern-docs-*.yml`](../.github/workflows/) |
| Make targets | [`../Makefile`](../Makefile) |

## Layout

```
fern/
├── fern.config.json          # Fern CLI pin (4.62.4+) and org slug
├── docs.yml                  # Site config: instances, versions, redirects, libraries, theme
├── main.css                  # NVIDIA-green theme overrides
├── assets/                   # Logos and shared SVGs
├── components/               # BadgeLinks.tsx, Tag.tsx, CustomFooter.tsx
├── versions/
│   ├── nightly.yml           # CANONICAL nav — bleeding-edge tree, edited on every PR
│   ├── latest.yml            # GA alias — currently mirrors nightly; will pin to v0.4.yml after a real v0.4.0 snapshot is migrated
│   ├── v0.4.yml              # Frozen GA pin (currently mirrors nightly until a real v0.4.0 snapshot is migrated from the tag)
│   └── nightly/pages/        # MDX content tree (130+ pages) — single source of truth
└── product-docs/             # GENERATED Python API reference (gitignored — `make docs` regenerates)
```

```
File path                                                    Published URL
───────────────────────────────────────────────────────────  ─────────────────────────────────────────────────
fern/versions/nightly/pages/get-started/installation.mdx     docs.nvidia.com/nemo/automodel/nightly/get-started/installation
                                                             docs.nvidia.com/nemo/automodel/latest/get-started/installation
                                                             docs.nvidia.com/nemo/automodel/v0.4/get-started/installation
```

The same MDX file backs all three version slugs: `nightly` is the canonical, continuously-edited tree; `latest.yml` and `v0.4.yml` are copies that also point at `./nightly/pages/...`. They'll diverge once we cut a frozen v0.4.0 snapshot from the tag.

## Local development

From the repo root:

```bash
make docs           # `fern docs md generate` + `fern docs dev` → http://localhost:3002
make docs-check     # `fern check` (config + MDX validation)
make docs-preview   # shared preview URL on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)
make docs-publish   # trigger the `Publish Fern Docs` workflow on origin/main
```

First-time setup — sign the Fern CLI in to the NVIDIA org once via Google SSO:

```bash
fern login   # opens browser → @nvidia.com Google account
```

`fern docs md generate` (run by `make docs`) populates `fern/product-docs/` from the `nemo_automodel` package source declared in the `libraries:` block of `docs.yml`. Without it, a cold `fern docs dev` will fail with `Folder not found: ./product-docs/...`. Re-run only when the upstream Python source changes — for prose-only iteration, `cd fern && fern docs dev` alone is enough.

## Sidebar fidelity rule

**The published v0.4.0 sidebar at docs.nvidia.com/nemo/automodel/latest is the source of truth for section captions, page titles, and Model Coverage child ordering.** Don't silently shorten "Install NeMo AutoModel" to "Installation" or rename a section caption — engineers and the docs PM diff this site against the published one and any drift looks like a content regression.

If you want a shorter or different sidebar label, change the toctree-derived display name in the source — never just retitle in the converted MDX.

## Authoring conventions

### Frontmatter

```yaml
---
title: "<Page Title>"        # required — used by Fern as the page title and breadcrumb
description: ""              # required (may be empty string) — SEO
position: 1                  # optional — orders auto-discovered folders
---
```

The MDX body should generally **not** repeat the title as a leading `# H1` — Fern renders the frontmatter title at the top of the page automatically, and a duplicate H1 doubles up the heading visually. The post-stage `remove_duplicate_h1.py` strips them when the title and H1 match exactly.

### Components

Use the bundled custom components in `components/`:

| Component | Purpose | Import |
|---|---|---|
| `<BadgeLinks ... />` | Header badge rows on landing pages (PyPI, license, GitHub, …) | `import { BadgeLinks } from "@/components/BadgeLinks";` |
| `<Tag variant="...">label</Tag>` | Card chips ("start here", "5 min", etc.) | `import { Tag } from "@/components/Tag";` |
| `<CustomFooter />` | Wired in `docs.yml` `footer:`; **required** for NVIDIA legal/privacy compliance | (auto) |

Standard Fern components are also available — `<Note>`, `<Tip>`, `<Info>`, `<Warning>`, `<Cards>` / `<Card>`, etc. Don't use GitHub `> [!NOTE]` syntax — it does not render in MDX.

### Internal links

Use **version-agnostic paths** (no `/latest/`, `/v0.4/`, or `/nightly/` prefix):

```mdx
[Install NeMo AutoModel](/get-started/installation)
[LLM model list](/model-coverage/large-language-models/overview)
```

The same MDX backs every version slug — a hard-coded prefix would jump readers across versions. Page slugs come from explicit `slug:` overrides in the version YAML, not from the (often verbose) display title — so `Install NeMo AutoModel` is at `/get-started/installation`, not `/get-started/install-nemo-automodel`.

### Cross-repo references (yaml configs, source files)

Repository source paths like `examples/llm_finetune/foo.yaml` or `nemo_automodel/components/...` are not part of the docs site. Link to them as **absolute GitHub URLs**:

```mdx
[foo.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/foo.yaml)
```

## Versioning

`docs.yml` `versions:` lists three entries (matching the published version dropdown):

| display-name | slug | availability | path |
|---|---|---|---|
| `nightly` | `nightly` | `beta` | `./versions/nightly.yml` |
| `Latest` | `latest` | `stable` | `./versions/latest.yml` |
| `0.4.0 · 26.04` | `v0.4` | `stable` | `./versions/v0.4.yml` |

**`nightly` is the canonical content tree** — it's where all PRs land and where the daily build (once wired up) will publish from. `latest.yml` and `v0.4.yml` are content copies whose `path:` lines also point at `./nightly/pages/...` until a real GA snapshot is cut.

When the next GA cuts (e.g. `v0.5`):

1. `cp -r versions/nightly versions/v0.5` — frozen snapshot of the bleeding-edge tree
2. `cp versions/nightly.yml versions/v0.5.yml`, then sed `./nightly/` → `./v0.5/` in the new file
3. Update `versions/latest.yml` to point at `./v0.5/pages/...` (`cp` it from the new `v0.5.yml` is fine)
4. Add the new frozen-pin entry to `docs.yml` `versions:` (`display-name: "0.5.0"`, `slug: v0.5`, `availability: stable`); demote the old `v0.4` per support policy
5. `versions/nightly/pages/` keeps moving forward as the bleeding-edge tree

## CI and publishing

| Workflow | Trigger | Purpose |
|---|---|---|
| `fern-docs-ci.yml` | `push: pull-request/[0-9]+` (FW-CI mirror) | `fern check` on PRs |
| `fern-docs-preview-build.yml` | `pull_request` | Untrusted half: collect `fern/` artifact (no secrets) |
| `fern-docs-preview-comment.yml` | `workflow_run` after build | Trusted half: build preview with `DOCS_FERN_TOKEN`, post 🌿 comment |
| `publish-fern-docs.yml` | push to `main` (`fern/**`), `docs/v*` tag, or manual | Publish to docs.nvidia.com/nemo/automodel |

Required org secret: **`DOCS_FERN_TOKEN`** (already wired for the existing `build-docs.yml`).

PRs that touch `fern/**` get an automatic preview URL posted as a 🌿 comment.

## Commits

DCO sign-off is required:

```bash
git commit -s -m "docs: <add|update|remove> <page-title>"
```

PR titles follow Conventional Commits (e.g. `docs(fern): add gemma4 fine-tuning guide`) — see [`AGENTS.md`](../AGENTS.md) for the full convention.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML |
| Page 404 in preview | `slug:` collision in the same section, or missing `slug:` override (default slugifies the long display title) |
| `Folder not found: ./product-docs/...` in `fern docs dev` | Run `make docs` once; library generation populates `product-docs/` |
| `[ERR_PNPM_IGNORED_BUILDS]` on first `fern docs dev` | pnpm 10+ blocks esbuild's postinstall — `pnpm config set onlyBuiltDependencies '["esbuild"]' --location global`, then `rm -rf ~/.fern/app-preview` and retry |
| Broken-link warning for version-agnostic path | `fern docs broken-links` false-positives on links without a version slug; the URLMap-based `validate_fern_internal_links.py` is authoritative |
| `JSX expressions must have one parent element` | Wrap multi-element JSX in `<>...</>` or a `<div>` |
| Card badges have no spacing | Use `<Tag>` (NeMo AutoModel landing pattern), not raw HTML; spacing is in `main.css` |
| Old Sphinx URL breaks | Add a `redirects:` entry in `docs.yml` |
| `<basepath>/<version>/index.html` 404s but deep paths work | `:path*` does not match the empty-path case ([NVIDIA-NeMo/Curator#1938](https://github.com/NVIDIA-NeMo/Curator/pull/1938)). Each version-root `index.html` needs its own explicit redirect rule — slot before the `:path*/index.html` catch-all |

## Reference

- [Fern docs (upstream)](https://buildwithfern.com/docs)
- [convert-to-fern toolkit](https://gitlab-master.nvidia.com/fern/documentation-scripts) — the migration pipeline used to scaffold this site
- [NeMo Gym Fern docs](https://github.com/NVIDIA-NeMo/Gym/tree/main/fern) — sister site with the same theme + CI pattern
