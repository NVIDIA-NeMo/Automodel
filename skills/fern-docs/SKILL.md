---
name: fern-docs
description: Maintain the NeMo AutoModel Fern docs site with authored MDX under docs/ and generated Fern infrastructure from .github/fern-template/.
when_to_use: Editing or adding documentation pages, fixing broken links, renaming a slug, updating the sidebar, adding a redirect, regenerating the API library reference, debugging fern check / broken-link errors, 'edit docs', 'add doc page', 'fern check failing', 'preview fails locally'.
---

# Fern Docs Maintenance — NeMo AutoModel

NeMo AutoModel uses Fern for publishing, but the repository does **not** track a
root-level `fern/` project.

## Source of truth

- `docs/` — authored MDX pages and page-scoped images. This is where docs edits
  happen.
- `.github/fern-template/` — Fern site configuration, theme assets,
  components, and the single `latest` navigation YAML.
- `.github/scripts/prepare-fern-docs.sh` — builds a disposable Fern project by
  copying `docs/` into `versions/latest/pages/`.
- `.fern-build/fern/` — local generated Fern project, gitignored.

There is one maintained docs tree: `latest`. Do not add `nightly`, `v0.4`, or
other frozen content trees to the main branch. Legacy `/nightly`, `/v0.4`,
`/0.4`, `/0.3.0`, `/0.2.0`, and `/0.1.0` URLs redirect to `/latest` in
`.github/fern-template/docs.yml`.

## Add or update a page

1. Edit or create the MDX file under `docs/`.
2. Include Fern frontmatter:

   ```mdx
   ---
   title: "<Page Title>"
   description: ""
   ---
   ```

3. Do not repeat the title as a leading `# H1`; Fern renders the frontmatter
   title.
4. If the page is part of the sidebar, add or update its `path:` entry in
   `.github/fern-template/versions/latest.yml`.
5. Use paths relative to the generated version YAML:

   ```yaml
   - page: "Install NeMo AutoModel"
     path: ./latest/pages/guides/installation.mdx
     slug: installation
   ```

## Move or remove a page

1. Find references with `rg "<old-slug-or-file>" docs .github/fern-template`.
2. Move or remove the file under `docs/`.
3. Update `.github/fern-template/versions/latest.yml`.
4. Add redirects in `.github/fern-template/docs.yml` if a published URL changes.

## Links and images

Use version-agnostic internal links:

```mdx
[Install NeMo AutoModel](/get-started/installation)
[Llama coverage](/model-coverage/large-language-models/llama)
```

For repository source files such as examples or Python modules, use absolute
GitHub URLs.

Page-scoped images live beside the MDX file and should be referenced with
relative paths such as `./image.png`.

## Components

Use Fern MDX components instead of GitHub alert syntax:

| Purpose | Component |
|---|---|
| Neutral aside | `<Note>...</Note>` |
| Tip | `<Tip>...</Tip>` |
| Information | `<Info>...</Info>` |
| Warning | `<Warning>...</Warning>` |
| Error / danger | `<Error>...</Error>` |
| Card grid | `<Cards>` with `<Card>` children |
| Badge chips | `<Tag variant="primary">label</Tag>` |

When using custom components, import them from the generated Fern component
alias:

```mdx
import { Tag } from "@/components/Tag";
import { BadgeLinks } from "@/components/BadgeLinks";
```

## Local commands

Run from the repository root:

```bash
make -f .github/fern-template/Makefile docs-check
scripts/serve_docs.sh
```

`scripts/serve_docs.sh` builds `.fern-build/fern/` and starts
`fern docs dev` without requiring Fern login. It uses a local placeholder for
the generated Python API reference. Pass Fern dev-server arguments after `--`,
for example `scripts/serve_docs.sh -- --port 3003`.

To build the full Python API reference locally, run
`scripts/serve_docs.sh --generate-api`; that path may require Fern login.

If `fern docs md generate` fails with an organization permission error:

```bash
make -f .github/fern-template/Makefile docs-login
```

## CI and publish

| Workflow | Purpose |
|---|---|
| `.github/workflows/fern-docs-ci.yml` | Prepares a temp Fern project and runs `fern check` on PR mirror branches |
| `.github/workflows/fern-docs-preview-build.yml` | Uploads `docs/`, `.github/fern-template/`, and metadata from PRs |
| `.github/workflows/fern-docs-preview-comment.yml` | Builds a Fern preview with `DOCS_FERN_TOKEN` and comments on PRs |
| `.github/workflows/publish-fern-docs.yml` | Publishes the generated Fern project |

The publish and preview workflows require the `DOCS_FERN_TOKEN` secret.

## Key files

| File | Purpose |
|---|---|
| `docs/` | Authored MDX content |
| `.github/fern-template/docs.yml` | Site config, redirects, theme, library reference |
| `.github/fern-template/versions/latest.yml` | Sidebar/navigation |
| `.github/fern-template/components/` | Custom MDX components |
| `.github/fern-template/main.css` | Theme overrides |
| `.github/scripts/prepare-fern-docs.sh` | Generates the disposable Fern project |
