---
name: fern-docs
description: "Edit or validate the NeMo AutoModel Fern docs (docs/*.mdx, docs/fern/): page edits, new, moved or removed pages, slugs, redirects, version cuts, fern check or preview failures."
---

# Fern Docs — NeMo AutoModel

Source for `docs.nvidia.com/nemo/automodel`. Nightly MDX lives at the top level
of `docs/`; `docs/fern/` holds only Fern build infrastructure. Frozen versions
live on the `docs-archive` branch, not on `main`.

## Every docs change

1. Edit the MDX under `docs/`, not under `docs/fern/`. Fern renders the
   frontmatter `title:` as the page H1, so don't add a leading `# H1`.
2. Use Fern components (`<Note>`, `<Tip>`, `<Info>`, `<Warning>`, `<Error>`),
   not GitHub `> [!NOTE]`. Raw HTML must be valid JSX (`<img ... />`).
3. Link internal pages without a version prefix (`/get-started/installation`,
   not `/latest/...`) and reference page images relatively (`./image.png`).
4. Check that changed claims, commands, config keys, and links match current code.
5. Before committing, run this from the repository root:

   ```bash
   uv run --script tools/docs/check_docs.py
   ```

   It needs `uv`, Node.js 20 with `npm`/`npx`, `make`, and full Git history,
   takes a minute or two, and cleans up after itself. Report its result when
   you hand off. CI runs the same script on PRs and posts a Fern preview link.

A wording change to an existing page needs no other reference. One exception:
model-card files that share a frontmatter `slug:` are compatibility copies and
must stay byte-identical, so apply a body-only edit to every copy
(`check_docs.py` rejects copies that differ).

## Other tasks

Read only the reference that matches the task:

| Task | Reference |
|---|---|
| Add, move, retitle, or remove a page; frontmatter, components, links; model-card renames | [references/pages.md](references/pages.md) |
| Slug renames and `redirects:` in `docs/fern/docs.yml` | [references/redirects.md](references/redirects.md) |
| Version trees, `docs-archive`, site layout, cutting a version | [references/versions.md](references/versions.md) |
| `check_docs.py` internals, docs tooling, library reference, preview, publish | [references/validation-and-publishing.md](references/validation-and-publishing.md) |
| `fern check`, preview, or broken-link errors; key file map | [references/troubleshooting.md](references/troubleshooting.md) |
