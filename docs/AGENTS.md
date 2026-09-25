# `docs/` — Fern documentation site

Adds to the repository root `AGENTS.md`.

**Load the `fern-docs` skill before editing documentation.** It covers slugs,
redirects, navigation, version aliases, and how to run previews and validation.

## Scope rule

- **Content** is MDX at the top level of `docs/` — e.g. `docs/index.mdx`,
  `docs/guides/`, `docs/model-coverage/`. New pages go here.
- **`docs/fern/` is build infrastructure only** — `docs.yml`, `fern.config.json`,
  theme, components, and the per-version mount files under
  `docs/fern/versions/`. Do not put page content there.

Only the nightly tree lives on `main`. Frozen backward-version snapshots live
on the `docs-archive` branch and are restored at build time. Back-porting to a
frozen version happens on `docs-archive`, not here; call out the divergence in
the PR description.

## Keep the architecture tour here

`docs/repository-structure.mdx` is the source of truth for repository layout.
The root `AGENTS.md` points at it deliberately instead of keeping its own copy.
When you move a top-level directory, update this page — it is what agents and
new contributors read.

## Links

Adding or renaming a page changes its slug. Add a redirect for the old slug and
run the Fern link check; a dangling link fails the docs build, not just the
page. Prefer site-relative links over hard-coded GitHub URLs for content that
lives in the docs tree.
