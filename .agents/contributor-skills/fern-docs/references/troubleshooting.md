# Troubleshooting and key files

## Debugging

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to `nightly.yml`'s location (so nightly entries reach back up via `../../`); `slug:` must not collide with siblings |
| Page 404 in preview | Missing `slug:` override (default slugifies the long display title) or `position:` collision in an auto-discovered folder |
| `Folder not found: ./product-docs/...` on `fern docs dev` | Run `make docs` once to populate the library reference |
| `[ERR_PNPM_IGNORED_BUILDS]` on first `fern docs dev` | pnpm 10+ blocks esbuild's postinstall — `pnpm config set onlyBuiltDependencies '["esbuild"]' --location global`, then `rm -rf ~/.fern/app-preview` and retry |
| Broken-link warning on version-agnostic path | `fern docs broken-links` false-positives; URLMap-based validator is authoritative |
| `JSX expressions must have one parent element` | Wrap multi-element JSX in `<>...</>` or a `<div>` |
| Old Sphinx URL breaks | Add a `redirects:` entry in `docs/fern/docs.yml`; the redirect generator already handles `/index.html` and `.html` legacy forms |
| Image not rendering | Use relative path (`./image.png`) for page-scoped images, not root-relative (`/image.png`) |
| Sidebar caption looks shortened vs published site | Compare against `docs.nvidia.com/nemo/automodel/latest` and restore the verbatim title in `docs/fern/versions/nightly.yml` |
| `path: ../../foo.mdx` doesn't resolve | Confirm the MDX file is at `docs/foo.mdx` (top level), not still under `docs/fern/versions/nightly/pages/` — that legacy tree no longer exists |
| `fern check` fatals on missing archived page paths | Run `uv run --script tools/docs/check_docs.py` from the repository root; it restores the frozen trees from `docs-archive` before validation |
| `archive ref '...' does not contain '...'` in CI | The `stitch-fern-versions` action couldn't find the version's pages on its registry ref. Confirm the `docs-archive` branch (or the configured tag) still holds `docs/fern/versions/<vdir>/pages` |

## Key references

| File | Purpose |
|---|---|
| `docs/fern/docs.yml` | Site config — `instances`, `versions`, `redirects`, `libraries`, theme |
| `docs/fern/versions/nightly.yml` | Canonical nav tree — paths reach up into `docs/` via `../../` |
| `docs/fern/versions/{latest,v0.4}.yml` | Frozen GA nav (mount `./v0.4/pages/...`) |
| `docs/` (top-level *.mdx) | Nightly MDX content (~140 pages + page-scoped images) |
| `docs/fern/versions/v0.4/pages/` | Frozen 0.4.0 snapshot — **on the `docs-archive` branch, not `main`**; stitched in at build time |
| `docs-archive` branch | Holds all frozen backward-version `pages/` trees; restored by `stitch-fern-versions` / `make docs-stitch` |
| `.github/actions/stitch-fern-versions/` | Composite action that restores archived version pages before any Fern build |
| `docs/fern/components/` | `BadgeLinks.tsx`, `Tag.tsx` (repo-specific; NVIDIA footer ships via `global-theme: nvidia`) |
| `docs/fern/README.md` | Human-facing orientation |
| `docs/fern/Makefile` | `make docs / docs-check / docs-preview / docs-publish` (run from `docs/fern/` or via `make -C docs/fern`) |
| `.github/workflows/fern-docs-*.yml` | CI: check, preview build, preview comment |
| `.github/workflows/publish-fern-docs.yml` | CI: publish to docs.nvidia.com/nemo/automodel |
