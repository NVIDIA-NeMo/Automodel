# Version trees and archives

## Scope rule

**Nightly MDX content lives at the top level of `docs/`** (e.g. `docs/index.mdx`, `docs/guides/llm/finetune.mdx`). **`docs/fern/` holds only Fern build infrastructure** — config, theme, and components. New pages, release notes, migration guides → add as a top-level `.mdx` under `docs/`.

**Only the nightly tree is kept on `main`.** Frozen backward-version snapshots live on the `docs-archive` branch and are restored at build time — see *Archived backward versions* below.

**Two real content trees, plus a GA alias YAML.**

- `docs/` — bleeding-edge (nightly) tree. Every PR lands here. Mounted at the `nightly` URL slug via `docs/fern/versions/nightly.yml` (paths reach back up via `../../<rel>.mdx`).
- `docs/fern/versions/v0.4/pages/` — frozen 0.4.0 GA snapshot. **Not on `main`**: it lives on the `docs-archive` branch and is restored under this path by `make docs-stitch` (local) or the `stitch-fern-versions` CI action (build). Mounted at the `v0.4` URL slug via `v0.4.yml`. Only changes via deliberate back-port (on `docs-archive`).
- `docs/fern/versions/latest.yml` — GA alias. Its `path:` lines mount the current GA's content (today: `./v0.4/pages/...`). Repointed at the next GA's tree when one is cut.

The nightly and v0.4 trees were byte-for-byte identical at the moment 0.4.0 shipped, but they will diverge as nightly accumulates post-release edits and v0.4 stays frozen. **Default editing target is `docs/` top-level.** Back-ports to a frozen version happen on the `docs-archive` branch, not here — call out the divergence in the PR description.

## Archived backward versions

Fern has no native way to source a version train's prose from another git ref: `fern generate --docs` reads the single local working tree, and publish is a full-site snapshot (a train missing from the tree is *unpublished*). So frozen GA pages are kept off `main` on the **`docs-archive` branch** and restored before every Fern build.

- **The registry** for checking and publishing lives in the Makefile's `docs-stitch` target, invoked by `tools/docs/check_docs.py`. Archived preview builds declare their archive inputs in their workflow. The `<git-ref>` is opaque: a **branch** (default — all frozen versions in one place, easy back-ports) or a **tag** like `docs/v0.4.0` (immutable snapshot). `latest` is an alias of an existing pages tree and needs **no** entry.
- **The mechanism** is the `.github/actions/stitch-fern-versions` composite action in archived preview builds, or the equivalent Makefile `docs-stitch` target in local checks, CI checks, and publishing: `git fetch --depth=1 origin <ref>` then `git restore --source=FETCH_HEAD -- docs/fern/versions/<vdir>/pages`. `docs.yml` + the nav YAMLs always come from the live checkout, never the archive ref, so frozen prose can't drift into the wrong train.
- **Locally**, `make docs` / `docs-preview` and the shared checker use `make docs-stitch`, which does the same restore. Override the check's archive ref with `ARCHIVE_REF=docs/v0.4.0 uv run --script tools/docs/check_docs.py`.
- The restored path is gitignored on `main`, so a local stitch won't show the pages as untracked.

## Layout at a glance

```
docs/                                ← nightly MDX (top level)
├── index.mdx, breaking-changes.mdx, release-notes.mdx, ...
├── about/, guides/, model-coverage/, launcher/, api-reference/
├── *.png / *.jpg                    ← page-scoped images
└── fern/                            ← infra only
    ├── fern.config.json             # Org slug + Fern CLI pin (5.29.0+)
    ├── docs.yml                     # Site config + global-theme: nvidia (inherits
    │                                #   logos / footer / theme CSS / fonts / OneTrust JS
    │                                #   from NVIDIA/fern-components)
    ├── components/                  # BadgeLinks.tsx, Tag.tsx
    │                                #   (repo-specific; NVIDIA footer ships in global theme)
    ├── versions/
    │   ├── nightly.yml              # Nav for nightly — paths → ../../<rel>.mdx (up into docs/)
    │   ├── v0.4.yml                 # Nav for frozen 0.4.0 — paths → ./v0.4/pages/
    │   ├── v0.4/pages/              # Frozen 0.4.0 MDX (back-ports only)
    │   └── latest.yml               # GA alias — paths → ./v0.4/pages/ today; repointed at next GA cut
    └── product-docs/                # GENERATED Python API reference (gitignored)
```

```
File                                                          URL
─────────────────────────────────────────────────────────────  ────────────────────────────────────────────
docs/get-started/installation.mdx                              /nightly/get-started/installation
docs/fern/versions/v0.4/pages/get-started/installation.mdx     /latest/get-started/installation
                                                               /v0.4/get-started/installation
```

## Cutting a new version train

When NeMo AutoModel ships a new GA (e.g. `v0.5`):

1. `mkdir -p docs/fern/versions/v0.5/pages && rsync -a --exclude='fern' docs/ docs/fern/versions/v0.5/pages/` — fresh frozen snapshot of nightly at release time.
2. `cp docs/fern/versions/nightly.yml docs/fern/versions/v0.5.yml` and rewrite `../../` path prefixes to `./v0.5/pages/`.
3. Update `docs/fern/versions/latest.yml` to point at the new train: `cp docs/fern/versions/v0.5.yml docs/fern/versions/latest.yml`. (`latest` is the auto-bumping GA alias.)
4. In `docs/fern/docs.yml` `versions:`, add a new frozen-pin entry (`display-name: "0.5.0 · 26.07"`, `slug: v0.5`, `availability: stable`) and keep the previous pin (`v0.4`) for permalink stability.
5. **Archive the frozen tree off `main`** (see *Archived backward versions*): commit `docs/fern/versions/v0.5/pages/` onto the `docs-archive` branch and push it, then `git rm -r docs/fern/versions/v0.5/pages` from `main` and add that path to `.gitignore`. The `v0.5.yml`/`latest.yml`/`docs.yml` config stays on `main`.
6. Add the new version to the archive inputs in preview workflows, the shared `docs-stitch` target in `docs/fern/Makefile`, and `.gitignore`. `docs/` keeps moving forward as the bleeding-edge tree; the new frozen snapshot only changes via deliberate back-port on `docs-archive`.
7. Promote `nightly` to `availability: stable` if and when its content tree gets cut over.
8. Tag `docs/v0.5.0` and push to publish.
