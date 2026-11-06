# Pages: add, update, move, remove

**Sidebar fidelity rule.** Section captions, page titles, and Model Coverage child ordering must match the **published v0.4.0 sidebar at docs.nvidia.com/nemo/automodel/latest** verbatim. Don't silently shorten a title or reorder siblings — the docs PM and content engineers diff against the published site and any drift is treated as a regression. If you want a shorter sidebar label, change the toctree-derived display name in the source — never just retitle in the MDX.

## Add a page

1. Gather: title, target section, filename (kebab-case `.mdx`), subdirectory under `docs/`.
2. Create the MDX at `docs/<subdir>/<filename>.mdx` with frontmatter:

   ```mdx
   ---
   title: "<Page Title>"
   description: "One-line SEO description (or empty string)"
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

4. Run `uv run --script tools/docs/check_docs.py` from the repository root and verify the URL resolves on `make docs` preview. There is no `nightly.yml` ↔ `latest.yml` alias-sync step under this layout — `latest.yml` mounts the frozen v0.4 tree, not nightly, so it's intentionally out of sync.

## Update a page

1. Locate by path, title, or keyword: `grep -rn "<keyword>" docs/ --include="*.mdx" --exclude-dir=fern`.
2. **Content only** — edit the single MDX file at `docs/<...>.mdx`.
3. **Title change** — update the frontmatter `title:` and update the `- page:` entry's display label in `docs/fern/versions/nightly.yml`.
4. **Section move** — `git mv` the file within `docs/`, update `path:` in `nightly.yml`, fix incoming links.
5. **Slug change** — set an explicit `slug:` while keeping the source filename stable. For model cards, set the full route in MDX frontmatter; keep the navigation `path:` pointing to that file. Add a `redirects:` entry in `docs/fern/docs.yml` so the old published URL keeps working.

## Model repository name becomes public

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

The full MDX slug determines the model category and provider independently of
the source directory. A previously shared file may stay anywhere under `docs/`
while navigation places it in the correct section. Existing files sharing a slug
are compatibility copies: select exactly one in nightly navigation and keep all
copies byte-identical; no path alias entry in `catalog.py` is required.

## Remove a page

1. Find incoming links: `grep -rn "<filename>" docs/ --include="*.mdx" --exclude-dir=fern`.
2. `git rm docs/<...>.mdx`.
3. Remove the `- page:` block from `docs/fern/versions/nightly.yml`.
4. Fix or delete incoming links.
5. Add a redirect in `docs/fern/docs.yml` if the URL was public.

## Worked example: add a guide

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

3. Run `uv run --script tools/docs/check_docs.py` from the repository root, then `make docs` in `docs/fern/` to preview at `http://localhost:3002/nightly/recipes-e2e-examples/qwen3-6-finetune`.

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

Page-scoped images live alongside the MDX file (e.g. `docs/guides/audio/qwen_omni_asr.png`). Reference them with relative paths (`./image.png`), not absolute (`/image.png`) — Fern's path resolver doesn't normalize root-relative image paths the same way as link targets. The NVIDIA logos and favicon come from the `nvidia` global theme; do not add them locally.

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
