# Slugs and redirects

## Redirect quirks

Four things to watch when editing `redirects:` in `docs/fern/docs.yml`:

1. **`:path*` does NOT match the empty-path case.** `/<basepath>/v0.4/:path*/index.html` will *not* match `/<basepath>/v0.4/index.html` (where `:path*` would have to be empty). Each version-root `index.html` needs its own explicit rule. NeMo Curator (NVIDIA-NeMo/Curator#1938) discovered this when their version-root URLs 404'd. AutoModel ships explicit rules for `latest`, `v0.4`, `nightly`, and the legacy `0.4` form — when you add a new version slug, add four new explicit rules: `<slug>/index.html`, `<slug>/index`, plus the same two for any legacy form (e.g. `0.5` → `v0.5`).
2. **Older un-migrated versions need a fallback.** Whatever versions the published Sphinx site exposed (check the version-switcher dropdown on `docs.nvidia.com/nemo/<product>/latest/`) but you didn't migrate into Fern still need to resolve. The pattern: redirect each old slug's URLs to the equivalent path under `/latest/` so external bookmarks and search results land on the closest current page instead of 404ing. Five rules per old version: `<slug>/index.html`, `<slug>/index`, `<slug>/:path*/index.html`, `<slug>/:path*`, `<slug>/:path*.html` — all destinations `/latest/...`. AutoModel ships these for `0.3.0`, `0.2.0`, `0.1.0`.
3. **Order matters.** Specific rules must come before catch-alls — Fern uses first-match. Slot new rules *before* the `:path*/index.html` and `:path*.html` catch-alls.
4. **Don't ship `redirects: []`** then re-run the redirect generator on top — it replaces the whole `redirects:` block. Edit by hand or back up the existing rules first.

## Worked example: rename a slug with a redirect

Request: *"Rename `/recipes-e2e-examples/sft-peft` to `/recipes-e2e-examples/fine-tuning`."*

1. Edit `docs/fern/versions/nightly.yml`, change the `slug:` on the SFT & PEFT entry from `sft-peft` to `fine-tuning`.
2. Add a redirect to `docs/fern/docs.yml`:

   ```yaml
   redirects:
     - source: "/:version/recipes-e2e-examples/sft-peft"
       destination: "/:version/recipes-e2e-examples/fine-tuning"
   ```

3. `grep -rn "/recipes-e2e-examples/sft-peft" docs/ --include="*.mdx" --exclude-dir=fern` and update incoming body links.
