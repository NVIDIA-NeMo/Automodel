# NeMo AutoModel — top-level convenience targets.
# Docs-related targets live here so contributors don't have to remember the
# exact `cd fern && fern …` invocations. CI workflows under
# `.github/workflows/fern-docs-*.yml` are the source of truth for the published
# pipeline; these targets just mirror the local-developer entry points.

FERN_DIR := fern
PUBLISH_WORKFLOW := Publish Fern Docs

.PHONY: help docs docs-preview docs-publish docs-check

help:
	@echo "Docs targets:"
	@echo "  make docs           Generate library reference + serve preview on http://localhost:3002"
	@echo "  make docs-check     Run 'fern check' (config + MDX validation)"
	@echo "  make docs-preview   Build a shared preview URL on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)"
	@echo "  make docs-publish   Trigger the 'Publish Fern Docs' workflow on origin/main"

# Local-only preview. `fern docs md generate` populates fern/product-docs/ from
# the nemo_automodel package source (declared under `libraries:` in fern/docs.yml);
# `fern docs dev` then serves the site on localhost:3002. Re-run `make docs` only
# when the library source changes — for prose-only iteration, `cd fern && fern docs dev`
# alone is enough after the first generate.
docs:
	cd $(FERN_DIR) && fern docs md generate && fern docs dev

docs-check:
	cd $(FERN_DIR) && fern check

# Shared preview hosted at <repo-slug>.docs.buildwithfern.com — useful for sharing a
# work-in-progress link before merge. Requires DOCS_FERN_TOKEN in the environment
# (org secret of the same name is wired into CI).
docs-preview:
	cd $(FERN_DIR) && fern generate --docs --preview

# Trigger the Publish Fern Docs workflow on origin/main via workflow_dispatch.
# Alternative: tag a release with `git tag docs/v0.4.0 && git push origin docs/v0.4.0`
# — the workflow also fires on `docs/v*` tag pushes.
docs-publish:
	gh workflow run "$(PUBLISH_WORKFLOW)" --ref main
