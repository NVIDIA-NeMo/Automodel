# NeMo AutoModel — top-level convenience targets.
# Docs-related targets live here so contributors don't have to remember the
# exact `cd fern && fern …` invocations. CI workflows under
# `.github/workflows/fern-docs-*.yml` are the source of truth for the published
# pipeline; these targets just mirror the local-developer entry points.
#
# First time on this machine? Run `make docs-login` before `make docs`.
# Without dashboard provisioning, `fern docs md generate` fails with
# `HTTP 403: User does not belong to organization`.

FERN_DIR := fern
PUBLISH_WORKFLOW := Publish Fern Docs

.DEFAULT_GOAL := help

.PHONY: help docs docs-login docs-check docs-preview docs-publish docs-generate-library

help:
	@echo ""
	@echo "NeMo AutoModel top-level Make targets"
	@echo "====================================="
	@echo ""
	@echo "  make docs-login             FIRST-TIME SETUP — provision Fern account + CLI auth"
	@echo "  make docs                   Generate library reference and start Fern dev server"
	@echo "  make docs-check             Run 'fern check' (config + MDX validation)"
	@echo "  make docs-preview           Build a shared preview URL on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)"
	@echo "  make docs-publish           Trigger the 'Publish Fern Docs' workflow on origin/main"
	@echo "  make docs-generate-library  Regenerate the autodoc library reference under fern/product-docs/"
	@echo ""
	@echo "First time? Run 'make docs-login' before 'make docs' or your"
	@echo "autodoc step will fail with HTTP 403."
	@echo ""

# First-time auth setup. Fern's CLI requires that the user already exist in
# Fern's user DB *before* the CLI login flow can complete; signing in to the
# dashboard is what creates that user record. Skipping step 1 leaves you with
# a CLI prompt that has no path forward, and skipping step 2 leaves the CLI
# unauthenticated so `fern docs md generate` returns:
#
#     HTTP 403: User does not belong to organization
#
# (Tracked upstream at Fern; ping #fern-cli on Slack if it changes.)
docs-login:
	@echo ""
	@echo "Fern auth — one-time setup per machine"
	@echo "======================================="
	@echo ""
	@echo "  1. Open https://dashboard.buildwithfern.com/login and sign in"
	@echo "     with your @nvidia.com email. Use the email/magic-link flow,"
	@echo "     not the Google SSO button — Google sign-in does not always"
	@echo "     provision the account that the CLI later needs."
	@echo ""
	@echo "     External contributors: any email works (Fern provisions a"
	@echo "     personal account); ask in #fern to be added to the 'nvidia'"
	@echo "     org if you want to push library autodoc generation."
	@echo ""
	@echo "  2. Confirm the 'nvidia' organization shows in the dashboard"
	@echo "     sidebar — if it doesn't, you signed in with the wrong"
	@echo "     account; sign out and retry step 1."
	@echo ""
	@echo "  3. The CLI 'fern login' step will run next; complete the browser"
	@echo "     flow with the SAME email you used in step 1."
	@echo ""
	@echo "Press Ctrl-C now if you have not done step 1 yet."
	@echo ""
	@sleep 2
	fern login

# Local-only preview. `fern docs md generate` populates fern/product-docs/ from
# the nemo_automodel package source (declared under `libraries:` in fern/docs.yml);
# `fern docs dev` then serves the site on localhost:3002. Re-run `make docs` only
# when the library source changes — for prose-only iteration, `cd fern && fern docs dev`
# alone is enough after the first generate.
docs: docs-generate-library
	cd $(FERN_DIR) && fern docs dev

docs-check:
	cd $(FERN_DIR) && fern check

# Wraps the autodoc generate step. If it fails (the most common cause is
# the HTTP 403 from missing dashboard provisioning), print the recovery
# pointer so contributors aren't left guessing.
docs-generate-library:
	@cd $(FERN_DIR) && fern docs md generate || { \
		status=$$?; \
		echo ""; \
		echo "✗ 'fern docs md generate' failed (exit $$status)."; \
		echo ""; \
		echo "If the error mentions 'HTTP 403: User does not belong to"; \
		echo "organization', your CLI auth is missing the dashboard"; \
		echo "provisioning step. Fix:"; \
		echo ""; \
		echo "    make docs-login"; \
		echo ""; \
		echo "Then re-run: make docs"; \
		echo ""; \
		exit $$status; \
	}

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
