# NeMo AutoModel Fern docs convenience targets.
#
# The repository does not track a root-level fern/ directory. These targets
# assemble a temporary Fern project from:
#
#   docs/                    authored MDX pages and images
#   .github/fern-template/   Fern config, theme, components, navigation
#
# Invoke from anywhere with:
#
#   make -f .github/fern-template/Makefile docs-check

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MAKEFILE_DIR)/../..)
FERN_SITE_DIR ?= $(REPO_ROOT)/.fern-build/fern
FERN_VERSION ?= 4.62.4
FERN := npx -y fern-api@$(FERN_VERSION)
PUBLISH_WORKFLOW := Publish Fern Docs

.DEFAULT_GOAL := help

.PHONY: help prepare docs docs-login docs-check docs-preview docs-publish docs-generate-library

help:
	@echo ""
	@echo "NeMo AutoModel Fern docs Make targets"
	@echo "====================================="
	@echo ""
	@echo "Run from anywhere in the repo with:"
	@echo ""
	@echo "  make -f .github/fern-template/Makefile <target>"
	@echo ""
	@echo "Targets:"
	@echo "  docs-login             First-time Fern account + CLI auth setup"
	@echo "  prepare                Build the temporary Fern project under .fern-build/fern"
	@echo "  docs                   Generate library reference and start Fern dev server"
	@echo "  docs-check             Run 'fern check' against the generated Fern project"
	@echo "  docs-preview           Build a shared preview URL (needs DOCS_FERN_TOKEN)"
	@echo "  docs-publish           Trigger the 'Publish Fern Docs' workflow on origin/main"
	@echo "  docs-generate-library  Regenerate the autodoc library reference"
	@echo ""

prepare:
	"$(REPO_ROOT)/.github/scripts/prepare-fern-docs.sh" "$(REPO_ROOT)" "$(FERN_SITE_DIR)"

# First-time auth setup. Fern's CLI requires that the user already exist in
# Fern's user DB before `fern login` can complete; signing in to the dashboard
# creates that user record.
docs-login:
	@echo ""
	@echo "Fern auth - one-time setup per machine"
	@echo "======================================="
	@echo ""
	@echo "  Step 1: open https://dashboard.buildwithfern.com/login"
	@echo "          and sign in with your @nvidia.com email."
	@echo ""
	@echo "  Step 2: confirm the 'nvidia' organization shows in the dashboard."
	@echo ""
	@echo "  Step 3: this target will run 'fern login' next; complete the"
	@echo "          browser flow with the same email."
	@echo ""
	@printf "Have you completed step 1? [yes/N]: "
	@read confirm; case "$$confirm" in \
		yes|YES|y|Y) ;; \
		*) echo ""; echo "Bailing. Complete dashboard sign-in, then re-run this target."; exit 1 ;; \
	esac
	@echo ""
	$(FERN) login

docs:
	"$(REPO_ROOT)/scripts/serve_docs.sh"

docs-check: prepare
	cd "$(FERN_SITE_DIR)" && $(FERN) check

docs-generate-library: prepare
	@cd "$(FERN_SITE_DIR)" && $(FERN) docs md generate || { \
		status=$$?; \
		echo ""; \
		echo "'fern docs md generate' failed (exit $$status)."; \
		echo ""; \
		echo "If the error mentions 'HTTP 403: User does not belong to organization',"; \
		echo "run:"; \
		echo ""; \
		echo "    make -f .github/fern-template/Makefile docs-login"; \
		echo ""; \
		exit $$status; \
	}

docs-preview: prepare
	cd "$(FERN_SITE_DIR)" && $(FERN) generate --docs --preview

docs-publish:
	gh workflow run "$(PUBLISH_WORKFLOW)" --ref main
