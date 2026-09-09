# NeMo AutoModel — docs/ convenience wrapper.
#
# The real Fern targets live in docs/fern/Makefile. This thin forwarder lets
# you run them from the docs/ directory (`cd docs && make docs-check`) without
# descending into docs/fern/. It is exactly equivalent to
# `make -C docs/fern <target>` from the repo root.
#
# See docs/fern/Makefile for the target implementations and docs/fern/README.md
# for first-time setup (`make docs-login`).

FERN_DIR := fern

.DEFAULT_GOAL := help

.PHONY: help docs docs-login docs-check docs-preview docs-publish docs-generate-library

help docs docs-login docs-check docs-preview docs-publish docs-generate-library:
	@$(MAKE) -C $(FERN_DIR) $@
