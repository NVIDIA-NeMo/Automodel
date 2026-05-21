# NeMo AutoModel Fern Template

This directory contains Fern-specific build configuration only. The repository
does not track a root-level `fern/` project.

Authors edit pages and page-scoped images under [`../../docs`](../../docs).
CI and local Make targets generate a temporary Fern project by combining:

| Source | Purpose |
|---|---|
| `docs/` | Authored MDX pages and images |
| `.github/fern-template/docs.yml` | Fern site config, redirects, theme, library reference |
| `.github/fern-template/versions/latest.yml` | Single `latest` navigation tree |
| `.github/fern-template/assets/` | Shared logos |
| `.github/fern-template/components/` | Custom MDX components |

The generated project is created at `.fern-build/fern` locally, or under
`$RUNNER_TEMP` in GitHub Actions. It is disposable.

## Local Commands

Install the Fern CLI once:

```bash
npm install -g fern-api@4.62.4
```

Run checks or preview from the repo root:

```bash
make -f .github/fern-template/Makefile docs-check
scripts/serve_docs.sh
```

`scripts/serve_docs.sh` prepares `.fern-build/fern` and starts the local Fern
dev server without requiring Fern login. It uses a local placeholder for the
generated Python API reference. Pass Fern dev-server arguments after `--`, for
example:

```bash
scripts/serve_docs.sh -- --port 3003
```

To build the full Python API reference locally, run:

```bash
scripts/serve_docs.sh --generate-api
```

First-time library-reference generation may require Fern dashboard
provisioning and CLI login:

```bash
make -f .github/fern-template/Makefile docs-login
```

## Publishing Model

Only one maintained docs tree is published: `latest`.

Legacy `/nightly`, `/v0.4`, `/0.4`, `/0.3.0`, `/0.2.0`, and `/0.1.0` routes are
redirected to the nearest `/latest` path in `docs.yml`; their content trees are
not tracked in this repo.
