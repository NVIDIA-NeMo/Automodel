# CUDA installation coverage and wheelhouse trust

`install-test.yml` runs the CUDA UV installation in `wheelhouse` mode by
default. Dependency and build changes also enable `source` mode. A daily
2:00 AM Pacific (`America/Los_Angeles`) run and the manual `source-build` input
provide clean-build coverage even when the most recent commit only changes
documentation. GitHub adjusts for daylight saving time; on the spring-forward
day, the nonexistent 2:00 AM advances to 3:00 AM.

If the scheduled workflow fails, a notification job opens a GitHub issue
mentioning `@NVIDIA-NeMo/automation` and `@NVIDIA-NeMo/core-am`, with the failed
run and commit. Further failures comment on the existing open alert issue;
after it is closed, a later failure opens a new one. Successful, cancelled,
push, and manual runs do not send these alerts. Team mention notifications
must be enabled in each team's GitHub settings. Alerts do not delete or
invalidate cached wheels.

The source mode uses the normal UV project build settings, disables UV's cache,
and forbids prebuilt distributions for the native wheelhouse packages selected by
the existing `all` extra. The wheel-only `transformer-engine` Python wrapper is
allowed; `transformer-engine-torch` still builds its native extension from source.
It also disables Mamba and causal-conv1d's own release-wheel
downloads. Both modes run the existing import checks. Source mode does not add
extras that the normal UV job does not test.

## Cache compatibility

The CUDA image must include an immutable SHA-256 digest. The cache fingerprint
includes that image reference, Python/platform and GPU architecture inputs,
resolved wheel and build-tool versions, UV settings and project build-system
configuration, and the build/helper/manifest scripts. Update the readable image
tag and its digest together. A changed fingerprint rebuilds the wheelhouse;
there is no older-key fallback.

## Release branches and cherry-picks

Pushes to `main` and release branches matching `r0.<minor>.0` populate shared
wheelhouse and container caches. Copied PRs read their actual target branch from
PR metadata. Source-validation selection compares a PR with that target's merge
base, so historical differences between a release and main do not trigger an
unnecessary source installation. Direct main/release pushes compare with the
pre-push commit. Missing history still enables source validation.

GitHub's Actions cache on a copied `pull-request/<number>` push can read main's
cache but cannot read a sibling release branch's cache. After an Actions cache
miss, the wheelhouse job therefore looks for a release artifact whose name
contains the exact cache key. It checks the originating branch, repository,
event, and installation workflow before selecting its immutable artifact/run
IDs. The producer downloads that artifact and forwards its original provenance;
pip and UV consumers download the same artifact directly. PRs never publish to
the shared release scope or copy a reused archive into a new full artifact.

A release run publishes the archive once per fingerprint with 30-day retention.
If the archive expires while its Actions cache survives, the next release run
republishes from cache without compilation. If both are missing, normal cold
build behavior applies. Deletion/expiry after selection fails the download;
API/verification errors fail the job rather than silently authorizing a rebuild.
The small provenance artifact is still transferred on every wheelhouse run.

Rollout requires backporting these workflow/scripts to each active release
branch and completing a first producer run there. Main-only changes cannot
populate a cache for an older release checkout. Subsequent compatible cherry-pick
PRs reuse that release's artifacts/images. Producer runs are not cancelled by
newer pushes to main or a release branch; PR supersession remains enabled.

## Verification before installation

1. A fresh build writes a manifest containing the compatibility fingerprint and
   SHA-256 of every wheel. The build job signs this manifest with GitHub's build
   provenance attestation action. The wheels, manifest, and signature bundle are
   cached together; only `main` and release branches save shared caches.
2. Cache hits preserve the original signature. The verifier requires the exact
   repository and `install-test.yml` signer. Cached manifests must originate
   from `refs/heads/main` or the selected release target. Fresh builds must match
   the current source ref and source/workflow commit. Other releases and PR
   builders are never accepted for reused wheelhouses.
3. On a cache hit, pip and UV restore the producer's exact cache key directly;
   the producer skips uploading the full wheelhouse artifact, except when a
   release needs to seed or refresh its shared archive. Fresh builds still
   upload an artifact, which consumers download by ID. All paths pass
   the small manifest/signature artifact through the provenance verifier.
   Before installing any wheel, consumers check the verified manifest digest,
   expected build fingerprint, complete wheel set, and every wheel's SHA-256.
   Verification errors stop the job. Legacy unsigned cache entries are excluded
   by the v4 cache namespace.

If a cache hit disappears before a consumer restores it, that consumer fails
instead of compiling or falling back to another key. Rerun the workflow to
select the cache or fresh-build path again. Source-validation jobs consume
neither the wheelhouse cache nor the artifact.

The attestation identifies the CI builder and the bytes it produced. It does
not establish that upstream source packages are benign or replace GPU runtime
correctness tests. OIDC/attestation write permissions belong only to the
wheelhouse builder; the verification job has read-only repository access.

## CI container dependency reuse

`cicd-main.yml` also reuses the Dockerfile's `automodel_dependencies` stage as
a registry image. This is separate from the installation-test wheelhouse:
the container uses NGC PyTorch and needs native AMD64 and ARM64 builds,
including TE, DeepEP, bitsandbytes, torchao, FlashMLA, FlashAttention, and the
UV-installed dependencies. The installation-test wheels cannot substitute for
that different architecture/PyTorch ABI.

The key includes the resolved NGC PyTorch image digest, target platform,
`.github/container-build-args.env`, the Dockerfile and `docker/` inputs,
both lockfiles, project metadata, the UCCL setup script, `.dockerignore`, and
the key-generation script. Dependency changes build and publish a new image;
application and test changes reuse the matching image. Keep any new dependency
`COPY` inputs in the key's file list. The build receives the same pinned base
digest and arguments used to compute the key.

Each branch writes its own image scope. Cherry-pick PRs look up their own image,
then their release target's image, then main's image, always using the exact
fingerprint and platform. Other PRs look up their own image then main's. Main
never reads PR/release images. On a miss the PR publishes only to its own scope.
BuildKit uses the first available donor in the same order, never a mixture of
donors. Release branches also save their Actions UV cache for subsequent direct
release builds; copied PRs reuse the registry images/layers instead. Registry access
uses the existing CI registry permissions; this is not the attested
installation-wheelhouse artifact. Missing images build normally, while registry
authentication and network failures stop the job instead of starting an
unexpected source build. The first build for a new key still compiles.

The final build consumes the dependency image by digest through a named build
context. It always copies the current checkout and runs the existing native
runtime and non-root checks. On a hit it skips the dependency build, UV cache
download/injection, and registry layer-cache export. On a miss the existing
BuildKit/UV caches still accelerate compilation. Ordinary Docker builds without
the named context continue to build all stages from source.
