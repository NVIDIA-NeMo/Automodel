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

## Verification before installation

1. A fresh build writes a manifest containing the compatibility fingerprint and
   SHA-256 of every wheel. The build job signs this manifest with GitHub's build
   provenance attestation action. The wheels, manifest, and signature bundle are
   cached together; only `main` saves the shared cache.
2. Cache hits preserve the original signature. The verifier requires the exact
   repository and `install-test.yml` signer. Cached manifests must originate
   from `refs/heads/main`; fresh builds must match the current source ref and
   source/workflow commit.
3. On a cache hit, pip and UV restore the producer's exact cache key directly;
   the producer skips uploading the full wheelhouse artifact. Fresh builds
   still upload an artifact, which consumers download by ID. Both paths pass
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

Only `main` writes the canonical image scope. Other branches read their own
scope or `main` and publish misses to their own scope, so release branches can
reuse their builds even without a matching Actions UV cache. Registry access
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
