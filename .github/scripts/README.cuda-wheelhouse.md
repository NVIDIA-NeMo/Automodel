# CUDA installation coverage and wheelhouse trust

`install-test.yml` runs the CUDA UV installation in `wheelhouse` mode by
default. Dependency and build changes also enable `source` mode. A weekly
Monday 04:17 UTC run and the manual `source-build` input provide clean-build
coverage even when the most recent commit only changes documentation.

The source mode uses the normal UV project build settings, disables UV's cache,
and forbids prebuilt distributions for the wheelhouse packages selected by the
existing `all` extra. It also disables Mamba and causal-conv1d's own release-wheel
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
3. Pip and UV download the producer's artifact by ID. Before installing any
   wheel, they check the verified manifest digest, expected build fingerprint,
   complete wheel set, and every wheel's SHA-256. Verification errors stop the
   job. Legacy unsigned cache entries are excluded by the v4 cache namespace.

The attestation identifies the CI builder and the bytes it produced. It does
not establish that upstream source packages are benign or replace GPU runtime
correctness tests. OIDC/attestation write permissions belong only to the
wheelhouse builder; the verification job has read-only repository access.
