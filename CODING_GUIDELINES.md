# Coding guidelines

These guidelines apply to human contributors and coding agents. Follow them when
adding or modifying code. Existing code is being refactored incrementally and may
not yet comply; use these guidelines to guide changes within your task's scope.

## Design

- **Separate concerns.** Give each function and class one cohesive responsibility.
- **Extract around responsibility.** Split code when responsibilities, dependencies, or lifetimes differ.
  Keep cohesive algorithms together; do not extract helpers based on line count alone.
- **Keep argument lists short.** Pass only necessary inputs; group related settings in typed configs.
- **Use explicit contracts for variation.** Prefer typed attributes, capabilities, and existing interfaces
  over repeated runtime introspection, ad-hoc string dispatch, or untyped parameter bags.
  Use narrow, typed callbacks or model-owned hooks when they express a necessary operation;
  document when they run and who owns their state.
- **Avoid globals.** Keep mutable state with its owner and pass dependencies explicitly.
- **Keep APIs small.** Reuse existing components and avoid speculative abstractions.
  Introduce shared abstractions only when existing callers share stable semantics.
- **Search before implementing.** Before adding a helper, backend, or adapter, search shared components and sibling
  models for the same capability and underlying library calls. Inspect candidate implementations and their production
  callers. Reuse or extend compatible code; justify a separate implementation with concrete differences in semantics
  or input contracts.
- **Design for produced inputs.** Trace supported input forms to their production callers. Do not duplicate
  infrastructure to support hypothetical layouts, metadata, or execution modes without a demonstrated requirement.
- **Keep model-specific logic in the model directory.** Architecture, initialization, weight layouts,
  and model-specific parallelization policy belong under `nemo_automodel/components/models/<model>/`.
- **Keep shared components as infrastructure.** Shared components must not branch on model names, families,
  or model-specific module paths to select behavior. Expose generic typed contracts and let model-owned
  implementations supply the policy. Keep recipes as thin orchestration.
- **Stop YAML at the typed-config boundary.** Parse and validate YAML into typed configs at the configuration
  boundary. Components must consume typed configs and explicit typed runtime arguments, never YAML or `ConfigNode`
  objects. Converting a `ConfigNode` into a plain dictionary, nested mapping, or `**kwargs` bag and passing it into
  a component is the same boundary violation; it does not make the configuration typed. Do not reconstruct or
  interpret the YAML schema inside components.
- **Let configs own construction.** Use typed `build(...)` methods; keep runtime objects out of serializable configs.
  Preserve typed configs internally, serialize at shared boundaries, and do not mutate configs during construction.
- **Keep artifact versions consistent.** Resolve related external artifacts, such as configs and weights,
  to a consistent immutable revision and propagate that identity through dependent loads.

## Code provenance

- **Implement independently.** Make every effort to avoid copying code from external repositories.
  Reuse compatible code already in this repository as described above.
- **Record unavoidable external reuse.** If copying or adapting external code is necessary, include the source URL,
  immutable revision, original file paths, and applicable licenses in the header of each affected file.
  Complete the PR description's "Code provenance" section with those details and the affected repository files.
- **Preserve attribution.** Retain all required license and copyright notices and attribution in affected files,
  and state in the PR description where those notices are retained or added.

## Types and contracts

- **Prefer explicit types.** Annotate public inputs and returns with concrete, meaningful types.
- **Type annotations still apply to excluded files.** Add types to new and modified
  signatures even when `pyproject.toml` excludes the file or relaxes its type checks.
- **Avoid `Any`.** It effectively bypasses type checking for affected values and hides errors.
- **Use contracts where applicable.** Define interfaces with typed signatures, dataclasses, or `Protocol`;
  document assumptions, ownership, and failure modes.
- **Document tensor contracts.** State shapes, axis order, relevant dtype/device requirements, and distributed placements.
- **Preserve semantic contracts.** Transformations must preserve token/label/media alignment, document boundaries,
  parameter identity, and distributed placements where applicable. Document these relationships alongside shapes and types.
- **Treat precision as part of correctness.** Distinguish storage, compute, and accumulation dtypes.
  Preserve precision-sensitive state through casts and serialization. Verify gradients and required reproducibility
  when rewriting tensor operations or reductions.
- **Validate boundaries.** Reject invalid inputs and unsupported combinations early at ownership and trust boundaries.
  Raise specific exceptions and preserve exception causes. Use safe serialization defaults, preserve argument boundaries
  when constructing commands, and localize compatibility handling to adapters.
- **Follow repository tooling.** Use Ruff, Google-style docstrings, `uv`, and `safe_import()` for optional dependencies.

## State and resource ownership

- **Make ownership explicit.** Do not mutate caller-owned inputs unless the API documents it.
  Document tensor aliasing and ownership transfer.
- **Bound resource lifetimes.** Keep asynchronous resources alive until completion, release transient state after
  consumption, and handle cleanup on failure or early termination, including activation-checkpoint recomputation.
- **Keep state transitions coherent and recoverable.** Establish valid state before use and publish results only after
  successful completion. Define behavior for partial failure, retries, and repeated initialization or restoration;
  do not expose partially updated state as a successful result.

## Changes and validation

- Keep changes focused and preserve compatibility, or provide a migration path.
- Review newly added implementations for overlap with existing shared and sibling code. Record the strongest reuse
  candidate and why it was reused, extended, or unsuitable. Passing correctness or performance tests does not establish
  that a separate implementation is necessary.
- Test observable behavior and failure cases. Regression tests should demonstrate the original failure.
  Exercise the production entry point when helper tests cannot establish integration correctness;
  check forward and backward parity for tensor math.
- Use real multi-process tests for distributed behavior that mocks cannot exercise.
- Support performance changes with representative measurements. Report the workload, environment, throughput,
  and memory tradeoffs; distinguish isolated improvements from end-to-end gains.
- When adding or changing a quality gate, verify that it covers the intended files and rejects a known violation.
  Keep suppressions and exceptions narrow, justified, and tracked for removal when temporary.
- Report checks run, results, and remaining gaps. Sign off commits with DCO.

Details: [contributing](CONTRIBUTING.md), [agent instructions](AGENTS.md),
[review policy](skills/pr-review/SKILL.md), [testing](.agents/contributor-skills/testing/SKILL.md),
[parity testing](.agents/contributor-skills/parity-testing/SKILL.md),
[distributed training](skills/nemo-automodel-distributed-training/SKILL.md),
and [checkpointing](docs/guides/checkpointing.mdx).
