# Coding guidelines

Draft for humans and coding agents. Existing repository policies remain authoritative.

## Design

- **Separate concerns.** Give each function and class one cohesive responsibility.
- **Keep functions small.** Extract cohesive helpers when control flow becomes hard to follow.
- **Keep argument lists short.** Pass only necessary inputs; group related settings in typed configs.
- **Avoid function pointers.** Prefer direct calls or typed object methods over callback chains;
  keep framework-required callbacks explicit and typed.
- **Avoid globals.** Keep mutable state with its owner and pass dependencies explicitly.
- **Keep APIs small.** Reuse existing components and avoid speculative abstractions.
- **Respect ownership.** Keep model-specific logic in its model package and recipes as thin orchestration.
- **Let configs own construction.** Use typed `build(...)` methods; keep runtime objects out of serializable configs.

## Types and contracts

- **Prefer explicit types.** Annotate public inputs and returns with concrete, meaningful types.
- **Avoid `Any`.** It effectively bypasses type checking for affected values and hides errors.
- **Use contracts where applicable.** Define interfaces with typed signatures, dataclasses, or `Protocol`;
  document assumptions, ownership, and failure modes.
- **Document tensor contracts.** State shapes, axis order, relevant dtype/device requirements, and distributed placements.
- **Validate boundaries.** Raise specific exceptions for invalid inputs and preserve exception causes.
- **Follow repository tooling.** Use Ruff, Google-style docstrings, `uv`, and `safe_import()` for optional dependencies.

## Changes and validation

- Keep changes focused and preserve compatibility, or provide a migration path.
- Test observable behavior and failure cases; check forward and backward parity for tensor math.
- Use real multi-process tests for distributed behavior that mocks cannot exercise.
- Report checks run, results, and remaining gaps. Sign off commits with DCO.

Details: [contributing](CONTRIBUTING.md), [agent instructions](AGENTS.md),
[review policy](skills/pr-review/SKILL.md), and [testing](.agents/contributor-skills/testing/SKILL.md).
