# `tests/` — test suites

Adds to the repository root `AGENTS.md`.

**Load the `testing` skill before adding, moving, or disabling a test.**

```
tests/
  unit_tests/        # fast, isolated, CPU-only
  functional_tests/  # GPU-required integration tests
  ci_tests/          # executed in the CI pipeline only
  integration/       # cross-component integration
  capability_registry/
```

Unit tests must run without a GPU. Mark GPU tests with `@pytest.mark.gpu` or
skip on `not torch.cuda.is_available()`.

## Tiers

| Tier | Trigger | Blocking |
|---|---|---|
| L0 | Every PR | Yes |
| L1 | PRs labeled `needs-more-tests`, scheduled | Yes |
| L2 | Scheduled only | Yes when triggered |

**Prefer unit tests over functional tests.** CI GPU capacity is limited and
functional tests are capped at **2 GPUs** per PR.

## Writing tests that can fail

- Assert on values, not on the absence of an exception. "It loaded" and "it ran"
  are the two weakest assertions in this repo and both hide real bugs.
- Keep configs tiny: small hidden dims, 1–2 layers, short sequences.
- Guard config overrides. `setattr(config, key, value)` on a dataclass that has
  no such field silently creates a phantom attribute — the test passes and the
  recipe breaks for a real user:

  ```python
  if not hasattr(config_obj, key):
      raise ValueError(f"Config has no field '{key}'")
  setattr(config_obj, key, value)
  ```

- Set `CUDA_VISIBLE_DEVICES` and `--master_port` explicitly for multi-GPU runs
  to avoid device and port collisions.

## Thresholds

A numerical threshold is a claim about the code, not a knob. When a comparison
fails, find the cause before widening the tolerance; a threshold loosened to
make CI green stops testing anything.
