# Qwen Image Edit 2511 Implementation Status

Last updated: 2026-07-22 UTC

## Objective

Port cached, full-parameter `Qwen/Qwen-Image-Edit-2511` training into the current NeMo AutoModel diffusion architecture, including generic Hugging Face image-edit preprocessing, a versioned cached dataset, Qwen-specific encoding and flow-matching adaptation, FSDP2, checkpoint/resume coverage, deterministic MagicBrush protocols, performance metrics, tests, and documentation.

## Completed implementation

- Added repeatable Hugging Face media-role mappings (`target`, `context`, `condition`), dataset revision forwarding, source-column deduplication, and a generic ordered image-edit manifest.
- Added the `image-edit` preprocessing CLI and configurable model-owned processor target.
- Added a versioned generic cached image-edit dataset using `BaseMultiresolutionDataset`, `SequentialBucketSampler`, and `StatefulDataLoader`.
- Added strict cache/tensor/provenance validation, compound target/context bucketing, padded prompt collation, named conditioning tensors, and CPU token-count metadata.
- Rejected unsafe dynamic image-edit batch sizing until a compound target/context cost policy is implemented.
- Added `QwenImageEditCacheEncoder` for deterministic Qwen VAE latent modes and Qwen2.5-VL prompt conditioning.
- Matched Diffusers' Qwen vision-conditioning resize policy and retained aspect-preserving or fixed-square VAE preprocessing modes.
- Added `QwenImageEditAdapter` with upstream-compatible 2x2 latent packing, ordered target/context concatenation, Qwen shape metadata, timestep scaling, target-only prediction slicing, and float32 flow loss.
- Added a model-owned Qwen FSDP2/activation-checkpoint strategy covering attention, image MLP, and text MLP branches without prototype regrouping or special last-layer resharding.
- Added canonical typed `flow_matching.adapter._target_` construction while preserving legacy `adapter_type` recipes.
- Added deterministic beta timestep sampling through a checkpointed rank-local `torch.Generator` without mutating global PyTorch RNG state.
- Updated the diffusion recipe to construct adapters before parallel planning, restore generator state, use correct FSDP/DDP gradient-sync contexts, and report sample/token throughput.
- Made scheduler and checkpoint accounting use completed optimizer steps, including exact 20-step warm-up plus 100-step timed benchmark windows.
- Added benchmark warm-up exclusion, CUDA-event phase timing, dataloader-wait timing, profiler ranges, logging-overhead exclusion, and explicit peak allocated/reserved memory plus gradient-clipping timing. NCCL profiler events remain the source for separately attributed communication time because communication overlaps backward.
- Added strict sparse-optimizer restore support for parameters that legitimately have no Adam state yet. The loader validates checkpoint metadata, rejects unexpected or partial state groups, and prunes only wholly absent lazy state groups before Distributed Checkpoint restore.
- Isolated `StatefulDataLoader` worker/base-seed generation behind a dedicated rank-local CPU generator. Iterator creation and resume no longer advance global CPU PyTorch RNG state.
- Centralized the generic image-edit cache schema version in `nemo_automodel.shared.image_edit_cache`; the Qwen writer and generic reader now share one owner.
- Removed the unused public `ModelAdapter.attach_pipeline()` no-op and its unconditional recipe call.
- Moved diffusion checkpoint construction to the typed `RecipeConfig.checkpoint` owner. Pre-shard model keys are runtime-only `build(...)` input owned by `Checkpointer`, while advanced declarative checkpoint fields are preserved.
- Hardened consolidated Diffusers export provenance: exact pinned snapshots and component directories are preferred, root Hugging Face weights retain precedence, and both Transformers and Diffusers safetensors indexes are supported. The pinned Qwen transformer resolves 1,933 keys across its original five shards.
- Hardened sparse-optimizer parameter-name normalization so only exact distributed wrapper path components are removed; legitimate names such as `submodule.weight` remain distinct and collision-checked.
- Pinned and validated immutable Hugging Face provenance end to end: MagicBrush revision `1d8d4629150d18ca50afab66391866f2085be989` and Qwen-Image-Edit-2511 revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` are forwarded into loading and recorded in cache metadata.
- Added the eight-GPU BF16 recipe at `examples/diffusion/finetune/qwen_image_edit_2511_flow.yaml`.
- Updated Qwen diffusion documentation and the model support table with pinned MagicBrush preprocessing, cache semantics, training/resume, benchmark protocol, license attribution, and scope exclusions.

## Local validation evidence

- Current integrated unit selection covering the Diffusers bridge, checkpointing, image-edit data/export/CLI, flow matching, Qwen model code, recipe metrics, and RNG: **661 passed, 14 skipped** in 12.55 seconds.
- Full checkpoint unit directory, including exact-revision Diffusers mapping, typed runtime ownership, strict sparse-optimizer metadata validation, and lazy-state restore: **261 passed, 12 skipped**.
- Real two-rank NCCL/FSDP2 tests cover gradients, activation checkpointing off/full, production Distributed Checkpoint save/load, a fresh empty resumed Adam optimizer, restored state, and continued training: **2 passed** in 35.10 seconds.
- Benchmark-accounting, typed checkpoint construction, and rendered-metric-log coverage: **46 passed**; the log distinguishes peak allocated from peak reserved memory and reports gradient-clipping time.
- The pinned 64-example MagicBrush development cache was generated with eight H100s and verified. It contains 64 finite target/context/prompt samples, one compatible bucket, the exact dataset/model revisions above, dataset config `default`, and aspect-preserving `max_pixels=1048576` preprocessing metadata.
- An eight-H100 BF16/FSDP2 run completed 10 finite optimizer steps and wrote `/tmp/qwen-image-edit-validation/checkpoints/split/epoch_1_step_10` with model, optimizer, scheduler, sampler, dataloader, and RNG state. Peak allocated memory was 23.57 GiB per the globally reduced metric.
- Restoring that checkpoint initially exposed six last-block text parameters that had legitimately never received gradients and therefore had no Adam state. The strict sparse-optimizer fix above restored all materialized state and resumed through step 20 with finite loss and gradient norm, writing `epoch_2_step_20`.
- A separate uninterrupted 20-step eight-H100 control completed successfully. Across resumed steps 11-20, the maximum printed loss difference was `0.000281` (`0.435%` relative). Comparing all 20,430,401,088 model elements gave maximum absolute difference `6.8665e-05` and mean absolute difference `1.6617e-08`; BF16/FSDP reductions are not expected to be bitwise identical across process restarts.
- After adding the dedicated dataloader generator, the real step-10 checkpoint was resumed again through step 20. On all eight ranks, Python RNG, NumPy RNG, global CPU PyTorch RNG, every CUDA RNG state, and the dedicated flow-matching generator state match the uninterrupted control exactly. Optimizer metadata and parameter-group/scalar structure also match exactly; sampled moment tensors show only small cross-run numerical variation.
- The pinned 1,024-example MagicBrush training cache was generated and verified with eight H100s. All target/context/prompt tensors are finite, all samples carry the exact pinned revision, and the fixed-square cache has one compound bucket with target and context latent shapes `[16, 128, 128]`.
- Three independent eight-H100 baseline runs completed 20 warm-up plus 100 measured optimizer steps with finite loss and gradient norms. No benchmark checkpoint artifacts were written.

| Run | Samples/s | Step time | Target latent tokens/s | Total latent tokens/s | Peak allocated | Peak reserved |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.02 | 2.646 s | 12,383.59 | 24,767.19 | 23.57 GiB | 45.97 GiB |
| 2 | 3.03 | 2.640 s | 12,413.68 | 24,827.35 | 23.57 GiB | 45.97 GiB |
| 3 | 3.07 | 2.607 s | 12,568.00 | 25,136.01 | 23.57 GiB | 45.97 GiB |
| **Median** | **3.03** | **2.640 s** | **12,413.68** | **24,827.35** | **23.57 GiB** | **45.97 GiB** |

- Median phase time per measured step was 0.000 s dataloader wait (rounded), 0.729 s forward, 1.718 s backward including overlapped FSDP communication, 0.081 s gradient clipping, and 0.061 s optimizer work.
- A one-step eight-H100 run successfully wrote a Distributed Checkpoint and reported successful five-shard Diffusers-compatible consolidation at `/tmp/qwen-image-edit-validation/checkpoints/consolidated-reload/epoch_0_step_1/model/consolidated`.
- Targeted `ty` validation passes for the new image-edit dataset, flow-matching config/pipeline/adapter APIs, Qwen model package, and checkpointed RNG extensions when pointed at the runtime Python environment.
- Fern content/navigation validation passes: 292 MDX files validated and repository-pinned Fern 5.29.0 reported 0 errors. Redirect validation was skipped because this workspace has no Fern authentication token.
- The direct Fern Make target could not fetch `docs-archive` over the configured SSH remote and lacks a global `fern` executable. Validation used the existing local `origin/docs-archive` ref and the repository-pinned CLI instead; these were environment limitations, not content failures.

## Remaining-work closure (2026-07-22)

All five items from the previous stop were completed on the same eight-H100 node. The `/tmp/qwen-image-edit-validation` artifacts from 2026-07-21 had been wiped between sessions, so the development cache and consolidated export were regenerated from the pinned revisions before the reload check.

1. **Consolidated reload verified.** The 64-example MagicBrush development cache was regenerated with the documented pinned-revision command (aspect-preserving `max_pixels=1048576`, dataset revision `1d8d4629…`, model revision `6f3ccc0b…`, `--verify`, 64/64 samples). A fresh one-step eight-H100 BF16/FSDP2 run produced a finite step (loss 0.009428, gradient norm 0.691) and exported the five-shard Diffusers-compatible consolidation. `QwenImageTransformer2DModel.from_pretrained(...)` reloaded the consolidated directory: all **1,933 tensors** (20,430,401,088 elements) match the pinned base snapshot's keys, shapes, and dtypes, and every value is finite.
2. **Two-rank functional module re-run.** `tests/functional_tests/models/qwen_image_edit/test_fsdp_distributed.py`: **2 passed** in 31.74 seconds after all audit fixes, including the provenance-read fix below.
3. **Diffusers deprecation warning removed.** `Checkpointer._get_original_model_path` now reads `name_or_path`/`_name_or_path` from the already-read config before probing the model, so the Diffusers config-proxy `__getattr__` deprecation path is never entered. A new unit test asserts the model attributes are not probed when config provenance exists, and the fresh one-step training log contains no deprecation warning.
4. **Final gates green.** `ruff format .` (one file reformatted) and `ruff check --fix .` pass repository-wide; `git diff --check` is clean; the read-only workspace audit confirmed no new untracked files beyond the branch's intended additions. The integrated unit selection across the Diffusers bridge, checkpointing, image-edit data/export/CLI, flow matching, Qwen model code, recipe metrics, and RNG now reports **991 passed, 12 skipped** in 64.7 seconds.
5. **MDX re-check not required.** Documentation was not modified in this session; the Fern environment caveat above still applies to full redirect validation.

Environment note: the runtime environment needed the `diffusion-media` extra re-synced (`uv run --extra diffusion-media …`) to restore `cv2` for the preprocessing CLI; `pyproject.toml` and `uv.lock` were not modified.

## Current validation environment

- Eight H100 80GB GPUs were used for cache regeneration, the one-step training/consolidation run, the consolidated reload verification, and the two-rank functional module.
- No training, preprocessing, consolidation, or benchmark process remains active.
- The benchmark numbers above are a validated baseline, not an optimization winner; no backend or kernel sweep was run.

## Workspace safety

- `pyproject.toml` and `uv.lock` were already modified before this task and are user-owned changes. Do not overwrite or revert them.
- Prototype PR reference used for comparison: original PR #2351 commit `b5acfa4317c3f5f3ac4d8963953c1b7900710320`, fetched locally as `origin/pr-2351-original`.
