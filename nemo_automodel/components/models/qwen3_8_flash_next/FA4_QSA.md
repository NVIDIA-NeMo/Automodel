# FA4 QSA on SM90

Select FA4 through the existing `cute` setting in the model-owned runtime backend configuration:

```yaml
model:
  backend:
    _target_: nemo_automodel.components.models.qwen3_8_flash_next.backend.Qwen3_8_FlashNextBackendConfig
    attn: cute
```

For example, change the backend target and attention selection in
`examples/llm_finetune/qwen/qwen3_8_flash_next_180b_hellaswag_ep64.yaml`,
retaining its other backend settings, then use its normal scheduler launch.
Plain backend mappings passed to the AutoModel factory are resolved by the
model's existing `backend_config_resolver` hook. Existing shared
`BackendConfig` instances remain accepted. The shared configuration is
unchanged: only this model's configuration declares the `"cute"` option.
The selection affects Qwen3.8-Flash-Next QSA layers. Gated DeltaNet layers
continue to use FLA. The default backend, indexer and checkpoint tensor layout
are unchanged.

## Implementation

One model-local file, `fa4_qsa.py`, validates the SM90/BF16/D256 contract,
lazily loads optional dependencies through `safe_import`, and calls FA4's
public `flash_attn_func`. FA4 owns attention forward, backward and autograd.

PyTorch constructs a uint8 membership table and forward/reverse block lists
for 128-query by 80-key forward tiles and 128-query by 64-key backward
tiles, matching the pinned FA4 public API defaults. Only these discrete operations are lazily
compiled with `torch.compile(dynamic=False)`; model layers are
not compiled. The small CuTe `mask_mod` callback reads membership inside FA4.
There are no model-owned CuTe preprocessing kernels or custom autograd classes.

Duplicate IDs select a token once. Invalid IDs write a separate padding column
and never overwrite a valid token's membership. Empty query rows have zero
output and query gradients. Routes encode causality, documents, physical
padding gaps and local-query/global-KV CP coordinates. No additional triangular
mask or host read of a device scalar is introduced.

The byte table occupies `batch * queries * round_up(keys + 1, 32)` bytes.
Compile caches retain code, not input/output tensors. PyTorch may execute
preprocessing eagerly after exhausting its shape-specialization budget; this
preserves correctness for varying sequence/stride layouts without changing
process-global compiler settings. Route membership and block
lists are rebuilt from each call's IDs, including checkpoint recomputation.
The historical `attn: cute` selector is retained for existing recipes; shared
BackendConfig and its defaults remain unchanged.

## Dependencies

The tested attention source matches `docker/Dockerfile` at the branch base:

- Dao-AILab/flash-attention commit
  `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`, package `flash_attn.cute`.
- `nvidia-cutlass-dsl==4.6.2`.
- `quack-kernels==0.6.4`.
- `apache-tvm-ffi==0.1.11`, matching AutoModel's existing compatibility cap.
- The container's Torch 2.13 development build and CUDA Python bindings.

The installed AutoModel CUDA image should supply these dependencies.
For an existing compatible environment missing only the FA4 package:

```bash
uv pip install --no-deps   "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820#subdirectory=flash_attn/cute"
```

The `--no-deps` installation matches the existing Docker build: FA4 declares
TVM FFI >=0.1.12, while this AutoModel revision caps it at 0.1.11 for TileLang
compatibility. This QSA forward/backward suite was explicitly tested against
0.1.11. Do not upgrade the shared FFI dependency merely to enable this backend.

## Scope

CUDA execution requires SM90, BF16, head dimension 256, positive dimensions,
matching K/V shapes and an integral positive Q-head/KV-head ratio. Inputs must
share one device; arbitrary strides are accepted. The padded byte mask must
contain fewer than 2^31 elements for the FA4 auxiliary-indexing contract.

The backend supports first-order autograd and non-reentrant activation
checkpointing. Backward uses atomic reductions and is not bitwise deterministic;
deterministic-algorithm mode is rejected explicitly. Higher-order gradients and
outer layer/model torch.compile or CUDA-graph integration are not advertised by this change.

The packed/CP attention contract is tested with disjoint document ranges,
physical gaps, differing Q/K lengths and local-query gradient composition.
The existing model-owned CP collectives are unchanged. A new multi-node
packed-CP training run is not part of this branch's validation.

## Validation

From the repository root, in a compatible environment:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit_tests/models/qwen3_8_flash_next -q
python -m pytest tests/functional_tests/models/test_qwen3_8_flash_next_fa4_qsa.py -q
```

The H100 suite checks output and all Q/K/V gradients against independent dense
FP64 attention, with a 0.4% relative-L2 bound established by the original BF16
kernel validation. It covers int32/int64 routes, large invalid IDs, duplicates,
empty rows and tiles, full tiles, strided inputs, physical gaps, checkpoint
recomputation, a non-default CUDA stream, local query slices, 4K route shapes,
and actual QSA-layer parameter gradients versus the PyTorch oracle.

The earlier 12.05% EP64 MFU result came from a different experiment checkout
with additional performance settings and a warm autotune cache. It is not a
measurement of this isolated branch, which contains only FA4 QSA integration.
