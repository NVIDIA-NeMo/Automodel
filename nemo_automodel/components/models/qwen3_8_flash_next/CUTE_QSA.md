# CuTe QSA on SM90

Select the backend through the existing model configuration:

```yaml
model:
  backend:
    attn: cute
```

For example, use the existing
`examples/llm_finetune/qwen/qwen3_8_flash_next_180b_hellaswag_ep64.yaml`
with `--model.backend.attn cute` in its normal scheduler launch.
The selection affects Qwen3.8-Flash-Next QSA layers. Gated DeltaNet layers
continue to use FLA. The default backend, indexer and checkpoint tensor layout
are unchanged.

## Implementation

- `cute_qsa.py`: validates the SM90/BF16/D256 contract and lazily loads the
  optional kernels through `safe_import`. CPU model dispatch keeps the
  existing numerical oracle without importing CuTe.
- `_cute_qsa.py`: builds one bit per physical K/V token in each query's
  route set, supplies a CuTe mask callback, and owns first-order autograd.
- `_cute_qsa_metadata.py`: classifies 128-query by 64-key tiles as absent,
  partial or full, then builds forward and reverse block lists on the GPU.
- FlashAttention's CuTe SM90 implementation supplies attention forward/backward
  arithmetic. Its sources are an external BSD-3-Clause dependency, not vendored
  into this branch.

Duplicate route IDs select a token once. Negative and out-of-range IDs are
ignored before converting int64 IDs to int32. Fully masked query rows and their
query gradients are zero; unselected K/V rows receive zero gradients.
Causality, packed documents and physical padding gaps are encoded by the
indexer's route IDs. Local query slices can reference gathered global K/V.
No device scalar is read on the host during route preprocessing.

The bitmap uses `4 * batch * queries * ceil(keys / 32)` bytes.
The two model-owned compile caches each retain at most 32 compiled callables,
without retaining input or output tensors.

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
share one device; arbitrary strides are accepted. The route bitmap is limited
to 48 KiB of shared memory per query and fewer than 2^31 total bitmap words.

The backend supports first-order autograd and non-reentrant activation
checkpointing. Backward uses atomic reductions and is not bitwise deterministic;
deterministic-algorithm mode is rejected explicitly. Higher-order gradients and
outer torch.compile/CUDA-graph integration are not advertised by this change.

The packed/CP attention contract is tested with disjoint document ranges,
physical gaps, differing Q/K lengths and local-query gradient composition.
The existing model-owned CP collectives are unchanged. A new multi-node
packed-CP training run is not part of this branch's validation.

## Validation

From the repository root, in a compatible environment:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit_tests/models/qwen3_8_flash_next -q
python -m pytest tests/functional_tests/models/test_qwen3_8_flash_next_cute_qsa.py -q
```

The H100 suite checks output and all Q/K/V gradients against independent dense
FP64 attention, with a 0.4% relative-L2 bound established by the original BF16
kernel validation. It covers int32/int64 routes, large invalid IDs, duplicates,
empty rows and tiles, full tiles, strided inputs, physical gaps, checkpoint
recomputation, a non-default CUDA stream, local query slices, 4K route shapes,
and actual QSA-layer parameter gradients versus the PyTorch oracle.

The earlier 12.05% EP64 MFU result came from a different experiment checkout
with additional performance settings and a warm autotune cache. It is not a
measurement of this isolated branch, which contains only CuTe QSA integration.
