# K3 vision and packed context-parallel validation

These checks distinguish vision-component parity, adversarial packed CUDA kernels,
and native distributed checkpoint loading/training. Component tests do not certify
native loading.

## Native preprocessing

The native VLM recipe accepts a model-owned tokenizer through its packing config:

    packed_sequence:
      pack_size: 1024
      max_length: 1025
      tokenize_hook_fn: nemo_automodel.components.models.kimi_k3.processing.tokenize_conversation

This hook emits unshifted, expanded image tokens. The native packer shifts each
document once. Shuffle source documents with a recorded seed before packing and
reuse exactly the same frozen packs across parallel configurations. Set
dataset.inject_fake_images to false when testing actual text-only documents.

## Required FLA source corrections

The validated environment used FLA 0.4.2 from the 26.10 RC5 arm64 container, with
two corrections included in dependencies/:

- CP state backward: include the document start and head offsets when loading
  each head's terminal key-decay value. Without this correction, the independent
  two-head gradient regression fails.
- FP32 KDA triangular solve: honor FLA_TRIL_PRECISION=ieee. The prior hard-coded
  TF32 path does not pass the independent FP64 recurrence gradient check.

These patches are external dependency corrections, not changes automatically
installed by importing AutoModel. The overlay helper checks the exact original
and patched file hashes and refuses different sources. It copies the package;
it does not edit the installed environment. A version number alone is insufficient
to establish that these fixes are present.

    python tests/functional_tests/models/kimi_k3/dependencies/prepare_fla_overlay.py       --source /absolute/path/to/installed/fla       --output /absolute/path/to/new/fla-overlay
    export PYTHONPATH=/absolute/path/to/new/fla-overlay:$PWD:$PYTHONPATH
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0
    export TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee
    export FLA_DISABLE_TENSOR_CACHE=1

Disabling FLA's process-global tensor cache is required for interleaved pipeline
microbatches with selective recomputation. AutoModel retains CP contexts and MLA
masks on the owning microbatch.

## CUDA regression commands

Run inside an allocated GPU node, from the repository root. Each invocation
needs a separate artifact directory; create it before launch.

    export K3_RUN=/absolute/path/to/artifacts
    export K3_KDA_HEAD_DIM=128 K3_RETAIN_CP_STATE=1 K3_PARITY_DTYPE=float32
    torchrun --standalone --nproc_per_node=2       tests/functional_tests/models/kimi_k3/run_packed_vision_cp.py
    torchrun --standalone --nproc_per_node=2       tests/functional_tests/models/kimi_k3/run_cp_decay_gradient.py
    pytest -q tests/functional_tests/models/kimi_k3/test_training_convolution.py

Repeat packed checks with four ranks and with K3_PARITY_DTYPE=bfloat16. They cover
image spans crossing CP cuts; document ends and starts immediately before, at,
and after cuts; singleton documents; unaligned lengths; and all-padding ranks.
They check independently processed documents, positions, shifted labels, every
parameter gradient, pixel gradients, poisoning isolation, and a deliberately
erased-boundary negative control.

FP32 uses strict per-parameter gradient comparisons. BF16 records those same
pairwise metrics but gates vision, language, and pixel gradients separately
against a same-rounded-weight FP32 control: CP error must not exceed 1.25 times
the CP1 precision error, with an absolute relative-error ceiling of 5%. The
unfixed CP decay kernel fails this gate. This is a calibrated precision test,
not a claim of exact BF16 gradient equality.

For vision-only HF parity:

    python tests/functional_tests/models/kimi_k3/run_vision_parity.py       --reference /absolute/path/to/pinned/HF/source-and-checkpoint       --output /absolute/path/to/vision-parity.json --real-checkpoint

The reference must be audited and its source/checkpoint revisions recorded.

## Native training evidence and limits

HSG artifacts are under
/lustre/fsw/portfolios/coreai/users/huiyingl/k3/vision_20261024.
The runs use the production VLM recipe, distributed loading, FSDP2 initialization,
backward, clipping, and optimizer step. Read-only observers inspect loaded state
and export comparison data; they do not load or repair model weights.

- Full 27-layer vision tower and projector: BF16 HF output, input-gradient, and
  parameter-gradient parity is exact for eager and actual FlashAttention-2.
- Four real decoder layers plus vision: FP32 HF maximum logit error 3.62e-5
  across 506 valid tokens. The independent HF graph consumes AutoModel-exported
  dequantized LM weights, so this does not independently certify dequantization.
- The whole four-layer BF16 HF graph does not meet the strict parity gate
  (mean KL 0.005, maximum KL 0.091). Do not generalize the vision-only result.
- Native four-layer EP8 CP1/2/4 all-parameter training ran. Unmodified FP32 CP
  logits do not pass the strict gate: near-tied MoE routes change. Vision
  gradient relative errors are 7.31e-5 and 5.79e-5 for CP2/4. A separately
  labeled routing-replay diagnostic passes strict logits; it is not an
  unmodified end-to-end pass.
- Native 13-layer PP2 EP8 CP2 packed and PP2 EP8 CP1 nonpacked training ran with
  an actual 12+1 decoder split. The first 12 decoder layers and token embeddings
  were frozen; vision, layer 12, and the output head were trainable.
- Four-layer EP2 PP2 versus EP4 PP1 uses an explicit 2+2 decoder split with the
  checkpoint's original residual-block size 12. Results are recorded separately
  after both native runs finish.
- The full 93-layer model run was canceled by the user and was never submitted.

Runtime: PyTorch 2.14.0a0+4fdf77b940.nv26.08 / CUDA 13.4; native Transformers
5.18.0; HF comparison Transformers 4.56.2. HF source revision:
f831ab66814297da540d832a5235f8e904f29d06. Checkpoint revision:
301be1b88c89c0d3a763da6301352cb8fe399e90. HF compatibility corrections and precision
settings are retained in the HSG artifacts rather than treated as an unmodified
reference.
