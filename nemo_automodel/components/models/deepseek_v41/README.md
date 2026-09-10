# DeepSeek V4.1 Flash

The native backbone follows the [official released inference code and report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40).
The numerical reference is that pinned HF-hosted implementation. The checkpoint
does not contain a Transformers modeling implementation.

The implementation retains all 40 causal encoder/decoder layers, differentiable
cross-layer compressed KV, Full/Reindex/Reuse attention, hierarchical selection,
single-pass mHC, both Engram modules, and the vision encoder/projector. Shared
AutoModel MoE handles experts and dispatch. DSV4 supplies the modality-aware
gate, grouped output projection, FP32 parameter holder, and vision modules.
Engram reuses the Qwen3.8 Flash Next row-owner embedding and its autograd
communications; V4.1 supplies its own tokenizer compression, hashes, and gates.

V4.1 enables the shared MoE's `combine_in_fp32` contract. Routed BF16 expert
outputs are accumulated with the shared expert in FP32, then cast once at the
MoE output. HybridEP dispatch is retained; sparse reverse
all-to-all returns individual expert contributions for FP32 accumulation at
their source. This preserves the released arithmetic at an extra communication
cost relative to HybridEP's usual BF16 combine.

The default `torch_linear` expert backend reuses the shared packed parameter
storage while executing separate gate/up/down projections and eager FP32
SwiGLU. This preserves rounding boundaries from the released implementation.
Fusing those operations introduces small differences that the model's KV
quantization can amplify across layers. The EP128 recipe has three local
experts per rank; this accurate backend executes those projections separately.

## Training configuration

[The HellaSwag recipe](../../../../examples/llm_finetune/deepseek_v41/deepseek_v41_flash_hellaswag_ep128.yaml)
configures full text-backbone fine-tuning, with EP128 and TP1/PP1/CP1. It includes
both trainable Engram tables. The vision tower is disabled for this text recipe.
The model's default configuration retains the vision tower.

FSDP must preserve FP32 mHC coefficients alongside BF16 residual streams:

```yaml
distributed:
  strategy: fsdp2
  ep_size: 128
  activation_checkpointing: true
  moe:
    lm_head_precision: float32
    mp_policy:
      _target_: torch.distributed.fsdp.MixedPrecisionPolicy
      param_dtype: bfloat16
      reduce_dtype: float32
      output_dtype: null
      cast_forward_inputs: false
```

The model owns arithmetic casts and rejects a carried mHC coefficient that has
been lowered to BF16. Engram defaults to WORLD row ownership; the FSDP owner
mesh must contain the same ranks in the same order. Each table is a registered
global `Shard(0)` DTensor, excluded from FSDP all-gathers. Its gradients return
to row owners and use the shared training utility's owner divisor.

Original weights use blockwise FP8 dense matrices, packed FP4 expert matrices,
and rowwise FP8 Engram tables. The adapter decodes these layouts into trainable
floating-point parameters. `checkpoint.dequantize_base_checkpoint: true`
enables released-weight initialization through the checkpoint engine. Exports
use released parameter names with unquantized trained weights. Owner-local
Engram conversion preserves logical rows and removes only allocation padding.

Window KV retains FP8 quantize/dequantize, compressed KV retains NVFP4, and
index Q/K retains MXFP4. These forward boundaries also apply when reference
weight compute is BF16. Training uses straight-through activation gradients.

## Validation runners

The functional runners under `tests/functional_tests/models/deepseek_v41`
record the pinned oracle hashes, source-tensor audits, inputs, and numerical
results. `run_reference_parity.py` compares pretrained prefixes at every token
and vocabulary position. `run_engram_reference_parity.py` separately checks
hashes, selected released rows, outputs, and gradients. `run_training.py`
exercises the actual EP/FSDP model construction and strict checkpoint resume,
including optimizer, scheduler, RNG, and the stateful dataloader.

The released-weight prefix comparison passes with bitwise-identical logits for
the first four layers, 4,096 tokens, and all 129,280 vocabulary entries. Mean,
P95, and maximum KL are zero; top-1 agreement is 100%. The unchanged official
TP1 reference receives one valid sample. Native EP4/Engram-owner4 uses HybridEP,
`torch_linear`, and TileLang; rank 0 receives that sample and the other ranks
participate with masked inputs, preserving the same global batch. All four
native state audits load 110/110 entries with no missing or mismatched tensors.

The contiguous decoder window at layers 20–24 also produces bitwise-identical
intermediate activations and logits over 4,096 tokens and the complete vocabulary.
That comparison starts from the unchanged official encoder's 20-layer prefix
state; it is a decoder-window comparison, not a complete native 25-layer run.

A continuous 40-layer comparison also passes with identical logits and all 205
observed intermediate tensors over 4,096 tokens. It streams one unchanged native
EP4/FSDP block at a time, carrying its own residual streams, mHC coefficients,
and CSA2 state through the entire backbone. Shared embedding, norm, and head
parameters are replicated. All four ranks load 1,006/1,006 native state entries
and pass parameter dtype audits matching the recipe's BF16 storage and protected
FP32 tensors. This tests the complete forward computation with streamed weights;
resident full-model training requires a separate multi-node run.

A reduced-width, 40-layer EP2 run with both Engram modules completes 50 training
steps and restores its step-25 checkpoint exactly, including model, Adam,
scheduler, RNG, dataloader, and progress. Resumed numerical trajectories are not
bitwise reproducible with TileLang backward. An independent uninterrupted run
from an exactly matching initial state also diverges numerically. The runner
reports strict trajectory failure without applying a numerical tolerance;
checkpoint restoration and finite-gradient checks are reported separately.

The same 50-step/step-25-resume schedule also restores the recipe's TE FusedAdam
state exactly, including FP32 moments and the BF16 parameters' int16 master
remainders on both Engram owners. Its resumed trajectory remains non-exact and
is reported separately; the AdamW repeat is not a numerical bound for FusedAdam.

To reproduce the first-four-layer comparison, set `DS41_CHECKPOINT` to the
complete pinned snapshot, including its unchanged `inference/` directory.
Run the official TP1 stage and native EP4 stage successively on a four-GB200
node. The saved reference artifact includes the exact input and source hashes.

```bash
mkdir -p parity-results
python tests/functional_tests/models/deepseek_v41/run_streaming_reference_parity.py \
  --mode reference --checkpoint "$DS41_CHECKPOINT" \
  --reference-dir "$DS41_CHECKPOINT/inference" --num-layers 4 --sequence-length 4096 \
  --artifact parity-results/official.safetensors \
  --output parity-results/official.json
torchrun --standalone --nproc-per-node 4 \
  tests/functional_tests/models/deepseek_v41/run_streaming_reference_parity.py \
  --mode native --checkpoint "$DS41_CHECKPOINT" \
  --reference-dir "$DS41_CHECKPOINT/inference" --num-layers 4 --sequence-length 4096 \
  --attention-backend tilelang --expert-backend torch_linear \
  --native-global-batch-one --artifact parity-results/official.safetensors \
  --output parity-results/native-ep4.json
```

For the continuous 40-layer comparison, generate a separate reference artifact
with `--num-layers 40`, then run `run_streaming_backbone_parity.py` under the same
four-process launcher with `--checkpoint`, `--artifact`, and `--output`. Its
defaults select 40 layers, 4,096 tokens, and the recipe's numerical backends.

Prefix parity covers only the configured prefix. Decoder-window and reduced
candidate-pool diagnostics must be identified separately from pretrained
end-to-end logits. A short structural training test does not establish
full-checkpoint convergence.

## Scope

The supported training path uses unpacked, zero-based, right-padded sequences,
`torch` linears, `torch_fp32` RMSNorm, and eager, SDPA, or TileLang attention. The
TileLang path reuses DSV4's sparse forward/backward kernels and retains their
BF16 probability boundary. TP, PP, CP,
packed sequences, incremental generation caches, and bounded decoder replay are
not implemented. Indexers and the eager/SDPA paths materialize dense scores or masks; million-token
context training is not a validated execution path.

Released indexers remain frozen during backbone fine-tuning because hard top-k
provides no language-model gradient. Indexer distillation is outside this port.
DSpark drafts are trained separately in the report and are excluded from the
backbone model and its checkpoint audit. The recipe uses Adam for fine-tuning;
it does not reproduce the report's momentum/Sinkhorn Engram pretraining optimizer.

`DeepseekV41Processor` supports ordinary text/image conversations and HF
AutoProcessor save/reload. For tool or reasoning-specific message encoding, use
the official encoder and pass its rendered text to the processor.
