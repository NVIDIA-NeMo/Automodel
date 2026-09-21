# Starting point: it does not run

```bash
torchrun --nproc-per-node 8 benchmark.py --config moonlight_v4_16b_hf.yaml
```

`transformers` ships a DeepSeek-V4 implementation. Pointed at the full model on 8 H100s it fails
before finishing one forward pass, at sequence 2048, 1024 and 512 alike.

**Why: the sparse attention is not actually sparse.**

```python
if self.compressor is not None:                   # compressed-sparse layer
    kv = torch.cat([kv, compressed_kv], dim=2)    # extend the key axis
attn_output, _ = attention_interface(self, q, kv, kv, attention_mask, ...)
```

The compressed entries are appended to the keys, one dense attention runs over the whole extended
length, and sparsity is applied only as a **mask**. Nothing is skipped. A layer that should be cheap
becomes more expensive than a dense one. There is also no FlashAttention for this model's
512-dimensional heads, so the score matrix is materialised and its softmax runs in fp32.

| per layer, one sequence of 2048 | memory |
| --- | ---: |
| attention probabilities, fp32, kept for the backward pass | 336 MB |
| the same layer computing only its 128-token window | 17 MB |

Across 27 layers that is 9.1 GB of probabilities alone, before weights, activations or gradients.
Halving the sequence length does not save you: the term is quadratic, but it is competing against
everything else that is linear.

**This is the lesson in miniature.** The implementation is correct. It was written for inference,
where you generate one token at a time and none of this is on the critical path.
