import time, torch, torch.nn.functional as F
from transformer_engine.pytorch import DotProductAttention
torch.cuda.set_device(0)
H, KV, D = 16, 2, 256
def bench(fn, reps=3):
    fn(); torch.cuda.synchronize()  # warmup
    t = time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time() - t) / reps * 1e3
for S in (4096, 8192, 32768):
    for mask in ("causal", "padding_causal"):
        dpa = DotProductAttention(H, D, num_gqa_groups=KV, attn_mask_type=mask, qkv_format="bshd").cuda()
        q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, S, KV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, S, KV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kw = {}
        if mask == "padding_causal":  # mirror the model: key padding mask [B,1,1,S], True = masked out
            kw = dict(attention_mask=torch.zeros(1, 1, 1, S, dtype=torch.bool, device="cuda"))
        def step():
            out = dpa(q, k, v, **kw); out.backward(torch.randn_like(out))
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        try:
            ms = bench(step)
            print(f"RESULT TE   S={S:6d} {mask:15s} {ms:8.1f} ms  peak_extra={(torch.cuda.max_memory_allocated()-base)/2**30:6.2f} GiB")
        except Exception as e:
            print(f"RESULT TE   S={S:6d} {mask:15s} FAIL {type(e).__name__}: {str(e)[:120]}")
    qs = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ks = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    vs = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    def sdpa_step():
        out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True); out.backward(torch.randn_like(out))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    ms = bench(sdpa_step)
    print(f"RESULT SDPA S={S:6d} {'causal(flash)':15s} {ms:8.1f} ms  peak_extra={(torch.cuda.max_memory_allocated()-base)/2**30:6.2f} GiB")
