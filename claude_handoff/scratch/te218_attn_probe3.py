import re, time, torch, torch.nn.functional as F
import transformer_engine
from transformer_engine.pytorch import DotProductAttention
torch.cuda.set_device(0)
print("RESULT mapped cudnn:", sorted({l.split()[-1] for l in open('/proc/self/maps') if re.search(r'libcudnn(_graph)?\.so', l)}))
print("RESULT torch cudnn runtime:", torch.backends.cudnn.version())
H, KV, D = 16, 2, 256
def bench(fn, reps=3):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time() - t) / reps * 1e3
for S in (4096, 8192, 32768):
    for mask in ("causal", "padding_causal"):
        dpa = DotProductAttention(H, D, num_gqa_groups=KV, attn_mask_type=mask, qkv_format="bshd").cuda()
        q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, S, KV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, S, KV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kw = dict(attention_mask=torch.zeros(1, 1, 1, S, dtype=torch.bool, device="cuda")) if mask == "padding_causal" else {}
        def step():
            out = dpa(q, k, v, **kw); out.backward(torch.randn_like(out))
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        try:
            ms = bench(step)
            print(f"RESULT TE   S={S:6d} {mask:15s} {ms:8.1f} ms  peak_extra={(torch.cuda.max_memory_allocated()-base)/2**30:6.2f} GiB")
        except Exception as e:
            print(f"RESULT TE   S={S:6d} {mask:15s} FAIL {type(e).__name__}: {str(e)[:150]}")
    qs = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ks = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    vs = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    def sdpa_step():
        out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True); out.backward(torch.randn_like(out))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    ms = bench(sdpa_step)
    print(f"RESULT SDPA S={S:6d} {'causal(flash)':15s} {ms:8.1f} ms  peak_extra={(torch.cuda.max_memory_allocated()-base)/2**30:6.2f} GiB")
