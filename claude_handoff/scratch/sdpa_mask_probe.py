import time, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
H, KV, D = 16, 2, 256
for S in (4096, 8192, 16384):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    q = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key_mask = torch.ones(1, S, dtype=torch.bool, device="cuda"); key_mask[:, S * 3 // 4:] = False  # right padding
    pos = torch.arange(S, device="cuda")
    mask = (pos[None, :] <= pos[:, None])[None, None] & key_mask[:, None, None, :]
    base = torch.cuda.memory_allocated()
    # mirror utils.py: explicit mask => repeat_interleave KV, enable_gqa=False
    kk, vv = k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)
    for name, bes in (("model-default(FLASH,EFF,MATH)", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), ("efficient-only", [SDPBackend.EFFICIENT_ATTENTION]), ("cudnn-only", [SDPBackend.CUDNN_ATTENTION])):
        torch.cuda.reset_peak_memory_stats()
        try:
            with sdpa_kernel(bes):
                torch.cuda.synchronize(); t = time.time()
                out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, is_causal=False)
                out.backward(torch.randn_like(out)); torch.cuda.synchronize()
            print(f"RESULT S={S:6d} {name:30s} ok peak_extra={(torch.cuda.max_memory_allocated()-base)/2**30:6.2f} GiB fwd+bwd={(time.time()-t)*1e3:8.1f} ms")
        except Exception as e:
            print(f"RESULT S={S:6d} {name:30s} FAIL {type(e).__name__}: {str(e)[:90]}")
