import time, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from nemo_automodel.components.attention.utils import preprocess_args_and_kwargs_for_attn
B, H, KV, D, S = 2, 16, 2, 256, 16384
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, KV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, KV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = torch.ones(B, S, dtype=torch.bool, device="cuda"); mask[1, S // 3:] = False  # right padding on row 1
qt, kt, vt, kw = preprocess_args_and_kwargs_for_attn(q, k, v, attention_mask=mask, attn_impl="sdpa")
print("RESULT kwargs:", {k_: (tuple(v_.shape) if torch.is_tensor(v_) else v_) for k_, v_ in kw.items()})
torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    for _ in range(2):
        torch.cuda.synchronize(); t = time.time()
        out = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True, **kw); out.backward(torch.randn_like(out))
        torch.cuda.synchronize(); dt = time.time() - t
print(f"RESULT flash-only ok: fwd+bwd={dt*1e3:.1f} ms peak_extra={(torch.cuda.max_memory_allocated()-base)/2**30:.2f} GiB")
