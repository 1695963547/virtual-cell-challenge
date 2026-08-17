#!/usr/bin/env python
"""验证 diff attention 的 sdpa 等价替换：数学等价性 + 18080 seq 内存/速度"""
import torch
import torch.nn.functional as F
import time

torch.manual_seed(0)
dev = 'cuda:0'
bsz, num_heads, head_dim = 4, 8, 16
embed_dim = num_heads * head_dim  # 128
scaling = head_dim ** -0.5


def orig_attn(q1, k1, q2, k2, v, lambda_full):
    """原实现：显式 (B,H,S,S) 矩阵"""
    attn_weights_1 = torch.matmul(q1, k1.transpose(-1, -2))
    attn_weights_2 = torch.matmul(q2, k2.transpose(-1, -2))
    attn_weights_1 = torch.nn.functional.softmax(attn_weights_1, dim=-1, dtype=torch.float32).type_as(attn_weights_1)
    attn_weights_2 = torch.nn.functional.softmax(attn_weights_2, dim=-1, dtype=torch.float32).type_as(attn_weights_2)
    attn_weights = attn_weights_1 - lambda_full * attn_weights_2
    attn = torch.matmul(attn_weights, v)
    return attn


def sdpa_attn(q1, k1, q2, k2, v, lambda_full):
    """新实现：两次 sdpa 再相减（matmul 线性性等价）"""
    attn1 = F.scaled_dot_product_attention(q1, k1, v)
    attn2 = F.scaled_dot_product_attention(q2, k2, v)
    attn = attn1 - lambda_full.unsqueeze(-1).unsqueeze(-1) * attn2
    return attn


# ---- 1. 数学等价性（小 seq）----
seq = 64
x = torch.randn(bsz, seq, embed_dim, device=dev)
q1 = torch.randn(bsz, num_heads, seq, head_dim, device=dev) * scaling
k1 = torch.randn(bsz, num_heads, seq, head_dim, device=dev)
q2 = torch.randn(bsz, num_heads, seq, head_dim, device=dev) * scaling
k2 = torch.randn(bsz, num_heads, seq, head_dim, device=dev)
v = torch.randn(bsz, num_heads, seq, head_dim, device=dev)
lambda_full = torch.randn(1, device=dev).exp().squeeze()  # blocks.py 中为标量

o_orig = orig_attn(q1, k1, q2, k2, v, lambda_full)
o_sdpa = sdpa_attn(q1, k1, q2, k2, v, lambda_full)
diff = (o_orig - o_sdpa).abs().max().item()
print(f'[等价性] seq={seq}: max abs diff = {diff:.3e}', 'PASS' if diff < 1e-5 else 'FAIL')

# ---- 2. 大 seq 内存/速度（fp32）----
seq = 18080
x = torch.randn(bsz, seq, embed_dim, device=dev)
wq, wk, wv = (torch.randn(embed_dim, embed_dim, device=dev) * 0.02 for _ in range(3))
with torch.no_grad():
    q = (x @ wq).view(bsz, seq, num_heads, head_dim).transpose(1, 2) * scaling
    k = (x @ wk).view(bsz, seq, num_heads, head_dim).transpose(1, 2)
    v = (x @ wv).view(bsz, seq, num_heads, head_dim).transpose(1, 2)

torch.cuda.empty_cache()
t0 = time.time()
with torch.no_grad():
    for _ in range(2):
        out = sdpa_attn(q, k, q, k, v, lambda_full)
torch.cuda.synchronize()
dt = (time.time() - t0) / 2
mem = torch.cuda.max_memory_allocated() / 1e9
print(f'[大seq] bsz={bsz} seq={seq}: 单次 {dt:.2f}s, 峰值显存 {mem:.1f}GB, out shape {out.shape}')

# ---- 3. 尝试 math 后端对比（确认 sdpa 实际后端）----
from torch.nn.attention import sdpa_kernel, SDPBackend
try:
    with sdpa_kernel(SDPBackend.MATH):
        t0 = time.time()
        with torch.no_grad():
            out_m = sdpa_attn(q, k, q, k, v, lambda_full)
        torch.cuda.synchronize()
        print(f'[math后端] 单次 {(time.time()-t0):.2f}s')
except Exception as e:
    print('math backend failed:', e)

# ---- 4. 精度对比（大 seq, fp32 vs 后端）----
o_ref = orig_attn(q, k, q, k, v, lambda_full)
diff2 = (o_ref - out).abs().max().item()
print(f'[大seq等价性] max abs diff = {diff2:.3e}', 'PASS' if diff2 < 1e-4 else 'FAIL')
