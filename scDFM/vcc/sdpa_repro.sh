#!/bin/bash
# SDPA 最小复现：模拟 scDFM cross-attention 的精确形状（不加载 h5ad 数据）
# 测试手动 Q/K/V projections + SDPA 替代 nn.MultiheadAttention 的修复
cd /home/zjh || exit 1
CUDA_VISIBLE_DEVICES=0 /home/zjh/state-vcc-local/state/.venv/bin/python3 - <<'EOF'
import torch

B, L, S, D, H = 256, 1000, 18084, 512, 8
head_dim = D // H
dev = 'cuda'
print('torch:', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))
print('显存 free: %.1f GB' % (torch.cuda.mem_get_info()[0]/1e9))

# 模拟 CrossAttentionTransformerLayer 的新实现（手动 projections + SDPA）
q_proj = torch.nn.Linear(D, D, bias=True).to(dev)
k_proj = torch.nn.Linear(D, D, bias=True).to(dev)
v_proj = torch.nn.Linear(D, D, bias=True).to(dev)
out_proj = torch.nn.Linear(D, D, bias=True).to(dev)
attn_dropout = torch.nn.Dropout(0.1)

tgt = torch.randn(B, L, D, device=dev)
memory = torch.randn(B, S, D, device=dev)
mask = torch.zeros(L, S, dtype=torch.bool, device=dev)
mask[:, :30] = True   # 模拟 topk=30 共表达图结构

# 转换为 SDPA float mask
sdpa_mask = torch.zeros_like(mask, dtype=torch.float32)
sdpa_mask.masked_fill_(mask, float('-inf'))

for trial in range(2):
    print(f'--- trial {trial} ---', flush=True)

    # 手动 Q/K/V projections → contiguous tensors
    q = q_proj(tgt).view(B, L, H, head_dim).transpose(1, 2)     # (B, H, L, head_dim)
    k = k_proj(memory).view(B, S, H, head_dim).transpose(1, 2)   # (B, H, S, head_dim)
    v = v_proj(memory).view(B, S, H, head_dim).transpose(1, 2)   # (B, H, S, head_dim)

    # SDPA with correct scale
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=sdpa_mask, dropout_p=0.1, scale=head_dim ** -0.5,
    )
    attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
    attn_out = out_proj(attn_out)
    out = tgt + attn_dropout(attn_out)

    torch.cuda.synchronize()
    print('Cross-attn OK, out:', tuple(out.shape), flush=True)

# bf16 autocast 对照
print('--- bf16 autocast ---', flush=True)
with torch.autocast('cuda', dtype=torch.bfloat16):
    q2 = q_proj(tgt).view(B, L, H, head_dim).transpose(1, 2)
    k2 = k_proj(memory).view(B, S, H, head_dim).transpose(1, 2)
    v2 = v_proj(memory).view(B, S, H, head_dim).transpose(1, 2)
    attn_out2 = torch.nn.functional.scaled_dot_product_attention(
        q2, k2, v2, attn_mask=sdpa_mask, dropout_p=0.1, scale=head_dim ** -0.5,
    )
    attn_out2 = attn_out2.transpose(1, 2).reshape(B, L, D)
    out2 = out_proj(attn_out2)
torch.cuda.synchronize()
print('bf16 OK, out:', tuple(out2.shape), flush=True)
print('ALL PASSED')
EOF
