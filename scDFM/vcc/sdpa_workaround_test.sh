#!/bin/bash
# 验证绕过 cuBLASLt 失败的方案（A: 限制 workspace  B: 禁 cublasLt）
cd /home/zjh || exit 1
for cfg in "A_CUBLAS_WS:export CUBLAS_WORKSPACE_CONFIG=:16:8" "B_NO_LT:export CUBLAS_WORKSPACE_CONFIG=:4096:2" "C_PREFER_CUBLAS:true"; do
  name="${cfg%%:*}"; extra="${cfg#*:}"
  echo "===== 方案 $name ====="
  CUDA_VISIBLE_DEVICES=0 /home/zjh/state-vcc-local/state/.venv/bin/python3 - "$extra" <<'EOF'
import sys, os
extra = sys.argv[1]
if extra.startswith('export '):
    os.environ[extra.split()[1]] = extra.split('=',1)[1]
import torch
if extra == 'true':
    try:
        torch.backends.cuda.preferred_blas_library('cublas')
        print('已设 preferred_blas_library=cublas（禁用 cublasLt）')
    except Exception as e:
        print('preferred_blas_library 不支持:', e)

B, L, S, D, H = 256, 1000, 18084, 512, 8
mha = torch.nn.MultiheadAttention(D, H, dropout=0.1, batch_first=True).cuda()
q = torch.randn(B, L, D, device='cuda')
k = torch.randn(B, S, D, device='cuda')
v = torch.randn(B, S, D, device='cuda')
mask = torch.zeros(L, S, dtype=torch.bool, device='cuda')
mask[:, :30] = True

with torch.autocast('cuda', dtype=torch.bfloat16):
    out, _ = mha(query=q, key=k, value=v, attn_mask=mask, need_weights=False)
torch.cuda.synchronize()
print('MHA bf16 OK, out:', tuple(out.shape))
print('方案通过')
EOF
  echo "exit=$?"
done
