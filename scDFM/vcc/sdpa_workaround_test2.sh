#!/bin/bash
# 修正版：系统级环境变量绕过方案
cd /home/zjh || exit 1
PY=/home/zjh/state-vcc-local/state/.venv/bin/python3
run_test() {
  local label="$1"; shift
  echo "===== 方案 $label ====="
  CUDA_VISIBLE_DEVICES=0 "$@" $PY - <<'EOF'
import torch
print('torch:', torch.__version__, '| flash_sdp:', torch.backends.cuda.flash_sdp_enabled())
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
}
run_test "A_DISABLE_ADDMM_CUDA_LT" env DISABLE_ADDMM_CUDA_LT=1
run_test "B_CUBLAS_WS_16"            env CUBLAS_WORKSPACE_CONFIG=:16:8
run_test "C_两者同时"                 env DISABLE_ADDMM_CUDA_LT=1 CUBLAS_WORKSPACE_CONFIG=:16:8
