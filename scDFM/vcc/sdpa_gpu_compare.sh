#!/bin/bash
# 维度验证：不同 GPU 卡 + fp32/bf16 对照
cd /home/zjh || exit 1
PY=/home/zjh/state-vcc-local/state/.venv/bin/python3
run_test() {
  local label="$1"; local gpu="$2"; local dtype="$3"
  echo "===== $label (GPU $gpu, $dtype) ====="
  CUDA_VISIBLE_DEVICES=$gpu $PY - "$dtype" <<'EOF'
import sys, torch
dtype = sys.argv[1]
B, L, S, D, H = 256, 1000, 18084, 512, 8
mha = torch.nn.MultiheadAttention(D, H, dropout=0.1, batch_first=True).cuda()
q = torch.randn(B, L, D, device='cuda')
k = torch.randn(B, S, D, device='cuda')
v = torch.randn(B, S, D, device='cuda')
mask = torch.zeros(L, S, dtype=torch.bool, device='cuda')
mask[:, :30] = True
if dtype == 'bf16':
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, _ = mha(query=q, key=k, value=v, attn_mask=mask, need_weights=False)
else:
    out, _ = mha(query=q, key=k, value=v, attn_mask=mask, need_weights=False)
torch.cuda.synchronize()
print('MHA OK, out:', tuple(out.shape), '| 通过')
EOF
  echo "exit=$?"
}
run_test "GPU0 bf16" 0 bf16
run_test "GPU4 bf16" 4 bf16
run_test "GPU0 fp32" 0 fp32
