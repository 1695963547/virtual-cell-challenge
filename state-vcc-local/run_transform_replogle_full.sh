#!/usr/bin/env bash
# Replogle K562 GWPS 全量 SE transform：1,989,578 细胞 -> obsm[X_state]
# 冒烟（2k 细胞，2026-08-08 21:27）已通过：7845/8248 基因对齐（95.1%），
# 吞吐 ~50 细胞/s => 全量约 11h。输出 ~25GB。
#
# 用法（Qoder 后台 或 用户终端）：
#   bash /home/zjh/state-vcc-local/run_transform_replogle_full.sh \
#     > /home/zjh/log/transform_replogle_full.log 2>&1
# 注意：不要 setsid nohup（沙箱内会丢 GPU，见 STATUS 4.3）。
set -uo pipefail

STATE=/home/zjh/state-vcc-local/state
PY=$STATE/.venv/bin/python
SRC=/home/zjh/vcc_official/replogle/replogle_2022_k562_gwps.h5ad
DST=$STATE/vci_pretrain/replogle_full_se.h5ad
SE_CKPT=/home/zjh/SE-600M/se600m_epoch16.ckpt
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export HDF5_USE_FILE_LOCKING=FALSE

log() { echo "[$(date '+%F %T')] $*"; }

if [[ -f "$DST" ]]; then
  log "!! $DST 已存在 —— 如需重算请先删除"
  exit 1
fi
for f in "$SRC" "$SE_CKPT" /home/zjh/SE-600M/protein_embeddings.pt; do
  [[ -f "$f" ]] || { log "!! 缺少 $f"; exit 1; }
done
if ! "$PY" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"; then
  log "!! torch.cuda.is_available()=False —— 已中止（CPU 跑 1.99M 细胞不可行）"
  exit 1
fi
log "GPU 校验通过：$("$PY" -c 'import torch;print(torch.cuda.get_device_name(0))')  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
df -h /home | tail -1

log "===== SE transform 全量启动（1,989,578 细胞，预计 ~11h） ====="
"$STATE/.venv/bin/state" emb transform \
  --model-folder /home/zjh/SE-600M \
  --checkpoint "$SE_CKPT" \
  --protein-embeddings /home/zjh/SE-600M/protein_embeddings.pt \
  --input "$SRC" \
  --output "$DST" || { log "!! transform 失败 rc=$?"; exit 1; }

log "===== 校验输出 ====="
"$PY" - <<'EOF'
import anndata as ad, numpy as np
a = ad.read_h5ad("/home/zjh/state-vcc-local/state/vci_pretrain/replogle_full_se.h5ad", backed="r")
assert "X_state" in a.obsm, "缺 X_state"
x = a.obsm["X_state"]
print("shape:", a.shape, " X_state:", x.shape, x.dtype)
s = np.asarray(x[:2000])
norms = np.linalg.norm(s, axis=1)
print("抽样 2000 行 NaN:", int(np.isnan(s).sum()),
      " 范数 mean/min: %.2f / %.2f" % (norms.mean(), norms.min()))
assert x.shape == (1989578, 2058)
assert np.isnan(s).sum() == 0 and norms.min() > 1e-3
print("OK 全量 transform 校验通过")
EOF
rc=$?
log "===== 结束 rc=$rc ====="
ls -la "$DST"
exit $rc
