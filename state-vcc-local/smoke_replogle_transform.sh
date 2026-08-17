#!/usr/bin/env bash
# Replogle 全量 SE transform 冒烟：2k 细胞验证管线。
# 与 validation 的差异：基因面板 8248（非 18080）、obs 列名 gene/batch（非 target_gene）。
# 冒烟过了才允许跑全量（1.99M 细胞 ~11h）。
set -uo pipefail
STATE=/home/zjh/state-vcc-local/state
PY=$STATE/.venv/bin/python
SMOKE_IN=/tmp/replogle_smoke_2k.h5ad
SMOKE_OUT=/tmp/replogle_smoke_2k_se.h5ad
SE_CKPT=/home/zjh/SE-600M/se600m_epoch16.ckpt
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export HDF5_USE_FILE_LOCKING=FALSE

log() { echo "[$(date '+%F %T')] $*"; }

log "===== 1. 切 2000 细胞冒烟样本 ====="
"$PY" - <<'EOF'
import anndata as ad
a = ad.read_h5ad("/home/zjh/vcc_official/replogle/replogle_2022_k562_gwps.h5ad", backed="r")
s = a[:2000].to_memory()
s.write_h5ad("/tmp/replogle_smoke_2k.h5ad")
print("smoke input:", s.shape, "X max(前500行):", float(s.X[:500].max()))
EOF

log "===== 2. SE transform（GPU $CUDA_VISIBLE_DEVICES） ====="
"$STATE/.venv/bin/state" emb transform \
  --model-folder /home/zjh/SE-600M \
  --checkpoint "$SE_CKPT" \
  --protein-embeddings /home/zjh/SE-600M/protein_embeddings.pt \
  --input "$SMOKE_IN" \
  --output "$SMOKE_OUT" || { log "!! transform 失败"; exit 1; }

log "===== 3. 校验输出 ====="
"$PY" - <<'EOF'
import anndata as ad, numpy as np
a = ad.read_h5ad("/tmp/replogle_smoke_2k_se.h5ad")
assert "X_state" in a.obsm, "缺 X_state"
x = np.asarray(a.obsm["X_state"])
norms = np.linalg.norm(x, axis=1)
print("X_state:", x.shape, x.dtype,
      "| NaN:", int(np.isnan(x).sum()),
      "| 行范数 mean/min/max: %.2f / %.2f / %.2f" % (norms.mean(), norms.min(), norms.max()))
assert x.shape == (2000, 2058), f"维度不对: {x.shape}"
assert np.isnan(x).sum() == 0, "有 NaN"
assert norms.min() > 1e-3, "有零向量行（基因对齐失败？）"
print("OK 冒烟通过")
EOF
log "===== 冒烟完成 ====="
