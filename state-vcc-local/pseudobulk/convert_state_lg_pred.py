"""state_lg 预测（log1p 空间）→ counts 整数空间，供 cell-eval 与迁移配方同口径对比。

用法：
    uv run --no-sync python convert_state_lg_pred.py \
        --input competition/prediction_val_state_lg.h5ad \
        --output competition/prediction_val_state_lg_counts.h5ad
"""
import argparse
import os

import anndata as ad
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()

print(f"读取: {args.input}")
a = ad.read_h5ad(args.input)
print(f"  shape={a.shape}, dtype={a.X.dtype}, sparse={sp.issparse(a.X)}")

X = a.X
if sp.issparse(X):
    X = X.copy()
    X.data = np.expm1(X.data.astype(np.float64))
    X.data = np.rint(np.clip(X.data, 0, 1e9)).astype(np.float32)
    X = X.tocsr()
else:
    X = np.expm1(X.astype(np.float64))
    X = np.rint(np.clip(X, 0, 1e9)).astype(np.float32)
    X = sp.csr_matrix(X)

out = ad.AnnData(X=X, obs=a.obs.copy(), var=a.var.copy())
os.makedirs(os.path.dirname(args.output), exist_ok=True)
out.write_h5ad(args.output)
print(f"输出: {args.output} ({out.shape})")
