"""state_lg 预测 × 迁移预测 线性混合（log1p 空间），val 上 α 扫描选优。

用法：
    uv run --no-sync python mix_preds.py \
        --a competition/prediction_val_state_lg.h5ad \
        --b competition/prediction_val_transfer_log1p.h5ad \
        --alpha 0.5 --output competition/prediction_val_mix_a05.h5ad
"""
import argparse
import os

import anndata as ad
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="state_lg 预测（log1p）")
ap.add_argument("--b", required=True, help="迁移预测（log1p）")
ap.add_argument("--alpha", type=float, required=True, help="state_lg 权重")
ap.add_argument("--output", required=True)
args = ap.parse_args()

print(f"读取 A: {args.a}")
A = ad.read_h5ad(args.a)
print(f"读取 B: {args.b}")
B = ad.read_h5ad(args.b)

Xa = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)
Xb = B.X.toarray() if sp.issparse(B.X) else np.asarray(B.X)
print(f"  A={Xa.shape} B={Xb.shape}")

X = args.alpha * Xa + (1.0 - args.alpha) * Xb
np.clip(X, 0, 14.99, out=X)

out = ad.AnnData(X=sp.csr_matrix(X.astype(np.float32)), obs=A.obs.copy(), var=A.var.copy())
os.makedirs(os.path.dirname(args.output), exist_ok=True)
out.write_h5ad(args.output)
print(f"输出: {args.output} (alpha={args.alpha})")
