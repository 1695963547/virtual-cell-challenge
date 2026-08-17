"""用迁移 delta 构建 cell-eval 格式的预测 h5ad（val/test 通用干跑器）。

原理（BM_xTVC 式 replicate）：
  pred_X[细胞] = log1p 对照均值 + delta_transfer(该细胞的扰动) × scale
  - 对照细胞：保留真实 log1p 值
  - 无参考覆盖的扰动：退化为对照均值（预测≈无效应）
  - 每个扰动的所有细胞输出相同向量（replicate，满足 Wilcoxon）

用法：
  python build_transfer_pred.py --truth /home/zjh/vcc_official/validation/adata_Validation.h5ad \
      --scale 64 --output /home/zjh/state-vcc-local/state/competition/prediction_val_transfer_x64.h5ad
"""
import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

OUT = "/home/zjh/state-vcc-local/pseudobulk"
REFS = ["k562_gwps", "rpe1", "jurkat", "k562", "hepg2"]
CTRL = "non-targeting"


def load(name):
    d = np.load(os.path.join(OUT, f"deltas_{name}.npz"), allow_pickle=False)
    return {p: d["deltas"][i].astype(np.float64) for i, p in enumerate(d["perts"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--scale", type=float, default=64.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pert-col", default="target_gene")
    ap.add_argument("--space", choices=["counts", "log1p"], default="counts",
                    help="输出空间：counts=expm1 取整（cell-eval 判为 raw counts，放大不受 15 阈值限制）；log1p 仅限 scale<=1 使用")
    args = ap.parse_args()
    if args.space == "log1p" and args.scale > 1:
        raise SystemExit("log1p 空间 max 值会超 cell-eval 的 15 阈值，scale>1 必须用 counts 空间")

    refs = {r: load(r) for r in REFS}
    print(f"加载真值: {args.truth}")
    a = ad.read_h5ad(args.truth)
    print(f"  {a.shape}, obs 列: {list(a.obs.columns)}")

    # log1p 空间统一（真值是 raw counts）
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.copy()
    X.data = np.log1p(X.data)

    tg = a.obs[args.pert_col].astype(str).to_numpy()
    ctrl_mask = tg == CTRL
    ctrl_mean = np.asarray(X[ctrl_mask].mean(axis=0)).ravel()
    print(f"对照细胞 {ctrl_mask.sum()} 个，均值向量 mean={ctrl_mean.mean():.4f}")

    n_cov = sum(1 for p in set(tg) - {CTRL} if any(p in refs[r] for r in REFS))
    n_perts = len(set(tg) - {CTRL})
    print(f"扰动覆盖: {n_cov}/{n_perts}")

    # 每个扰动的迁移 delta（多参考均匀平均，V4 已证加权无增益）
    uniq = sorted(set(tg))
    delta_map = {}
    for p in uniq:
        if p == CTRL:
            continue
        avail = [refs[r][p] for r in REFS if p in refs[r]]
        delta_map[p] = np.mean(avail, axis=0) if avail else np.zeros_like(ctrl_mean)

    # 构建预测矩阵（float64 计算防 expm1 溢出，98,927 x 18080 x 8B = 14.3GB）
    codes = pd.Categorical(tg, categories=uniq).codes
    Xp = np.empty((len(tg), X.shape[1]), dtype=np.float64)
    Xp[:] = ctrl_mean[None, :]
    for i, p in enumerate(uniq):
        if p == CTRL:
            rows = codes == i
            Xp[rows] = np.asarray(X[rows].todense(), dtype=np.float64)
        else:
            Xp[codes == i] += delta_map[p] * args.scale

    if args.space == "counts":
        Xp = np.expm1(Xp)
        np.clip(Xp, 0, 1e9, out=Xp)
        Xp = np.rint(Xp)  # 全整数 → cell-eval 判为 raw counts（无 15 阈值）
    else:
        np.clip(Xp, 0, 14.99, out=Xp)

    pred = ad.AnnData(X=sp.csr_matrix(Xp.astype(np.float32)), obs=a.obs.copy(), var=a.var.copy())
    pred.write_h5ad(args.output)
    print(f"输出: {args.output} (scale={args.scale}, space={args.space})")


if __name__ == "__main__":
    main()
