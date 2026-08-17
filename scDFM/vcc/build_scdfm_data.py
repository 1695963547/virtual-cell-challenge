#!/usr/bin/env python
"""build_scdfm_data.py: cell_load h5 -> scDFM 训练 h5ad（VCC 6 文件合并版，hepg2 留出）"""
import argparse
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix

FILES = [
    ("competition_train", "H1"),
    ("k562_gwps", "K562"),
    ("rpe1", "RPE1"),
    ("jurkat", "Jurkat"),
    ("k562", "K562"),
]
CELL_LINE_ID = {"H1": 0, "K562": 1, "RPE1": 2, "Jurkat": 3}
CONTROL_NAMES = {"non-targeting", "control", "non_targeting"}
MIN_CELLS = 48


def read_obs_col(obs_group, name, n):
    g = obs_group[name]
    if isinstance(g, h5py.Dataset):
        return g[:n].astype(str)
    cats = g["categories"][:].astype(str)
    return cats[g["codes"][:n]]


def read_csr(f, n, n_genes):
    X = f["X"]
    if not hasattr(X, "keys"):  # dense Dataset（k562_gwps/rpe1/jurkat/k562/hepg2）
        arr = np.asarray(X[:n, :], dtype=np.float32)
        return csr_matrix(arr)
    indptr = X["indptr"][: n + 1]
    data = X["data"][: int(indptr[-1])]
    indices = X["indices"][: int(indptr[-1])]
    return csr_matrix((data, indices, indptr), shape=(n, n_genes))


def load_one(path, name, cell_line, n_genes, var_names):
    with h5py.File(path, "r") as f:
        n = f["obs"]["target_gene"]["codes"].shape[0]
        target = read_obs_col(f["obs"], "target_gene", n)
        cell_type = read_obs_col(f["obs"], "cell_type", n) if "cell_type" in f["obs"] else np.full(n, cell_line)
        batch = read_obs_col(f["obs"], "batch", n) if "batch" in f["obs"] else np.full(n, cell_line)
        X = read_csr(f, n, n_genes)
        # h5 的 X 是 log1p(raw counts) → 还原 raw → normalize_total(1e4) → log1p，
        # 与 val/test 推理口径（log1p(CP10K)）保持一致（scDFM 原版惯例）
        X.data = np.expm1(X.data)

    is_ctrl = np.isin(target, list(CONTROL_NAMES))
    condition = np.where(is_ctrl, "control", target)

    df = pd.DataFrame({
        "condition": condition,
        "Drug1": condition,
        "Drug2": np.full(n, "control"),
        "cell_line": np.full(n, cell_line),
        "cell_line_id": np.full(n, CELL_LINE_ID[cell_line], dtype=np.int64),
        "batch": batch,
        "target_gene": target,
        "cell_type": cell_type,
    }, index=[f"{name}_{i}" for i in range(n)])
    a = ad.AnnData(X=X, obs=df, var=pd.DataFrame(index=var_names))
    sc.pp.normalize_total(a, target_sum=1e4)  # CP10K，消除测序深度差异
    sc.pp.log1p(a)                            # log1p(CP10K)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/zjh/state-vcc-local/state/vci_pretrain")
    ap.add_argument("--out", default="/home/zjh/scDFM/vcc/data/vcc_train.h5ad")
    ap.add_argument("--min_cells", type=int, default=MIN_CELLS)
    ap.add_argument("--smoke", type=int, default=0, help=">0 时每文件只取前 N 细胞（冒烟用）")
    args = ap.parse_args()

    with h5py.File(f"{args.data_dir}/{FILES[0][0]}.h5", "r") as f:
        var_col = "_index" if "_index" in f["var"] else "gene_name"
        var_g = f["var"][var_col]
        if isinstance(var_g, h5py.Dataset):
            var_names = var_g[:].astype(str)
        else:
            var_names = var_g["categories"][:].astype(str)
        n_genes = len(var_names)

    ads = []
    stats = []
    for name, cell_line in FILES:
        a = load_one(f"{args.data_dir}/{name}.h5", name, cell_line, n_genes, var_names)
        if args.smoke > 0:
            a = a[: args.smoke].copy()
        else:
            cnt = a.obs["condition"].value_counts()
            keep = (a.obs["condition"] == "control") | (a.obs["condition"].map(cnt) >= args.min_cells)
            n_drop = int((~keep).sum())
            a = a[keep].copy()
            stats.append(f"{name}: kept={len(a.obs)} (dropped {n_drop} small-pert cells)")
        ads.append(a)

    print("\n".join(stats))

    adata = ad.concat(ads, join="outer", index_unique=None)
    print(f"merged: {adata.n_obs} cells x {adata.n_vars} genes")
    print("condition unique:", adata.obs["condition"].nunique(),
          "| control cells:", int((adata.obs["condition"] == "control").sum()))
    print("cell_line counts:", adata.obs["cell_line"].value_counts().to_dict())

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    adata.write_h5ad(args.out, compression="gzip", compression_opts=4)
    print(f"saved to {args.out} ({os.path.getsize(args.out)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
