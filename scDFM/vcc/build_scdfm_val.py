#!/usr/bin/env python
"""build_scdfm_val.py: 官方 validation h5ad -> scDFM 验证格式 vcc_val.h5ad
- obs 构造 condition/Drug1/Drug2/cell_line（官方 val 是 H1 细胞）
- var 对齐到训练数据（vcc_train.h5ad）的基因顺序
- X 转 log1p：官方 val X 是原始 count，训练 X 是 normalize_total(1e4)+log1p，必须同口径
"""
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

CONTROL_NAMES = {"non-targeting", "control", "non_targeting"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="/home/zjh/vcc_official/validation/adata_Validation.h5ad")
    ap.add_argument("--train", default="/home/zjh/scDFM/vcc/data/vcc_train.h5ad")
    ap.add_argument("--out", default="/home/zjh/scDFM/vcc/data/vcc_val.h5ad")
    args = ap.parse_args()

    print("loading val ...")
    av = ad.read_h5ad(args.val)
    print(f"val: {av.n_obs} cells x {av.n_vars} genes")

    # 训练 var 顺序（对齐）
    at = ad.read_h5ad(args.train, backed="r")
    train_var = list(at.var_names)
    at.file.close()
    val_var = list(av.var_names)
    missing = set(train_var) - set(val_var)
    if missing:
        print(f"WARNING: {len(missing)} train genes missing in val:", list(missing)[:10])
    av = av[:, train_var].copy()  # 重排到训练顺序
    print(f"after reorder: {av.n_vars} genes, var match = {list(av.var_names) == train_var}")

    # 官方 val X 是原始 count，转成与训练同口径的 log1p（normalize_total 1e4 + log1p）
    sc.pp.normalize_total(av, target_sum=1e4)
    sc.pp.log1p(av)
    print(f"val X after log1p: max={av.X.data.max() if hasattr(av.X, 'data') else av.X.max():.2f}")

    # obs 列构造
    tgt = av.obs["target_gene"].astype(str) if "target_gene" in av.obs.columns else av.obs["guide_id"].astype(str)
    is_ctrl = tgt.isin(CONTROL_NAMES)
    condition = np.where(is_ctrl, "control", tgt.to_numpy())
    av.obs["condition"] = condition
    av.obs["Drug1"] = condition
    av.obs["Drug2"] = "control"
    av.obs["cell_line"] = "H1"
    av.obs["cell_line_id"] = 0
    print("condition unique:", av.obs["condition"].nunique(),
          "| control cells:", int(is_ctrl.sum()))

    av.write_h5ad(args.out, compression="gzip", compression_opts=4)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
