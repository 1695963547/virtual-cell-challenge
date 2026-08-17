#!/usr/bin/env python
"""check_vcc_data.py: 验证 vcc_train/vcc_val.h5ad 的兼容性（vocab 安全、cell line 交叉）"""
import anndata as ad
import numpy as np
import pandas as pd

DATA_DIR = "/home/zjh/scDFM/vcc/data"

train = ad.read_h5ad(f"{DATA_DIR}/vcc_train.h5ad")
val = ad.read_h5ad(f"{DATA_DIR}/vcc_val.h5ad")

print(f"train: {train.n_obs} x {train.n_vars}, val: {val.n_obs} x {val.n_vars}")
print(f"var names equal: {list(train.var_names) == list(val.var_names)}")

var_set = set(train.var_names)

# 1. train condition 是否都在 var_names（vocab.encode 安全）
train_conds = [c for c in train.obs["condition"].unique() if c != "control"]
missing_train = [c for c in train_conds if c not in var_set]
print(f"\n[train] {len(train_conds)} perturbations, not-in-var_names: {missing_train}")

val_conds = [c for c in val.obs["condition"].unique() if c != "control"]
missing_val = [c for c in val_conds if c not in var_set]
print(f"[val] {len(val_conds)} perturbations, not-in-var_names: {missing_val}")

# 2. val condition 是否都被 train 覆盖（perturbation_dict union 后反查安全）
train_cond_set = set(train.obs["condition"].unique())
uncovered = [c for c in val.obs["condition"].unique() if c not in train_cond_set]
print(f"[val] conditions not in train: {uncovered}")

# 3. condition x cell_line 交叉（是否跨系）
cross = train.obs.groupby("condition")["cell_line"].nunique()
multi = cross[cross > 1]
print(f"\n[train] conditions spanning multiple cell lines: {len(multi)}")
if len(multi):
    for c in multi.index:
        sub = train.obs[train.obs["condition"] == c]
        print(f"  {c}: {sub['cell_line'].value_counts().to_dict()}")

# 4. control 数量按 cell line
ctrl = train.obs[train.obs["condition"] == "control"]
print(f"\ncontrol by cell line:\n{ctrl['cell_line'].value_counts().to_dict()}")
print(f"val control cells: {(val.obs['condition'] == 'control').sum()}, val cell_line: {val.obs['cell_line'].unique()}")

# 5. 每 cell line 的扰动数
pt = train.obs[train.obs["condition"] != "control"]
print(f"perturbations by cell line:\n{pt.groupby('cell_line')['condition'].nunique().to_dict()}")

# 6. X 数值范围抽查（确认 log1p）
import scipy.sparse as sp
X = train.X
if sp.issparse(X):
    xs = X.data
else:
    xs = np.asarray(X).ravel()
print(f"\ntrain X: max={xs.max():.2f}, nnz_frac={1.0 if not sp.issparse(X) else len(xs) / (train.n_obs * train.n_vars):.4f}")
