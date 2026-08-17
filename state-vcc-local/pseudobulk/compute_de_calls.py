"""计算 H1 训练集 150 个扰动的真 DE 基因集（Wilcoxon，仿 cell-eval 真值侧口径）。

输出 de_calls_H1.npz：
  perts: (150,) 扰动名
  de_idx: object array，每个元素为该扰动 FDR<=0.05 的 DE 基因列索引（按 |log2fc| 降序）
  de_lfc: object array，对应的 log2fc
为控制耗时：每扰动最多抽 1000 细胞，对照最多 5000 细胞（功效足够）。
"""
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

BASE = "/home/zjh/state-vcc-local/state/vci_pretrain"
OUT = "/home/zjh/state-vcc-local/pseudobulk"
CTRL = "non-targeting"
MAX_PERT_CELLS = 1000
MAX_CTRL_CELLS = 5000
rng = np.random.default_rng(0)

t0 = time.time()
a = ad.read_h5ad(os.path.join(BASE, "competition_train.h5"))
tg = a.obs["target_gene"]
cats = np.array(tg.cat.categories)
codes = tg.cat.codes.to_numpy()
print(f"loaded {a.shape}, {time.time()-t0:.0f}s", flush=True)

keep_idx = []
for c in cats:
    idx = np.where(codes == list(cats).index(c))[0]
    cap = MAX_CTRL_CELLS if c == CTRL else MAX_PERT_CELLS
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    keep_idx.append(idx)
sub = a[np.sort(np.concatenate(keep_idx))].copy()
del a
print(f"subsampled {sub.shape}, {time.time()-t0:.0f}s", flush=True)

sc.tl.rank_genes_groups(
    sub, groupby="target_gene", reference=CTRL, method="wilcoxon",
    n_genes=sub.n_vars, rankby_abs=False, corr_method="benjamini-hochberg",
)
print(f"wilcoxon done, {time.time()-t0:.0f}s", flush=True)

names = sub.uns["rank_genes_groups"]["names"]
pvals = sub.uns["rank_genes_groups"]["pvals_adj"]
lfc = sub.uns["rank_genes_groups"]["logfoldchanges"]
groups = names.dtype.names

perts, de_idx_list, de_lfc_list = [], [], []
for gname in groups:
    if gname == CTRL:
        continue
    gnames = np.array(names[gname])
    gp = np.array(pvals[gname])
    gl = np.array(lfc[gname])
    sig = gp <= 0.05
    idx = np.array([sub.var.index.get_loc(x) for x in gnames[sig]])
    order = np.argsort(-np.abs(gl[sig]))
    perts.append(gname)
    de_idx_list.append(idx[order])
    de_lfc_list.append(gl[sig][order])

np.savez(
    os.path.join(OUT, "de_calls_H1.npz"),
    perts=np.array(perts),
    de_idx=np.array(de_idx_list, dtype=object),
    de_lfc=np.array(de_lfc_list, dtype=object),
)
stats = pd.Series([len(x) for x in de_idx_list])
print(f"saved. DE 基因数分布: median={stats.median():.0f}, "
      f"q10={stats.quantile(0.1):.0f}, q90={stats.quantile(0.9):.0f}, zero={(stats==0).sum()}", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
