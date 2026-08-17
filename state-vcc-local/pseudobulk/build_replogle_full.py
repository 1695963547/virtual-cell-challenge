"""Replogle GWPS 全量 → VCC 面板 delta 库。

处理：1,989,578 细胞 × 8,248 基因（raw counts）→ 每细胞 CPM 式归一化 + log1p
→ 按 perturbation 分组均值 − 'control' 均值 → 映射到 VCC 18,080 面板（var.gene_name 匹配，未映射列置 0）
输出：pseudobulk/deltas_replogle_full.npz（deltas: (n_perts, 18080) float32, perts, counts, panel_genes）
"""
import os
import time

import h5py
import numpy as np

SRC = "/home/zjh/vcc_official/replogle/replogle_2022_k562_gwps.h5ad"
OUT = "/home/zjh/state-vcc-local/pseudobulk"
PANEL_H5 = "/home/zjh/state-vcc-local/state/vci_pretrain/competition_train.h5"
CTRL = "control"

t0 = time.time()
with h5py.File(SRC, "r") as h:
    cats = np.array([c.decode() for c in h["obs"]["perturbation"]["categories"][:]])
    codes = h["obs"]["perturbation"]["codes"][:]
    gene_names = np.array([g.decode() for g in h["var"]["gene_name"][:]])
    print(f"obs/var 读取 {time.time()-t0:.0f}s；{len(cats)} 扰动, {len(codes)} 细胞", flush=True)
    X = h["X"][:]  # 65GB 一次性读入（机器内存充足）
print(f"X 读取完成 {X.shape} {time.time()-t0:.0f}s", flush=True)

# 每细胞总 counts 归一化 → log1p（原位操作省内存）
totals = X.sum(axis=1, keepdims=True)
med = np.median(totals[totals > 0])
np.divide(X, np.maximum(totals, 1), out=X)
X *= med
np.log1p(X, out=X)
del totals
print(f"归一化+log1p 完成 {time.time()-t0:.0f}s", flush=True)

# 分组均值
order = np.argsort(codes, kind="stable")
counts = np.bincount(codes, minlength=len(cats))
bounds = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
Xs = X[order]
del X
sums = np.add.reduceat(Xs, bounds[:-1], axis=0)
means = (sums / counts.astype(np.float64)[:, None]).astype(np.float32)
del Xs, sums
print(f"分组均值完成 {means.shape} {time.time()-t0:.0f}s", flush=True)

ci = list(cats).index(CTRL)
ctrl = means[ci]
keep = [i for i, c in enumerate(cats) if c != CTRL]
deltas_small = means[keep] - ctrl[None, :]
perts = cats[keep]
counts_keep = counts[keep]

# 映射到 VCC 18,080 面板
with h5py.File(PANEL_H5, "r") as h:
    panel = np.array([x.decode() for x in h["var"]["_index"][:]])
panel_idx = {g: i for i, g in enumerate(panel)}
deltas = np.zeros((len(perts), len(panel)), dtype=np.float32)
mapped = 0
seen = set()
for j, g in enumerate(gene_names):
    if g in panel_idx and g not in seen:
        deltas[:, panel_idx[g]] = deltas_small[:, j]
        seen.add(g)
        mapped += 1
print(f"面板映射: {mapped}/8248 基因列映射到 18080 面板", flush=True)

np.savez_compressed(
    os.path.join(OUT, "deltas_replogle_full.npz"),
    deltas=deltas, perts=perts, counts=counts_keep,
)
print(f"保存完成 {time.time()-t0:.0f}s", flush=True)
