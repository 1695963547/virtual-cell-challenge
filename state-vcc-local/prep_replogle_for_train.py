"""Replogle GWPS 全量（带 X_state）→ cell_load 训练管线格式。

输入：vci_pretrain/replogle_full_se.h5ad
  1,989,578 细胞 × 8,248 基因（X = dense raw counts，gzip 压缩 ~65GB 解压）
  + obsm[X_state]（1.99M × 2058）+ obs[gene/batch/...]
输出：vci_pretrain/replogle_full_trainready.h5ad
  1.99M × 18,080 csr（log1p，无压缩连续存储）+ X_state + 训练 obs 列

处理（黄金标准 = k562_gwps.h5，同源数据的官方预处理版）：
1. X：raw counts → log1p 直转（与 prep_val_for_train.py / k562_gwps 的尺度一致；
   k562_gwps X max 6.85 ≈ 直接 log1p 而非 normalize_total）
2. 列映射 8,248 → 18,080 面板：var/gene_name ↔ competition_train.h5 var/_index，
   未映射列丢弃（v7 已验证 7,581/8,248 可映射）
3. obs：target_gene ← gene（对照名 non-targeting，与管线 control_pert 一致）
        batch_var ← batch（267 类，与 k562_gwps 完全同源，one-hot 复用）
        cell_type = "k562"（小写，复用 k562_gwps.h5 已有 one-hot 维度）
4. 过滤缺 ESM2 特征的扰动细胞（158 基因 ≈ 2.8 万细胞，1.4%）：
   训练管线对缺特征的扰动补零向量 —— 与对照同 embedding，会污染对照信号
   注意：non-targeting 也不在 ESM2 keys 里，但它是对照（零向量本就正确），
   必须从过滤名单排除 —— 否则 7.5 万对照细胞全被误删（2026-08-09 踩过）
5. uns["log1p"] = {}（output_space=all 下 mixed 标记直接 ValueError）
6. 无压缩写盘：gzip+chunk 会让 cell_load 随机行读放大数百倍
   （64 workers 单 worker 读 293GB 出不了第一个 batch 的坑，见 prep_val_for_train.py）

内存策略：X 分块（100k 行/块）解压 → log1p → csr → 列重映射，峰值 ~70GB。
运行时机：run_transform_replogle_full.sh 完成后（replogle_full_se.h5ad 就绪）。
冒烟：python prep_replogle_for_train.py /tmp/replogle_smoke_2k_se.h5ad /tmp/out.h5ad
"""
import sys
import time

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import torch

# argv 可覆盖输入输出做小样冒烟
SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/zjh/state-vcc-local/state/vci_pretrain/replogle_full_se.h5ad"
DST = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/zjh/state-vcc-local/state/vci_pretrain/replogle_full_trainready.h5ad"
PANEL_H5 = "/home/zjh/state-vcc-local/state/vci_pretrain/competition_train.h5"
ESM2 = "/home/zjh/state-vcc-local/state/competition_support_set/ESM2_pert_features.pt"
CHUNK = 100_000

t0 = time.time()


def ts():
    return f"{time.time()-t0:7.0f}s"


# ---------- 1. 面板与列映射 ----------
with h5py.File(PANEL_H5, "r") as h:
    panel = np.array([x.decode() for x in h["var/_index"][:]])
panel_idx = {g: i for i, g in enumerate(panel)}
print(f"[{ts()}] 面板 {len(panel)} 基因", flush=True)

with h5py.File(SRC, "r") as h:
    src_genes = np.array([x.decode() for x in h["var/gene_name"][:]])
    X_shape = h["X"].shape
print(f"[{ts()}] 源 {X_shape[0]:,} 细胞 × {X_shape[1]} 基因", flush=True)
n_obs = X_shape[0]

col_map = np.full(len(src_genes), -1, dtype=np.int32)
seen = set()
n_mapped = 0
for j, g in enumerate(src_genes):
    if g in panel_idx and g not in seen:
        col_map[j] = panel_idx[g]
        seen.add(g)
        n_mapped += 1
print(f"[{ts()}] 列映射 {n_mapped}/{len(src_genes)} → {len(panel)}", flush=True)

# ---------- 2. obs / X_state（一次性读） ----------
print(f"[{ts()}] 读 obs / X_state ...", flush=True)
a_obs = ad.read_h5ad(SRC, backed="r")
obs = a_obs.obs.copy()
X_state = np.asarray(a_obs.obsm["X_state"][:], dtype=np.float32)
print(f"[{ts()}] obs {obs.shape}, X_state {X_state.shape}", flush=True)

# ---------- 3. X 分块：解压 → log1p → csr → 列重映射 ----------
blocks = []
with h5py.File(SRC, "r") as h:
    X_ds = h["X"]
    for s in range(0, n_obs, CHUNK):
        e = min(s + CHUNK, n_obs)
        Xb = np.asarray(X_ds[s:e])           # dense (chunk, 8248)
        np.log1p(Xb, out=Xb)
        c = sp.csr_matrix(Xb)
        del Xb
        mapped = col_map[c.indices]          # (nnz,) 向量化列重映射
        valid = mapped >= 0
        row_ids = np.repeat(np.arange(e - s, dtype=np.int64), np.diff(c.indptr))
        new_counts = np.bincount(row_ids[valid], minlength=e - s)
        new_indptr = np.concatenate([[0], np.cumsum(new_counts)]).astype(np.int64)
        blocks.append(sp.csr_matrix(
            (c.data[valid], mapped[valid].astype(np.int32), new_indptr),
            shape=(e - s, len(panel)),
        ))
        del c, mapped, valid, row_ids
        print(f"[{ts()}]   块 {s:>9,}:{e:>9,} 完成", flush=True)

print(f"[{ts()}] vstack {len(blocks)} 块 ...", flush=True)
X_full = sp.vstack(blocks, format="csr")
del blocks
print(f"[{ts()}] X_full {X_full.shape}, nnz={X_full.nnz:,}", flush=True)

# ---------- 4. 过滤缺 ESM2 特征的扰动 ----------
esm2_keys = set(torch.load(ESM2, map_location="cpu").keys())
pert_all = obs["gene"].astype(str)
CTRL = "non-targeting"  # 对照不是基因，不在 ESM2 keys，但必须保留
missing = sorted(set(pert_all.unique()) - esm2_keys - {CTRL})
keep_mask = ~pert_all.isin(missing).to_numpy()
print(f"[{ts()}] 缺 ESM2 扰动 {len(missing)} 个，过滤 {(~keep_mask).sum():,} / {len(keep_mask):,} 细胞", flush=True)
X_full = X_full[keep_mask]
obs = obs.loc[keep_mask].copy()
X_state = X_state[keep_mask]

# ---------- 5. obs 列对齐 + 写盘 ----------
obs["target_gene"] = obs["gene"].astype(str).astype("category")
obs["batch_var"] = obs["batch"].astype(str).astype("category")
obs["cell_type"] = "k562"  # 小写，复用 k562_gwps.h5 的 one-hot 维度
obs["cell_type"] = obs["cell_type"].astype("category")

out = ad.AnnData(X=X_full, obs=obs)
out.obsm["X_state"] = X_state
out.uns["log1p"] = {}

print(f"[{ts()}] 写盘（无压缩）→ {DST}", flush=True)
out.write_h5ad(DST)  # 必须不压缩，见文件头注释
print(f"[{ts()}] 完成 {out.shape}", flush=True)
