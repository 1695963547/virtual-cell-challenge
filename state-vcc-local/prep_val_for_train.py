"""把 vci_pretrain/adata_Validation.h5ad（带 X_state）改造成可进 cell_load 训练管线的文件：

1. obs 补 cell_type="ARC_H1"（复用训练集 H1 的细胞类型 one-hot，避免新增未训练维度；
   TOML 里用 dataset 级 key "official_val.ARC_H1"=val 只把这个文件切到 val，
   competition_train 的 ARC_H1 不受影响）
2. obs 补 batch_var（复制 batch 列，训练管线 batch_col=batch_var）
3. X 从 raw counts 转 log1p，并补 uns["log1p"] 标记 —— 训练文件全部带该标记，
   output_space=all 下 mixed 标记会直接 ValueError（_setup_global_maps）
4. 输出新文件 adata_Validation_trainready.h5ad，原嵌入版保留不动
"""
import anndata as ad
import numpy as np
import scipy.sparse as sp

SRC = "/home/zjh/state-vcc-local/state/vci_pretrain/adata_Validation.h5ad"
DST = "/home/zjh/state-vcc-local/state/vci_pretrain/adata_Validation_trainready.h5ad"

print("读取", SRC)
a = ad.read_h5ad(SRC)
print("  shape:", a.shape, "| obs:", list(a.obs.columns), "| obsm:", list(a.obsm.keys()))

X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X))
print("  X max(raw):", float(X.max()))
X = X.copy()
X.data = np.log1p(X.data)
a.X = X
print("  X max(log1p):", float(X.max()))

a.obs["batch_var"] = a.obs["batch"].astype(str).astype("category")
a.obs["cell_type"] = "ARC_H1"
a.obs["cell_type"] = a.obs["cell_type"].astype("category")
a.uns["log1p"] = {}

# 必须不压缩：cell_load 随机行读，gzip+chunk 会把每次行读放大成整 chunk 解压
# （实测 64 workers 下每 worker 读了 293GB 仍出不了第一个 batch）；
# 训练文件与 transform 原始输出均为无压缩连续存储
a.write_h5ad(DST)
print("写出", DST)
