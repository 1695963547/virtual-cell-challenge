#!/usr/bin/env python
"""check_hvg_coverage.py: 实证回答"<48 过滤"和"HVG 5000 截断"对得分的影响。
1) val 47 个扰动基因在 5 个训练 h5 中的扰动细胞数（是否 >=48）
2) test 真值扰动效应基因在 HVG 5000 中的覆盖率（DES/PDS 损失量化）
"""
import h5py
import numpy as np
import pandas as pd

DATA_DIR = "/home/zjh/state-vcc-local/state/vci_pretrain"
FILES = [
    ("competition_train", "H1"),
    ("k562_gwps", "K562"),
    ("rpe1", "RPE1"),
    ("jurkat", "Jurkat"),
    ("k562", "K562"),
]
TEST_H5AD = "/home/zjh/vcc_official/adata_Test.h5ad"
VAL_H5AD = "/home/zjh/vcc_official/validation/adata_Validation.h5ad"


def read_obs_col(obs_group, name, n):
    g = obs_group[name]
    if isinstance(g, h5py.Dataset):
        return g[:n].astype(str)
    cats = g["categories"][:].astype(str)
    return cats[g["codes"][:n]]


def step1_val_pert_counts():
    """val 扰动的基因名，在 5 训练文件里作为扰动出现过吗？细胞数多少？"""
    # val 扰动列表（obs 列名是 target_gene）
    with h5py.File(VAL_H5AD, "r") as f:
        obs_g = f["obs"]
        col = "target_gene" if "target_gene" in obs_g else "condition"
        n = obs_g[col]["codes"].shape[0]
        tgt = read_obs_col(obs_g, col, n)
        val_perts = sorted(set(tgt) - {"control", "non-targeting"})
    print(f"[1] val 扰动数: {len(val_perts)}")

    # 各文件扰动 -> 细胞数
    pert_cells = {}  # gene -> {file: count}
    for name, cl in FILES:
        path = f"{DATA_DIR}/{name}.h5"
        with h5py.File(path, "r") as f:
            n = f["obs"]["target_gene"]["codes"].shape[0]
            tgt = read_obs_col(f["obs"], "target_gene", n)
            vals, cnts = np.unique(tgt, return_counts=True)
        for g, c in zip(vals, cnts):
            if g in val_perts:
                pert_cells.setdefault(g, {})[name] = int(c)

    rows = []
    for g in val_perts:
        if g in pert_cells:
            for name, c in pert_cells[g].items():
                rows.append((g, name, c))
        else:
            rows.append((g, "-", 0))
    df = pd.DataFrame(rows, columns=["val_gene", "file", "pert_cells"])
    small = df[df["pert_cells"] > 0]
    print(f"[1] val 扰动在训练文件中出现: {len(small[small.pert_cells>0].drop_duplicates('val_gene'))} 个")
    print(f"[1] 其中 <48 细胞的: {len(small[small.pert_cells < 48])} 条")
    if len(small[small.pert_cells < 48]):
        print(small[small.pert_cells < 48].to_string(index=False))
    print()


def step2_hvg_coverage():
    """test 真值 delta 效应基因在 HVG 5000 的覆盖率"""
    import anndata as ad
    import scanpy as sc

    print("[2] 加载 test h5ad ...")
    adata = ad.read_h5ad(TEST_H5AD, backed="r")
    print(f"    cells={adata.n_obs}, genes={adata.n_vars}")
    obs = adata.obs
    cond_col = "target_gene" if "target_gene" in obs.columns else "condition"
    cond = obs[cond_col].astype(str).to_numpy()
    ctrl_mask = np.isin(cond, ["control", "non-targeting"])
    perts = sorted(set(cond) - {"control", "non-targeting"})
    print(f"    test 扰动数: {len(perts)}, control 细胞: {ctrl_mask.sum()}")

    # 基因名
    var_names = adata.var_names.to_numpy().astype(str)

    # 读全 X（稀疏）
    X = adata.X[:]
    adata.file.close()
    import scipy.sparse as sp
    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = sp.csr_matrix(X)
    print(f"    X 读入: {X.shape}, nnz={X.nnz/1e6:.0f}M")

    # 官方 test 的 X 不是 log1p（max=2406, min=1）——先 log1p 到训练/预测口径
    X.data = np.log1p(X.data)
    print("    X -> log1p 完成")

    # control 均值
    ctrl_mean = np.asarray(X[ctrl_mask].mean(axis=0)).ravel()
    print("    control mean 完成")

    # HVG：一次算 top 12000，再按 rank 截断出不同规模（flavor=seurat，log1p 口径）
    adata2 = ad.AnnData(X=X, var=pd.DataFrame(index=var_names))
    sc.pp.highly_variable_genes(adata2, n_top_genes=12000)
    # 旧版 scanpy 无 rank 列：按 dispersions_norm 降序即 HVG 排名
    disp = adata2.var["dispersions_norm"].to_numpy()
    rank = np.argsort(np.argsort(-disp))  # rank 0 = 最可变
    print("    HVG top-12000 计算完成 (在 test 数据上，作为代理)")

    # 每个扰动：真值 delta = pert_mean - ctrl_mean
    topk_list = [50, 100, 200, 500]
    n_hvg_list = [3000, 5000, 8000, 10000, 12000]
    delta_per_pert = []
    for p in perts:
        m = cond == p
        pert_mean = np.asarray(X[m].mean(axis=0)).ravel()
        delta = pert_mean - ctrl_mean
        delta_per_pert.append(delta)
    delta_mat = np.stack(delta_per_pert)  # (100, 18080)

    print("\n[2] 真值 |delta| top-K 基因落在各规模 HVG 内的平均比例:")
    print(f"    {'HVG':>7} | " + " | ".join(f"top-{k:>4}" for k in topk_list) + " | 非HVG效应占比")
    for n in n_hvg_list:
        hvg_mask = rank < n
        cov = []
        for i, p in enumerate(perts):
            absd = np.abs(delta_mat[i])
            order = np.argsort(absd)[::-1]
            cov.append(hvg_mask[order[:100]].mean())
        non_hvg_frac = np.abs(delta_mat[:, ~hvg_mask]).sum() / np.abs(delta_mat).sum()
        print(f"    {n:>7} | {np.mean(cov)*100:>6.1f}%  (min={min(cov)*100:.0f}%) | {non_hvg_frac*100:.1f}%")
    print("    -> 非 HVG 预测填 0（无效应）时，PDS L1 距离中该部分贡献占比 ≈ 非HVG效应占比")


if __name__ == "__main__":
    step1_val_pert_counts()
    step2_hvg_coverage()
