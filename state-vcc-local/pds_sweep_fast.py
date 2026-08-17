#!/usr/bin/env python3
"""
快速 PDS 放大倍数扫描：不跑 cell-eval，直接在内存中计算 PDS。

原理（与 cell-eval discrimination_score_l1 完全一致）：
  1. 对每个扰动 p，计算 delta = pert_mean - control_mean（pred 和 real 各一份）
  2. 排除目标基因（gene == p）
  3. 对每个 pred_delta[p]，用 L1 距离在 real_delta 中找最近的
  4. score = 1 - rank/n_perts，PDS = mean(scores)

用法：
  python pds_sweep_fast.py \
    --pred competition/prediction_test_state_lg.h5ad \
    --truth /home/zjh/vcc_official/adata_Test.h5ad \
    --scales 1,2,3,4,5,6,8,10,15,20
"""
import argparse
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist


def to_dense(X):
    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def compute_pert_means(adata, pert_col="target_gene"):
    """按扰动分组计算均值表达"""
    perts = adata.obs[pert_col].values
    unique_perts = np.unique(perts)
    X = to_dense(adata.X)

    pert_means = {}
    for p in unique_perts:
        mask = perts == p
        pert_means[p] = X[mask].mean(axis=0)

    return pert_means, unique_perts


def compute_pds(pred_deltas, real_deltas, perts, genes, exclude_target=True):
    """
    计算 PDS（与 cell-eval discrimination_score_l1 一致）

    pred_deltas: dict[pert] -> np.ndarray (n_genes,)
    real_deltas: dict[pert] -> np.ndarray (n_genes,)
    perts: list of perturbation names
    genes: list of gene names
    exclude_target: 是否排除目标基因
    """
    n_perts = len(perts)
    scores = []

    # 预计算所有 real_delta 的矩阵
    real_matrix = np.vstack([real_deltas[p] for p in perts])  # (n_perts, n_genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    for p_idx, p in enumerate(perts):
        pred_delta = pred_deltas[p]

        # 排除目标基因
        if exclude_target and p in gene_to_idx:
            target_idx = gene_to_idx[p]
            mask = np.ones(len(genes), dtype=bool)
            mask[target_idx] = False
            pred_vec = pred_delta[mask].reshape(1, -1)
            real_mat = real_matrix[:, mask]
        else:
            pred_vec = pred_delta.reshape(1, -1)
            real_mat = real_matrix

        # L1 距离
        distances = cdist(real_mat, pred_vec, metric="cityblock").flatten()

        # 排序找正确扰动的 rank
        sorted_indices = np.argsort(distances)
        rank = np.flatnonzero(sorted_indices == p_idx)[0]

        # score = 1 - rank / n_perts
        score = 1.0 - rank / n_perts
        scores.append(score)

    return np.mean(scores)


def main():
    parser = argparse.ArgumentParser(description="快速 PDS 放大倍数扫描")
    parser.add_argument("--pred", required=True, help="预测 h5ad 文件")
    parser.add_argument("--truth", required=True, help="真值 h5ad 文件")
    parser.add_argument("--scales", default="1,2,3,4,5,6,8,10,15,20",
                        help="逗号分隔的放大倍数列表")
    parser.add_argument("--pert-col", default="target_gene")
    parser.add_argument("--control-pert", default="non-targeting")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]

    print(f"加载预测: {args.pred}")
    t0 = time.time()
    pred = ad.read_h5ad(args.pred)
    print(f"  形状: {pred.shape}, 耗时: {time.time()-t0:.1f}s")

    print(f"加载真值: {args.truth}")
    t0 = time.time()
    truth = ad.read_h5ad(args.truth)
    print(f"  形状: {truth.shape}, 耗时: {time.time()-t0:.1f}s")

    genes = list(pred.var_names)
    print(f"基因数: {len(genes)}")

    # 计算每个扰动的均值表达
    print("\n计算扰动均值表达...")
    t0 = time.time()
    pred_means, pred_perts = compute_pert_means(pred, args.pert_col)
    real_means, real_perts = compute_pert_means(truth, args.pert_col)
    print(f"  预测扰动数: {len(pred_perts)}, 真值扰动数: {len(real_perts)}, 耗时: {time.time()-t0:.1f}s")

    # 对照组均值
    ctrl_pred = pred_means.get(args.control_pert)
    ctrl_real = real_means.get(args.control_pert)
    if ctrl_pred is None or ctrl_real is None:
        print(f"!! 找不到对照组 '{args.control_pert}'")
        return

    # 只保留两边都有的扰动（排除对照组）
    common_perts = sorted(set(pred_perts) & set(real_perts) - {args.control_pert})
    print(f"共同扰动数（排除对照）: {len(common_perts)}")

    # 计算 delta
    pred_deltas = {p: pred_means[p] - ctrl_pred for p in common_perts}
    real_deltas = {p: real_means[p] - ctrl_real for p in common_perts}

    # 扫描不同放大倍数
    print(f"\n{'Scale':>6} | {'PDS':>8} | {'Delta_mean':>10} | {'Delta_std':>10}")
    print("-" * 50)

    for scale in scales:
        # 放大 pred delta
        amplified_deltas = {p: pred_deltas[p] * scale for p in common_perts}

        # 计算 PDS
        t0 = time.time()
        pds = compute_pds(amplified_deltas, real_deltas, common_perts, genes,
                          exclude_target=True)
        elapsed = time.time() - t0

        # 统计 delta 幅度
        all_deltas = np.vstack([amplified_deltas[p] for p in common_perts])
        delta_mean = np.abs(all_deltas).mean()
        delta_std = all_deltas.std()

        print(f"{scale:>6.1f} | {pds:>8.4f} | {delta_mean:>10.4f} | {delta_std:>10.4f}  ({elapsed:.1f}s)")

    print("\n完成！选择 PDS 最高的倍数，然后用 pds_amplify.py 生成放大后的 h5ad 文件，")
    print("再跑完整 cell-eval 获得 DES/PDS/MAE 和 Overall Score。")


if __name__ == "__main__":
    main()
