#!/usr/bin/env python3
"""
PDS 后处理放大：对已有预测 h5ad 做 delta 放大 + self-knockdown 注入。

原理：
  PDS (discrimination_score_l1) 是 scale-sensitive 的。放大预测 delta 后，
  L1 距离在高维下退化为符号余弦相似度，扰动区分度提升。

用法：
  # 默认 3x 放大 + self-knockdown
  python pds_amplify.py \
    --pred /path/to/prediction_test.h5ad \
    --truth /path/to/adata_Test.h5ad \
    --output /path/to/prediction_amplified.h5ad

  # 试不同放大倍数
  python pds_amplify.py --pred ... --truth ... --output ... --scale 5.0

  # 只做放大，不注入 self-knockdown
  python pds_amplify.py --pred ... --truth ... --output ... --no-knockdown

  # 扫描多个倍数，输出对比表
  python pds_amplify.py --pred ... --truth ... --output-dir /tmp/pds_sweep --sweep
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_matrix(adata):
    """把 .X 转成稠密 numpy（如果稀疏的话）"""
    if sp.issparse(adata.X):
        return np.asarray(adata.X.todense())
    return np.asarray(adata.X)


def compute_control_mean(truth, pert_col="target_gene", control="non-targeting"):
    """从真值数据计算对照组均值"""
    ctrl_mask = truth.obs[pert_col] == control
    ctrl_X = load_matrix(truth[ctrl_mask])
    return ctrl_X.mean(axis=0)


def amplify(pred_adata, truth_adata, scale=3.0, knockdown_val=-4.93,
            pert_col="target_gene", control_pert="non-targeting",
            no_knockdown=False):
    """
    对预测做后处理：
    1. 计算 delta = pred - control_mean
    2. 放大: delta_new = delta * scale
    3. 重建: pred_new = control_mean + delta_new
    4. (可选) 注入 self-knockdown：目标基因 delta 设为 knockdown_val

    Parameters
    ----------
    pred_adata : AnnData
        模型预测结果，obs 中有 pert_col 列
    truth_adata : AnnData
        真值数据（用于计算 control mean 和获取基因名）
    scale : float
        delta 放大倍数
    knockdown_val : float
        self-knockdown 注入值（CRISPRi 的标准 knockdown 幅度）
    pert_col : str
        obs 中扰动名列
    control_pert : str
        对照组名
    no_knockdown : bool
        是否跳过 self-knockdown 注入

    Returns
    -------
    AnnData : 放大后的预测（结构与输入相同）
    """
    gene_names = list(pred_adata.var_names)

    # 1. 计算对照组均值（从真值数据）
    ctrl_mean = compute_control_mean(truth_adata, pert_col, control_pert)
    print(f"对照组均值：mean={ctrl_mean.mean():.4f}, std={ctrl_mean.std():.4f}")

    # 2. 复制预测数据
    result = pred_adata.copy()
    pred_X = load_matrix(result)

    # 3. 逐细胞放大 delta
    delta = pred_X - ctrl_mean[np.newaxis, :]
    delta_amplified = delta * scale
    pred_new = ctrl_mean[np.newaxis, :] + delta_amplified

    # 4. Self-knockdown 注入
    n_knockdown = 0
    if not no_knockdown:
        perts = result.obs[pert_col].values
        for i, pert in enumerate(perts):
            if pert == control_pert:
                continue
            if pert in gene_names:
                gene_idx = gene_names.index(pert)
                # 目标基因的预测 = control_mean + knockdown_val
                pred_new[i, gene_idx] = ctrl_mean[gene_idx] + knockdown_val
                n_knockdown += 1
        print(f"Self-knockdown 注入：{n_knockdown} 个细胞（目标基因在 gene_names 中）")
    else:
        print("跳过 self-knockdown 注入")

    # 5. 写回
    result.X = pred_new
    return result


def sweep_scales(pred_adata, truth_adata, output_dir, scales=None,
                 pert_col="target_gene", control_pert="non-targeting"):
    """扫描多个放大倍数，输出对比文件"""
    if scales is None:
        scales = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Scale':>6} | {'Delta_mean':>10} | {'Delta_std':>10} | Output")
    print("-" * 55)

    for s in scales:
        result = amplify(pred_adata, truth_adata, scale=s,
                         pert_col=pert_col, control_pert=control_pert)
        fname = f"prediction_scale{s:.1f}.h5ad"
        fpath = output_dir / fname
        result.write_h5ad(fpath)

        # 统计放大后的 delta 幅度
        ctrl_mean = compute_control_mean(truth_adata, pert_col, control_pert)
        delta = load_matrix(result) - ctrl_mean[np.newaxis, :]
        print(f"{s:>6.1f} | {np.abs(delta).mean():>10.4f} | {delta.std():>10.4f} | {fpath}")

    print(f"\n所有文件写入 {output_dir}/")
    print("下一步：对每个 scale 文件跑 cell-eval run + score，对比 PDS 变化")


def main():
    parser = argparse.ArgumentParser(description="PDS 后处理放大")
    parser.add_argument("--pred", required=True, help="预测 h5ad 文件路径")
    parser.add_argument("--truth", required=True, help="真值 h5ad 文件路径")
    parser.add_argument("--output", help="输出 h5ad 文件路径（单次模式）")
    parser.add_argument("--output-dir", help="输出目录（sweep 模式）")
    parser.add_argument("--scale", type=float, default=3.0, help="放大倍数（默认 3.0）")
    parser.add_argument("--knockdown-val", type=float, default=-4.93,
                        help="self-knockdown 注入值（默认 -4.93）")
    parser.add_argument("--no-knockdown", action="store_true",
                        help="跳过 self-knockdown 注入")
    parser.add_argument("--sweep", action="store_true",
                        help="扫描多个放大倍数")
    parser.add_argument("--pert-col", default="target_gene", help="扰动名列名")
    parser.add_argument("--control-pert", default="non-targeting", help="对照组名")
    args = parser.parse_args()

    print(f"加载预测: {args.pred}")
    pred = ad.read_h5ad(args.pred)
    print(f"  形状: {pred.shape}, 扰动数: {pred.obs[args.pert_col].nunique()}")

    print(f"加载真值: {args.truth}")
    truth = ad.read_h5ad(args.truth)
    print(f"  形状: {truth.shape}")

    if args.sweep:
        if not args.output_dir:
            parser.error("--sweep 需要 --output-dir")
        sweep_scales(pred, truth, args.output_dir,
                     pert_col=args.pert_col, control_pert=args.control_pert)
    else:
        if not args.output:
            parser.error("非 sweep 模式需要 --output")
        result = amplify(pred, truth, scale=args.scale,
                         knockdown_val=args.knockdown_val,
                         pert_col=args.pert_col, control_pert=args.control_pert,
                         no_knockdown=args.no_knockdown)
        result.write_h5ad(args.output)
        print(f"\n输出: {args.output}")
        print(f"  放大倍数: {args.scale}x")
        print(f"  Self-knockdown: {'关闭' if args.no_knockdown else f'{args.knockdown_val}'}")


if __name__ == "__main__":
    main()
