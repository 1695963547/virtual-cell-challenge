"""FCN 推理：生成 cell-eval 兼容的预测 h5ad。

对测试集的每个扰动：
  1. 用 H1 ctrl_mean + ESM-2(gene) 预测 delta
  2. pred = expm1(ctrl_mean + delta) → 取整为 raw counts
  3. 非靶向对照保留真实表达值
  4. 复制给该扰动所有细胞（replicate，满足 Wilcoxon）

用法：
  cd /home/zjh/state-vcc-local/state && uv run --no-sync python ../pseudobulk/infer_fcn.py \
      --truth /home/zjh/vcc_official/adata_Test.h5ad \
      --ckpt /home/zjh/state-vcc-local/pseudobulk/fcn_best.pt \
      --output /home/zjh/state-vcc-local/state/competition/prediction_test_fcn.h5ad
"""
import argparse
import os
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_fcn import PerturbFCN

ESM_FILE = "/home/zjh/state-vcc-local/state/competition_support_set/ESM2_pert_features.pt"
CTRL_MEAN_FILE = "/home/zjh/state-vcc-local/pseudobulk/ctrl_mean_H1.npy"
GENE_DIM = 18080
ESM_DIM = 5120
CTRL = "non-targeting"


def load_model(ckpt_path, device):
    """加载 FCN 模型 checkpoint。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = PerturbFCN(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


def load_esm2():
    esm = torch.load(ESM_FILE, map_location="cpu", weights_only=True)
    return {k: v.numpy().astype(np.float32) for k, v in esm.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="测试真值 h5ad")
    ap.add_argument("--ckpt", required=True, help="FCN checkpoint")
    ap.add_argument("--output", required=True, help="输出预测 h5ad")
    ap.add_argument("--ctrl-mean", default=CTRL_MEAN_FILE, help="H1 ctrl_mean npy")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    model = load_model(args.ckpt, device)

    # 加载 ESM-2
    esm2 = load_esm2()
    print(f"ESM-2 genes: {len(esm2)}")

    # 加载 H1 ctrl_mean（log1p 空间）
    ctrl_mean = np.load(args.ctrl_mean).astype(np.float64)
    print(f"H1 ctrl_mean: shape={ctrl_mean.shape}, norm={np.linalg.norm(ctrl_mean):.2f}")

    # 加载测试真值
    print(f"Loading truth: {args.truth}")
    a = ad.read_h5ad(args.truth)
    print(f"  {a.shape}, obs columns: {list(a.obs.columns)}")

    # 转换到 log1p 空间
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.copy()
    X.data = np.log1p(X.data.astype(np.float64))

    tg = a.obs["target_gene"].astype(str).to_numpy()
    ctrl_mask = tg == CTRL

    # 收集所有扰动基因
    uniq_perts = sorted(set(tg) - {CTRL})
    print(f"Perturbations: {len(uniq_perts)}, Control cells: {ctrl_mask.sum()}")

    # 预测 delta
    esm_dim = ESM_DIM
    zero_esm = np.zeros(esm_dim, dtype=np.float32)
    delta_map = {}
    n_covered = 0

    ctrl_tensor = torch.from_numpy(ctrl_mean.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        for p in uniq_perts:
            esm = esm2.get(p, zero_esm)
            esm_tensor = torch.from_numpy(esm).unsqueeze(0).to(device)
            pred_delta = model(ctrl_tensor, esm_tensor).cpu().numpy().ravel()
            delta_map[p] = pred_delta.astype(np.float64)
            if p in esm2:
                n_covered += 1

    print(f"ESM-2 coverage: {n_covered}/{len(uniq_perts)}")

    # 构建预测矩阵（log1p 空间 → counts 空间）
    # 向量化：对照细胞保留真实值，扰动细胞 = ctrl_mean + delta
    print("Building prediction matrix...")
    Xp = np.empty((len(tg), X.shape[1]), dtype=np.float64)

    # 扰动细胞：按扰动分组填充
    for p in uniq_perts:
        mask = tg == p
        Xp[mask] = ctrl_mean + delta_map[p]

    # 对照细胞：保留真实 log1p 值
    Xp[ctrl_mask] = np.asarray(X[ctrl_mask].todense(), dtype=np.float64)

    # log1p → counts（expm1 + 取整）
    Xp = np.expm1(Xp)
    np.clip(Xp, 0, 1e9, out=Xp)
    Xp = np.rint(Xp)

    print(f"Prediction range: [{Xp.min():.0f}, {Xp.max():.0f}], mean={Xp.mean():.2f}")

    # 构建 AnnData
    pred = ad.AnnData(
        X=sp.csr_matrix(Xp.astype(np.float32)),
        obs=a.obs.copy(),
        var=a.var.copy(),
    )

    pred.write_h5ad(args.output)
    print(f"Output: {args.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
