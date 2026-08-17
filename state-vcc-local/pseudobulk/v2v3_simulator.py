"""V2/V3：未见扰动模拟器 —— 基因空间迁移 vs 程序空间迁移。

设定：H1 的 150 个扰动中，凡在参考数据集（k562_gwps/rpe1/jurkat/k562/hepg2）出现过
的，轮流当作"未见测试扰动"：只用参考细胞系的 delta 做迁移预测，与 H1 真值比较。

预测器（全部由简到繁）：
  zero        : 全零（对照）
  h1_mean     : 其余 H1 扰动的均值 delta（全局响应先验 ≈ 扰动均值 baseline）
  gene_direct : 参考细胞系 delta 直接迁移（多参考取平均）
  gene_scaled : gene_direct × 保守性标量 s（s 由训练对最小二乘估计，思路②）
  prog        : 程序空间迁移——联合 PCA 子空间投影去噪 + 逐程序 H1/参考方差校准（思路③）

评测（严格仿官方 PDS 定义）：
  PDS：预测 delta 与全部 150 个 H1 真值 delta 的 L1 距离排序，PDS_g = 1-(r-1)/N
  另报：全向量 cosine、top500-union pearson
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

OUT = "/home/zjh/state-vcc-local/pseudobulk"
REFS = ["k562_gwps", "rpe1", "jurkat", "k562", "hepg2"]
K_PROG = 50  # 程序数


def load(name):
    d = np.load(os.path.join(OUT, f"deltas_{name}.npz"), allow_pickle=False)
    # NpzFile 不缓存：先取出再切片，避免每行一次全量解压
    deltas = d["deltas"]
    perts = d["perts"]
    return {p: deltas[i].astype(np.float64) for i, p in enumerate(perts)}


def pds_rank(pred, pool, true_idx):
    d = np.abs(pool - pred).sum(axis=1)
    order = np.argsort(d, kind="stable")
    rank = int(np.where(order == true_idx)[0][0]) + 1
    return 1.0 - (rank - 1) / len(pool)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def top_pearson(a, b, k=500):
    idx = np.union1d(np.argsort(-np.abs(a))[:k], np.argsort(-np.abs(b))[:k])
    if idx.size < 10 or a[idx].std() == 0 or b[idx].std() == 0:
        return 0.0
    return float(np.corrcoef(a[idx], b[idx])[0, 1])


def main():
    h1 = load("H1")
    refs = {r: load(r) for r in REFS}
    h1_perts = sorted(h1)
    H1mat = np.array([h1[p] for p in h1_perts])  # 150 x 18080, PDS 真值池
    perts_g = {p: i for i, p in enumerate(h1_perts)}

    heldout = [p for p in h1_perts if any(p in refs[r] for r in REFS)]
    print(f"H1 扰动 {len(h1_perts)} 个，可在参考库中找到的留出扰动 {len(heldout)} 个")

    refmat_all = np.vstack([np.array(list(refs[r].values())) for r in REFS])
    rows = []
    for g in heldout:
        true = h1[g]
        gi = perts_g[g]
        train_idx = [i for i, p in enumerate(h1_perts) if p != g]
        H1_train = H1mat[train_idx]

        pred = {}
        pred["zero"] = np.zeros_like(true)
        pred["h1_mean"] = H1_train.mean(axis=0)

        dref = np.mean([refs[r][g] for r in REFS if g in refs[r]], axis=0)
        pred["gene_direct"] = dref

        # 保守性标量：用训练扰动中 H1∩参考 的配对估计 s = <dH1,dref>/<dref,dref> 的中位
        pairs = [p for p in h1_perts if p != g and any(p in refs[r] for r in REFS)]
        ss = []
        for p in pairs:
            dp = np.mean([refs[r][p] for r in REFS if p in refs[r]], axis=0)
            denom = dp @ dp
            if denom > 0:
                ss.append((h1[p] @ dp) / denom)
        s = float(np.median(ss)) if ss else 1.0
        pred["gene_scaled"] = dref * s

        # 稀疏迁移：只保留参考 delta 中 |效应| 最大的 top-k 个基因，其余置零
        for k in [50, 100, 200, 500]:
            idx = np.argsort(-np.abs(dref))[:k]
            sp_vec = np.zeros_like(dref)
            sp_vec[idx] = dref[idx]
            pred[f"gene_top{k}"] = sp_vec
            pred[f"gene_top{k}_scaled"] = sp_vec * s

        # 全局缩放扫描（验证 Outlier 式 scaling：放大后 PDS 排序是否转向方向主导）
        for sc in [2, 4, 8, 16, 32, 64, 128, 256]:
            pred[f"gene_direct_x{sc}"] = dref * sc
        # 稀疏 × 缩放交互
        for k in [100, 200]:
            idx = np.argsort(-np.abs(dref))[:k]
            sp_vec = np.zeros_like(dref)
            sp_vec[idx] = dref[idx]
            for sc in [16, 64, 256]:
                pred[f"gene_top{k}_x{sc}"] = sp_vec * sc

        # 程序空间：联合 PCA（参考库全部 + H1 训练扰动，不含留出 g）
        stack = np.vstack([H1_train, refmat_all])
        svd = TruncatedSVD(n_components=K_PROG, random_state=0).fit(stack)
        W = svd.components_  # K x G 共享程序
        c_ref = W @ dref  # 留出扰动在程序空间的激活（来自参考）
        # 逐程序校准：H1 与参考在每个程序上的响应幅度比
        A_h1 = (W @ H1_train.T).T  # n_train x K
        A_ref = (W @ refmat_all.T).T
        ratio = (np.abs(A_h1).mean(axis=0) + 1e-8) / (np.abs(A_ref).mean(axis=0) + 1e-8)
        pred["prog"] = (c_ref * ratio) @ W

        for name, pr in pred.items():
            rows.append(dict(
                pert=g, model=name,
                pds=pds_rank(pr, H1mat, gi),
                cosine=cosine(pr, true),
                pearson_top500=top_pearson(pr, true),
                h1_l1=float(np.abs(true).sum()),
                ref_l1=float(np.abs(dref).sum()),
            ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v2v3_simulator.csv"), index=False)
    summ = df.groupby("model")[["pds", "cosine", "pearson_top500"]].mean().round(4)
    summ["pds_median"] = df.groupby("model")["pds"].median().round(4)
    print("\n===== V2/V3 模拟器结果（%d 个留出扰动均值）=====" % len(heldout))
    print(summ.sort_values("pds", ascending=False).to_string())

    # 按 H1 效应强度分桶看迁移效果（验证 "subtle 扰动应预测≈0" 的判断）
    df["h1_effect_tier"] = pd.qcut(df["h1_l1"], 3, labels=["weak", "mid", "strong"])
    key_models = ["zero", "h1_mean", "gene_direct", "gene_direct_x16", "gene_direct_x64",
                  "gene_direct_x256", "gene_top100_x64", "gene_top200_x64", "prog"]
    piv = (df[df.model.isin(key_models)]
           .groupby(["model", "h1_effect_tier"], observed=True)["pds"].mean().round(4)
           .unstack())
    print("\n===== PDS 按 H1 效应强度分桶 =====")
    print(piv.to_string())
    # 保守性与参考效应强度的关系（迁移 gate 的特征依据）
    sub = df[df.model == "gene_direct"]
    print("\ncorr(ref_l1, pds) =", round(np.corrcoef(sub["ref_l1"], sub["pds"])[0, 1], 3),
          " corr(h1_l1, pds) =", round(np.corrcoef(sub["h1_l1"], sub["pds"])[0, 1], 3))
    print("\n判定：gene_scaled > gene_direct > h1_mean → 思路②成立；prog > gene_scaled → 思路③成立")


if __name__ == "__main__":
    main()
