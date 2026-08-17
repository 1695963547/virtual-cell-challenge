"""V4：相似度加权多参考聚合 + 弱效应 gate 验证（在 V2 模拟器框架内）。

A. gate 特征分析：ref_l1 / 靶基因 H1 基础表达 对 H1 效应强度(h1_l1)的预测力
B. 新增预测器：
   wavg_x64  : 多参考按"与H1的保守性"加权（权重在训练配对上估计，无泄漏），再 x64
   gate_x64  : 用训练配对学 ref_l1→h1_l1 线性映射，预测 h1 弱效应则输出≈0，否则 dref x64
判定：wavg_x64 > gene_direct_x64 → 加权聚合有效；gate_x64 > gene_direct_x64 → gate 有效
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2v3_simulator import OUT, REFS, load, pds_rank, cosine, top_pearson

SCALE = 64.0
CTRL = "non-targeting"


def h1_basal_expr():
    """H1 对照细胞的平均表达（log1p 空间），18080 维。"""
    cache = os.path.join(OUT, "h1_ctrl_mean.npy")
    if os.path.exists(cache):
        return np.load(cache)
    a = ad.read_h5ad("/home/zjh/state-vcc-local/state/vci_pretrain/competition_train.h5")
    codes = a.obs["target_gene"].cat.codes.to_numpy()
    ci = list(a.obs["target_gene"].cat.categories).index(CTRL)
    import scipy.sparse as sp
    Xc = a.X[codes == ci]
    m = np.asarray(Xc.mean(axis=0)).ravel() if sp.issparse(Xc) else Xc.mean(axis=0)
    np.save(cache, m.astype(np.float32))
    return m


def main():
    h1 = load("H1")
    refs = {r: load(r) for r in REFS}
    genes = np.load("/home/zjh/state-vcc-local/pseudobulk/deltas_H1.npz")["perts"]  # noqa: F841 (仅触发文件存在)
    basal = h1_basal_expr()
    h1_perts = sorted(h1)
    H1mat = np.array([h1[p] for p in h1_perts])
    perts_g = {p: i for i, p in enumerate(h1_perts)}
    heldout = [p for p in h1_perts if any(p in refs[r] for r in REFS)]

    # 基因名→列索引（靶基因基础表达查表用）；var 顺序 = competition_train 的 var
    import h5py
    with h5py.File("/home/zjh/state-vcc-local/state/vci_pretrain/competition_train.h5", "r") as h:
        var_index = [x.decode() for x in h["var"]["_index"][:]]
    g2i = {g: i for i, g in enumerate(var_index)}

    # ---------- A. gate 特征分析 ----------
    rows_a = []
    for p in heldout:
        rl1 = max((np.abs(refs[r][p]).sum() for r in REFS if p in refs[r]), default=0)
        rows_a.append(dict(pert=p, h1_l1=np.abs(h1[p]).sum(), ref_l1=rl1,
                           basal_expr=basal[g2i[p]] if p in g2i else np.nan))
    A = pd.DataFrame(rows_a)
    print("===== A. 弱效应 gate 特征分析（136 对）=====")
    print("corr(ref_l1, h1_l1)        =", round(np.corrcoef(A.ref_l1, A.h1_l1)[0, 1], 3))
    ok = A.basal_expr.notna()
    print("corr(basal_expr, h1_l1)    =", round(np.corrcoef(A.basal_expr[ok], A.h1_l1[ok])[0, 1], 3))
    # 联合线性模型的 R²
    X = np.column_stack([A.ref_l1, A.basal_expr.fillna(A.basal_expr.median()), np.ones(len(A))])
    beta, *_ = np.linalg.lstsq(X, A.h1_l1, rcond=None)
    pred = X @ beta
    ss_res = ((A.h1_l1 - pred) ** 2).sum()
    ss_tot = ((A.h1_l1 - A.h1_l1.mean()) ** 2).sum()
    print("R²(ref_l1 + basal_expr → h1_l1) =", round(1 - ss_res / ss_tot, 3))

    # ---------- B. 模拟器新增预测器 ----------
    rows = []
    for g in heldout:
        true = h1[g]
        gi = perts_g[g]
        train_pairs = [p for p in heldout if p != g]

        # 每个参考细胞系的保守性权重（训练配对上的中位 pearson_top500）
        w = {}
        for r in REFS:
            if g not in refs[r]:
                continue
            cors = []
            for p in train_pairs:
                if p in refs[r]:
                    d1, d2 = h1[p], refs[r][p]
                    idx = np.union1d(np.argsort(-np.abs(d1))[:500], np.argsort(-np.abs(d2))[:500])
                    if d1[idx].std() > 0 and d2[idx].std() > 0:
                        cors.append(np.corrcoef(d1[idx], d2[idx])[0, 1])
            w[r] = max(np.median(cors), 0.01) if cors else 0.0
        avail = [r for r in REFS if g in refs[r]]
        wsum = sum(w.get(r, 0) for r in avail)
        dref_w = (np.mean([refs[r][g] for r in avail], axis=0) if wsum == 0
                  else sum(w[r] * refs[r][g] for r in avail) / wsum)

        # gate：训练配对上 ref_l1 → h1_l1 线性回归，预测弱效应则输出 0
        tp_r = np.array([max((np.abs(refs[r][p]).sum() for r in REFS if p in refs[r]), default=0)
                         for p in train_pairs])
        tp_h = np.array([np.abs(h1[p]).sum() for p in train_pairs])
        A_ = np.column_stack([tp_r, np.ones(len(tp_r))])
        b, *_ = np.linalg.lstsq(A_, tp_h, rcond=None)
        g_rl1 = max((np.abs(refs[r][g]).sum() for r in avail), default=0)
        h1_hat = b[0] * g_rl1 + b[1]
        weak_cut = np.quantile(tp_h, 1 / 3)  # 与 V2 分桶同口径的弱效应阈值

        dref = np.mean([refs[r][g] for r in avail], axis=0)
        pred = {
            "gene_direct_x64": dref * SCALE,
            "wavg_x64": dref_w * SCALE,
            "gate_x64": np.zeros_like(true) if h1_hat < weak_cut else dref * SCALE,
        }
        for name, pr in pred.items():
            rows.append(dict(pert=g, model=name, pds=pds_rank(pr, H1mat, gi),
                             cosine=cosine(pr, true), pearson_top500=top_pearson(pr, true),
                             h1_l1=np.abs(true).sum()))
    df = pd.DataFrame(rows)
    df["tier"] = pd.qcut(df.h1_l1, 3, labels=["weak", "mid", "strong"])
    print("\n===== B. 新预测器 PDS（总体 / 分层）=====")
    print(df.groupby("model")["pds"].agg(["mean", "median"]).round(4).to_string())
    print(df.groupby(["model", "tier"], observed=True)["pds"].mean().round(4).unstack().to_string())
    df.to_csv(os.path.join(OUT, "v4_gate_wavg.csv"), index=False)


if __name__ == "__main__":
    main()
