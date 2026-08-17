"""V5：模拟器 DES 半边 —— 迁移预测的 top-N 基因集 vs H1 真 DE 基因集（overlap_at_N 口径）。

真 DE 集来自 compute_de_calls.py（Wilcoxon FDR≤0.05，按 |log2fc| 排序）。
对每个留出扰动 g：k = |真 DE 集|，DES = |pred_topk ∩ true_DE| / k，
pred_topk = 预测 |delta| 最大的 k 个基因（正缩放不改变排序 → 稠密版一个就够）。

候选：gene_direct（稠密）/ gene_top{50,100,200,500}（稀疏）/ h1_mean / prog
另报：sign-only top-k（只按符号方向不看幅度一致性，上限参考）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2v3_simulator import OUT, REFS, load


def main():
    de = np.load(os.path.join(OUT, "de_calls_H1.npz"), allow_pickle=True)
    true_de = {p: de["de_idx"][i] for i, p in enumerate(de["perts"])}
    h1 = load("H1")
    refs = {r: load(r) for r in REFS}
    heldout = [p for p in sorted(h1) if any(p in refs[r] for r in REFS)]

    n_de = pd.Series({p: len(true_de.get(p, [])) for p in sorted(h1)})
    print("H1 真 DE 基因数分布: median=%d, q25=%d, q75=%d, zero=%d/150"
          % (n_de.median(), n_de.quantile(0.25), n_de.quantile(0.75), (n_de == 0).sum()))

    rows = []
    skipped = 0
    for g in heldout:
        tde = true_de.get(g)
        if tde is None or len(tde) == 0:
            skipped += 1
            continue
        k = len(tde)
        tset = set(tde.tolist())
        true = h1[g]
        avail = [refs[r][g] for r in REFS if g in refs[r]]
        dref = np.mean(avail, axis=0)

        pred = {"gene_direct": dref, "h1_mean": np.mean(
            [h1[p] for p in sorted(h1) if p != g], axis=0)}
        for kk in [50, 100, 200, 500]:
            idx = np.argsort(-np.abs(dref))[:kk]
            v = np.zeros_like(dref)
            v[idx] = dref[idx]
            pred[f"gene_top{kk}"] = v

        for name, pr in pred.items():
            topk = set(np.argsort(-np.abs(pr))[:k].tolist())
            des = len(topk & tset) / k
            rows.append(dict(pert=g, model=name, des=des, k_true=k,
                             h1_l1=float(np.abs(true).sum())))

    df = pd.DataFrame(rows)
    df["tier"] = pd.qcut(df.h1_l1, 3, labels=["weak", "mid", "strong"])
    df.to_csv(os.path.join(OUT, "v5_des.csv"), index=False)
    print(f"\n===== V5 DES（overlap_at_N，{df.pert.nunique()} 个留出扰动，跳过 {skipped} 个无 DE 扰动）=====")
    print(df.groupby("model")["des"].agg(["mean", "median"]).round(4)
          .sort_values("mean", ascending=False).to_string())
    print("\n===== DES 按效应强度分层 =====")
    print(df.groupby(["model", "tier"], observed=True)["des"].mean().round(4).unstack().to_string())
    print("\n判定：gene_topK > gene_direct → 稀疏化对 DES 有利（与 PDS 侧结论相反则按指标分治）")


if __name__ == "__main__":
    main()
