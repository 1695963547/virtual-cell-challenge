"""V6：PDS×DES 联合调和 —— 迁移 + h1_mean 混合比例扫描。

背景：PDS 侧 gene_direct×64 最优（0.667），h1_mean 随机（0.48）；
     DES 侧 h1_mean 最优（0.316），gene_direct 次之（0.252）。
提交只能给一份预测 → 扫描 pred = (dref + λ·h1_mean) × 64 的 λ 对两个指标的联合影响。
注意模拟器 DES 的 h1_mean 优势含批次水分（H1 训练集内部共享批次响应），
真实比例留到 val 上复核；这里先看联合曲线形状。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2v3_simulator import OUT, REFS, load, pds_rank

SCALE = 64.0
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def main():
    de = np.load(os.path.join(OUT, "de_calls_H1.npz"), allow_pickle=True)
    true_de = {p: set(de["de_idx"][i].tolist()) for i, p in enumerate(de["perts"])}
    h1 = load("H1")
    refs = {r: load(r) for r in REFS}
    h1_perts = sorted(h1)
    H1mat = np.array([h1[p] for p in h1_perts])
    perts_g = {p: i for i, p in enumerate(h1_perts)}
    heldout = [p for p in h1_perts if any(p in refs[r] for r in REFS)]

    rows = []
    for g in heldout:
        gi = perts_g[g]
        tset = true_de.get(g, set())
        dref = np.mean([refs[r][g] for r in REFS if g in refs[r]], axis=0)
        hmean = np.mean([h1[p] for p in h1_perts if p != g], axis=0)
        k = len(tset)
        for lam in LAMBDAS:
            pr = (dref + lam * hmean) * SCALE
            pds = pds_rank(pr, H1mat, gi)
            des = (len(set(np.argsort(-np.abs(pr))[:k].tolist()) & tset) / k) if k > 0 else np.nan
            rows.append(dict(pert=g, lam=lam, pds=pds, des=des))
    df = pd.DataFrame(rows)
    summ = df.groupby("lam")[["pds", "des"]].mean().round(4)
    print("===== V6 混合比例扫描（136 留出扰动，pred=(dref+λ·h1_mean)×64）=====")
    print(summ.to_string())
    print("\nλ=0: 纯迁移；λ→∞: 纯 h1_mean。找 PDS 损失小、DES 增益大的 λ")
    df.to_csv(os.path.join(OUT, "v6_blend.csv"), index=False)


if __name__ == "__main__":
    main()
