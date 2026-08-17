"""V7：Replogle 全量参考 vs 184 子集 —— 方向质量对照（模拟器核心问题）。

对照：k562_gwps（184 扰动子集，V2 原结果 cosine=0.145 / PDS_x64=0.666）
实验：replogle_full（9,867 扰动全量）单独做参考
另试：replogle_full + 4 个小参考联合均匀平均
输出：PDS(×64)、cosine、pearson_top500、分层 PDS
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2v3_simulator import OUT, REFS, load, pds_rank, cosine, top_pearson

SCALE = 64.0


def load_full():
    d = np.load(os.path.join(OUT, "deltas_replogle_full.npz"), allow_pickle=False)
    # NpzFile.__getitem__ 不缓存：d["deltas"] 每访问一次就重新解压整个数组，
    # 必须先取出再切片（否则 9867 次全量解压 ≈ 6.8TB，跑不完）
    deltas = d["deltas"]
    perts = d["perts"]
    return {p: deltas[i].astype(np.float64) for i, p in enumerate(perts)}


def run(h1, H1mat, perts_g, heldout, ref_dict, name):
    rows = []
    for g in heldout:
        if g not in ref_dict:
            continue
        gi = perts_g[g]
        true = h1[g]
        dref = ref_dict[g]
        pr = dref * SCALE
        rows.append(dict(pert=g, pds=pds_rank(pr, H1mat, gi),
                         cosine=cosine(dref, true),
                         pearson_top500=top_pearson(dref, true),
                         h1_l1=float(np.abs(true).sum())))
    df = pd.DataFrame(rows)
    df["tier"] = pd.qcut(df.h1_l1, 3, labels=["weak", "mid", "strong"])
    print(f"\n--- {name}（{len(df)} 个留出扰动）---")
    print("PDS_x64 mean=%.4f median=%.4f | cosine=%.4f | pearson_top500=%.4f"
          % (df.pds.mean(), df.pds.median(), df.cosine.mean(), df.pearson_top500.mean()))
    print("分层 PDS:", df.groupby("tier", observed=True).pds.mean().round(4).to_dict())
    return df


def main():
    h1 = load("H1")
    full = load_full()
    sub = load("k562_gwps")
    small_refs = {r: load(r) for r in ["rpe1", "jurkat", "k562", "hepg2"]}
    h1_perts = sorted(h1)
    H1mat = np.array([h1[p] for p in h1_perts])
    perts_g = {p: i for i, p in enumerate(h1_perts)}
    heldout = [p for p in h1_perts if p in full]
    print(f"H1 ∩ replogle_full = {len(heldout)}/150（子集参考时为 136）")

    run(h1, H1mat, perts_g, heldout, {p: sub[p] for p in heldout if p in sub},
        "A. k562_gwps 184 子集（基线复现）")
    run(h1, H1mat, perts_g, heldout, full, "B. replogle_full 全量")
    # C. 全量 + 小参考联合（可用即平均）
    combo = {}
    for g in heldout:
        vs = [full[g]] + [small_refs[r][g] for r in small_refs if g in small_refs[r]]
        combo[g] = np.mean(vs, axis=0)
    run(h1, H1mat, perts_g, heldout, combo, "C. 全量+4小参考 均匀平均")


if __name__ == "__main__":
    main()
