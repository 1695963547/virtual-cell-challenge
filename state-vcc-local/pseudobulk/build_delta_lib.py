"""构建 pseudo-bulk delta 库 + V1 跨细胞系保守性分析。

输入：state/vci_pretrain/ 下 6 个 h5（log1p 表达，18080 基因面板一致）
输出：state-vcc-local/pseudobulk/deltas_<name>.npz（deltas, perts, genes）
     state-vcc-local/pseudobulk/v1_conservation.csv / 控制台摘要

delta = mean(log1p expr | pert) - mean(log1p expr | non-targeting)，等价 logFC 方向。
"""
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE = "/home/zjh/state-vcc-local/state/vci_pretrain"
OUT = "/home/zjh/state-vcc-local/pseudobulk"
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "H1": "competition_train.h5",
    "k562_gwps": "k562_gwps.h5",
    "rpe1": "rpe1.h5",
    "jurkat": "jurkat.h5",
    "k562": "k562.h5",
    "hepg2": "hepg2.h5",
}
CTRL = "non-targeting"


def group_means(X, codes, n_cat):
    """按 codes 分组求均值，返回 (n_cat, n_genes) float32。"""
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n_cat)
    bounds = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    Xs = X[order]
    if sp.issparse(Xs):
        Xs = Xs.tocsr()
        sums = np.vstack(
            [np.asarray(Xs[bounds[i]:bounds[i + 1]].sum(axis=0)).ravel() for i in range(n_cat)]
        )
    else:
        sums = np.add.reduceat(Xs, bounds[:-1], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts.astype(np.float64)[:, None]
    return means.astype(np.float32), counts


def build(name, fname):
    t0 = time.time()
    path = os.path.join(BASE, fname)
    a = ad.read_h5ad(path)
    tg = a.obs["target_gene"]
    cats = list(tg.cat.categories)
    codes = tg.cat.codes.to_numpy()
    X = a.X
    print(f"[{name}] loaded {a.shape}, {len(cats)} cats, {time.time()-t0:.0f}s", flush=True)
    means, counts = group_means(X, codes, len(cats))
    del X, a
    if CTRL not in cats:
        raise ValueError(f"{name}: no control label")
    ci = cats.index(CTRL)
    ctrl = means[ci]
    keep = [i for i, c in enumerate(cats) if c != CTRL]
    perts = np.array([cats[i] for i in keep])
    deltas = means[keep] - ctrl[None, :]
    np.savez_compressed(
        os.path.join(OUT, f"deltas_{name}.npz"),
        deltas=deltas, perts=perts, counts=counts[keep],
    )
    print(f"[{name}] deltas {deltas.shape}, saved, {time.time()-t0:.0f}s", flush=True)


def load_genes():
    with __import__("h5py").File(os.path.join(BASE, "competition_train.h5"), "r") as h:
        return np.array([x.decode() for x in h["var"]["_index"][:]])


def topk_union(d1, d2, k):
    return np.union1d(np.argsort(-np.abs(d1))[:k], np.argsort(-np.abs(d2))[:k])


def v1_analysis():
    genes = load_genes()
    H1 = np.load(os.path.join(OUT, "deltas_H1.npz"), allow_pickle=False)
    h1 = {p: H1["deltas"][i] for i, p in enumerate(H1["perts"])}
    rows = []
    for name in ["k562_gwps", "rpe1", "jurkat", "k562", "hepg2"]:
        D = np.load(os.path.join(OUT, f"deltas_{name}.npz"), allow_pickle=False)
        other = {p: D["deltas"][i] for i, p in enumerate(D["perts"])}
        overlap = sorted(set(h1) & set(other))
        print(f"\n=== H1 ∩ {name}: {len(overlap)} 个重叠扰动 ===", flush=True)
        for p in overlap:
            d1, d2 = h1[p], other[p]
            pr_full = np.corrcoef(d1, d2)[0, 1]
            idx = topk_union(d1, d2, 500)
            pr_top = np.corrcoef(d1[idx], d2[idx])[0, 1]
            idx200 = topk_union(d1, d2, 200)
            s = np.sign(d1[idx200]) * np.sign(d2[idx200])
            sign_cos = float((np.sign(d1[idx200]) @ np.sign(d2[idx200])) / len(idx200))
            rows.append(dict(ref=name, pert=p, pearson_full=pr_full,
                             pearson_top500=pr_top, sign_agree_top200=s.mean(),
                             sign_cos_top200=sign_cos,
                             h1_l1=np.abs(d1).sum(), ref_l1=np.abs(d2).sum()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v1_conservation.csv"), index=False)
    print("\n===== V1 保守性摘要（按参考数据集）=====")
    print(df.groupby("ref")[["pearson_full", "pearson_top500", "sign_agree_top200"]]
            .agg(["count", "median", "mean"]).round(3).to_string())
    for th in [0.1, 0.2, 0.3, 0.5]:
        frac = (df["pearson_top500"] > th).mean()
        print(f"pearson_top500 > {th}: {frac:.1%}")


if __name__ == "__main__":
    import sys
    if "build" in sys.argv:
        for name, fn in DATASETS.items():
            if not os.path.exists(os.path.join(OUT, f"deltas_{name}.npz")):
                build(name, fn)
    if "v1" in sys.argv:
        v1_analysis()
