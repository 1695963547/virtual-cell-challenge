#!/usr/bin/env python
"""pred_de 全局 fdr 校准：对齐 n_sig 分布，修复 SPEARMAN (de_spearman_sig)。

背景（VCC_CHALLENGE_STATUS.md 4.x）：
- de_spearman_sig = 每扰动"显著基因数"(fdr<0.05) 的 Spearman 秩相关
- 模型预测的 fdr 普遍过小 → n_sig 饱和（pred 12663 vs real 3235，~4 倍虚高）
  → 每个扰动的 n_sig 都接近"全部基因数"，秩信息丢失 → SPEARMAN 低
- 全局缩放 fdr（fdr' = clip(fdr*s, 0, 1)）把 n_sig 拉回非饱和区间
- 缩放因子 s 在 val 上选定（有真值），应用到 test（不依赖真值）

影响矩阵（缩放 fdr 是单调变换）：
- SPEARMAN  n_sig 秩相关           -> 目标指标（受饱和效应影响）
- SPEARMAN_LFC  real 显著集 ∩ pred 全量 的 LFC 秩相关
                                    -> pred 的 fdr 不参与过滤 -> 不受影响
- DES overlap_at_N                 -> pred 侧 fdr 过滤后 |LFC| topk -> 受影响
- AUPRC     -log10(fdr) 排序 PR     -> 单调 -> 不变
- MAE/PDS/PEARSON                  -> 只依赖 adata.X -> 不变

用法：
    # 验证：扫 s 输出 SPEARMAN/n_sig 响应曲线（判断校准是否有效）
    python calibrate_nsig.py --pred-de pred_de.csv --real-de real_de.csv --mode verify

    # 拟合：在 val 上选 s（对齐 n_sig 中位数或总和），输出校准 csv + 报告
    python calibrate_nsig.py --pred-de pred_de.csv --real-de real_de.csv --mode fit \
        --target median --out pred_de_calibrated.csv --report nsig_report.json

    # 应用：给定 s 直接校准（test 上用 val 选定的 s）
    python calibrate_nsig.py --pred-de pred_de.csv --s 8.2 \
        --out pred_de_calibrated.csv --report nsig_report.json
"""
import argparse
import json
import sys

import numpy as np
import polars as pl
from scipy.stats import spearmanr

FDR_CUT = 0.05


def read_de(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, schema_overrides={"target": pl.Utf8, "feature": pl.Utf8})
    for col in ("target", "feature", "p_value", "fdr", "log2_fold_change"):
        if col not in df.columns:
            sys.exit(f"[calibrate_nsig] {path} 缺少列 {col}")
    return df


def nsig_map(df: pl.DataFrame) -> dict[str, int]:
    """每扰动显著基因数（fdr < 0.05）。"""
    d = df.filter(pl.col("fdr") < FDR_CUT).group_by("target").len()
    return {r["target"]: int(r["len"]) for r in d.iter_rows(named=True)}


def nsig_at_scale(df: pl.DataFrame, s: float) -> dict[str, int]:
    """给定全局缩放 s，fdr' = clip(fdr*s, 0, 1) 下的每扰动显著基因数。"""
    d = (
        df.filter((pl.col("fdr") * s) < FDR_CUT)
        .group_by("target")
        .len()
    )
    return {r["target"]: int(r["len"]) for r in d.iter_rows(named=True)}


def spearman_between(a: dict[str, int], b: dict[str, int]) -> float:
    """两个 per-pert 计数 map 的 Spearman 秩相关（按 target 对齐，缺失补 0）。"""
    keys = sorted(set(a) | set(b))
    if len(keys) < 2:
        return np.nan
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    if np.all(va == va[0]) or np.all(vb == vb[0]):
        return np.nan  # 秩相关无定义（常数）
    return float(spearmanr(va, vb).statistic)


def des_at_scale(pred_de: pl.DataFrame, real_de: pl.DataFrame, s: float) -> float:
    """复现 de_overlap_metric(metric='overlap', k=None, fdr_threshold=0.05, sort_by=|LFC|)：
    对每个扰动取 real 显著基因数为 k，比较 pred 显著基因 topk 的重叠率，再 mean。
    返回 0 表示无显著基因（cell-eval 对每个 pert 记 0）。"""
    scores = {}
    for pert in real_de["target"].unique().to_list():
        real_sig = real_de.filter(
            (pl.col("target") == pert) & (pl.col("fdr") < FDR_CUT)
        ).sort("log2_fold_change", descending=True)  # |LFC| 排序约等于 abs 降序
        k_eff = real_sig.height
        if k_eff == 0:
            scores[pert] = 0.0
            continue
        pred_sig = pred_de.filter(
            (pl.col("target") == pert) & ((pl.col("fdr") * s) < FDR_CUT)
        ).sort("log2_fold_change", descending=True)
        pred_top = set(pred_sig["feature"].to_list()[:k_eff])
        real_top_set = set(real_sig["feature"].to_list()[:k_eff])
        scores[pert] = len(pred_top & real_top_set) / k_eff
    vals = [v for v in scores.values()]
    return float(np.mean(vals)) if vals else 0.0


def spearman_lfc_at_scale(pred_de: pl.DataFrame, real_de: pl.DataFrame, s: float) -> float:
    """复现 DESpearmanLFC：real 显著集（fdr<0.05）join pred 全量 LFC，
    per-pert Spearman 后均值。pred 的 fdr 不参与过滤 -> 理论上不随 s 变。"""
    corrs = []
    for pert in real_de["target"].unique().to_list():
        real_sig = real_de.filter(
            (pl.col("target") == pert) & (pl.col("fdr") < FDR_CUT)
        )
        if real_sig.height < 2:
            continue
        pred = pred_de.filter(pl.col("target") == pert)
        merged = real_sig.join(
            pred.select(["target", "feature", "log2_fold_change"]),
            on=["target", "feature"],
            suffix="_pred",
            how="left",
        ).with_columns(pl.col("log2_fold_change_pred").fill_null(0.0))
        if merged.height < 2:
            continue
        c = spearmanr(
            merged["log2_fold_change"].to_numpy(),
            merged["log2_fold_change_pred"].to_numpy(),
        ).statistic
        if not np.isnan(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else np.nan


def verify(pred_de: pl.DataFrame, real_de: pl.DataFrame) -> None:
    real_n = nsig_map(real_de)
    real_arr = np.array(list(real_n.values()), dtype=float)
    print(f"[verify] 真值 n_sig: n_pert={len(real_n)} median={np.median(real_arr):.0f} "
          f"sum={real_arr.sum():.0f} min={real_arr.min():.0f} max={real_arr.max():.0f}")

    base_pred = nsig_map(pred_de)
    base_arr = np.array(list(base_pred.values()), dtype=float)
    print(f"[verify] 原始 pred n_sig: median={np.median(base_arr):.0f} "
          f"sum={base_arr.sum():.0f} min={base_arr.min():.0f} max={base_arr.max():.0f}")

    print(f"\n{'s':>8s} {'n_sig_med':>10s} {'n_sig_sum':>10s} {'SPEARMAN':>9s} "
          f"{'SPEARMAN_LFC':>13s} {'DES':>7s}")
    for s in np.geomspace(1.0, 200.0, 25):
        pred_n = nsig_at_scale(pred_de, s)
        arr = np.array(list(pred_n.values()), dtype=float)
        sp = spearman_between(pred_n, real_n)
        slfc = spearman_lfc_at_scale(pred_de, real_de, s)
        des = des_at_scale(pred_de, real_de, s)
        print(f"{s:8.3f} {np.median(arr):10.0f} {arr.sum():10.0f} {sp:9.3f} {slfc:13.3f} {des:7.3f}")


def fit_scale(pred_de: pl.DataFrame, real_de: pl.DataFrame,
              target: str = "median") -> float:
    """二分找 s：n_sig 统计量（median 或 sum）随 s 单调递减，对齐真值。"""
    real_n = nsig_map(real_de)
    real_arr = np.array(list(real_n.values()), dtype=float)
    target_val = float(np.median(real_arr)) if target == "median" else float(real_arr.sum())

    def stat(s: float) -> float:
        arr = np.array(list(nsig_at_scale(pred_de, s).values()), dtype=float)
        return float(np.median(arr)) if target == "median" else float(arr.sum())

    lo, hi = 1e-3, 1e4
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))
        if stat(mid) > target_val:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def apply_scale(pred_de: pl.DataFrame, s: float) -> pl.DataFrame:
    """fdr' = clip(fdr*s, 0, 1)，p' = clip(p*s, 0, 1) 保持一致。"""
    return pred_de.with_columns(
        [
            (pl.col("p_value") * s).clip(upper_bound=1.0).alias("p_value"),
            (pl.col("fdr") * s).clip(upper_bound=1.0).alias("fdr"),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-de", required=True)
    ap.add_argument("--real-de", required=True, help="真值 DE（fit/verify 需要）")
    ap.add_argument("--mode", default="verify", choices=["verify", "fit", "apply"])
    ap.add_argument("--target", default="median", choices=["median", "sum"])
    ap.add_argument("--s", type=float, help="apply 模式：直接用给定 s")
    ap.add_argument("--out", help="校准后的 pred_de.csv")
    ap.add_argument("--report", help="报告 json 输出路径")
    args = ap.parse_args()

    pred_de = read_de(args.pred_de)
    real_de = read_de(args.real_de)

    if args.mode == "verify":
        verify(pred_de, real_de)
        return

    if args.mode == "apply":
        assert args.s is not None, "apply 模式需要 --s"
        s = args.s
    else:  # fit
        s = fit_scale(pred_de, real_de, args.target)
        print(f"[fit] target={args.target} 选定 s={s:.4f}")

    cal = apply_scale(pred_de, s)
    before = nsig_map(pred_de)
    after = nsig_map(cal)
    b_arr = np.array(list(before.values()), dtype=float)
    a_arr = np.array(list(after.values()), dtype=float)
    r_arr = np.array(list(nsig_map(real_de).values()), dtype=float)

    report = {
        "s": s,
        "target": args.target,
        "n_sig_pred_before": {"median": float(np.median(b_arr)), "sum": float(b_arr.sum())},
        "n_sig_pred_after": {"median": float(np.median(a_arr)), "sum": float(a_arr.sum())},
        "n_sig_real": {"median": float(np.median(r_arr)), "sum": float(r_arr.sum())},
    }
    if args.out:
        cal.write_csv(args.out)
        print(f"[ok] 校准后 pred_de -> {args.out}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[ok] 报告 -> {args.report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
