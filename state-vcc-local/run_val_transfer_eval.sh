#!/usr/bin/env bash
# 迁移预测的 validation 端到端评测：建预测 -> cell-eval run -> baseline -> score
#
# 用法（validation h5ad 下载完成后）：
#   setsid nohup bash /home/zjh/state-vcc-local/run_val_transfer_eval.sh \
#     > /home/zjh/log/val_transfer_eval.log 2>&1 < /dev/null &
#
# 环境变量：
#   SCALES="1 64"      扫描的放大倍数（默认 "1 64"）
#   THREADS=32         pdex 差异表达并行线程数

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
PY=$STATE/.venv/bin/python
CELL_EVAL_PY=(
  "$PY" -c
  'import logging,sys;logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s");sys.argv[0]="cell-eval";from cell_eval.__main__ import main;main()'
)

TRUTH=/home/zjh/vcc_official/validation/adata_Validation.h5ad
COUNTS=/home/zjh/vcc_official/validation/pert_counts_Validation.csv
TRAIN_DATA=$STATE/vci_pretrain/competition_train.h5
OUT=$STATE/competition/val_transfer_eval
BASE_ADATA=$OUT/baseline.h5ad
BASE_DE=$OUT/baseline_de.csv
SCALES=${SCALES:-"1 64"}
THREADS=${THREADS:-32}

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "!! $*"; exit 1; }
mkdir -p "$OUT"
cd "$STATE" || die "进不去 $STATE"

[[ -f "$TRUTH" ]] || die "真值不存在: $TRUTH（等下载）"
[[ -f "$COUNTS" ]] || die "缺少 $COUNTS"

for SCALE in $SCALES; do
  PRED=$STATE/competition/prediction_val_transfer_x${SCALE}.h5ad
  log "===== 构建迁移预测（scale=$SCALE）====="
  $PY "$ROOT/pseudobulk/build_transfer_pred.py" \
      --truth "$TRUTH" --scale "$SCALE" --output "$PRED" || die "构建预测失败"

  log "===== cell-eval run（scale=$SCALE）====="
  DE_ARGS=()
  if [[ -f "$OUT/x${SCALE}/real_de.csv" && -f "$OUT/x${SCALE}/pred_de.csv" ]]; then
    DE_ARGS=(-dr "$OUT/x${SCALE}/real_de.csv" -dp "$OUT/x${SCALE}/pred_de.csv")
  elif [[ -f "$OUT/real_de.csv" ]]; then
    DE_ARGS=(-dr "$OUT/real_de.csv")
  fi
  "${CELL_EVAL_PY[@]}" run \
      -ap "$PRED" -ar "$TRUTH" "${DE_ARGS[@]}" \
      -o "$OUT/x${SCALE}" \
      --pert-col target_gene --control-pert non-targeting \
      --profile vcc --num-threads "$THREADS" || die "cell-eval run 失败"
  # 真值侧 DE 全局缓存一份（各 scale 复用）
  [[ -f "$OUT/real_de.csv" ]] || cp "$OUT/x${SCALE}/real_de.csv" "$OUT/real_de.csv"
done

# 官方规范 baseline：训练数据 + val pert_counts（只建一次）
if [[ ! -f "$BASE_ADATA" || ! -f "$BASE_DE" ]]; then
  log "===== cell-eval baseline（训练数据 + val counts）====="
  "${CELL_EVAL_PY[@]}" baseline \
      -a "$TRAIN_DATA" -c "$COUNTS" -o "$BASE_ADATA" -O "$BASE_DE" \
      --pert-col target_gene --control-pert non-targeting \
      --counts-col n_cells --num-threads "$THREADS" || die "baseline 失败"
fi
if [[ ! -f "$OUT/baseline/agg_results.csv" ]]; then
  log "===== cell-eval run（baseline）====="
  "${CELL_EVAL_PY[@]}" run \
      -ap "$BASE_ADATA" -ar "$TRUTH" \
      -dp "$BASE_DE" -dr "$OUT/real_de.csv" \
      -o "$OUT/baseline" \
      --pert-col target_gene --control-pert non-targeting \
      --profile vcc --num-threads "$THREADS" || die "baseline run 失败"
fi

log "===== 归一化打分汇总 ====="
for SCALE in $SCALES; do
  "${CELL_EVAL_PY[@]}" score \
      -i "$OUT/x${SCALE}/agg_results.csv" \
      -I "$OUT/baseline/agg_results.csv" \
      -o "$OUT/score_x${SCALE}.csv" || die "score 失败"
  echo "--- scale=$SCALE 原始指标（mean 行）---"
  $PY - <<PY
import polars as pl
d = pl.read_csv("$OUT/x${SCALE}/agg_results.csv")
row = d.filter(pl.col("statistic") == "mean")
for c in d.columns[1:]:
    print(f"  {c:28s} {row[c][0]}")
PY
  echo "--- scale=$SCALE 归一化分 ---"
  cat "$OUT/score_x${SCALE}.csv"
done
log "===== 完成 ====="
