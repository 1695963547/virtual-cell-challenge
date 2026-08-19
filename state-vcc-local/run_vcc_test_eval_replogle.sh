#!/bin/bash
# Replogle 全量合训模型（state_emb_all_lg_cs512_replogle_full）VCC 测试集评测（7 项指标）
# 流程：等 scDFM 训练完成(GPU4 释放) → 单卡推理 → cell-eval 7 项指标 → score_vs_baseline
# 复用：baseline（vcc_test_eval_state_lg_cs256 已生成）+ 真值 DE（跨 run 安全）
# 用法：nohup bash /home/zjh/state-vcc-local/run_vcc_test_eval_replogle.sh \
#         > /home/zjh/log/vcc_test_eval_replogle_full.log 2>&1 &
set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
PY=$STATE/.venv/bin/python
CELL_EVAL=(
  "$PY" -c
  'import logging,sys;logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s");sys.argv[0]="cell-eval";from cell_eval.__main__ import main;main()'
)

TRUTH=/home/zjh/vcc_official/adata_Test.h5ad
COUNTS=/home/zjh/vcc_official/pert_counts_Test.csv
TEMPLATE=$STATE/vci_pretrain/adata_Test.h5ad      # 带 X_state 嵌入的测试模板（13G）
RUN_DIR=competition/state_emb_all_lg_cs512_replogle_full
CKPT=${CKPT:-final.ckpt}
EMBED_KEY=X_state
PRED=$STATE/competition/prediction_test_replogle_full.h5ad
OUT=$STATE/competition/vcc_test_eval_replogle_full
# 复用 state_lg_cs256 评测时基于训练数据构建的官方基线（跨模型通用）
BASE_ADATA=$STATE/competition/vcc_test_eval_state_lg_cs256/baseline.h5ad
BASE_DE=$STATE/competition/vcc_test_eval_state_lg_cs256/baseline_de.csv
BASE_AGG=$STATE/competition/vcc_test_eval_state_lg_cs256/baseline/agg_results.csv
# 真值侧差异表达跨 run 安全复用（cell-eval 官方说明）
REAL_DE=$STATE/competition/vcc_test_eval_state_lg_cs256/state/real_de.csv
LOG=/home/zjh/log/vcc_test_eval_replogle_full.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "!! $*"; exit 1; }

# ── 0. 前置检查 ──
mkdir -p "$OUT"
[[ -f "$TRUTH" ]] || die "真值缺失 $TRUTH"
[[ -f "$TEMPLATE" ]] || die "模板缺失 $TEMPLATE"
[[ -f "$STATE/$RUN_DIR/checkpoints/$CKPT" ]] || die "checkpoint 缺失 $RUN_DIR/checkpoints/$CKPT"
[[ -f "$BASE_ADATA" && -f "$BASE_DE" && -f "$BASE_AGG" ]] || die "baseline 复用文件缺失"
[[ -f "$REAL_DE" ]] || die "真值 DE 缺失 $REAL_DE"
log "===== 前置检查 OK：ckpt=$CKPT / 复用 baseline+real_de ====="

# ── 1. 等 GPU4 释放（scDFM ESM2 训练完成）──
log "===== 等待 GPU4 释放（scDFM 训练进行中）====="
for i in $(seq 1 240); do
  used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
  if [[ "${used:-99999}" -lt 10000 ]]; then
    log "GPU4 已释放（${used}MB），继续等 60s 稳定"
    sleep 60
    break
  fi
  if (( i % 6 == 0 )); then log "GPU4 仍占用 ${used}MB（已等 $((i/6*5)) 分钟）..."; fi
  sleep 300
done
used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
[[ "${used:-99999}" -lt 10000 ]] || { log "!! 等待超时：GPU4 仍未释放（${used}MB）"; exit 1; }

cd "$STATE" || die "进不去 $STATE"

# ── 2. 推理（单卡 GPU4，170846 细胞，约 15-30 分钟）──
if [[ -f "$PRED" ]]; then
  log "预测文件已存在，跳过推理：$PRED"
else
  log "===== 步骤1 测试集推理 ====="
  CUDA_VISIBLE_DEVICES=4 $PY "$ROOT/infer_local.py" \
      --checkpoint "$STATE/$RUN_DIR/checkpoints/$CKPT" \
      --model-dir "$RUN_DIR" \
      --adata "$TEMPLATE" \
      --output "$PRED" \
      --pert-col target_gene \
      --embed-key "$EMBED_KEY" \
    >> "$LOG" 2>&1 || die "推理失败，见 $LOG"
  log "OK  预测=$PRED ($(stat -c %s "$PRED" | awk '{printf "%.1f GB", $1/1e9}'))"
fi

# ── 3. cell-eval run：模型预测 vs 真值（真值 DE 复用）──
log "===== 步骤2 cell-eval run（Replogle 模型）====="
"${CELL_EVAL[@]}" run \
    -ap "$PRED" \
    -ar "$TRUTH" \
    -dr "$REAL_DE" \
    -o "$OUT/state" \
    --pert-col target_gene \
    --control-pert non-targeting \
    --profile full \
    --skip-metrics pearson_edistance \
    --num-threads 32 \
    >> "$LOG" 2>&1 || die "cell-eval run（模型）失败"
log "OK  $OUT/state/agg_results.csv"

# ── 4. score：相对 baseline 归一化（baseline 评测结果复用）──
log "===== 步骤4 cell-eval score ====="
"${CELL_EVAL[@]}" score \
    -i "$OUT/state/agg_results.csv" \
    -I "$BASE_AGG" \
    -o "$OUT/score_vs_baseline.csv" \
    >> "$LOG" 2>&1 || die "cell-eval score 失败"
[[ -f "$OUT/score_vs_baseline.csv" ]] || die "score_vs_baseline.csv 未生成"

log "===== 全部完成 ====="
echo
echo "--- 7 项原始指标（Replogle 模型 vs 真值, mean）---"
$PY - "$OUT/state/agg_results.csv" <<'PYEOF'
import sys, polars as pl
COLS = ["discrimination_score_l1", "de_direction_match", "mae",
        "de_spearman_sig", "de_spearman_lfc_sig", "pr_auc", "pearson_delta"]
d = pl.read_csv(sys.argv[1])
row = d.filter(pl.col("statistic") == "mean")
for c in COLS:
    if c in row.columns:
        print(f"  {c:26s} {float(row[c][0]):.4f}")
PYEOF
echo
echo "--- score_vs_baseline.csv（7 项归一化分）---"
$PY - "$OUT/score_vs_baseline.csv" <<'PYEOF'
import sys, polars as pl
COLS = ["discrimination_score_l1", "de_direction_match", "mae",
        "de_spearman_sig", "de_spearman_lfc_sig", "pr_auc", "pearson_delta"]
d = pl.read_csv(sys.argv[1])
for c in COLS:
    if c in d.columns:
        print(f"  {c:26s} {float(d.row(0)[d.columns.index(c)]):.4f}")
PYEOF
