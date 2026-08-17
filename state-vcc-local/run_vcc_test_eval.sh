#!/usr/bin/env bash
# VCC 测试集本地复现：推理 -> 算指标 -> 造基线 -> 归一化打分
#
# 用法（推荐后台跑，日志按约定放 /home/zjh/log）：
#   setsid nohup bash /home/zjh/state-vcc-local/run_vcc_test_eval.sh \
#     > /home/zjh/log/vcc_test_eval.log 2>&1 < /dev/null &
#
# 可用环境变量覆盖：
#   CKPT=final.ckpt|best.ckpt|last.ckpt   使用的 checkpoint（默认 final.ckpt）
#   RUN_DIR=competition/full_run           训练 run 目录
#   PROFILE=full|vcc                       full=7 项指标，vcc=榜单 3 项（默认 full）
#   THREADS=32                             pdex 差异表达并行线程数
#   SKIP_INFER=1                           跳过推理（预测文件已存在时）
#   SKIP_METRICS=pearson_edistance         逗号分隔，跳过的指标（默认跳 pearson_edistance）
#   REUSE_DE=1                             复用已存在的 real_de.csv/pred_de.csv，不重算差异表达
#   REUSE_REAL_DE=/path/real_de.csv        仅复用真值侧差异表达（预测侧重算；真值 DE 跨 run 可安全复用）
#
# ⚠ 默认跳过 pearson_edistance：它是唯一按细胞两两算距离(O(n_cells²))的指标，
#   而真值 adata_Test.h5ad 是 48% 稠密的 CSR 稀疏矩阵，sklearn 走不到 BLAS，
#   实测比稠密慢 24 倍 —— 单是 38176 个 control 的 sigma 就要约 1 小时，
#   100 个扰动再各来一遍，全跑要好几个小时。它也不属于榜单那 7 项指标。

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
PY=$STATE/.venv/bin/python
# 用 python 直接进 cell_eval.__main__，好处是能打开 INFO 日志看到当前在算哪个指标
# （cell-eval 的 console script 自己不调 logging.basicConfig，什么都不打印）
CELL_EVAL_PY=(
  "$PY" -c
  'import logging,sys;logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s");sys.argv[0]="cell-eval";from cell_eval.__main__ import main;main()'
)
CELL_EVAL=("${CELL_EVAL_PY[@]}")

TRUTH=/home/zjh/vcc_official/adata_Test.h5ad
COUNTS=/home/zjh/vcc_official/pert_counts_Test.csv
TRUTH_SIZE=11950739168
# baseline 必须基于训练数据构建（VCC 官方规范），不能用测试集真值
TRAIN_DATA=$STATE/vci_pretrain/competition_train.h5

TEMPLATE=${TEMPLATE:-competition_support_set/competition_test_template.h5ad}
RUN_DIR=${RUN_DIR:-competition/full_run}
CKPT=${CKPT:-final.ckpt}
PROFILE=${PROFILE:-full}
THREADS=${THREADS:-32}
SKIP_METRICS=${SKIP_METRICS:-pearson_edistance}
# SE 路线：TEMPLATE 换成带 X_state 的嵌入版，EMBED_KEY=X_state 让推理读 obsm
EMBED_KEY=${EMBED_KEY:-}

PRED=${PRED:-$STATE/competition/prediction_test.h5ad}
OUT=${OUT:-$STATE/competition/vcc_test_eval}
BASE_ADATA=${BASE_ADATA:-$OUT/baseline.h5ad}
BASE_DE=${BASE_DE:-$OUT/baseline_de.csv}

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "!! $*"; exit 1; }

mkdir -p "$OUT"
cd "$STATE" || die "进不去 $STATE"

# ---------- 前置检查 ----------
log "===== 前置检查 ====="
[[ -f "$TRUTH" ]] || die "真值文件不存在: $TRUTH"
sz=$(stat -c %s "$TRUTH")
[[ "$sz" -eq "$TRUTH_SIZE" ]] || die "真值文件不完整: $sz / $TRUTH_SIZE，等下载完再跑"
[[ -f "$COUNTS" ]] || die "缺少 $COUNTS"
[[ -f "$TEMPLATE" ]] || die "缺少测试模板 $TEMPLATE"
[[ -f "$RUN_DIR/checkpoints/$CKPT" ]] || die "缺少 checkpoint $RUN_DIR/checkpoints/$CKPT"
log "OK  真值 $((sz/1000000000)) GB / checkpoint=$RUN_DIR/checkpoints/$CKPT / profile=$PROFILE"
df -h /home | tail -1

# ---------- 1. 在测试模板上推理 ----------
if [[ "${SKIP_INFER:-0}" == "1" && -f "$PRED" ]]; then
  log "===== 步骤1 跳过（SKIP_INFER=1，已有 $PRED）====="
else
  log "===== 步骤1 在测试模板上推理（170846 细胞，注意不是 val 模板）====="
  EMBED_ARGS=()
  [[ -n "$EMBED_KEY" ]] && EMBED_ARGS=(--embed-key "$EMBED_KEY")
  $PY "$ROOT/infer_local.py" \
      --checkpoint "$RUN_DIR/checkpoints/$CKPT" \
      --model-dir "$RUN_DIR" \
      --adata "$TEMPLATE" \
      --output "$PRED" \
      --pert-col target_gene \
      "${EMBED_ARGS[@]}" \
    || die "推理失败"
  log "OK  预测写到 $PRED ($(stat -c %s "$PRED" | awk '{printf "%.2f GB", $1/1e9}'))"
fi

# ---------- 2. 算你的模型的指标 ----------
log "===== 步骤2 cell-eval run：预测 vs 真值 ====="
# 差异表达是整个流程最贵的一步之一，已经算过就直接复用（REUSE_DE=1）
STATE_DE_ARGS=()
if [[ "${REUSE_DE:-0}" == "1" && -f "$OUT/state/real_de.csv" && -f "$OUT/state/pred_de.csv" ]]; then
  STATE_DE_ARGS=(-dr "$OUT/state/real_de.csv" -dp "$OUT/state/pred_de.csv")
  log "复用已有差异表达：$OUT/state/{real,pred}_de.csv"
elif [[ -n "${REUSE_REAL_DE:-}" && -f "${REUSE_REAL_DE:-}" ]]; then
  STATE_DE_ARGS=(-dr "$REUSE_REAL_DE")
  log "仅复用真值差异表达：$REUSE_REAL_DE（预测侧 DE 重算）"
fi
"${CELL_EVAL[@]}" run \
    -ap "$PRED" \
    -ar "$TRUTH" \
    "${STATE_DE_ARGS[@]}" \
    -o "$OUT/state" \
    --pert-col target_gene \
    --control-pert non-targeting \
    --profile "$PROFILE" \
    --skip-metrics "$SKIP_METRICS" \
    --num-threads "$THREADS" \
  || die "cell-eval run（模型）失败"
log "OK  $OUT/state/agg_results.csv"

# ---------- 3. 造官方基线（全扰动均值） ----------
# VCC 规范：baseline 必须基于 training data (competition_train.h5) 构建，
# 不能用测试集真值（会导致 baseline 人为过强，压制 scaled scores 到 ~0）
if [[ -f "$BASE_ADATA" && -f "$BASE_DE" ]]; then
  log "===== 步骤3 跳过（基线已存在）====="
else
  log "===== 步骤3 cell-eval baseline：用训练数据+pert_counts 造全扰动均值基线 ====="
  [[ -f "$TRAIN_DATA" ]] || die "训练数据不存在: $TRAIN_DATA"
  "${CELL_EVAL[@]}" baseline \
      -a "$TRAIN_DATA" \
      -c "$COUNTS" \
      -o "$BASE_ADATA" \
      -O "$BASE_DE" \
      --pert-col target_gene \
      --control-pert non-targeting \
      --counts-col n_cells \
      --num-threads "$THREADS" \
    || die "cell-eval baseline 失败"
  log "OK  $BASE_ADATA（基于训练数据 $TRAIN_DATA）"
fi

# ---------- 4. 算基线的指标 ----------
log "===== 步骤4 cell-eval run：基线 vs 真值 ====="
# 真值侧的差异表达和步骤2 完全一样，能复用就别重算
BASE_DE_ARGS=(-dp "$BASE_DE")
if [[ -f "$OUT/state/real_de.csv" ]]; then
  BASE_DE_ARGS+=(-dr "$OUT/state/real_de.csv")
  log "复用步骤2 的真值差异表达：$OUT/state/real_de.csv"
fi
"${CELL_EVAL[@]}" run \
    -ap "$BASE_ADATA" \
    -ar "$TRUTH" \
    "${BASE_DE_ARGS[@]}" \
    -o "$OUT/baseline" \
    --pert-col target_gene \
    --control-pert non-targeting \
    --profile "$PROFILE" \
    --skip-metrics "$SKIP_METRICS" \
    --num-threads "$THREADS" \
  || die "cell-eval run（基线）失败"
log "OK  $OUT/baseline/agg_results.csv"

# ---------- 5. 相对基线归一化 ----------
log "===== 步骤5 cell-eval score：相对基线归一化 ====="
"${CELL_EVAL[@]}" score \
    -i "$OUT/state/agg_results.csv" \
    -I "$OUT/baseline/agg_results.csv" \
    -o "$OUT/score_vs_baseline.csv" \
  || die "cell-eval score 失败"

log "===== 完成 ====="
echo
echo "--- 原始指标（你的模型，mean 行）---"
$PY - <<PY
import polars as pl
d = pl.read_csv("$OUT/state/agg_results.csv")
row = d.filter(pl.col("statistic") == "mean")
for c in d.columns[1:]:
    print(f"  {c:28s} {row[c][0]}")
PY
echo
echo "--- 相对基线归一化分（榜单口径）---"
cat "$OUT/score_vs_baseline.csv"
