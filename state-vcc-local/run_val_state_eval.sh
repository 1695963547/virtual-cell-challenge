#!/usr/bin/env bash
# 老路线（SE+decoder state_lg）在官方 validation split 上的端到端评测：
#   SE transform（补 X_state）-> state_lg 推理 -> cell-eval run -> baseline -> score
#
# 背景：validation 是 raw counts 无 obsm[X_state]，SE 路线推理前必须先跑
#       state emb transform（SE-600M）。obs 列与 test 模板一致（batch/guide_id/target_gene），
#       基因面板与 test 完全一致（18080），val 50 扰动 ∩ H1 训练 = 0（零泄漏未见场景）。
#
# 用法（后台）：
#   setsid nohup bash /home/zjh/state-vcc-local/run_val_state_eval.sh \
#     > /home/zjh/log/val_state_eval.log 2>&1 < /dev/null &
#
# 环境变量：
#   GPU=1                transform+推理性能用的卡
#   RUN_DIR=competition/state_emb_all_lg_cs512   模型 run 目录（state_lg）
#   CKPT=best.ckpt       checkpoint
#   THREADS=32           pdex 差异表达并行线程数
#   SKIP_TRANSFORM=1     X_state 已存在时跳过步骤1

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
EMB_ADATA=$STATE/vci_pretrain/adata_Validation.h5ad
SE_CKPT=/home/zjh/SE-600M/se600m_epoch16.ckpt

RUN_DIR=${RUN_DIR:-competition/state_emb_all_lg_cs512}
CKPT=${CKPT:-best.ckpt}
THREADS=${THREADS:-32}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export HDF5_USE_FILE_LOCKING=FALSE

# PRED/OUT 可用环境变量覆盖，避免不同 run 的评测互相覆盖锚点结果
PRED=${PRED:-$STATE/competition/prediction_val_state_lg.h5ad}
OUT=${OUT:-$STATE/competition/val_state_eval}
BASE_ADATA=$OUT/baseline.h5ad
BASE_DE=$OUT/baseline_de.csv

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "!! $*"; exit 1; }
mkdir -p "$OUT"
cd "$STATE" || die "进不去 $STATE"

log "===== 前置检查 ====="
for f in "$TRUTH" "$COUNTS" "$SE_CKPT" "$RUN_DIR/checkpoints/$CKPT"; do
  [[ -f "$f" ]] || die "缺少 $f"
done
"$PY" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "GPU 不可用（CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES）"
log "OK  GPU=$CUDA_VISIBLE_DEVICES 模型=$RUN_DIR/$CKPT"
df -h /home | tail -1

# ---------- 1. SE transform：给 validation 补 X_state ----------
if [[ "${SKIP_TRANSFORM:-0}" == "1" && -f "$EMB_ADATA" ]]; then
  log "===== 步骤1 跳过（SKIP_TRANSFORM=1，已有 $EMB_ADATA）====="
elif [[ -f "$EMB_ADATA" ]]; then
  log "===== 步骤1 跳过（$EMB_ADATA 已存在；如需重算请删除后再跑）====="
else
  log "===== 步骤1 SE transform：validation -> X_state（98927 细胞）====="
  # ckpt 内嵌 config 里的蛋白嵌入路径是 Arc 原始环境的死路径，
  # 必须给 --model-folder（CLI 会优先在里面找 protein_embeddings.pt）
  "$STATE/.venv/bin/state" emb transform \
      --model-folder /home/zjh/SE-600M \
      --checkpoint "$SE_CKPT" \
      --protein-embeddings /home/zjh/SE-600M/protein_embeddings.pt \
      --input "$TRUTH" \
      --output "$EMB_ADATA" \
    || die "SE transform 失败"
  log "OK  $EMB_ADATA ($(stat -c %s "$EMB_ADATA" | awk '{printf "%.2f GB", $1/1e9}'))"
fi

# ---------- 2. state_lg 推理 ----------
if [[ -f "$PRED" ]]; then
  log "===== 步骤2 跳过（已有 $PRED）====="
else
  log "===== 步骤2 state_lg 推理（embed_key=X_state）====="
  "$PY" "$ROOT/infer_local.py" \
      --checkpoint "$RUN_DIR/checkpoints/$CKPT" \
      --model-dir "$RUN_DIR" \
      --adata "$EMB_ADATA" \
      --output "$PRED" \
      --pert-col target_gene \
      --embed-key X_state \
    || die "推理失败"
  log "OK  $PRED ($(stat -c %s "$PRED" | awk '{printf "%.2f GB", $1/1e9}'))"
fi

# ---------- 3. cell-eval run：预测 vs 真值 ----------
log "===== 步骤3 cell-eval run：state_lg 预测 vs validation 真值 ====="
"${CELL_EVAL_PY[@]}" run \
    -ap "$PRED" -ar "$TRUTH" \
    -o "$OUT/state_lg" \
    --pert-col target_gene --control-pert non-targeting \
    --profile full --skip-metrics pearson_edistance \
    --num-threads "$THREADS" || die "cell-eval run（模型）失败"
log "OK  $OUT/state_lg/agg_results.csv"

# ---------- 4. 官方规范 baseline（训练数据 + val counts，只建一次） ----------
if [[ ! -f "$BASE_ADATA" || ! -f "$BASE_DE" ]]; then
  log "===== 步骤4 cell-eval baseline（competition_train + val counts）====="
  "${CELL_EVAL_PY[@]}" baseline \
      -a "$TRAIN_DATA" -c "$COUNTS" -o "$BASE_ADATA" -O "$BASE_DE" \
      --pert-col target_gene --control-pert non-targeting \
      --counts-col n_cells --num-threads "$THREADS" || die "baseline 失败"
fi
if [[ ! -f "$OUT/baseline/agg_results.csv" ]]; then
  log "===== 步骤5 cell-eval run（baseline）====="
  "${CELL_EVAL_PY[@]}" run \
      -ap "$BASE_ADATA" -ar "$TRUTH" \
      -dp "$BASE_DE" -dr "$OUT/state_lg/real_de.csv" \
      -o "$OUT/baseline" \
      --pert-col target_gene --control-pert non-targeting \
      --profile full --skip-metrics pearson_edistance \
      --num-threads "$THREADS" || die "baseline run 失败"
fi

# ---------- 6. 归一化打分 ----------
log "===== 步骤6 cell-eval score ====="
"${CELL_EVAL_PY[@]}" score \
    -i "$OUT/state_lg/agg_results.csv" \
    -I "$OUT/baseline/agg_results.csv" \
    -o "$OUT/score_state_lg_vs_trainbase.csv" || die "score 失败"

log "===== 完成 ====="
echo
echo "--- state_lg @ validation 原始指标（mean 行）---"
"$PY" - <<PY
import polars as pl
d = pl.read_csv("$OUT/state_lg/agg_results.csv")
row = d.filter(pl.col("statistic") == "mean")
for c in d.columns[1:]:
    print(f"  {c:28s} {row[c][0]}")
PY
echo
echo "--- 归一化分（vs 训练数据 baseline）---"
cat "$OUT/score_state_lg_vs_trainbase.csv"
