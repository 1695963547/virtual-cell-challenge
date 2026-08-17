#!/usr/bin/env bash
# FCN Pseudo-bulk Pipeline：构建数据 → 训练 → 推理 → cell-eval 评测
#
# 参照 XLearning Lab (VCC 2025 #2) 方案：
#   FCN + ctrl_mean + ESM-2 + 残差学习 + pseudo-bulk 训练
#
# 用法：
#   bash /home/zjh/state-vcc-local/run_fcn_pipeline.sh
#
# 日志：/home/zjh/log/fcn_pipeline.log

set -euo pipefail

cd /home/zjh/state-vcc-local/state

TRUTH="/home/zjh/vcc_official/adata_Test.h5ad"
CKPT="/home/zjh/state-vcc-local/pseudobulk/fcn_best.pt"
PRED_OUT="/home/zjh/state-vcc-local/state/competition/prediction_test_fcn.h5ad"
EVAL_DIR="/home/zjh/state-vcc-local/state/competition/vcc_test_eval_fcn"
BASELINE_DIR="/home/zjh/state-vcc-local/state/competition/vcc_test_eval/baseline_train"
SCORE_OUT="/home/zjh/state-vcc-local/state/competition/vcc_test_eval/score_fcn_vs_trainbase.csv"
LOG="/home/zjh/log/fcn_pipeline.log"

PY="uv run --no-sync python"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ===== Step 1: 构建训练数据 =====
log "===== Step 1: Build FCN training data ====="
$PY ../pseudobulk/build_fcn_data.py 2>&1 | tee -a "$LOG"

# ===== Step 2: 训练 FCN =====
log "===== Step 2: Train FCN (500 epochs, hidden=2048, lr=1e-3, wd=0.05) ====="
$PY ../pseudobulk/train_fcn.py \
    --epochs 500 \
    --lr 1e-3 \
    --hidden 2048 \
    --wd 0.05 \
    --dropout 0.15 \
    --batch_size 64 \
    --dir_weight 0.1 \
    2>&1 | tee -a "$LOG"

# ===== Step 3: 推理 =====
log "===== Step 3: Inference on test set ====="
$PY ../pseudobulk/infer_fcn.py \
    --truth "$TRUTH" \
    --ckpt "$CKPT" \
    --output "$PRED_OUT" \
    2>&1 | tee -a "$LOG"

# ===== Step 4: cell-eval 评测 =====
log "===== Step 4: cell-eval run ====="
mkdir -p "$EVAL_DIR"

$PY -c "
import sys, logging
logging.basicConfig(level=logging.INFO)
sys.argv = ['cell-eval', 'run',
    '-a', '$TRUTH',
    '-p', '$PRED_OUT',
    '-o', '$EVAL_DIR',
    '--n-cores', '16']
from cell_eval.__main__ import main; main()
" 2>&1 | tee -a "$LOG"

# ===== Step 5: cell-eval score（归一化） =====
log "===== Step 5: cell-eval score (vs baseline_train) ====="
$PY -c "
import sys, logging
logging.basicConfig(level=logging.INFO)
sys.argv = ['cell-eval', 'score',
    '-i', '$EVAL_DIR/agg_results.csv',
    '-I', '$BASELINE_DIR/agg_results.csv',
    '-o', '$SCORE_OUT']
from cell_eval.__main__ import main; main()
" 2>&1 | tee -a "$LOG"

log "===== Done ====="
log "Score file: $SCORE_OUT"
log "Eval dir:   $EVAL_DIR"
