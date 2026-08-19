#!/bin/bash
# scDFM ESM2 VCC 测试集自动评测（R2 训练完成后自动触发）
# 流程：等待训练完成 → 等 GPU 释放 → test_inference.py 推理(esm) → cell-eval 7 项指标 → score_vs_baseline
# 用法：nohup bash /home/zjh/scDFM/vcc/auto_eval_scdfm_esm.sh > /home/zjh/log/scdfm_vcc_auto_eval_esm_agent.log 2>&1 &
set -uo pipefail

ROOT=/home/zjh/scDFM
STATE=/home/zjh/state-vcc-local/state
PY=$STATE/.venv/bin/python
ESM_FEATURES=/home/zjh/competition_support_set/ESM2_pert_features.pt
EXP_NAME=flow-fusion_differential_perceiver-vcc-origin-predict_y-gamma_0.5-perturbation_function_esm-lr_0.0001-dim_model_512-infer_top_gene_1000-split_method_additive-use_mmd_loss_True-fold_0-use_negative_edge_True-topk_30
EXP_DIR=$ROOT/result/vcc_esm_r2/$EXP_NAME
FINAL_CKPT=$EXP_DIR/iteration_35000/checkpoint.pt
INFER_OUT=$ROOT/result/vcc/test_eval_esm35000
EVAL_OUT=$INFER_OUT/cell_eval
LOG=/home/zjh/log/scdfm_vcc_auto_eval_esm.log
TRUTH=/home/zjh/vcc_official/adata_Test.h5ad
COUNTS=/home/zjh/vcc_official/pert_counts_Test.csv
TRAIN_DATA=$STATE/vci_pretrain/competition_train.h5
BASE_DIR=$STATE/competition/vcc_test_eval_state_lg_cs256
BASE_AGG=$BASE_DIR/baseline/agg_results.csv

CELL_EVAL=($PY -c 'import logging,sys;logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s");sys.argv[0]="cell-eval";from cell_eval.__main__ import main;main()')

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ── 1. 等待 ESM2 R2 训练完成（35000 ckpt 出现 + 训练进程退出）──
log "===== 等待训练完成 ====="
log "FINAL_CKPT=$FINAL_CKPT"
done=0
for i in $(seq 1 120); do
  if [[ -f "$FINAL_CKPT" ]] && ! pgrep -af "master_port=29502" >/dev/null 2>&1; then
    done=1; break
  fi
  sleep 120
done
if [[ $done -ne 1 ]]; then
  log "!! 等待超时（4h）：iteration_35000/checkpoint.pt 未出现或训练进程未退出"
  log "!! 当前 ckpt 状态：$(ls "$EXP_DIR"/iteration_*/checkpoint.pt 2>/dev/null | tail -3)"
  exit 1
fi
log "训练完成 ✓ checkpoint=$FINAL_CKPT"

# ── 2. 等训练用 GPU（4/5/6）显存释放 ──
sleep 90
for i in $(seq 1 15); do
  used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
  if [[ "${used:-99999}" -lt 10000 ]]; then break; fi
  log "GPU4 仍占用 ${used}MB，等待释放... ($i/15)"
  sleep 60
done

# ── 2.5 等 Replogle 评测完成（优先排队，避免抢卡）──
# Replogle 合训测试集评测（run_vcc_test_eval_replogle.sh）单卡 30-60 分钟，先让它跑完
for i in $(seq 1 90); do
  if ! pgrep -af "infer_local.py|vcc_test_eval_replogle" >/dev/null 2>&1; then break; fi
  log "等待 Replogle 评测完成... ($i/90, 每1分钟)"
  sleep 60
done

# ── 3. 探测空闲 GPU（训练完 4/5/6 必空；其余看现场）──
FREE_GPUS=()
for g in 2 3 4 5 6 7; do
  mem=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
  if [[ "${mem:-99999}" -lt 5000 ]]; then FREE_GPUS+=($g); fi
done
NP=${#FREE_GPUS[@]}
if [[ $NP -lt 3 ]]; then
  log "空闲卡不足(仅 $NP 张)，退回等待 4/5/6 并固定使用"
  FREE_GPUS=(4 5 6); NP=3
fi
GPUS=$(IFS=,; echo "${FREE_GPUS[*]}")
log "推理用 GPU=$GPUS nproc=$NP"

# ── 4. test_inference.py 推理 + 内部评测（esm 版）──
if [[ ! -f "$INFER_OUT/agg_results.csv" ]]; then
  mkdir -p "$INFER_OUT"
  log "启动 test_inference.py（checkpoint=$FINAL_CKPT n_cells=128 ode_steps=10 bf16）"
  CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$ROOT \
  setsid nohup $PY -m torch.distributed.run --nproc_per_node=$NP --master_port=29516 \
    "$ROOT/vcc/test_inference.py" \
    --checkpoint_path="$FINAL_CKPT" \
    --perturbation_function=esm --esm_dim=5120 --esm_features_path="$ESM_FEATURES" \
    --n_cells=128 --ode_steps=10 --bf16 --batch_size=64 \
    --output_dir="$INFER_OUT" \
    > "$LOG.infer" 2>&1 < /dev/null &
  log "推理后台启动 PID=$!（约 4-5h），日志=$LOG.infer"
  for i in $(seq 1 240); do
    [[ -f "$INFER_OUT/agg_results.csv" ]] && break
    if ! pgrep -af "test_inference.py" >/dev/null 2>&1; then
      log "!! 推理进程已退出但无 agg_results.csv，检查 $LOG.infer"; sleep 5
      if [[ ! -f "$INFER_OUT/agg_results.csv" ]]; then exit 1; fi
    fi
    sleep 120
  done
fi
[[ -f "$INFER_OUT/agg_results.csv" ]] || { log "!! 推理未完成/失败"; tail -30 "$LOG.infer" >> "$LOG"; exit 1; }
log "推理完成 ✓ agg_results.csv 已生成"

# ── 5. 修补 pred.h5ad（补 var_names + target_gene 列，供外部 cell-eval）──
$PY - "$TRUTH" "$INFER_OUT/pred.h5ad" "$INFER_OUT/pred_fixed.h5ad" <<'PYEOF'
import sys
import anndata as ad
import scanpy as sc
ref = sc.read_h5ad(sys.argv[1])
p = ad.read_h5ad(sys.argv[2])
n = min(p.shape[1], ref.shape[1])
p.var_names = ref.var_names[:n]
if 'perturbation' in p.obs.columns:
    p.obs['target_gene'] = p.obs['perturbation'].astype(str)
p.write_h5ad(sys.argv[3])
print(f"pred fixed: {p.shape}, var={list(p.var_names[:3])}... obs cols={list(p.obs.columns)}")
PYEOF

# ── 6. cell-eval run（模型侧，CLI full 口径与 baseline 对齐）──
log "===== cell-eval run（模型）====="
mkdir -p "$EVAL_OUT/state"
"${CELL_EVAL[@]}" run \
    -ap "$INFER_OUT/pred_fixed.h5ad" -ar "$TRUTH" \
    -o "$EVAL_OUT/state" \
    --pert-col target_gene --control-pert non-targeting \
    --profile full --skip-metrics pearson_edistance --num-threads 32 \
    >> "$LOG" 2>&1 || { log "!! cell-eval run 失败"; exit 1; }
log "OK $EVAL_OUT/state/agg_results.csv"

# ── 7. baseline（复用 state 路线共用基线，缺失才现场生成）──
if [[ ! -f "$BASE_AGG" ]]; then
  log "===== 生成 baseline ====="
  mkdir -p "$EVAL_OUT"
  "${CELL_EVAL[@]}" baseline \
      -a "$TRAIN_DATA" -c "$COUNTS" \
      -o "$EVAL_OUT/baseline.h5ad" -O "$EVAL_OUT/baseline_de.csv" \
      --pert-col target_gene --control-pert non-targeting --counts-col n_cells \
      --num-threads 32 >> "$LOG" 2>&1
  "${CELL_EVAL[@]}" run \
      -ap "$EVAL_OUT/baseline.h5ad" -ar "$TRUTH" \
      -o "$EVAL_OUT/baseline" \
      --pert-col target_gene --control-pert non-targeting \
      --profile full --skip-metrics pearson_edistance --num-threads 32 \
      >> "$LOG" 2>&1
  BASE_AGG=$EVAL_OUT/baseline/agg_results.csv
fi
log "baseline agg=$BASE_AGG"

# ── 8. score（7 项归一化分）──
log "===== cell-eval score ====="
"${CELL_EVAL[@]}" score \
    -i "$EVAL_OUT/state/agg_results.csv" -I "$BASE_AGG" \
    -o "$INFER_OUT/score_vs_baseline.csv" >> "$LOG" 2>&1
[[ -f "$INFER_OUT/score_vs_baseline.csv" ]] || { log "!! score 失败"; exit 1; }

log "===== 全部完成 ====="
$PY - "$EVAL_OUT/state/agg_results.csv" "$INFER_OUT/score_vs_baseline.csv" <<'PYEOF'
import sys, polars as pl
COLS = ["discrimination_score_l1", "de_direction_match", "mae",
        "de_spearman_sig", "de_spearman_lfc_sig", "pr_auc", "pearson_delta"]
print("--- 7 项原始指标（模型 vs 真值, mean）---")
d = pl.read_csv(sys.argv[1])
row = d.filter(pl.col("statistic") == "mean")
for c in COLS:
    if c in row.columns:
        print(f"  {c:26s} {float(row[c][0]):.4f}")
print("--- score_vs_baseline.csv ---")
print(pl.read_csv(sys.argv[2]))
PYEOF
