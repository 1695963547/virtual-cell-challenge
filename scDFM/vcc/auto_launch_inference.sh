#!/bin/bash
# 监控训练进程，结束后自动启动 test_inference.py（6 卡并行）
# 用法（训练过程中启动）：
#   bash vcc/auto_launch_inference.sh
# 或后台运行：
#   nohup bash vcc/auto_launch_inference.sh > /home/zjh/log/scdfm_vcc_monitor.log 2>&1 &

TRAIN_PID=3194483
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPTS_DIR")"
cd "$ROOT_DIR" || { echo "ERROR: cannot cd to $ROOT_DIR"; exit 1; }

EXP_DIR="./result/vcc/flow-fusion_differential_perceiver-vcc-origin-predict_y-gamma_0.5-perturbation_function_crisper-lr_5e-05-dim_model_512-infer_top_gene_1000-split_method_additive-use_mmd_loss_True-fold_0-use_negative_edge_True-topk_30"
CKPT_PATH="$EXP_DIR/iteration_20000/checkpoint.pt"
LOG_FILE="/home/zjh/log/scdfm_vcc_test_inference.log"
LOGGING_DIR="/home/zjh/log"

echo "[$(date)] ============================================="
echo "[$(date)] Monitor started"
echo "[$(date)]   Training PID: $TRAIN_PID"
echo "[$(date)]   Working dir : $(pwd)"
echo "[$(date)]   Expected ckpt: $CKPT_PATH"
echo "[$(date)]   Inference log: $LOG_FILE"
echo "[$(date)] ============================================="

# ── 第 1 步：等待训练进程结束 ──
sleep_count=0
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep_count=$((sleep_count + 1))
    if [ $((sleep_count % 10)) -eq 0 ]; then
        echo "[$(date)] Still waiting for training PID=$TRAIN_PID to finish... (${sleep_count}min)"
    fi
    sleep 60
done

echo "[$(date)] Training process exited. Waiting 90s for GPU memory cleanup..."
sleep 90

# ── 第 2 步：确认 checkpoint 存在 ──
if [ ! -f "$CKPT_PATH" ]; then
    echo "[$(date)] WARNING: $CKPT_PATH not found! Searching for latest checkpoint..."
    CKPT_PATH=$(find "./result/vcc" -name "checkpoint.pt" -path "*/iteration_*/*" | sort | tail -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "[$(date)] ERROR: No checkpoint found. Aborting."
        exit 1
    fi
    echo "[$(date)] Found: $CKPT_PATH"
else
    echo "[$(date)] Checkpoint found: $CKPT_PATH"
fi

# ── 第 3 步：确认 GPU 已释放（训练用过的卡空闲）──
TRAIN_GPUS="2,3,4"
echo "[$(date)] Verifying GPUs $TRAIN_GPUS are free..."
for attempt in 1 2 3; do
    ALL_FREE=true
    for gpu in 2 3 4; do
        mem=$(nvidia-smi --id=$gpu --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        if [ "$mem" -gt 10000 ] 2>/dev/null; then
            echo "[$(date)] GPU $gpu still in use (${mem}MB), waiting... (attempt $attempt)"
            ALL_FREE=false
            break
        fi
    done
    if $ALL_FREE; then break; fi
    sleep 30
done

if ! $ALL_FREE; then
    echo "[$(date)] WARNING: GPUs may still be occupied. Proceeding anyway."
fi

# ── 第 4 步：启动推理 ──
echo "[$(date)] ============================================="
echo "[$(date)] Launching test inference on 6 GPUs (2,3,4,5,6,7)..."
echo "[$(date)]   checkpoint: $CKPT_PATH"
echo "[$(date)]   n_cells=128, ode_steps=10, bf16"
echo "[$(date)]   log: $LOG_FILE"
echo "[$(date)] ============================================="

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export PYTHONPATH=./

setsid nohup /home/zjh/state-vcc-local/state/.venv/bin/torchrun \
    --nproc_per_node=6 --master_port=29515 \
    vcc/test_inference.py \
    --checkpoint_path="$CKPT_PATH" \
    --n_cells=128 --ode_steps=10 --bf16 \
    > "$LOG_FILE" 2>&1 < /dev/null &

INFER_PID=$!
echo "[$(date)] Inference launched! PID=$INFER_PID"

# 快速检查进程是否存活
sleep 5
if kill -0 "$INFER_PID" 2>/dev/null; then
    echo "[$(date)] Inference process is running (PID=$INFER_PID)"
    echo "[$(date)] Monitor complete. Check progress with: tail -f $LOG_FILE"
else
    echo "[$(date)] ERROR: Inference process exited quickly! Check logs:"
    tail -5 "$LOG_FILE" 2>/dev/null
    exit 1
fi