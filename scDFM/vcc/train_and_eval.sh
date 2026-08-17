#!/bin/bash
# scDFM VCC 训练 + 自动测评流水线
# 训练完成后自动寻找最新 checkpoint 并在测试集上评测
set -o pipefail

cd /home/zjh/scDFM || exit 1
export PYTHONPATH=./

LOG_DIR=/home/zjh/log
TRAIN_LOG="${LOG_DIR}/scdfm_vcc_train_20k.log"
EVAL_LOG="${LOG_DIR}/scdfm_vcc_test_eval_20k.log"
PYTHON=/home/zjh/state-vcc-local/state/.venv/bin/python3
TORCHRUN=/home/zjh/state-vcc-local/state/.venv/bin/torchrun

echo "=========================================="
echo " scDFM VCC 训练 + 测评流水线"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ---- 第一阶段：训练（20000 steps） ----
echo ""
echo ">>> [阶段1] 开始训练 (20000 steps, 4 GPU) ..."
echo ">>> 训练日志: ${TRAIN_LOG}"
echo ">>> 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN --nproc_per_node=4 --master_port=29500 \
  src/script/run.py \
  --batch_size=256 --model_type=origin --d_model=512 --ntoken=18084 --lr=1e-4 --steps=20000 --eta_min=1e-06 \
  --data_name=vcc --perturbation_function=crisper --noise_type=Gaussian --mode=predict_y \
  --result_path=./result/vcc_cp10k --fusion_method=differential_perceiver \
  --infer_top_gene=1000 --n_top_genes=18080 --gamma=0.5 --split_method=additive \
  --use_mmd_loss --fold=0 --use_negative_edge --topk=30 \
  --eval_n_cells=64 --eval_ode_steps=10 --eval_bf16 --eval_subset_perturbations=15 --seed=42 \
  2>&1 | tee "${TRAIN_LOG}"

TRAIN_EXIT=${PIPESTATUS[0]}
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "!!! 训练异常退出 (exit code: ${TRAIN_EXIT})"
  exit $TRAIN_EXIT
fi

echo ">>> [阶段1] 训练完成！ $(date '+%Y-%m-%d %H:%M:%S')"

# ---- 第二阶段：查找最新 checkpoint ----
EXP_DIR="./result/vcc_cp10k"
# 修复：原 sort -t_ -k2 -n 按路径中第 2 个 '_' 字段排序，字段为实验名（differential），
# 数值排序下全相等，tail -1 变成随机选择。改用 sort -V 按 version 顺序正确取最大迭代号。
CKPT_PATH=$(find "${EXP_DIR}" -name "checkpoint.pt" -path "*/iteration_*/checkpoint.pt" | sort -V | tail -1)

if [ -z "${CKPT_PATH}" ]; then
  echo "!!! 未找到 checkpoint 文件"
  exit 1
fi

echo ">>> 找到 checkpoint: ${CKPT_PATH}"

# ---- 第三阶段：测试集评测 ----
echo ""
echo ">>> [阶段2] 开始测试集评测 ..."
echo ">>> 评测日志: ${EVAL_LOG}"
echo ">>> 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

CUDA_VISIBLE_DEVICES=0,1,2 $TORCHRUN --nproc_per_node=3 \
  vcc/test_inference.py \
  --checkpoint_path="${CKPT_PATH}" \
  --n_cells=128 --ode_steps=10 --bf16 \
  2>&1 | tee "${EVAL_LOG}"

EVAL_EXIT=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo " 流水线执行完毕！ $(date '+%Y-%m-%d %H:%M:%S')"
echo " 训练 exit code: ${TRAIN_EXIT}"
echo " 评测 exit code: ${EVAL_EXIT}"
echo " Checkpoint: ${CKPT_PATH}"
echo "=========================================="
