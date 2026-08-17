#!/bin/bash
# scDFM VCC 单卡冒烟测试：3 步前向+反向，验证数据链路与模型无 CUDA 非法访问
# 与 train_cp10k.sh 参数一致，仅 steps 缩减、单卡、开 CUDA_LAUNCH_BLOCKING 精确定位
cd /home/zjh/scDFM || exit 1
export PYTHONPATH=./
export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0 nohup \
  /home/zjh/state-vcc-local/state/.venv/bin/python3 \
  src/script/run.py \
  --batch_size=256 --model_type=origin --d_model=512 --ntoken=18084 --lr=1e-4 --steps=3 --eta_min=1e-06 \
  --data_name=vcc --perturbation_function=crisper --noise_type=Gaussian --mode=predict_y \
  --result_path=./result/vcc_cp10k_smoke --fusion_method=differential_perceiver \
  --infer_top_gene=1000 --n_top_genes=18080 --gamma=0.5 --split_method=additive \
  --use_mmd_loss --fold=0 --use_negative_edge --topk=30 \
  --eval_n_cells=64 --eval_ode_steps=10 --eval_bf16 --eval_subset_perturbations=15 --seed=42 \
  > /home/zjh/log/scdfm_vcc_smoke_test.log 2>&1 < /dev/null &
echo "smoke test started PID=$!"
