#!/bin/bash
# scDFM VCC ESM-2 条件化训练（加固重启版 R2）
# - 从 esm checkpoint(20000) 续训（原训练 15000-15500 已完成投影层解冻，freeze_esm_steps=0 避免二次冻结）
# - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 根治缓存碎片化（15500 eval 曾 OOM）
# - eval_batch_size=32：eval ODE 生成峰值显存减半（9.5GB -> ~4.7GB），eval 统计 64 细胞不变
cd /home/zjh/scDFM || exit 1
export PYTHONPATH=./
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=/home/zjh/scDFM/result/vcc_esm/flow-fusion_differential_perceiver-vcc-origin-predict_y-gamma_0.5-perturbation_function_esm-lr_0.0001-dim_model_512-infer_top_gene_1000-split_method_additive-use_mmd_loss_True-fold_0-use_negative_edge_True-topk_30/iteration_20000/checkpoint.pt

CUDA_VISIBLE_DEVICES=4,5,6 setsid nohup \
  /home/zjh/state-vcc-local/state/.venv/bin/torchrun --nproc_per_node=3 --master_port=29502 \
  src/script/run.py \
  --batch_size=256 --model_type=origin --d_model=512 --ntoken=18084 --lr=1e-4 --steps=35000 --eta_min=1e-06 \
  --data_name=vcc --perturbation_function=esm --esm_features_path=/home/zjh/competition_support_set/ESM2_pert_features.pt --esm_dim=5120 \
  --noise_type=Gaussian --mode=predict_y \
  --result_path=./result/vcc_esm_r2 --fusion_method=differential_perceiver \
  --infer_top_gene=1000 --n_top_genes=18080 --gamma=0.5 --split_method=additive \
  --use_mmd_loss --fold=0 --use_negative_edge --topk=30 \
  --checkpoint_path="$CKPT" --freeze_esm_steps=0 \
  --eval_n_cells=64 --eval_ode_steps=10 --eval_bf16 --eval_subset_perturbations=15 --eval_batch_size=32 --seed=42 \
  > /home/zjh/log/scdfm_vcc_train_esm_r2.log 2>&1 < /dev/null &
echo "esm train R2 started PID=$!"
