#\!/bin/bash
# scDFM VCC 重训：全链路 log1p(CP10K) 口径（修复训练/推理归一化不一致）
cd /home/zjh/scDFM || exit 1
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup \
  /home/zjh/state-vcc-local/state/.venv/bin/torchrun --nproc_per_node=4 --master_port=29500 \
  src/script/run.py \
  --batch_size=256 --model_type=origin --d_model=512 --ntoken=18084 --lr=1e-4 --steps=20000 --eta_min=1e-06 \
  --data_name=vcc --perturbation_function=crisper --noise_type=Gaussian --mode=predict_y \
  --result_path=./result/vcc_cp10k --fusion_method=differential_perceiver \
  --infer_top_gene=1000 --n_top_genes=18080 --gamma=0.5 --split_method=additive \
  --use_mmd_loss --fold=0 --use_negative_edge --topk=30 \
  --eval_n_cells=64 --eval_ode_steps=10 --eval_bf16 --eval_subset_perturbations=15 --seed=42 \
  > /home/zjh/log/scdfm_vcc_train_cp10k.log 2>&1 < /dev/null &
echo "train started PID=$\!"
