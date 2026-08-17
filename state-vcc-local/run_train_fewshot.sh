#!/usr/bin/env bash
# 本地验证集版训练：从 ARC_H1 留出 20 个扰动做 val，并从所有细胞系剔除这些扰动。
#
# 与上一轮 full_run 的差别：
#   1. toml 换成 starter_fewshot.toml —— val 是同细胞系未见扰动，任务形状和 VCC 一致
#      （旧的 hepg2 val_loss 量的是跨细胞系，且 hepg2 的 68 个扰动 100% 在训练池里）
#   2. max_steps 40000 -> 20000 —— 上一轮 val_loss 在 12000 步就触底，之后 28000 步全在过拟合
#   3. keep_all_ckpts=false —— 只保留 best.ckpt + last.ckpt + final.ckpt，不再存中间 step ckpt
#
# 用法：
#   setsid nohup bash /home/zjh/state-vcc-local/run_train_fewshot.sh \
#     > /home/zjh/log/train_fewshot.log 2>&1 < /dev/null &

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
NAME=${NAME:-fewshot_run}
MAX_STEPS=${MAX_STEPS:-20000}

log() { echo "[$(date '+%F %T')] $*"; }

cd "$STATE" || { log "!! 进不去 $STATE"; exit 1; }

log "===== 前置检查 ====="
for f in competition_support_set/starter_fewshot.toml \
         competition_support_set/ESM2_pert_features.pt \
         "$ROOT/val_perts.json"; do
  [[ -e "$f" ]] || { log "!! 缺少 $f"; exit 1; }
done
if [[ -d "competition/$NAME" ]]; then
  log "!! competition/$NAME 已存在 —— 会从 last.ckpt 续训。若要重跑请先改 NAME 或移走该目录"
fi
log "run 名称=$NAME  max_steps=$MAX_STEPS"
df -h /home | tail -1

log "===== 启动训练 ====="
"$STATE/.venv/bin/state" tx train \
  data.kwargs.toml_config_path=competition_support_set/starter_fewshot.toml \
  data.kwargs.num_workers=48 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=competition_support_set/ESM2_pert_features.pt \
  data.kwargs.output_space=gene \
  training.max_steps="$MAX_STEPS" \
  training.val_freq=2000 \
  training.ckpt_every_n_steps=2000 \
  training.keep_all_ckpts=false \
  training.devices=1 \
  training.batch_size=64 \
  model=state \
  model.kwargs.cell_set_len=128 \
  output_dir=competition \
  name="$NAME" \
  use_wandb=false

rc=$?
log "===== 训练结束 rc=$rc ====="
ls -la "competition/$NAME/checkpoints/" 2>/dev/null
exit $rc
