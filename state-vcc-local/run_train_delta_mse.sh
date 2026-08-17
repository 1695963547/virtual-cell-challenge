#!/usr/bin/env bash
# Route B 实验：decoder_loss 从 energy_distance 换成 delta MSE
# 基于 state_lg 的 main network（init_from），decoder 随机初始化重新学 delta MSE。
#
# 与 run_train_state_emb_all.sh 的唯一差异：
#   1. model.kwargs.decoder_loss_type="delta_mse"  （新增参数，默认 "energy" = 原行为）
#   2. INIT_FROM=state_lg best.ckpt（复用 main network，只训 decoder）
#   3. NAME=state_emb_all_lg_delta_mse
#
# 回滚：删掉 decoder_loss_type 行或改为 "energy" 即恢复原 loss
#
# 用法：
#   bash /home/zjh/state-vcc-local/run_train_delta_mse.sh
# 日志：/home/zjh/log/train_delta_mse.log

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
NAME=${NAME:-state_emb_all_lg_delta_mse_cs256}
MODEL=${MODEL:-state_lg}
MAX_STEPS=${MAX_STEPS:-8000}
NUM_WORKERS=${NUM_WORKERS:-64}
CELL_SET_LEN=${CELL_SET_LEN:-256}
INIT_FROM=${INIT_FROM:-competition/state_emb_all_lg_cs512/checkpoints/best.ckpt}
export CELL_LOAD_H5_RDCC_NBYTES=${H5_CACHE:-1073741824}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${GPU:-3}

log() { echo "[$(date '+%F %T')] $*"; }

cd "$STATE" || { log "!! 进不去 $STATE"; exit 1; }

cleanup_pymp() { rm -rf "$STATE"/pymp-* 2>/dev/null; }
trap cleanup_pymp EXIT
cleanup_pymp

log "===== 前置检查 ====="
for f in vci_pretrain/starter.toml \
         competition_support_set/ESM2_pert_features.pt \
         "$INIT_FROM"; do
  [[ -e "$f" ]] || { log "!! 缺少 $f"; exit 1; }
done
if [[ -d "competition/$NAME" ]]; then
  log "!! competition/$NAME 已存在 —— 会从 last.ckpt 续训。若要重跑请先改 NAME 或移走该目录"
fi

if ! "$STATE/.venv/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  log "!! torch.cuda.is_available()=False —— 训练会退化到 CPU，已中止"
  exit 1
fi
log "GPU 校验通过：$("$STATE/.venv/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))')"

log "run=$NAME  model=$MODEL  max_steps=$MAX_STEPS  decoder_loss_type=delta_mse  init_from=$INIT_FROM"

log "===== 启动训练（delta MSE loss） ====="
"$STATE/.venv/bin/state" tx train \
  data.kwargs.toml_config_path=vci_pretrain/starter.toml \
  data.kwargs.embed_key="X_state" \
  data.kwargs.output_space=all \
  data.kwargs.num_workers=$NUM_WORKERS \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=competition_support_set/ESM2_pert_features.pt \
  training.max_steps="$MAX_STEPS" \
  training.val_freq=1000 \
  training.ckpt_every_n_steps=1000 \
  training.keep_all_ckpts=false \
  training.devices=1 \
  training.batch_size=128 \
  model=$MODEL \
  model.kwargs.cell_set_len=$CELL_SET_LEN \
  +model.kwargs.decoder_loss_type="delta_mse" \
  +model.kwargs.freeze_main=true \
  model.kwargs.init_from="$INIT_FROM" \
  output_dir=competition \
  name="$NAME" \
  use_wandb=false

rc=$?
log "===== 训练结束 rc=$rc ====="
ls -la "competition/$NAME/checkpoints/" 2>/dev/null
exit $rc
