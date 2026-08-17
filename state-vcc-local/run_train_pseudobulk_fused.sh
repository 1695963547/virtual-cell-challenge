#!/usr/bin/env bash
# Route B-2: state model + Fused PDS/DES Proxy Loss
# 与 full_run 唯一区别：loss 从 energy distance 换成 fused PDS/DES proxy
# 使用 FusedPDSDESLoss: weighted_mse + contrastive_pds + direction_loss
#
# 关键参数（与 full_run 对齐）：
#   MODEL=state, embed_key=None, output_space=gene
#   loss=fused, cell_set_len=128, batch_size=64
#
# 用法：
#   bash /home/zjh/state-vcc-local/run_train_pseudobulk_fused.sh
# 日志：/home/zjh/log/train_pseudobulk_fused.log

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
NAME=${NAME:-state_fused_run2}
MODEL=${MODEL:-state}
MAX_STEPS=${MAX_STEPS:-8000}
NUM_WORKERS=${NUM_WORKERS:-48}
CELL_SET_LEN=${CELL_SET_LEN:-128}
export CELL_LOAD_H5_RDCC_NBYTES=${H5_CACHE:-1073741824}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${GPU:-4}

log() { echo "[$(date '+%F %T')] $*"; }

cd "$STATE" || { log "!! 进不去 $STATE"; exit 1; }

cleanup_pymp() { rm -rf "$STATE"/pymp-* 2>/dev/null; }
trap cleanup_pymp EXIT
cleanup_pymp

log "===== 前置检查 ====="
for f in competition_support_set/starter.toml \
         competition_support_set/ESM2_pert_features.pt; do
  [[ -e "$f" ]] || { log "!! 缺少 $f"; exit 1; }
done
if [[ -d "competition/$NAME" ]]; then
  log "!! competition/$NAME 已存在 —— 会从 last.ckpt 续训。若要重跑请先改 NAME 或移走该目录"
fi

if ! "$STATE/.venv/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  log "!! torch.cuda.is_available()=False"
  exit 1
fi
log "GPU 校验通过：$("$STATE/.venv/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))')"

log "run=$NAME  model=$MODEL  max_steps=$MAX_STEPS  loss=fused  cell_set_len=$CELL_SET_LEN"

log "===== 启动训练（state + Fused PDS/DES Loss） ====="
"$STATE/.venv/bin/state" tx train \
  data.kwargs.toml_config_path=competition_support_set/starter.toml \
  data.kwargs.embed_key=null \
  data.kwargs.output_space=gene \
  data.kwargs.num_workers=$NUM_WORKERS \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=competition_support_set/ESM2_pert_features.pt \
  training.max_steps="$MAX_STEPS" \
  training.ckpt_every_n_steps=2000 \
  training.val_freq=2000 \
  training.keep_all_ckpts=false \
  training.devices=1 \
  training.batch_size=64 \
  model=$MODEL \
  model.kwargs.cell_set_len=$CELL_SET_LEN \
  model.kwargs.loss=fused \
  output_dir=competition \
  name="$NAME" \
  use_wandb=false

rc=$?
log "===== 训练结束 rc=$rc ====="
ls -la "competition/$NAME/checkpoints/" 2>/dev/null
exit $rc
