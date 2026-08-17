#!/usr/bin/env bash
# valsplit 版 state_lg（SE+decoder）训练：与 state_emb_all_lg_cs512 完全同配方，
# 唯一变量 = toml 换成 starter_valsplit.toml（官方 validation 作为训练时 val 集，
# val_loss 在 H1 官方 50 个未见扰动上计算，best.ckpt 由它选出）。
#
# 复刻 state_lg 配方（competition/state_emb_all_lg_cs512/config.yaml）：
#   model=state_lg（301M，hidden 1488，llama backbone）/ cell_set_len=512 /
#   output_space=all / embed_key=X_state / 8000 步 / batch_size=64 / val_freq=1000
#
# 用法（Qoder 后台或用户终端）：
#   bash /home/zjh/state-vcc-local/run_train_state_emb_all_valsplit.sh \
#     > /home/zjh/log/train_state_emb_all_valsplit.log 2>&1
# 注意：不要 setsid nohup（沙箱内会丢 GPU，见 STATUS 4.3）。

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
NAME=${NAME:-state_emb_all_lg_cs512_valsplit}
MODEL=${MODEL:-state_lg}
MAX_STEPS=${MAX_STEPS:-8000}
NUM_WORKERS=${NUM_WORKERS:-64}
CELL_SET_LEN=${CELL_SET_LEN:-512}
export CELL_LOAD_H5_RDCC_NBYTES=${H5_CACHE:-1073741824}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${GPU:-1}

log() { echo "[$(date '+%F %T')] $*"; }

cd "$STATE" || { log "!! 进不去 $STATE"; exit 1; }

cleanup_pymp() { rm -rf "$STATE"/pymp-* 2>/dev/null; }
trap cleanup_pymp EXIT
cleanup_pymp

log "===== 前置检查 ====="
for f in vci_pretrain/starter_valsplit.toml \
         vci_pretrain/adata_Validation_trainready.h5ad \
         competition_support_set/ESM2_pert_features.pt; do
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

log "run 名称=$NAME  model=$MODEL  max_steps=$MAX_STEPS  num_workers=$NUM_WORKERS  cell_set_len=$CELL_SET_LEN  GPU=$CUDA_VISIBLE_DEVICES"
df -h /home | tail -1

log "===== 启动训练（X_state + output_space=all + 官方 validation 作 val） ====="
"$STATE/.venv/bin/state" tx train \
  data.kwargs.toml_config_path=vci_pretrain/starter_valsplit.toml \
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
  training.batch_size=64 \
  model=$MODEL \
  model.kwargs.cell_set_len=$CELL_SET_LEN \
  output_dir=competition \
  name="$NAME" \
  use_wandb=false

rc=$?
log "===== 训练结束 rc=$rc ====="
ls -la "competition/$NAME/checkpoints/" 2>/dev/null
exit $rc
