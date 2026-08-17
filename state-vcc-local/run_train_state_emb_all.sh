#!/usr/bin/env bash
# output_space=all 版 ST 训练：输入 SE-600M 的 2058 维 X_state embedding，
# main network 预测 2058 维 latent，gene_decoder（2058→18080）自动建并训练，
# 输出可对 18080 维真值（adata_Test.h5ad）本地评测。
#
# 与 run_train_state_emb.sh（output_space=embedding）的差别：
#   1. output_space embedding -> all
#      - _train.py 自动注入 decoder_cfg（latent_dim=2058, gene_dim=18080,
#        hidden_dims=[1024,1024,512]），模型 __init__ 建 LatentToGeneDecoder
#      - cell_load 自动 store_raw_expression=True（embed_key 非空 + output_space=all），
#        batch 带 pert_cell_counts（fetch_gene_expression 读 .X，18080 维）
#      - training_step：decoder 用 pred.detach() 解码到 18080，和 pert_cell_counts 算
#        decoder_loss，total_loss = main_loss + decoder_loss（decoder_loss 梯度不回传
#        main network；main network 仍由 main_loss 更新）
#   2. 5 个辅助文件（k562/rpe1/jurkat/k562_gwps/hepg2）已补 uns/log1p 标记 ——
#      output_space=all 的 log1p 一致性检查（_setup_global_maps）通过
#   3. NAME 换成 state_emb_all_run，避免和 state_emb_run 冲突
#
# 注意：state_emb_run（embedding 版）的 main network 任务与此版完全一致（都是
# 2058 latent loss），只是多了 decoder。若想复用其已收敛的 main network 只训 decoder，
# 设 INIT_FROM=competition/state_emb_run/checkpoints/step8000.ckpt（_train.py 的
# init_from 路径会用 strict=False + shape 过滤加载 main network，decoder 随机初始化）。
#
# 用法（从头训，推荐，~1h）：
#   setsid nohup bash /home/zjh/state-vcc-local/run_train_state_emb_all.sh \
#     > /home/zjh/log/train_state_emb_all.log 2>&1 < /dev/null &
# 用法（finetune decoder，复用 main network）：
#   INIT_FROM=competition/state_emb_run/checkpoints/step8000.ckpt \
#     setsid nohup bash /home/zjh/state-vcc-local/run_train_state_emb_all.sh \
#     > /home/zjh/log/train_state_emb_all.log 2>&1 < /dev/null &

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
NAME=${NAME:-state_emb_all_full}
MODEL=${MODEL:-state}
MAX_STEPS=${MAX_STEPS:-8000}
NUM_WORKERS=${NUM_WORKERS:-64}
CELL_SET_LEN=${CELL_SET_LEN:-128}
# h5 chunk cache（CSR 随机读提速）。cell_load 默认 64MB，这里拉到 1GB；
# 2TB 内存 ×64 worker 也撑得住。用 setdefault 设，这里显式 export 会覆盖。
export CELL_LOAD_H5_RDCC_NBYTES=${H5_CACHE:-1073741824}
# 关 HDF5 文件锁：多 worker 并发读同一 h5 时会死锁（症状：卡在 "Total FLOPs: 0"，GPU 0%）。
# 2026-08-03 验证：加了 FALSE 后 worker 不再死锁。
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${GPU:-3}

log() { echo "[$(date '+%F %T')] $*"; }

cd "$STATE" || { log "!! 进不去 $STATE"; exit 1; }

# multiprocessing 临时目录残留清理（同 run_train_state_emb.sh）
cleanup_pymp() { rm -rf "$STATE"/pymp-* 2>/dev/null; }
trap cleanup_pymp EXIT
cleanup_pymp

log "===== 前置检查 ====="
for f in vci_pretrain/starter.toml \
         competition_support_set/ESM2_pert_features.pt \
         "$ROOT/val_perts.json"; do
  [[ -e "$f" ]] || { log "!! 缺少 $f"; exit 1; }
done
if [[ -d "competition/$NAME" ]]; then
  log "!! competition/$NAME 已存在 —— 会从 last.ckpt 续训。若要重跑请先改 NAME 或移走该目录"
fi

# GPU 可见性硬校验。_train.py 里 accelerator 是 torch.cuda.is_available() 三元决定的，
# 看不到卡就静默退化成 accelerator="cpu" 继续训（0.18 it/s，比 GPU 慢 15 倍）且不报错。
# 2026-08-03 那次就是这样：日志里只有一行 "GPU available: False"，白跑 4 分钟。宁可不启动。
if ! "$STATE/.venv/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  log "!! torch.cuda.is_available()=False —— 训练会退化到 CPU，已中止"
  log "   检查 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 是否指向有效卡，以及启动环境能否访问 /dev/nvidia*"
  exit 1
fi
log "GPU 校验通过：$("$STATE/.venv/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))')"

log "run 名称=$NAME  model=$MODEL  max_steps=$MAX_STEPS  num_workers=$NUM_WORKERS  cell_set_len=$CELL_SET_LEN  h5_cache=$((CELL_LOAD_H5_RDCC_NBYTES/1048576))MB  GPU=$CUDA_VISIBLE_DEVICES  HDF5_USE_FILE_LOCKING=$HDF5_USE_FILE_LOCKING"
df -h /home | tail -1

INIT_FROM_ARGS=()
if [[ -n "${INIT_FROM:-}" ]]; then
  log "INIT_FROM=$INIT_FROM —— 复用已训 main network，decoder 随机初始化从头训"
  INIT_FROM_ARGS=(model.kwargs.init_from="$INIT_FROM")
fi

log "===== 启动训练（X_state + output_space=all 路线） ====="
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
  training.batch_size=64 \
  model=$MODEL \
  model.kwargs.cell_set_len=$CELL_SET_LEN \
  ${INIT_FROM_ARGS[@]+"${INIT_FROM_ARGS[@]}"} \
  output_dir=competition \
  name="$NAME" \
  use_wandb=false

rc=$?
log "===== 训练结束 rc=$rc ====="
ls -la "competition/$NAME/checkpoints/" 2>/dev/null
exit $rc
