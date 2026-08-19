#!/usr/bin/env bash
# X_state embedding 版 few-shot 训练：输入是 SE-600M 的 2058 维 embedding（obsm/X_state），
# val 切分与基因表达版完全一致（ARC_H1 留出 20 扰动），两条路线公平可比。
#
# 与 run_train_fewshot.sh 的差别：
#   1. toml 换成 vci_pretrain/starter_fewshot.toml —— 数据文件带 obsm/X_state
#   2. 新增 data.kwargs.embed_key="X_state" + output_space=embedding —— 输入输出都在
#      2058 维 embedding 空间（不建 gene decoder，也不会去读 X_hvg）
#   3. max_steps 20000 -> 8000 —— 官方 README 的 X_state 路线建议步数
#      （官方 8000 步即交出 leaderboard normalized 0.072）
#
# 注意：不要加 output_space=gene —— gene 模式下 cell_load 会尝试读 obsm/X_hvg
# （vci_pretrain 的文件没有 X_hvg），直接 KeyError。
# 同样不能留空：output_space 默认 all，会在 _setup_global_maps 阶段撞 log1p 元数据
# 一致性检查（vci_pretrain 里 competition_train 有 uns/log1p、辅助数据没有）而拒训。
# 所以必须显式写 output_space=embedding。
#
# 用法：
#   setsid nohup bash /home/zjh/state-vcc-local/run_train_state_emb.sh \
#     > /home/zjh/log/train_state_emb.log 2>&1 < /dev/null &

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
STATE=$ROOT/state
NAME=${NAME:-state_emb_run}
MAX_STEPS=${MAX_STEPS:-8000}
export CUDA_VISIBLE_DEVICES=${GPU:-3}

log() { echo "[$(date '+%F %T')] $*"; }

cd "$STATE" || { log "!! 进不去 $STATE"; exit 1; }

# 钉住 multiprocessing 临时目录：无论外层 shell 的 TMPDIR 是什么，
# pymp-*（内含 listener socket）一律落在 /tmp，由系统定期清理，不混入代码目录。
export TMPDIR=/tmp

# 双保险：trap EXIT 保证本次训练结束（含异常退出）自动清理；开头再清一次
# 兜底上一次的残留。
cleanup_pymp() { rm -rf "$STATE"/pymp-* 2>/dev/null; }
trap cleanup_pymp EXIT
cleanup_pymp

log "===== 前置检查 ====="
for f in vci_pretrain/starter_fewshot.toml \
         competition_support_set/ESM2_pert_features.pt \
         "$ROOT/val_perts.json"; do
  [[ -e "$f" ]] || { log "!! 缺少 $f"; exit 1; }
done
if [[ -d "competition/$NAME" ]]; then
  log "!! competition/$NAME 已存在 —— 会从 last.ckpt 续训。若要重跑请先改 NAME 或移走该目录"
fi
log "run 名称=$NAME  max_steps=$MAX_STEPS  GPU=$CUDA_VISIBLE_DEVICES"
df -h /home | tail -1

log "===== 启动训练（X_state embedding 路线） ====="
"$STATE/.venv/bin/state" tx train \
  data.kwargs.toml_config_path=vci_pretrain/starter_fewshot.toml \
  data.kwargs.embed_key="X_state" \
  data.kwargs.output_space=embedding \
  data.kwargs.num_workers=16 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=competition_support_set/ESM2_pert_features.pt \
  data.kwargs.output_space=embedding \
  training.max_steps="$MAX_STEPS" \
  training.val_freq=1000 \
  training.ckpt_every_n_steps=1000 \
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
