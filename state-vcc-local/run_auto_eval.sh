#!/usr/bin/env bash
# 训练完成后自动跑测试集评测
# 用法：setsid bash run_auto_eval.sh > /home/zjh/log/auto_eval.log 2>&1 < /dev/null &
#
# 监控 3 个实验，训练结束后自动调用 run_vcc_test_eval.sh

set -uo pipefail

ROOT=/home/zjh/state-vcc-local
EVAL_SCRIPT=$ROOT/run_vcc_test_eval.sh
LOG=/home/zjh/log/auto_eval.log
POLL_INTERVAL=120  # 每 2 分钟检查一次

log() { echo "[$(date '+%F %T')] $*"; }

# 实验配置：名称 | 训练目录 | checkpoint | 模板 | embed_key
declare -a EXPERIMENTS=(
  "state_lg_fused_adamw|competition/state_lg_fused_adamw|last.ckpt|competition_support_set/competition_test_template.h5ad|"
  "state_lg_cs256|competition/state_emb_all_lg_cs256_energy|last.ckpt|vci_pretrain/adata_Test.h5ad|X_state"
)

# 记录已完成的评测
declare -A EVALUATED

log "===== 自动评测监控启动 ====="
log "监控 ${#EXPERIMENTS[@]} 个实验，每 ${POLL_INTERVAL}s 检查一次"

while true; do
  all_done=true

  for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name run_dir ckpt template embed_key <<< "$exp"

    # 跳过已评测的
    if [[ "${EVALUATED[$name]:-}" == "1" ]]; then
      continue
    fi

    # 检查训练是否完成：看 last.ckpt 是否存在且训练进程已结束
    ckpt_path="$ROOT/state/$run_dir/checkpoints/$ckpt"
    if [[ ! -f "$ckpt_path" ]]; then
      all_done=false
      log "[$name] 等待 checkpoint: $ckpt_path"
      continue
    fi

    # 检查是否还有训练进程在跑（用 run_dir 名称匹配进程）
    dir_basename=$(basename "$run_dir")
    running=$(ps -eo args --no-headers 2>/dev/null | grep "$dir_basename" | grep -v grep | head -1)
    if [[ -n "$running" ]]; then
      all_done=false
      log "[$name] checkpoint 存在但训练仍在跑，继续等待"
      continue
    fi

    # 训练完成，开始评测
    log "[$name] ===== 训练完成，开始测试集评测 ====="
    EVALUATED[$name]=1

    # 设置评测参数
    export RUN_DIR="$run_dir"
    export CKPT="$ckpt"
    export TEMPLATE="$ROOT/state/$template"
    export EMBED_KEY="${embed_key}"
    export PRED="$ROOT/state/competition/prediction_test_${name}.h5ad"
    export OUT="$ROOT/state/competition/vcc_test_eval_${name}"
    export BASE_ADATA="$OUT/baseline.h5ad"
    export BASE_DE="$OUT/baseline_de.csv"

    log "[$name] RUN_DIR=$RUN_DIR CKPT=$CKPT TEMPLATE=$TEMPLATE EMBED_KEY=$EMBED_KEY"
    log "[$name] PRED=$PRED OUT=$OUT"

    # 运行评测
    bash "$EVAL_SCRIPT" > "/home/zjh/log/vcc_test_eval_${name}.log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
      log "[$name] ===== 评测完成 rc=$rc ====="
      # 输出结果摘要
      if [[ -f "$OUT/agg_results.csv" ]]; then
        log "[$name] 结果："
        cat "$OUT/agg_results.csv" | tail -3
      fi
    else
      log "[$name] ===== 评测失败 rc=$rc ====="
    fi

    # 清理环境变量
    unset RUN_DIR CKPT TEMPLATE EMBED_KEY PRED OUT BASE_ADATA BASE_DE
  done

  # 所有实验都评测完了
  if $all_done; then
    log "===== 所有实验评测完毕，监控退出 ====="
    break
  fi

  sleep $POLL_INTERVAL
done

log "===== auto_eval 结束 ====="
