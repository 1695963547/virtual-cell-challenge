#!/bin/bash
# gpu_watch_train.sh: GPU 空闲监控 + 环境自检 + 自动启动 scDFM 训练
#
# 流程：
#   1) 每 interval 秒检查 GPU 0-3 是否空闲（无任何 compute 进程）
#   2) 空闲后先跑最小复现（sdpa_repro.sh）确认 cuBLAS/SDPA 环境已恢复
#   3) 复现通过 → 备份旧日志 → 启动 train_cp10k.sh
#   4) 启动后验证 20 分钟：进程存活且日志出现 loss 输出才算成功，
#      否则视为环境仍异常，按 --retries 重试
#
# 用法:
#   bash gpu_watch_train.sh [--interval 60] [--max-wait-hours 48] [--retries 1] [--gpus 0,1,2,3] [--dry-run]
#
# 后台运行:
#   setsid nohup bash gpu_watch_train.sh > /home/zjh/log/scdfm_vcc_gpu_watch.out 2>&1 &

cd /home/zjh/scDFM/vcc || exit 1

INTERVAL=60
MAX_WAIT_HOURS=48
RETRIES=1
GPUS="0,1,2,3"
DRY_RUN=0
WATCH_LOG=/home/zjh/log/scdfm_vcc_gpu_watch.log
TRAIN_LOG=/home/zjh/log/scdfm_vcc_train_cp10k.log

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --max-wait-hours) MAX_WAIT_HOURS="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCH_LOG"; }

# 是否已有训练进程在跑（避免重复启动）
train_already_running() {
  pgrep -f "src/script/run.py" >/dev/null 2>&1
}

# GPU 空闲判定：指定卡上没有任何 compute 进程
gpus_idle() {
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPUS" 2>/dev/null | tr -d ' ')
  [ -z "$pids" ]
}

# 单次最小复现，通过返回 0
run_sdpa_repro() {
  bash /home/zjh/scDFM/vcc/sdpa_repro.sh > /tmp/scdfm_sdpa_repro.out 2>&1
  grep -q "ALL PASSED" /tmp/scdfm_sdpa_repro.out
}

# 启动训练并验证。成功返回 0，失败返回 1
start_and_verify_training() {
  # 备份上次的训练日志（> 重定向会覆盖）
  if [ -f "$TRAIN_LOG" ]; then
    mv "$TRAIN_LOG" "${TRAIN_LOG}.$(date '+%Y%m%d_%H%M%S').prev"
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] 环境已恢复，跳过训练启动（--dry-run）"
    return 0
  fi

  bash /home/zjh/scDFM/vcc/train_cp10k.sh
  # 训练进程需要几秒启动（python 解释器 + import），轮询最多 60s 找进程
  local train_pid=""
  local i
  for i in $(seq 1 12); do
    sleep 5
    train_pid=$(pgrep -f "src/script/run.py" | head -1)
    [ -n "$train_pid" ] && break
  done
  if [ -z "$train_pid" ]; then
    log "训练启动失败：60s 内未找到 run.py 进程"
    return 1
  fi
  log "训练已启动 PID=$train_pid，进入 20 分钟验证期..."

  # 验证期：最多 20 分钟，每 60s 检查一次
  local waited=0
  local saw_loss=0
  while [ "$waited" -lt 1200 ]; do
    sleep 60
    waited=$((waited + 60))
    if ! kill -0 "$train_pid" 2>/dev/null; then
      log "训练进程在 ${waited}s 内退出（疑似再次崩溃）"
      tail -5 "$TRAIN_LOG" >> "$WATCH_LOG" 2>/dev/null
      return 1
    fi
    if [ "$saw_loss" -eq 0 ] && grep -q "loss:" "$TRAIN_LOG" 2>/dev/null; then
      saw_loss=1
      log "训练已跑出第一个 step 的 loss 输出（+${waited}s），环境正常"
    fi
  done

  if [ "$saw_loss" -eq 1 ]; then
    log "验证通过：训练稳定运行（PID=$train_pid）。监控任务完成"
    return 0
  fi
  log "验证期超时：进程存活但 20 分钟内无 loss 输出，请人工检查 $TRAIN_LOG"
  return 1
}

# ---------- 主流程 ----------
log "=========================================="
log "GPU 监控启动 | 监控卡: $GPUS | 间隔: ${INTERVAL}s | 最长等待: ${MAX_WAIT_HOURS}h | 重试: $RETRIES | dry-run: $DRY_RUN"

if train_already_running; then
  log "检测到已有 run.py 训练进程在运行，退出（避免重复启动）"
  exit 0
fi

MAX_SECONDS=$((MAX_WAIT_HOURS * 3600))
elapsed=0

while true; do
  if [ "$elapsed" -ge "$MAX_SECONDS" ]; then
    log "已达到最长等待时间 ${MAX_WAIT_HOURS}h，GPU 仍未空闲，退出"
    exit 1
  fi

  if ! gpus_idle; then
    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
    continue
  fi

  log "GPU $GPUS 已空闲（等待了 $((elapsed / 60)) 分钟），开始环境自检..."
  if run_sdpa_repro; then
    log "最小复现通过，cuBLAS/SDPA 环境已恢复"
  else
    log "最小复现仍失败，继续等待（详见 /tmp/scdfm_sdpa_repro.out）"
    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
    continue
  fi

  if start_and_verify_training; then
    exit 0
  fi

  if [ "$RETRIES" -le 0 ]; then
    log "重试次数用尽，退出。请人工检查训练日志"
    exit 1
  fi
  RETRIES=$((RETRIES - 1))
  log "训练验证失败，剩余重试次数: $RETRIES，10 分钟后重试..."
  sleep 600
done
