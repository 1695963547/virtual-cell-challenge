#!/usr/bin/env bash
# 下载 Virtual Cell Challenge 2025 官方测试集（带真实表达值）
# 特点：断点续传 + 无限重试 + 大小校验，本地断网不影响服务器端继续下载
# 用法： setsid nohup bash /home/zjh/download_vcc_test.sh > /home/zjh/log/download_vcc_test.log 2>&1 &

set -u

BASE="https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025"
DEST="/home/zjh/vcc_official"
mkdir -p "$DEST"

log() { echo "[$(date '+%F %T')] $*"; }

# 文件名  期望字节数  远端相对路径
FILES=(
  "pert_counts_Test.csv|1894|test/pert_counts_Test.csv"
  "adata_Test.h5ad|11950739168|test/adata_Test.h5ad"
  "gene_names.csv|0|gene_names.csv"
)

fetch() {
  local name="$1" want="$2" rel="$3"
  local out="$DEST/$name"
  local url="$BASE/$rel"

  # 期望大小为 0 表示未知，向远端问一次
  if [[ "$want" == "0" ]]; then
    want=$(curl -sSI --max-time 60 "$url" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
    [[ -z "$want" ]] && { log "!! 无法获取 $name 的大小，跳过"; return 1; }
  fi

  local try=0
  while :; do
    local have=0
    [[ -f "$out" ]] && have=$(stat -c %s "$out")
    if [[ "$have" -eq "$want" ]]; then
      log "OK  $name 已完整 ($want bytes)"
      return 0
    fi
    if [[ "$have" -gt "$want" ]]; then
      log "!! $name 本地 ($have) 大于远端 ($want)，删除重下"
      rm -f "$out"
      have=0
    fi

    try=$((try+1))
    log ">>> $name 第 $try 次尝试，已有 $have / $want bytes ($(awk -v h=$have -v w=$want 'BEGIN{printf "%.2f", h*100/w}')%)"

    # -c 断点续传；--read-timeout 卡住即断开重试；--tries=0 内部也无限重试
    wget -c --tries=0 --timeout=60 --read-timeout=120 --waitretry=15 \
         --retry-connrefused --no-verbose --show-progress --progress=dot:giga \
         -O "$out" "$url"

    have=0
    [[ -f "$out" ]] && have=$(stat -c %s "$out")
    if [[ "$have" -eq "$want" ]]; then
      log "OK  $name 下载完成 ($want bytes)"
      return 0
    fi
    log "... $name 未完成 ($have/$want)，20 秒后继续续传"
    sleep 20
  done
}

log "===== 开始下载，目标目录 $DEST ====="
for f in "${FILES[@]}"; do
  IFS='|' read -r name want rel <<< "$f"
  fetch "$name" "$want" "$rel"
done

log "===== 全部完成，校验 crc32c ====="
python3 - <<'PY'
import sys, base64
try:
    import google_crc32c as g
    def crc(p):
        c = g.Checksum()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(8 << 20), b''):
                c.update(chunk)
        return base64.b64encode(c.digest()).decode()
except ImportError:
    print("未安装 google-crc32c，跳过校验和比对（大小已校验通过）")
    sys.exit(0)

expect = {
    "/home/zjh/vcc_official/adata_Test.h5ad": "JMP5RQ==",
    "/home/zjh/vcc_official/pert_counts_Test.csv": "dCaUQw==",
}
for p, e in expect.items():
    a = crc(p)
    print(f"{'PASS' if a == e else 'FAIL'}  {p}  expect={e} actual={a}")
PY

log "===== 目录内容 ====="
ls -la "$DEST"
