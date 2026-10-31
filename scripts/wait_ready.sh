#!/bin/bash
# 轮询等待 vLLM 服务就绪（也可以探测 trtllm-serve）
# 用法: wait_ready.sh [port] [log_file]
#   port      服务端口（默认 8000）
#   log_file  失败时回显的服务日志（可选）
set -u
PORT="${1:-8000}"
LOG="${2:-/dev/null}"
for i in $(seq 1 36); do
  if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "READY_AT_$((i*5))s"
    curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 300
    echo
    exit 0
  fi
  if ! pgrep -f "vllm serve" >/dev/null && ! pgrep -f "trtllm-serve" >/dev/null; then
    echo "DEAD"
    [ -f "$LOG" ] && tail -25 "$LOG"
    exit 1
  fi
  sleep 5
done
echo "TIMEOUT_180s"
[ -f "$LOG" ] && tail -10 "$LOG"
exit 1
