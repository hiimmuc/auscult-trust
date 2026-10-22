#!/usr/bin/env bash
# Run the API on this host, detached (127.0.0.1:8000). Start a tunnel separately (see app/README.md).
#   bash app/deploy/serve.sh start|stop|status
# Reads app/deploy/.env (AT_APP_TOKEN, AT_APP_ORIGINS). Log: logs/app-api.log. PIDs: logs/app.pids.
set -euo pipefail
cd "$(dirname "$0")/../.."
PIDS=logs/app.pids

stop() {
  [ -f "$PIDS" ] && while read -r p; do kill "$p" 2>/dev/null || true; done < "$PIDS"
  rm -f "$PIDS"
}

start() {
  stop
  mkdir -p logs
  set -a; source app/deploy/.env; set +a
  PYTHONPATH=. setsid nohup .venv-train/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 \
    > logs/app-api.log 2>&1 < /dev/null & echo $! >> "$PIDS"
  echo "started: $(tr '\n' ' ' < "$PIDS")"
}

status() {
  [ -f "$PIDS" ] && ps -o pid,etime,args -p "$(paste -sd, "$PIDS")" || echo "not running"
  curl -s -m 3 127.0.0.1:8000/v1/health; echo
}

"${1:-status}"
