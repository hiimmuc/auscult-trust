#!/usr/bin/env bash
# Run the demo on this host, detached: API (127.0.0.1:8000), static web build (127.0.0.1:8090), Cloudflare tunnel.
#   bash app/deploy/serve.sh start|stop|status
# Reads app/deploy/.env (AT_APP_TOKEN, AT_APP_ORIGINS). Logs: logs/app-{api,web,tunnel}.log. PIDs: logs/app.pids.
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
  PYTHONPATH=. setsid nohup .venv-train/bin/uvicorn app.api.server:app --host 127.0.0.1 --port 8000 \
    > logs/app-api.log 2>&1 < /dev/null & echo $! >> "$PIDS"
  setsid nohup .venv-train/bin/python -m http.server 8090 --bind 127.0.0.1 --directory app/web/dist \
    > logs/app-web.log 2>&1 < /dev/null & echo $! >> "$PIDS"
  setsid nohup cloudflared --config app/deploy/cloudflared.yml tunnel run \
    > logs/app-tunnel.log 2>&1 < /dev/null & echo $! >> "$PIDS"
  echo "started: $(tr '\n' ' ' < "$PIDS")"
}

status() {
  [ -f "$PIDS" ] && ps -o pid,etime,args -p "$(paste -sd, "$PIDS")" || echo "not running"
  curl -s -m 3 127.0.0.1:8000/v1/health; echo
}

"${1:-status}"
