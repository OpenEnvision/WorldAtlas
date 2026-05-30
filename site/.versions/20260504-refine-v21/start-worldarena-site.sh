#!/usr/bin/env bash
set -euo pipefail
SITE=/share/project/bench/WorldArena-Site
NODE=/share/project/bench/tools/node-v20.11.1-linux-x64/bin/node
PID=$SITE/worldarena-site.pid
LOG=$SITE/worldarena-site.log
PORT=${PORT:-4173}
cd "$SITE"
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running pid=$(cat "$PID")"
  exit 0
fi
if ss -ltnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "port ${PORT} is already in use" >&2
  ss -ltnp 2>/dev/null | grep ":${PORT} " >&2 || true
  exit 1
fi
: > "$LOG"
PORT="$PORT" nohup "$NODE" "$SITE/server.cjs" > "$LOG" 2>&1 &
echo $! > "$PID"
echo "started pid=$(cat "$PID") port=$PORT"
