#!/usr/bin/env bash
set -euo pipefail
SITE=/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena/site
NODE=node
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
if command -v setsid >/dev/null 2>&1; then
  setsid env PORT="$PORT" "$NODE" "$SITE/server.cjs" > "$LOG" 2>&1 < /dev/null &
else
  PORT="$PORT" nohup "$NODE" "$SITE/server.cjs" > "$LOG" 2>&1 < /dev/null &
fi
echo $! > "$PID"
sleep 1
if ! kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "server failed to stay running; log follows" >&2
  cat "$LOG" >&2
  exit 1
fi
echo "started pid=$(cat "$PID") port=$PORT"
