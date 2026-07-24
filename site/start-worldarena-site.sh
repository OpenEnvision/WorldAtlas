#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SITE="$SCRIPT_DIR"
NODE=${NODE:-$(command -v node 2>/dev/null || true)}
if [ -z "$NODE" ]; then
  NODE=$(ls -1 /root/.cursor-server/bin/linux-x64/*/node 2>/dev/null | tail -1 || true)
fi
if [ -z "$NODE" ] || [ ! -x "$NODE" ]; then
  echo "node executable not found; set NODE=/path/to/node and retry" >&2
  exit 1
fi
PID="$SITE/worldarena-site.pid"
LOG="$SITE/worldarena-site.log"
PORT=${PORT:-4173}
WORLDARENA_GENERATED_ROOT=${WORLDARENA_GENERATED_ROOT:-"$(cd "$SITE/.." && pwd)/docs/generated"}
ENV_FILE="$SITE/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
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
  setsid env PORT="$PORT" WORLDARENA_GENERATED_ROOT="$WORLDARENA_GENERATED_ROOT" "$NODE" "$SITE/server.cjs" > "$LOG" 2>&1 < /dev/null &
else
  PORT="$PORT" WORLDARENA_GENERATED_ROOT="$WORLDARENA_GENERATED_ROOT" nohup "$NODE" "$SITE/server.cjs" > "$LOG" 2>&1 < /dev/null &
fi
echo $! > "$PID"
sleep 1
if ! kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "server failed to stay running; log follows" >&2
  cat "$LOG" >&2
  exit 1
fi
echo "started pid=$(cat "$PID") port=$PORT"
