#!/usr/bin/env bash
# Launch, stop or restart the trajdata Web UI
set -e
cd "$(dirname "$0")"

PORT=5006
COMMAND="run"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    stop)    COMMAND="stop";   shift ;;
    restart) COMMAND="restart"; shift ;;
    --port)  PORT="$2";        shift 2 ;;
    --port=*) PORT="${1#*=}";  shift ;;
    --no-browser) EXTRA_ARGS="$EXTRA_ARGS --no-browser"; shift ;;
    *) shift ;;
  esac
done

function stop_server() {
  PID=$(lsof -ti :$PORT || true)
  if [ -n "$PID" ]; then
    echo "Stopping server on port $PORT (PID: $PID)..."
    kill -9 $PID 2>/dev/null || true
    sleep 1
  else
    echo "No server running on port $PORT."
  fi
}

function run_server() {
  # Auto-kill if port is busy before running
  PID=$(lsof -ti :$PORT || true)
  if [ -n "$PID" ]; then
    echo "Force-closing existing process on port $PORT..."
    kill -9 $PID 2>/dev/null || true
    sleep 0.5
  fi
  echo "Launching trajdata Web UI at http://localhost:$PORT/..."
  exec .venv/bin/python trajdata_webui/main.py --port "$PORT" $EXTRA_ARGS
}

case $COMMAND in
  stop)
    stop_server
    ;;
  restart)
    stop_server
    run_server
    ;;
  run)
    run_server
    ;;
esac
