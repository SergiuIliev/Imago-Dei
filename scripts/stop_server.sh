#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PID_FILE="$PROJECT_ROOT/logs/timecapsule.pid"
STOPPED=0

# 1. Kill via PID file if present
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "Stopping Time Capsule server (PID $PID)..."
    kill "$PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      kill -9 "$PID" 2>/dev/null || true
    fi
    STOPPED=1
  fi
  rm -f "$PID_FILE"
fi

# 2. Check any lingering listening process on port 7860
PORT_PIDS=$(lsof -ti tcp:7860 -sTCP:LISTEN 2>/dev/null || echo "")
if [ -n "$PORT_PIDS" ]; then
  for p in $PORT_PIDS; do
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      echo "Terminating server process on port 7860 (PID $p)..."
      kill "$p" 2>/dev/null || true
      sleep 1
      if kill -0 "$p" 2>/dev/null; then
        kill -9 "$p" 2>/dev/null || true
      fi
      STOPPED=1
    fi
  done
fi

if [ $STOPPED -eq 1 ]; then
  echo "✔ Personal AI Time Capsule server stopped successfully."
  osascript -e 'display notification "Time Capsule server has been stopped." with title "Time Capsule" subtitle "Server Stopped"' 2>/dev/null || true
else
  echo "ℹ️ Time Capsule server is not currently running."
  osascript -e 'display notification "Time Capsule server is not currently running." with title "Time Capsule" subtitle "Status Check"' 2>/dev/null || true
fi
