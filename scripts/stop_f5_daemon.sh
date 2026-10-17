#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PID_FILE="$PROJECT_ROOT/logs/f5_daemon.pid"
STOPPED=0

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "Stopping F5-TTS Voice Daemon (PID $PID)..."
    kill "$PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      kill -9 "$PID" 2>/dev/null || true
    fi
    STOPPED=1
  fi
  rm -f "$PID_FILE"
fi

PORT_PIDS=$(lsof -ti tcp:7861 -sTCP:LISTEN 2>/dev/null || echo "")
if [ -n "$PORT_PIDS" ]; then
  for p in $PORT_PIDS; do
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      echo "Terminating F5 process on port 7861 (PID $p)..."
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
  echo "✔ F5-TTS Voice Daemon stopped successfully."
else
  echo "ℹ️ F5-TTS Voice Daemon is not running."
fi
