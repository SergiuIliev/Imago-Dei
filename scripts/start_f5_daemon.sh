#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/f5_daemon.pid"
LOG_FILE="$LOG_DIR/f5_daemon.log"
mkdir -p "$LOG_DIR"

PYTHON_BIN="/Users/Sergiu/AI/venv-voice/bin/python"

# Check if already running on port 7861
if curl -s -m 1 http://127.0.0.1:7861/health > /dev/null 2>&1; then
  echo "✔ F5-TTS voice daemon is already running on http://127.0.0.1:7861"
  exit 0
fi

export PYTHONHASHSEED="0"
export PYTHONPATH="$PROJECT_ROOT"

echo "Starting F5-TTS Metal MPS Voice Daemon..."
nohup "$PYTHON_BIN" "$PROJECT_ROOT/scripts/f5_daemon.py" --host 127.0.0.1 --port 7861 < /dev/null > "$LOG_FILE" 2>&1 &
DAEMON_PID=$!
echo "$DAEMON_PID" > "$PID_FILE"

# Wait for daemon to become ready (model loads into MPS in ~10-20s)
for i in {1..40}; do
  if curl -s -m 1 http://127.0.0.1:7861/health | grep -q '"status":\s*"ready"'; then
    echo "✔ F5-TTS Voice Daemon is ONLINE and WARM on http://127.0.0.1:7861 (PID: $DAEMON_PID)"
    exit 0
  fi
  sleep 1
done

echo "⚠️ F5-TTS daemon started with PID $DAEMON_PID, but health endpoint is still initializing. Check logs at: $LOG_FILE"
