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
export PYTORCH_ENABLE_MPS_FALLBACK="1"
export PYTHONPATH="$PROJECT_ROOT"

echo "Starting F5-TTS Metal MPS Voice Daemon..."
DAEMON_PID=$("$PYTHON_BIN" -c '
import subprocess, sys
logfile, pidfile = sys.argv[1], sys.argv[2]
cmd = sys.argv[3:]
with open(logfile, "ab", buffering=0) as out:
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
with open(pidfile, "w") as pf:
    pf.write(str(p.pid) + "\n")
print(p.pid)
' "$LOG_FILE" "$PID_FILE" "$PYTHON_BIN" "$PROJECT_ROOT/scripts/f5_daemon.py" --host 127.0.0.1 --port 7861)

# Wait for daemon to become ready (model loads into MPS in ~10-20s)
for i in {1..40}; do
  if curl -s -m 1 http://127.0.0.1:7861/health | grep -q '"status":\s*"ready"'; then
    echo "✔ F5-TTS Voice Daemon is ONLINE and WARM on http://127.0.0.1:7861 (PID: $DAEMON_PID)"
    exit 0
  fi
  sleep 1
done

echo "⚠️ F5-TTS daemon started with PID $DAEMON_PID, but health endpoint is still initializing. Check logs at: $LOG_FILE"
