#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/timecapsule.pid"
LOG_FILE="$LOG_DIR/timecapsule.log"
mkdir -p "$LOG_DIR"

PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
  PYTHON_BIN="/Users/Sergiu/AI/venv-webui/bin/python"
fi
if [ ! -f "$PYTHON_BIN" ]; then
  PYTHON_BIN="/Users/Sergiu/.local/bin/python3.11"
fi
if [ ! -f "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(which python3)"
fi

# Check if already running on port 7860
if curl -s -m 2 http://127.0.0.1:7860/api/health > /dev/null 2>&1; then
  echo "✔ Time Capsule server is already running on http://127.0.0.1:7860"
  open "http://127.0.0.1:7860"
  osascript -e 'display notification "Time Capsule is already running. Opening browser..." with title "Time Capsule" subtitle "Already Active"' 2>/dev/null || true
  exit 0
fi

# Check Ollama
if ! curl -s -m 2 http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
  echo "⚠️ Notice: Ollama is not responding on 127.0.0.1:11434. Checking background services..."
  if [ -f "$PROJECT_ROOT/bin/ollama-service.sh" ]; then
    bash "$PROJECT_ROOT/bin/ollama-service.sh" start 2>/dev/null || true
  fi
fi

# Check F5-TTS Metal MPS Voice Daemon
if ! curl -s -m 1 http://127.0.0.1:7861/health > /dev/null 2>&1; then
  echo "Starting background F5-TTS Metal MPS Voice Daemon..."
  bash "$PROJECT_ROOT/scripts/start_f5_daemon.sh" > /dev/null 2>&1 &
fi

export PYTHONPATH="$PROJECT_ROOT"
export VIRTUAL_ENV="/Users/Sergiu/AI/venv-webui"
$PYTHON_BIN -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from core.config import CapsuleConfig
from memory.database import DatabaseManager
config = CapsuleConfig.load_or_create()
db = DatabaseManager(config.storage.sqlite_db_path)
stats = db.get_archive_stats()
if stats['total_memories'] == 0:
    print('[Launch] Fresh database detected. Seeding baseline test fixtures...')
    from scripts.seed_fixtures import seed_fixtures
    seed_fixtures()
" 2>/dev/null || true

echo "Starting Personal AI Time Capsule server..."
nohup "$PYTHON_BIN" -m uvicorn app.main:app --host 127.0.0.1 --port 7860 < /dev/null > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait for server to become responsive
for i in {1..20}; do
  if curl -s -m 1 http://127.0.0.1:7860/api/health > /dev/null 2>&1; then
    echo "✔ Time Capsule server is ONLINE at http://127.0.0.1:7860 (PID: $SERVER_PID)"
    open "http://127.0.0.1:7860"
    osascript -e 'display notification "Time Capsule is online at http://127.0.0.1:7860" with title "Time Capsule" subtitle "Server Started"' 2>/dev/null || true
    exit 0
  fi
  sleep 0.5
done

echo "⚠️ Server started with PID $SERVER_PID, but health endpoint is still initializing. Check logs at: $LOG_FILE"
open "http://127.0.0.1:7860" 2>/dev/null || true
