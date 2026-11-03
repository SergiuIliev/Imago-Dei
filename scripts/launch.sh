#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(which python3)"
fi

HOST="127.0.0.1"
PORT="7860"

echo "=========================================================="
echo " Personal AI Time Capsule — Launching Application"
echo " Host: Apple Silicon M4 / macOS 27.0"
echo " Python Runtime: $($PYTHON_BIN --version)"
echo " Network Binding: http://$HOST:$PORT (Loopback Only)"
echo "=========================================================="

# Check Ollama status
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
  echo "✔ Local Ollama inference engine detected at http://127.0.0.1:11434"
else
  echo "⚠️ Notice: Ollama is not responding on 127.0.0.1:11434. The app will use Mock LLM fallback until Ollama is started."
fi

# Ensure fixtures are seeded if DB is fresh
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
"

echo "Starting Uvicorn server on http://$HOST:$PORT ..."
exec $PYTHON_BIN -m uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level info
