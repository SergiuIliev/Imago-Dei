#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
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

LABEL="${1:-manual}"

echo "Creating Personal AI Time Capsule backup snapshot [label: $LABEL]..."
$PYTHON_BIN -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from core.config import CapsuleConfig
from memory.database import DatabaseManager
from memory.backup import BackupManager

config = CapsuleConfig.load_or_create()
db = DatabaseManager(config.storage.sqlite_db_path)
mgr = BackupManager(config, db)
res = mgr.create_full_backup(label='$LABEL')
print('Backup created successfully:')
print('File:', res['backup_file'])
print('Size:', res['size_bytes'], 'bytes')
print('SHA-256:', res['sha256'])
"
