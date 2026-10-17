#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_backup_archive.tar.gz>"
  exit 1
fi

BACKUP_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

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

echo "Restoring Personal AI Time Capsule from: $BACKUP_FILE ..."
$PYTHON_BIN -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from core.config import CapsuleConfig
from memory.database import DatabaseManager
from memory.backup import BackupManager

config = CapsuleConfig.load_or_create()
db = DatabaseManager(config.storage.sqlite_db_path)
mgr = BackupManager(config, db)
res = mgr.restore_backup('$BACKUP_FILE')
print('Restore completed successfully:', res)
"
