#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
  PYTHON_BIN="/Users/Sergiu/AI/venv-webui/bin/python"
fi

TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
  echo "Usage: $0 <path_to_video_or_audio_directory>"
  echo "Example: $0 /Volumes/Storage/PersonalVlogs"
  exit 1
fi

echo "Initiating media ingestion for directory: $TARGET_DIR ..."
$PYTHON_BIN -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from core.config import CapsuleConfig
from memory.database import DatabaseManager
from memory.vector_store import VectorStore
from ingestion.pipeline import MediaIngestionPipeline

config = CapsuleConfig.load_or_create()
db = DatabaseManager(config.storage.sqlite_db_path)
vs = VectorStore(config.storage.chroma_dir)
pipeline = MediaIngestionPipeline(config, db, vs)
res = pipeline.run_ingestion_job(target_directory='$TARGET_DIR')
print('Ingestion Job Summary:')
import json
print(json.dumps(res, indent=2))
"
