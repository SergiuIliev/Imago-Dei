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

echo "=========================================================="
echo " Running Personal AI Time Capsule Test Suite (Project 30) "
echo " Environment: Apple Silicon M4 / macOS $(sw_vers -productVersion 2>/dev/null || echo '27.0')"
echo " Python: $($PYTHON_BIN --version)"
echo "=========================================================="

$PYTHON_BIN -m unittest discover -s "$PROJECT_ROOT/tests" -p "test_*.py" -v
