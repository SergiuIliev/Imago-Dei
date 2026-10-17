#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if port 7860 is running
if curl -s -m 1 http://127.0.0.1:7860/api/health > /dev/null 2>&1 || lsof -ti tcp:7860 -sTCP:LISTEN > /dev/null 2>&1; then
  echo "Server is currently running. Stopping..."
  bash "$PROJECT_ROOT/scripts/stop_server.sh"
else
  echo "Server is currently stopped. Starting..."
  bash "$PROJECT_ROOT/scripts/start_server.sh"
fi
