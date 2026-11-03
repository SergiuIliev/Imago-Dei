#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PLIST_PATH="/Users/Sergiu/AI/config/com.sergiu.timecapsule.plist"
UID_NUM="$(id -u)"

# Check if port 7860 is running
if curl -s -m 1 http://127.0.0.1:7860/api/health > /dev/null 2>&1 || lsof -ti tcp:7860 -sTCP:LISTEN > /dev/null 2>&1; then
  echo "Server is currently running. Stopping..."
  bash "$PROJECT_ROOT/scripts/stop_server.sh"
else
  echo "Server is currently stopped. Starting via launchd (AbandonProcessGroup=true)..."
  launchctl bootout "gui/$UID_NUM/com.sergiu.timecapsule" 2>/dev/null || true
  launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH" 2>/dev/null || true
  launchctl kickstart -k "gui/$UID_NUM/com.sergiu.timecapsule"
  for i in {1..30}; do
    if curl -s -m 1 http://127.0.0.1:7860/api/health > /dev/null 2>&1; then
      exit 0
    fi
    sleep 0.5
  done
fi
