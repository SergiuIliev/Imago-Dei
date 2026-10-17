#!/usr/bin/env bash
# Double-click this file in Finder to stop the Personal AI Time Capsule server

DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$DIR/scripts/stop_server.sh"
