#!/usr/bin/env bash
# Double-click this file in Finder to toggle the Personal AI Time Capsule server (start if stopped, stop if running)

DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$DIR/scripts/toggle_server.sh"
