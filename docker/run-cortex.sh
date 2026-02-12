#!/bin/bash
# Backward-compatible alias to the maintained compose launcher.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run-compose.sh" "$@"
