#!/usr/bin/env bash
# Runs plate_blur.py inside its venv, creating the venv on first use.
# All arguments are passed through: run.sh <video-or-folder> [options]
set -euo pipefail

HOME_DIR="${PLATE_BLUR_HOME:-$HOME/.local/share/plate-blur}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$HOME_DIR/venv/bin/python" ]; then
  bash "$SCRIPTS/setup.sh"
fi

# Keep macOS awake during long batches.
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i "$HOME_DIR/venv/bin/python" "$SCRIPTS/plate_blur.py" "$@"
fi
exec "$HOME_DIR/venv/bin/python" "$SCRIPTS/plate_blur.py" "$@"
