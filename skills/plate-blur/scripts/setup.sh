#!/usr/bin/env bash
# Creates (or updates) the plate-blur Python environment.
# Location: $PLATE_BLUR_HOME (default ~/.local/share/plate-blur) — outside the
# skill folder, so plugin updates don't wipe it and nothing large lands in git.
set -euo pipefail

HOME_DIR="${PLATE_BLUR_HOME:-$HOME/.local/share/plate-blur}"
VENV="$HOME_DIR/venv"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

mkdir -p "$HOME_DIR"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r "$SKILL_DIR/requirements.txt"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found — output will have no audio. Install it (macOS: brew install ffmpeg)." >&2
fi
echo "plate-blur ready: $VENV/bin/python"
