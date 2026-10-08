#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Could not find $PYTHON. Set PYTHON to Python 3.10, 3.11, or 3.12." >&2
  exit 1
fi

"$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' || {
  echo "Python 3.10, 3.11, or 3.12 is required." >&2
  exit 1
}

VENV_DIR="${VENV_DIR:-.venv}"
if [[ "$SCRIPT_DIR" == *" "* && "$VENV_DIR" == ".venv" ]]; then
  VENV_DIR="${HOME}/.venvs/lightdp-flower-project"
  echo "Project path contains spaces; using a space-free environment at $VENV_DIR for Ray compatibility."
fi

if [[ ! -d "$VENV_DIR" ]]; then
  mkdir -p "$(dirname -- "$VENV_DIR")"
  "$PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$SCRIPT_DIR"

echo
echo "Setup complete. Activate the environment with: source \"$VENV_DIR/bin/activate\""
echo "Then start the smoke run with: flwr run . --stream"
