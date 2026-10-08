#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10, 3.11, or 3.12 is required. Set PYTHON to the interpreter to use." >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)'; then
  echo "Python 3.10, 3.11, or 3.12 is required." >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"
if [[ "$PROJECT_ROOT" == *" "* && "$VENV_DIR" == ".venv" ]]; then
  VENV_DIR="${HOME}/.venvs/lightdp-flower-project"
  echo "Project path contains spaces; using $VENV_DIR so Ray can start workers."
elif [[ "$VENV_DIR" != /* ]]; then
  VENV_DIR="$PROJECT_ROOT/$VENV_DIR"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  mkdir -p "$(dirname -- "$VENV_DIR")"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$PROJECT_ROOT"

echo
echo "Setup complete. Activate with: source \"$VENV_DIR/bin/activate\""
echo "Then run a smoke experiment from the project directory: flwr run . --stream"
