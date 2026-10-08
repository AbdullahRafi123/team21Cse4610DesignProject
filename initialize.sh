#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
      "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON must name an installed interpreter executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)'; then
  echo "Python 3.10, 3.11, 3.12, or 3.13 is required. Set PYTHON to an installed interpreter." >&2
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

if ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)'; then
  echo "The existing virtual environment uses unsupported Python. Remove $VENV_DIR or set VENV_DIR to a new location." >&2
  exit 1
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$PROJECT_ROOT"

echo
echo "Setup complete. Activate with: source \"$VENV_DIR/bin/activate\""
echo "Then run a smoke experiment from the project directory: flwr run . --stream"
