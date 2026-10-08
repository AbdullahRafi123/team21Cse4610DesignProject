#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/lightdp-flower-project}"
if [[ "$VENV_DIR" != /* ]]; then
  VENV_DIR="$PROJECT_ROOT/$VENV_DIR"
fi
export VENV_DIR
cd "$PROJECT_ROOT"

if [[ ! -x "$VENV_DIR/bin/flwr" ]]; then
  bash "$PROJECT_ROOT/setup/initialize.sh" || exit $?
fi

"$VENV_DIR/bin/flwr" run . --stream
status=$?
if [[ -t 0 ]]; then
  read -r -p "Run finished (exit $status). Press Enter to close this window. " _
fi
exit "$status"
