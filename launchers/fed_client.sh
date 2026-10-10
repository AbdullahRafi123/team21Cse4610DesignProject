#!/usr/bin/env bash
# Start one Flower SuperNode for the networked FedAbba deployment.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/lightdp-flower-gpu}"
CLIENT_ID=""
CLIENT_COUNT="10"
COORDINATOR_HOST=""
CA_CERT="${HOME}/fedabba-certs/ca.crt"
CLIENT_DATA=""
APPIO_PORT="9094"
LOG_FILE=""

usage() {
  cat <<'EOF'
Usage: launchers/fed_client.sh --id ID --host HOST [options]

Required:
  --id ID                 Client ID from 0 through CLIENT_COUNT - 1.
  --host HOST             Coordinator hostname or IP address.

Options:
  --clients COUNT         Number of connected SuperNodes (default: 10).
  --ca-cert PATH          Trusted coordinator CA certificate.
  --data PATH             Local NPZ dataset for client-data-mode=local_npz.
  --appio-port PORT       Local ClientAppIO port (default: 9094).
  --log-file PATH         Append this SuperNode's terminal output to PATH.
  --venv PATH             Python virtual environment.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --id) CLIENT_ID="$2"; shift 2 ;;
    --clients) CLIENT_COUNT="$2"; shift 2 ;;
    --host) COORDINATOR_HOST="$2"; shift 2 ;;
    --ca-cert) CA_CERT="$2"; shift 2 ;;
    --data) CLIENT_DATA="$2"; shift 2 ;;
    --appio-port) APPIO_PORT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$CLIENT_ID" =~ ^[0-9]+$ ]] || { echo "--id must be a non-negative integer" >&2; exit 2; }
[[ "$CLIENT_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "--clients must be a positive integer" >&2; exit 2; }
(( CLIENT_ID < CLIENT_COUNT )) || { echo "--id must be smaller than --clients" >&2; exit 2; }
[[ -n "$COORDINATOR_HOST" ]] || { echo "--host is required" >&2; exit 2; }
[[ -f "$CA_CERT" ]] || { echo "CA certificate does not exist: $CA_CERT" >&2; exit 2; }

if [[ ! -x "$VENV_DIR/bin/flower-supernode" ]]; then
  echo "Creating the client environment at $VENV_DIR"
  VENV_DIR="$VENV_DIR" bash "$PROJECT_ROOT/setup/initialize.sh"
fi

if [[ -n "$CLIENT_DATA" ]]; then
  [[ -f "$CLIENT_DATA" ]] || { echo "Client dataset does not exist: $CLIENT_DATA" >&2; exit 2; }
  export FEDABBA_CLIENT_DATA="$CLIENT_DATA"
else
  unset FEDABBA_CLIENT_DATA || true
fi

source "$VENV_DIR/bin/activate"
echo "Starting client $CLIENT_ID/$((CLIENT_COUNT - 1)) -> $COORDINATOR_HOST:9092"
client_command=(flower-supernode \
  --root-certificates "$CA_CERT" \
  --superlink "$COORDINATOR_HOST:9092" \
  --clientappio-api-address "127.0.0.1:$APPIO_PORT" \
  --node-config="partition-id=$CLIENT_ID num-partitions=$CLIENT_COUNT")
if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname -- "$LOG_FILE")"
  "${client_command[@]}" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi
exec "${client_command[@]}"
