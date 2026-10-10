#!/usr/bin/env bash
# Clickable coordinator launcher for a ten-client Flower deployment.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/lightdp-flower-gpu}"
CERTS_DIR="${HOME}/fedabba-certs"
CLIENTS=10
SMOKE=false
HEADLESS=false

usage() {
  cat <<'EOF'
Usage: launchers/start_10_client_lab.sh [--smoke] [--headless]

Without options, opens ten client terminal windows and interactively starts a
fresh or resumed ten-client Flower run. --smoke runs one IID FedAvg round with
eight records per client. --headless starts the ten SuperNodes in the
background instead of opening graphical terminal windows; it is useful for CI
or an unattended smoke check.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=true; shift ;;
    --headless) HEADLESS=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

prompt() {
  local text="$1" default="$2" value
  read -r -p "$text [$default]: " value
  printf '%s' "${value:-$default}"
}

require_free_ports() {
  local ports='9091|9092|9093|9094|9095|9096|9097|9098|9099|9100|9101|9102|9103'
  if ss -ltn | grep -Eq ":($ports)[[:space:]]"; then
    echo "A required Flower port is already in use. Stop the existing Flower process first." >&2
    exit 1
  fi
}

create_certificate() {
  local host="$1" subject san
  mkdir -p "$CERTS_DIR"
  if [[ -f "$CERTS_DIR/ca.crt" && -f "$CERTS_DIR/ca.key" && \
        -f "$CERTS_DIR/server.pem" && -f "$CERTS_DIR/server.key" ]]; then
    if openssl x509 -in "$CERTS_DIR/server.pem" -noout -ext subjectAltName | grep -Fq "$host"; then
      return
    fi
    echo "Existing TLS certificate does not include '$host' in its Subject Alternative Name." >&2
    echo "Use that certificate's host, or move the four existing certificate files aside and rerun." >&2
    exit 1
  fi
  if compgen -G "$CERTS_DIR/ca.*" >/dev/null || compgen -G "$CERTS_DIR/server.*" >/dev/null; then
    echo "Incomplete certificate set in $CERTS_DIR; refusing to overwrite it." >&2
    exit 1
  fi
  if [[ "$host" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ || "$host" == *:* ]]; then
    san="IP:$host,IP:127.0.0.1"
  else
    san="DNS:$host,IP:127.0.0.1"
  fi
  subject="/CN=$host"
  (
    umask 077
    openssl req -x509 -newkey rsa:4096 -nodes -days 3650 \
      -keyout "$CERTS_DIR/ca.key" -out "$CERTS_DIR/ca.crt" \
      -subj '/CN=FedAbba Lab CA' \
      -addext 'basicConstraints=critical,CA:TRUE' \
      -addext 'keyUsage=critical,keyCertSign,cRLSign'
    openssl req -new -newkey rsa:2048 -nodes \
      -keyout "$CERTS_DIR/server.key" -out "$CERTS_DIR/server.csr" \
      -subj "$subject"
    printf 'subjectAltName=%s\nextendedKeyUsage=serverAuth\nkeyUsage=digitalSignature,keyEncipherment\n' "$san" \
      > "$CERTS_DIR/server.ext"
    openssl x509 -req -in "$CERTS_DIR/server.csr" \
      -CA "$CERTS_DIR/ca.crt" -CAkey "$CERTS_DIR/ca.key" -CAcreateserial \
      -out "$CERTS_DIR/server.pem" -days 825 -sha256 -extfile "$CERTS_DIR/server.ext"
  )
  chmod 600 "$CERTS_DIR/ca.key" "$CERTS_DIR/server.key"
}

open_local_clients() {
  local client_id port command_line terminal log_file
  if [[ "$HEADLESS" == true ]]; then
    for client_id in $(seq 0 $((CLIENTS - 1))); do
      port=$((9094 + client_id))
      log_file="$CLIENT_LOG_DIR/client_$(printf '%02d' "$client_id").log"
      nohup env "VENV_DIR=$VENV_DIR" bash "$PROJECT_ROOT/launchers/fed_client.sh" \
        --id "$client_id" --clients "$CLIENTS" --host 127.0.0.1 \
        --ca-cert "$CERTS_DIR/ca.crt" --appio-port "$port" --log-file "$log_file" \
        >/dev/null 2>&1 &
      echo $! >> "$CLIENT_LOG_DIR/client_pids.txt"
    done
    return
  fi
  if command -v gnome-terminal >/dev/null 2>&1; then
    terminal='gnome-terminal'
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    terminal='x-terminal-emulator'
  elif command -v xterm >/dev/null 2>&1; then
    terminal='xterm'
  else
    echo "No supported terminal emulator was found (tried gnome-terminal, x-terminal-emulator, xterm)." >&2
    exit 1
  fi
  for client_id in $(seq 0 $((CLIENTS - 1))); do
    port=$((9094 + client_id))
    log_file="$CLIENT_LOG_DIR/client_$(printf '%02d' "$client_id").log"
    printf -v command_line '%q ' env "VENV_DIR=$VENV_DIR" bash \
      "$PROJECT_ROOT/launchers/fed_client.sh" --id "$client_id" --clients "$CLIENTS" \
      --host 127.0.0.1 --ca-cert "$CERTS_DIR/ca.crt" --appio-port "$port" \
      --log-file "$log_file"
    case "$terminal" in
      gnome-terminal)
        gnome-terminal --title="FedAbba client $client_id" -- bash -lc "$command_line; exec bash" &
        ;;
      x-terminal-emulator)
        x-terminal-emulator -T "FedAbba client $client_id" -e bash -lc "$command_line; exec bash" &
        ;;
      xterm)
        xterm -T "FedAbba client $client_id" -e bash -lc "$command_line; exec bash" &
        ;;
    esac
  done
}

cd "$PROJECT_ROOT"
if [[ ! -x "$VENV_DIR/bin/flower-superlink" ]]; then
  echo "Creating the coordinator environment at $VENV_DIR"
  VENV_DIR="$VENV_DIR" bash "$PROJECT_ROOT/setup/initialize.sh"
fi
source "$VENV_DIR/bin/activate"

echo "FedAbba: ten-client Flower deployment"
echo "This host will run one coordinator and ten simultaneous local Flower clients."
coordinator_host='127.0.0.1'
client_data_mode='partitioned_cifar10'

if [[ "$SMOKE" == true ]]; then
  run_mode='fresh'
else
  echo '1) Fresh run'
  echo '2) Continue an interrupted ten-client run'
  run_mode="$(prompt 'Run mode' '1')"
  case "$run_mode" in
    1) run_mode='fresh' ;;
    2) run_mode='continue' ;;
    *) echo 'Choose 1 or 2.' >&2; exit 2 ;;
  esac
fi

if [[ "$run_mode" == 'continue' ]]; then
  resume_tag="$(prompt 'Interrupted run tag' '')"
  [[ -n "$resume_tag" ]] || { echo 'A run tag is required to continue.' >&2; exit 2; }
  run_tag="$resume_tag"
  max_records_per_client='0'
else
  run_tag="local10_$(date -u +%Y%m%dT%H%M%SZ)"
fi
CLIENT_LOG_DIR="$PROJECT_ROOT/results/network-client-logs/$run_tag"
mkdir -p "$CLIENT_LOG_DIR"

if [[ "$run_mode" == 'fresh' ]]; then
  if [[ "$SMOKE" == true ]]; then
    choice='1'
  else
    echo
    echo "Experiment type"
    echo "1) IID FedAvg"
    echo "2) Non-IID FedAvg: Dirichlet label distribution"
    echo "3) Non-IID FedAvg: label shards"
    echo "4) Clipped FedAvg"
    echo "5) Vanilla DP extension"
    echo "6) SMPC-DP extension"
    echo "7) LightDP extension"
    echo "8) Clipped FedAvg + Vanilla DP"
    echo "9) Clipped FedAvg + SMPC-DP"
    echo "10) Clipped FedAvg + LightDP"
    choice="$(prompt 'Select an experiment' '1')"
  fi

  algorithm='fedavg'
  method='no_dp'
  partition='iid'
  epsilon='0.0'
  rounds='30'
  case "$choice" in
    1) ;;
    2) partition='dirichlet' ;;
    3) partition='label_shards' ;;
    4) algorithm='clipped_fedavg' ;;
    5) method='vanilla_dp'; epsilon='6'; rounds='8' ;;
    6) method='smpc_dp'; epsilon='6'; rounds='8' ;;
    7) method='lightdp'; epsilon='6'; rounds='8' ;;
    8) algorithm='clipped_fedavg'; method='vanilla_dp'; epsilon='6'; rounds='8' ;;
    9) algorithm='clipped_fedavg'; method='smpc_dp'; epsilon='6'; rounds='8' ;;
    10) algorithm='clipped_fedavg'; method='lightdp'; epsilon='6'; rounds='8' ;;
    *) echo 'Choose a number from 1 through 10.' >&2; exit 2 ;;
  esac

  if [[ "$partition" == 'dirichlet' ]]; then
    dirichlet_alpha="$(prompt 'Dirichlet alpha' '0.5')"
  else
    dirichlet_alpha='0.5'
  fi
  if [[ "$SMOKE" == true ]]; then
    rounds='1'
    device='cuda'
    max_records_per_client='8'
  else
    rounds="$(prompt 'Server rounds' "$rounds")"
    device="$(prompt 'Client device (cuda, cpu, or auto)' 'cuda')"
    max_records_per_client="$(prompt 'Max records per client (0 means full partition)' '0')"
  fi
  [[ "$device" == 'cuda' || "$device" == 'cpu' || "$device" == 'auto' ]] || {
    echo 'Device must be cuda, cpu, or auto.' >&2; exit 2;
  }
fi

require_free_ports
create_certificate "$coordinator_host"

nohup flower-superlink \
  --ssl-ca-certfile "$CERTS_DIR/ca.crt" \
  --ssl-certfile "$CERTS_DIR/server.pem" \
  --ssl-keyfile "$CERTS_DIR/server.key" \
  --fleet-api-address "${coordinator_host}:9092" \
  --serverappio-api-address 127.0.0.1:9091 \
  --control-api-address 127.0.0.1:9093 \
  > "$HOME/fedabba-superlink.log" 2>&1 &
superlink_pid=$!
echo "$superlink_pid" > "$HOME/fedabba-superlink.pid"

for attempt in $(seq 1 30); do
  if ss -ltn | grep -q ':9092'; then break; fi
  if ! kill -0 "$superlink_pid" 2>/dev/null; then
    cat "$HOME/fedabba-superlink.log" >&2
    exit 1
  fi
  sleep 1
done

open_local_clients
echo 'Ten local client terminals are opening. The coordinator waits for all ten SuperNodes.'

echo "Submitting run tag=$run_tag. Results and exported client logs stay on this coordinator."
if [[ "$run_mode" == 'continue' ]]; then
  python "$PROJECT_ROOT/scripts/run_network_experiment.py" \
    --ca-cert "$CERTS_DIR/ca.crt" --clients "$CLIENTS" --resume-tag "$resume_tag" \
    --client-log-dir "$CLIENT_LOG_DIR"
else
  python "$PROJECT_ROOT/scripts/run_network_experiment.py" \
    --ca-cert "$CERTS_DIR/ca.crt" --clients "$CLIENTS" --rounds "$rounds" \
    --algorithm "$algorithm" --method "$method" --epsilon "$epsilon" \
    --partition-method "$partition" --dirichlet-alpha "$dirichlet_alpha" \
    --client-data-mode "$client_data_mode" --device "$device" \
    --max-records-per-client "$max_records_per_client" --tag "$run_tag" \
    --client-log-dir "$CLIENT_LOG_DIR"
fi
