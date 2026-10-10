#!/usr/bin/env bash
# Run one named same-host ten-client Flower experiment without prompts.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: launchers/run_10_client_variant.sh EXPERIMENT [launcher options]

EXPERIMENT: iid, dirichlet, label_shards, clipped_fedavg, vanilla_dp,
            smpc_dp, lightdp, clipped_vanilla_dp, clipped_smpc_dp, or
            clipped_lightdp.

Examples:
  ./launchers/run_10_client_variant.sh iid --rounds 30
  ./launchers/run_10_client_variant.sh dirichlet --rounds 30 --dirichlet-alpha 0.5
  ./launchers/run_10_client_variant.sh clipped_lightdp --rounds 8

Each invocation replaces an existing local ten-client deployment, starts ten
background SuperNodes, and streams coordinator progress in this terminal.
EOF
}

if [[ $# -eq 0 || "$1" == '--help' || "$1" == '-h' ]]; then
  usage
  exit 0
fi

experiment="$1"
shift
case "$experiment" in
  iid|dirichlet|label_shards|clipped_fedavg|vanilla_dp|smpc_dp|lightdp|clipped_vanilla_dp|clipped_smpc_dp|clipped_lightdp) ;;
  *)
    echo "Unknown experiment: $experiment" >&2
    usage >&2
    exit 2
    ;;
esac

exec "$PROJECT_ROOT/launchers/start_10_client_lab.sh" \
  --replace --headless --fresh --experiment "$experiment" "$@"
