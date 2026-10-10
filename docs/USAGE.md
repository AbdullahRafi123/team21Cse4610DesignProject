# Installation and experiment usage

## Requirements

Python 3.10–3.13 and internet access are needed for initial CIFAR-10 and
pretrained-weight downloads. Initializers prefer Python 3.12 when available and
accept `PYTHON` to select another supported interpreter. They create an
environment outside the checkout by default and install dependencies; they do
not start training.

Clone the repository, enter its root, then run the platform initializer:

```bash
git clone https://github.com/AbdullahRafi123/team21Cse4610DesignProject.git
cd team21Cse4610DesignProject
bash setup/initialize.sh
source "$HOME/.venvs/lightdp-flower-project/bin/activate"
flwr run . --stream
```

On Windows, run `setup\initialize.bat`, then activate the environment using
the command it prints. To select a Python interpreter, for example:

```bash
PYTHON=python3.11 bash setup/initialize.sh
```

For a clean GPU environment on Linux, give the initializer a separate venv
path, then install the CUDA 12.8 PyTorch wheels. The example pins PyTorch 2.8
and torchvision 0.23, which are compatible with the project's minimum
versions and the RTX 30-series CUDA 12.8 runtime. Choose another supported
wheel pair with the official [PyTorch install selector](https://pytorch.org/get-started/locally/)
if your hardware or driver requires it.

```bash
VENV_DIR="$HOME/.venvs/lightdp-flower-gpu" bash setup/initialize.sh
source "$HOME/.venvs/lightdp-flower-gpu/bin/activate"
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU")'
```

Run the five-client CUDA smoke profile explicitly:

```bash
flwr run . local-simulation-gpu-smoke --stream
```

This profile reserves one GPU per client worker, so Flower schedules the five
clients one at a time on a single GPU. The 50-client GPU profile reserves a
whole GPU per worker too, which also serializes workers on a one-GPU host. The
half and quarter GPU profiles are throughput pilots: Flower's reservations
schedule workers but do not isolate GPU memory, so monitor VRAM before using
them for larger runs. A CUDA-enabled server alone does not make client training
use CUDA; the federation profile must request GPU resources and each client
must report `selected_device=cuda` in `training.log`.

## Smoke run

The clone commands above launch the default smoke run. To run it later from
the repository root with the environment active:

```bash
flwr run . --stream
```

The default Flower configuration is an intentionally small two-round,
five-client FedAvg smoke run on Flower's single-machine simulation engine. The
first execution may download CIFAR-10 and pretrained weights. It checks
software operation; it is not a training benchmark or confirmatory result.

## Four-machine Flower deployment

This deployment uses five machines and five terminals: one coordinator runs
the SuperLink and submits each run; four separate client machines each run one
SuperNode and train on data that stays on that machine. This uses Flower's
networked Deployment Runtime, not the single-machine Simulation Runtime.
Commands below target the project's Flower 1.25 environment.

Before starting, choose the coordinator's LAN DNS name or static IP (shown
below as `COORDINATOR_HOST`), make sure all five machines can reach it, and
prepare a TLS certificate whose Subject Alternative Name includes that DNS
name/IP. There are no certificates in the repository yet. For a lab network,
use a lab-approved CA or create a private CA and server certificate; protect
the private CA key on the coordinator. The coordinator keeps the server
certificate and private key; copy only the CA certificate to the client
machines. Install the same project
revision and compatible dependencies on all five machines. On GPU clients,
use the separate CUDA environment described above. With `device="auto"`, the
ClientApp should then select that client's GPU; verify `selected_device=cuda`
in its logs after the first run. The coordinator needs access to the
public CIFAR-10 validation/test data and pretrained model weights for server
evaluation and model initialization.

For a small, trusted research LAN, these OpenSSL commands create a private CA
and one server certificate. Replace `192.168.1.10` with the coordinator IP
and add its DNS name if clients connect by DNS. The certificate includes
`127.0.0.1` because the coordinator's Flower CLI connects to the local Control
API using that address. Keep `ca.key` private and do not commit these files.

```bash
mkdir -p "$HOME/fedabba-certs"
cd "$HOME/fedabba-certs"
openssl req -x509 -newkey rsa:4096 -nodes -days 3650 \
  -keyout ca.key -out ca.crt -subj "/CN=FedAbba Lab CA"
openssl req -new -newkey rsa:2048 -nodes \
  -keyout server.key -out server.csr -subj "/CN=192.168.1.10"
cat > server.ext <<'EOF'
subjectAltName=IP:192.168.1.10,IP:127.0.0.1
extendedKeyUsage=serverAuth
EOF
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.pem -days 825 -sha256 -extfile server.ext
chmod 600 ca.key server.key
```

Copy `ca.crt` to each client over a trusted channel; do not copy `ca.key` to
clients. Use organization-issued certificates instead when available.

Each client needs a local `.npz` with `x_train` and `y_train` arrays (or
`images` and `labels`). Images must be NHWC RGB, integer values in `[0,255]`
or floating values in `[0,1]`; labels are CIFAR-10 IDs from 0 to 9. For
example, create the file on that client machine with:

```python
import numpy as np
np.savez_compressed("client.npz", images=images, labels=labels)
```

Do not copy these client files to the coordinator. The server uses the public
CIFAR-10 training holdout for validation and the CIFAR-10 test set for final
evaluation; those are centralized public evaluation sets, not client-private
records. This data loader is specific to this CIFAR-10 classifier.

Allow inbound TCP `9092` on the coordinator from the four clients (Flower's
Fleet API). Keep TCP `9093` (the Control API used by `flwr run`) restricted to
the coordinator itself. If using a firewall, allow outbound client traffic to
coordinator port `9092`. Do not expose the unencrypted `network-demo` profile
to the network.

This TLS configuration encrypts connections and verifies the SuperLink
certificate. It does not configure Flower SuperNode authentication, so use it
for a trusted lab network; add Flower's node authentication before accepting
nodes from an untrusted network.

### Terminal 1 — coordinator: start SuperLink

From the project root with the project's venv active, start the coordinator
process. Use the actual certificate and key paths:

```bash
source "$HOME/.venvs/lightdp-flower-gpu/bin/activate"
nohup flower-superlink \
  --ssl-ca-certfile "$HOME/fedabba-certs/ca.crt" \
  --ssl-certfile "$HOME/fedabba-certs/server.pem" \
  --ssl-keyfile "$HOME/fedabba-certs/server.key" \
  --fleet-api-address 0.0.0.0:9092 \
  --serverappio-api-address 127.0.0.1:9091 \
  --control-api-address 127.0.0.1:9093 \
  > "$HOME/superlink.log" 2>&1 &
echo $! > "$HOME/superlink.pid"
tail -n 20 "$HOME/superlink.log"
```

Keep this terminal open. SuperLink runs in the background so this same
coordinator terminal can submit the run later. It serves clients on Fleet API
port `9092` and the local Flower CLI on Control API port `9093`. Stop it after
the experiment with `kill "$(cat "$HOME/superlink.pid")"`.

### Terminals 2–5 — one client on each machine

On each of the four client machines, activate its project venv, set the path to
that machine's own dataset, and start its SuperNode. Replace
`COORDINATOR_HOST` with the coordinator's reachable DNS name/IP, and set the
CA certificate path available on that client. Give each machine its own ID:

```bash
source "$HOME/.venvs/lightdp-flower-gpu/bin/activate"
export FEDABBA_CLIENT_DATA=/data/client-0.npz
flower-supernode \
  --root-certificates /path/to/ca.crt \
  --superlink COORDINATOR_HOST:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config="partition-id=0 num-partitions=4"
```

Use this mapping for the four client terminals:

| Client machine | Local data file | `partition-id` | `num-partitions` |
|---|---|---:|---:|
| Client 1 | `/data/client-0.npz` | 0 | 4 |
| Client 2 | `/data/client-1.npz` | 1 | 4 |
| Client 3 | `/data/client-2.npz` | 2 | 4 |
| Client 4 | `/data/client-3.npz` | 3 | 4 |

Keep all four SuperNode terminals running. The app code and run configuration
are delivered by Flower; the local data paths and files remain on their client
machines. When all four SuperNodes are connected, submit the run from the
coordinator.

### Terminal 1 — coordinator: submit a four-client FedAvg run

In the coordinator checkout, `[tool.flwr.federations.network-tls]` in
`pyproject.toml` should point to the local Control API (`127.0.0.1:9093`) and
the trusted CA certificate path. Ensure the SuperLink certificate SAN also
matches the coordinator hostname/IP used by client SuperNodes. For example:

```toml
[tool.flwr.federations.network-tls]
address = "127.0.0.1:9093"
root-certificates = "/home/USER/fedabba-certs/ca.crt"
```

Use the actual coordinator account's home path. Also include `127.0.0.1` in
the certificate SAN for this local CLI connection. With the same venv active,
submit a short two-round end-to-end check from the coordinator terminal:

```bash
flwr run . network-tls --stream \
  --run-config 'num-clients=4 num-server-rounds=2 training-algorithm="fedavg" method="no_dp" epsilon=0.0 client-data-mode="local_npz" max-records-per-client=0 simulate-stragglers=false'
```

When this succeeds, increase `num-server-rounds` for the research run. Keep
`num-clients=4` matched to `num-partitions=4` on every SuperNode and use a
unique client `partition-id` from 0 through 3. Results and checkpoints are
written on the coordinator; client datasets remain local.

For privacy experiments, keep the same networked architecture and change the
method/configuration in a separate run. For example, LightDP uses
`method="lightdp" epsilon=6 client-update-clip=1.0 max-colluders=1
max-stragglers=1`; its calibration limitations are described in
[method details](FEDERATED_METHODS.md#privacy-extensions-on-fedavg-client-updates).
The privacy methods are research mechanisms, not production cryptography.

### Run every maintained method through the same connected clients

After the SuperLink and four SuperNodes are connected, use the coordinator
runner below. It submits one ServerApp through the `network-tls` federation;
the connected SuperNodes receive each round in real time, train locally, and
return their updates. It works unchanged for the same-PC five-process setup
and for four physical client machines. Each invocation creates a separate,
machine-labelled run directory when `--tag` is left at its default `auto`.

```bash
source "$HOME/.venvs/lightdp-flower-gpu/bin/activate"

# Standard IID FedAvg baseline.
python scripts/run_network_experiment.py \
  --ca-cert "$HOME/fedabba-certs/ca.crt" \
  --algorithm fedavg --method no_dp --partition-method iid --rounds 30

# Same algorithm and training budget with a reproducible non-IID partition.
python scripts/run_network_experiment.py \
  --ca-cert "$HOME/fedabba-certs/ca.crt" \
  --algorithm fedavg --method no_dp --partition-method dirichlet \
  --dirichlet-alpha 0.5 --rounds 30

# Per-example clipped local-SGD FedAvg extension.
python scripts/run_network_experiment.py \
  --ca-cert "$HOME/fedabba-certs/ca.crt" \
  --algorithm clipped_fedavg --method no_dp --partition-method iid --rounds 30

# Privacy research extensions on the same Flower client/server path.
python scripts/run_network_experiment.py \
  --ca-cert "$HOME/fedabba-certs/ca.crt" \
  --algorithm fedavg --method vanilla_dp --epsilon 6 --rounds 8

python scripts/run_network_experiment.py \
  --ca-cert "$HOME/fedabba-certs/ca.crt" \
  --algorithm fedavg --method smpc_dp --epsilon 6 --rounds 8

python scripts/run_network_experiment.py \
  --ca-cert "$HOME/fedabba-certs/ca.crt" \
  --algorithm fedavg --method lightdp --epsilon 6 --rounds 8
```

For separate physical machines, use `--client-data-mode local_npz` and set
`FEDABBA_CLIENT_DATA` to a distinct client-local dataset before starting each
SuperNode. In that mode, the coordinator never reads those NPZ files. The
benchmark mode, `partitioned_cifar10`, is useful when every connected client
uses a deterministic CIFAR-10 partition. The deployment remains synchronous:
each new global model is issued only after the required clients return for the
current round. It is a real networked FL round protocol, rather than an
asynchronous production device fleet.

Clickable launchers are available under `launchers/` for Windows, macOS, and
Linux. They initialize the environment if needed and launch the default smoke
configuration. They also add the environment's executable directory to
`PATH`, which lets Flower find its simulation worker executable.

### One-click ten-client deployment launcher

The maintained same-host deployment guide and its copyable commands are in
[Ten-client same-host Flower lab](TEN_CLIENT_LAB.md). It provides a
non-interactive named-variant launcher, all ten maintained algorithms, smoke
mode, and safe continuation instructions.

`launchers/start_10_client_lab.sh` remains the interactive launcher. It
creates or validates the coordinator TLS certificate, starts SuperLink, starts
ten SuperNodes, and submits the chosen networked experiment. Clients receive
IDs 0 through 9 and use unique AppIO ports 9094 through 9103. Add `--headless`
to keep the SuperNodes in the background rather than opening ten terminal
windows.

The launcher gives synchronous rounds a 300-second timeout. It records actual
client failures in `train_rounds.csv`; plain FedAvg aggregates the valid,
example-weighted client updates when at least one is returned. Relaunch with
`--replace`, choose **Continue**, and enter that run tag to reconnect all ten
local clients and resume an interrupted run. Flower 1.25's Fleet API supports
one request worker, while the ten SuperNodes still train in separate processes.

Use `--smoke --headless` to verify the full ten-client path without opening
terminals. It runs one IID FedAvg round with eight records per client, stores
the normal run records, preserves ten per-client SuperNode logs, and writes
`network_run_export.json` containing the Git revision, branch, and working-tree
status. Fresh and completed runs automatically stage canonical metrics and
provenance with Git; they never create a Git commit.

The menu covers IID FedAvg, both non-IID partitions, clipped FedAvg, each
privacy extension, and each privacy extension with clipped FedAvg. All menu
choices retain Flower's synchronous client/update/aggregation protocol. Ten
CUDA processes on one host share VRAM, so choose `cpu` or reduce the client
count through the scripts when GPU memory is insufficient. This launcher uses
the deterministic CIFAR-10 partitions on the same host, so it validates the
networked Flower process topology but does not emulate separate hardware or
network links.

## Experiment suites

The maintained runners are:

```bash
# Plain FedAvg and all client-update privacy variants (script default: eight rounds)
python scripts/run_main_suite.py --algorithm fedavg --rounds 30

# Per-example clipped local-SGD FedAvg with its privacy variants
python scripts/run_main_suite.py --algorithm clipped_fedavg --epsilon 6 --rounds 8

# Full FedAvg privacy-method comparison and update-parameter sweep
python scripts/run_main_suite.py --algorithm fedavg --rounds 8
python scripts/run_sweeps.py
python scripts/collect_results.py
```

Add `--gpu` to either runner for the `local-simulation-gpu` profile. It
requires CUDA-enabled PyTorch and a visible NVIDIA GPU. Each suite run writes
under `results/experiments/<machine-id>/<tag>/`; suite tags include a UTC
timestamp. The FedAvg suite runs plain FedAvg and the three client-update
privacy variants at epsilon 3, 6, and 9 unless `--epsilon` selects one value.
`clipped_fedavg` adds per-example clipping during local SGD. The former
single-gradient/server-momentum algorithm is no longer an active run option;
historical outputs remain under their original labels.
Select a client
partition with `--partition-method iid`, `--partition-method label_shards`,
or `--partition-method dirichlet`; the latter accepts `--dirichlet-alpha`.
Private FedAvg clips final client model deltas and uniformly averages them;
plain FedAvg uses example-count weighting. Their aggregation rules differ and
must be reported separately.
For a one-GPU throughput pilot, `run_main_suite.py --gpu-fraction 0.5` or
`--gpu-fraction 0.25` selects a profile that schedules multiple client workers
per GPU. These are soft scheduling reservations; watch GPU memory and compare
the result against `--gpu-fraction 1.0` before using a profile in a full run.

The single-configuration Flower profile can be run directly:

```bash
flwr run . local-simulation-suite --stream \
  --run-config configs/full_lightdp.toml
```

For the GPU profile, replace `local-simulation-suite` with
`local-simulation-gpu`. The GPU profile reserves one GPU per worker; this is a
scheduling hint, not a VRAM limit. Do not compare its timings with other
concurrency settings without recording those settings. Linux is recommended
for GPU simulations; native Windows support through Ray is experimental.

## Resume an interrupted run

Resume by the existing tag, using the same total round count and settings:

```bash
python scripts/resume_run.py --tag my_run
```

Add `--gpu` for a GPU run. Checkpoints are machine-local; the runner rejects a
checkpoint from another machine. Do not copy a checkpoint across machines.
An already-completed target round cannot be continued.

## Run outputs

Each run directory retains:

- `run_config.json`: effective parameters, run and machine IDs, software and
  platform details, dataset/model provenance, and recorded Git revision.
- `history.csv`: per-round validation metrics and the configured epsilon
  estimate when a privacy method is selected. FedAvg privacy calibration is
  documented in `run_config.json` and is not yet a validated formal guarantee.
- `final_metrics.json`: one final test evaluation after training completes on
  the checkpoint selected by validation accuracy; round-by-round metrics are
  validation-only.
- `train_rounds.csv`: aggregation rounds, active/missing clients, example
  counts, aggregation rule, private update clipping statistics, local training
  metadata, timing, and failures.
- `training.log`: setup and round progress records.
- `checkpoints/`: machine-local checkpoints, excluded from Git.
- `git_tracking` in `run_config.json`: whether the canonical records were
  staged automatically.

Client progress JSON files are transient diagnostic state. The collection
script writes combined CSVs and plots under `results/summary/`. On successful
completion, the app stages only `run_config.json`, `history.csv`,
`train_rounds.csv`, `training.log`, and `final_metrics.json` from that run.
It never commits. Use `git status` to review those staged files before making a
commit. Set `git-track-results=false` in run configuration to disable automatic
staging. Runs outside a Git checkout are recorded locally and marked
`unavailable` in `run_config.json`. Do not hand-edit generated metrics. Keep
datasets, credentials, environments, private data, client progress, and
generated checkpoints out of Git.

The maintained model is a hybrid CIFAR-10 classifier: frozen pretrained
ResNet-18 features feed a trainable CNN branch and classifier head. FedAvg
performs local SGD on the trainable parameters. It is a standard FedAvg
aggregation baseline, while the architecture remains project-specific; it is
not end-to-end ResNet fine-tuning. See
[federated method definitions](FEDERATED_METHODS.md) before comparing the two
training algorithms.

To download/prepare data and weights without training:

```bash
python -m lightdp_fl.prepare_data
```

The optional test dependency can be installed with
`pip install -e '.[test]'`, after which repository tests run with
`python -m pytest`.

## Layout

- `src/lightdp_fl/`: maintained application and reusable training behavior.
- `configs/`: Flower configuration files.
- `scripts/`: thin experiment, collection, and maintenance entry points.
- `notebooks/`: methodology and source notebooks; older notebooks are under
  `notebooks/archive/`.
- `results/archive/`: historical records; do not rewrite their metrics.
- `results/experiments/`: machine-partitioned active run records.
- `results/summary/`: collected summaries and plots.

Use `python scripts/check_redundancies.py` (or
`setup\check_redundancies.bat` on Windows) to inspect duplicate files,
repeated top-level declarations, and broken local Markdown links. The checker
does not modify files.
