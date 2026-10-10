# Ten-client same-host Flower lab

This lab starts one Flower SuperLink coordinator and ten separately running
Flower SuperNodes on one computer. Every round follows the networked Flower
deployment path: the coordinator broadcasts the global parameters, each
SuperNode trains its assigned client partition locally, and the server
aggregates the returned model updates.

It demonstrates concurrent client processes and the project’s deployment
runtime. Since all processes share one host, GPU, filesystem, and loopback
network, it does not measure the performance, availability, or security of ten
physical devices.

## Start a run

Use the named-variant launcher. It replaces any previous local lab processes,
keeps all ten SuperNodes in the background, and streams the coordinator in the
current terminal:

```bash
cd "$HOME/Desktop/Fed Abba"
./launchers/run_10_client_variant.sh iid --rounds 30
```

The first run creates `$HOME/fedabba-certs/` with a local TLS CA and server
certificate. These files are local credentials and are excluded from Git.

## Copyable experiment commands

Run one command at a time. Each command creates a unique `local10_<UTC>` run
tag and writes records under `results/experiments/<machine-id>/<tag>/`.

```bash
# Standard FedAvg baselines
./launchers/run_10_client_variant.sh iid --rounds 30
./launchers/run_10_client_variant.sh dirichlet --rounds 30 --dirichlet-alpha 0.5
./launchers/run_10_client_variant.sh label_shards --rounds 30

# Per-example clipped local-SGD extension
./launchers/run_10_client_variant.sh clipped_fedavg --rounds 30

# Privacy-method extensions (epsilon 6; the launcher default is eight rounds)
./launchers/run_10_client_variant.sh vanilla_dp --rounds 8
./launchers/run_10_client_variant.sh smpc_dp --rounds 8
./launchers/run_10_client_variant.sh lightdp --rounds 8

# Clipped local-SGD with each privacy-method extension
./launchers/run_10_client_variant.sh clipped_vanilla_dp --rounds 8
./launchers/run_10_client_variant.sh clipped_smpc_dp --rounds 8
./launchers/run_10_client_variant.sh clipped_lightdp --rounds 8
```

## Interactive launcher answers

If you use `./launchers/start_10_client_lab.sh --replace --headless`, enter
**Fresh run** for each profile. Keep `device=cuda`,
`max-records-per-client=0`, and `round-timeout=300` unless the experiment
protocol records a different value.

| Menu choice | Method | Run mode | Dirichlet alpha | Server rounds | Device | Max records/client | Timeout (s) |
|---:|---|---:|---:|---:|---|---:|---:|
| 1 | IID FedAvg | 1 | — | 30 | cuda | 0 | 300 |
| 2 | Non-IID FedAvg, Dirichlet | 1 | 0.5 | 30 | cuda | 0 | 300 |
| 3 | Non-IID FedAvg, label shards | 1 | — | 30 | cuda | 0 | 300 |
| 4 | Clipped FedAvg | 1 | — | 30 | cuda | 0 | 300 |
| 5 | Vanilla DP extension | 1 | — | 8 | cuda | 0 | 300 |
| 6 | SMPC-DP extension | 1 | — | 8 | cuda | 0 | 300 |
| 7 | LightDP extension | 1 | — | 8 | cuda | 0 | 300 |
| 8 | Clipped FedAvg + Vanilla DP | 1 | — | 8 | cuda | 0 | 300 |
| 9 | Clipped FedAvg + SMPC-DP | 1 | — | 8 | cuda | 0 | 300 |
| 10 | Clipped FedAvg + LightDP | 1 | — | 8 | cuda | 0 | 300 |

For example, choice 2 uses these answers:

```text
Run mode [1]: 1
Select an experiment [1]: 2
Dirichlet alpha [0.5]: 0.5
Server rounds [30]: 30
Client device [cuda, cpu, or auto] [cuda]: cuda
Max records per client (0 means full partition) [0]: 0
Round timeout seconds [300]: 300
```

The commands use CUDA, full deterministic CIFAR-10 client partitions, and a
300-second synchronous round timeout by default. Supply explicit overrides
when a protocol requires them:

```bash
./launchers/run_10_client_variant.sh dirichlet \
  --rounds 30 --dirichlet-alpha 0.1 --device cuda \
  --max-records-per-client 0 --round-timeout 600
```

`max-records-per-client=0` uses the entire assigned partition. Reducing it is
useful for a smoke check but changes the training protocol and must be
recorded when results are compared.

## Interactive and smoke modes

The interactive launcher remains available when selecting a run from a menu:

```bash
./launchers/start_10_client_lab.sh --replace --headless
```

For a one-round deployment check with eight records per client:

```bash
./launchers/start_10_client_lab.sh --replace --smoke --headless
```

A smoke run verifies startup, client connection, aggregation, and artifact
creation. It is not a research result.

## Continue an interrupted run

Use the exact run tag only when the checkpoint is below its configured round
count:

```bash
./launchers/start_10_client_lab.sh --replace --headless \
  --resume-tag local10_YYYYMMDDTHHMMSSZ
```

The launcher checks this before it starts the coordinator or client processes.
A completed run is immutable and must not be resumed; start a new run instead.

## Records and diagnosis

The coordinator stores `run_config.json`, `history.csv`, `train_rounds.csv`,
`training.log`, final metrics, and `network_run_export.json` in the run
directory. The latter includes the Git revision and working-tree status.
SuperNode logs are copied to `client_logs/` when a run completes and are kept
out of Git. Use those logs together with `training.log` when investigating a
failed client or timed-out round. Plain FedAvg records client failures and
aggregates the valid example-weighted updates; record failure counts when
comparing experiments. Privacy extensions retain their configured missing-client
bound and stop when that bound is exceeded.

See [federated method definitions](FEDERATED_METHODS.md) for the difference
between standard example-weighted FedAvg and the clipped/privacy extensions.
