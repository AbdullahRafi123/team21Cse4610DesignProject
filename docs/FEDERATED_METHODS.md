# Federated training methods and research practice

The application separates Flower orchestration from the client update rule.
Choose the algorithm explicitly with `training-algorithm`; do not treat all
Flower runs as the same federated method.

The maintained baseline follows the core FedAvg procedure: clients perform
local optimization and send model parameters, and the server averages them
according to each client's number of examples. This is the established
FedAvg baseline introduced by
[McMahan et al. (AISTATS 2017)](https://proceedings.mlr.press/v54/mcmahan17a.html).
Flower provides the client/server orchestration and FedAvg strategy; Flower is
the framework, while FedAvg is the learning algorithm. The implementation uses
Flower `ClientApp`/`ServerApp` and its FedAvg strategy
([Flower FedAvg API](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html);
[Flower framework paper](https://arxiv.org/abs/2007.14390)).

These are standard baseline practices, not a claim that this project uses the
best or newest method for every federated-learning research question. The
hybrid CIFAR-10 model is project-specific, the privacy mechanisms remain
research extensions, and publication-quality conclusions require the protocol,
replication, and reporting steps in [Research Standards](RESEARCH_STANDARDS.md).

## Shared Flower architecture

The maintained path uses Flower `ClientApp`/`ServerApp` for every method. Each
client receives the current global parameters, trains on its own local
partition, and returns an update through Flower; the server coordinates rounds
and evaluation. Flower's local simulation can run multiple client workers at
once, subject to the configured CPU/GPU resources. A SuperLink/SuperNode
deployment runs distinct client processes on separate machines. A local
simulation is useful for repeatable experiments, but it does not itself model
network delay, disconnections, or separate physical machines.

The intended experiment sequence is plain `fedavg` as the baseline, followed
by `clipped_fedavg` and the privacy mechanisms as research variants. These
variants keep the Flower round and client/server structure. The prior
single-gradient client upload and server-momentum algorithm has been removed
from active code. Its archived outputs remain historical records.

The recommended comparison sequence is:

1. Establish plain FedAvg on an IID partition as the reference run.
2. Change one factor at a time: first compare a declared non-IID partition,
   then compare `clipped_fedavg`, then evaluate each privacy extension.
3. Keep the architecture, data split, rounds, client participation, optimizer,
   and compute profile fixed within each comparison unless that factor is the
   one being studied.
4. Define the protocol and independent seeds before confirmatory runs. Report
   all runs, uncertainty, failures, and deviations; treat one-off runs and
   smoke tests as engineering evidence only.

This sequence makes comparisons interpretable; it does not by itself validate
the privacy accounting or prove a method is superior.

## `fedavg`: standard weighted model averaging

For each selected client and round, the client loads the current global model,
performs the configured number of local SGD epochs, and returns its updated
trainable parameters plus its number of local training examples. Flower's
`FedAvg` strategy computes the example-count-weighted mean of client model
parameters. The local optimizer is recreated each round. Current controls are
`local-epochs`, `batch-size`, `local-learning-rate`, `local-momentum`, and
`fraction-fit`.

Plain FedAvg uses `method = "no_dp"`, example-count-weighted model averaging,
and no client-update clipping. It aggregates valid client updates if a
connected client fails during a round, and records the failure count with that
round. Privacy variants also use this local-SGD
training path, then clip each client's final model delta to
`client-update-clip` before applying noise or the simulated mask mechanism.
The current privacy strategy uniformly averages protected client updates.
That aggregation choice is part of the privacy research variant; it differs
from the example-count-weighted plain FedAvg baseline, especially with unequal
client dataset sizes. Results from those variants should not be described as
plain weighted FedAvg.

Example plain IID baseline:

```bash
python scripts/run_main_suite.py --algorithm fedavg --partition-method iid --rounds 30
```

To run the same FedAvg method with a label-skewed Dirichlet partition:

```bash
python scripts/run_main_suite.py --algorithm fedavg --partition-method dirichlet \
  --dirichlet-alpha 0.5 --rounds 30
```

## Privacy extensions on FedAvg client updates

Select `vanilla_dp`, `smpc_dp`, or `lightdp` with
`training-algorithm="fedavg"`. Each client trains locally for the configured
epochs, forms its model delta from the round's global model, clips the delta's
L2 norm, and applies its configured noise/mask mechanism. The server takes a
uniform client mean and adds it to the global model. `max-stragglers` bounds
missing contributors for the simulated dropout covariance model;
`simulate-stragglers=false` disables scheduled dropout for a connected
multi-machine run, while actual missing clients are still checked against the
bound. Privacy calibration requires `fraction-fit=1.0` and records the client
update clip and calibration in `run_config.json`.

The calibration code is adapted from the earlier clipped-gradient pipeline.
It has not received an independent formal derivation for local-SGD model
updates, so reported epsilon is a research calibration value, not a validated
formal guarantee. LightDP still uses deterministic pair masks without key
exchange; SMPC-DP remains idealized. Neither is production cryptography.

Example local-simulation comparison:

```bash
python scripts/run_main_suite.py --algorithm fedavg --epsilon 6 --rounds 8
```

## `clipped_fedavg`: clipped local-gradient extension

This variant keeps the FedAvg client/server round structure and
example-weighted aggregation, while clipping each per-example gradient at
every local SGD minibatch using `clip`. It is an algorithmic clipping
experiment by itself, not a differential privacy guarantee. Combine it with
`vanilla_dp`, `smpc_dp`, or `lightdp` to additionally clip and protect the final
client model update; that private form uses the uniform client-mean aggregator
and calibration limitations described above.

```bash
python scripts/run_main_suite.py --algorithm clipped_fedavg --epsilon 6 --rounds 8
```

LightDP simulates pairwise masks with matching seeds; it does not implement key
exchange. SMPC-DP remains an idealized aggregation baseline. Neither path is a
production cryptographic transport.

## Model and data protocol

All paths currently use the repository's same hybrid classifier: frozen
ImageNet-pretrained ResNet-18 features, a trainable image CNN branch, and a
classifier head. Local augmentation randomly chooses between original and
horizontally flipped inputs, with matching cached public features. The CNN
branch and classifier head are trained; the ResNet feature extractor is frozen.
This provides a controlled model within this implementation but is not a
canonical end-to-end CIFAR ResNet benchmark.

The default partition is IID. `label_shards` reproduces the existing sorted
label-shard style partition; `dirichlet` allocates each class across clients
using a seeded Dirichlet distribution and `dirichlet-alpha`. A deterministic,
class-stratified 10% holdout is drawn from CIFAR-10's training split before
partitioning clients. This validation split is evaluated each round, and the
highest-validation-accuracy checkpoint is selected. The CIFAR test split is
evaluated once at the final configured round on that selected checkpoint and
written to `final_metrics.json`.

Use the validation set for round-by-round selection and reserve the test set
for the final evaluation. Report validation and test results separately. For
confirmatory comparisons, use multiple predeclared seeds and summarize
variation across runs; a single seed does not establish a method ranking.

Changing the holdout, partition method, client optimizer, or algorithm changes
comparability with archived notebook outputs and prior Flower runs. Run a
fresh experiment under a clean revision; do not resume old checkpoints across
this protocol/schema change.

## Runtime tuning

Flower's simulation runtime schedules `ClientApp` workers according to their
declared CPU and GPU resources. The current GPU profile reserves one GPU per
worker, so a single-GPU machine schedules one GPU client worker at a time.
Fractional reservations can increase concurrency, but Flower treats these as
scheduling hints rather than VRAM limits. Benchmark a few rounds at several
resource settings, monitor GPU memory and utilization, and retain the chosen
settings in the protocol before running the full suite. For a one-GPU pilot,
compare these FedAvg runs:

```bash
python scripts/run_main_suite.py --algorithm fedavg --rounds 2 --gpu
python scripts/run_main_suite.py --algorithm fedavg --rounds 2 --gpu --gpu-fraction 0.5
python scripts/run_main_suite.py --algorithm fedavg --rounds 2 --gpu --gpu-fraction 0.25
```

Each command gets its own timestamped output. Compare the round times in
`training.log` and monitor `nvidia-smi`; lower reservations permit more
concurrent workers and can cause memory pressure. See Flower's
[simulation resource guide](https://flower.ai/docs/framework/how-to-run-simulations.html).

Do not use a faster simulation profile to claim algorithmic improvement. It
only changes execution resource allocation; compare model and privacy outcomes
under identical effective training configurations.

## Networked clients on separate machines

The same `ClientApp` and FedAvg strategy can run through Flower's deployment
engine. In that mode a coordinator runs the SuperLink and submits the
ServerApp, while independent machines run SuperNodes and execute their local
ClientApps. When `FEDABBA_CLIENT_DATA` points to a client-local NPZ file, the
client reads and trains on that file locally; the server receives the local
model parameters, example count, and configured training metrics for
aggregation. This is a networked multi-machine experiment, unlike the local
Flower simulation profiles.

The server-side validation and final test currently use public CIFAR-10 data.
This keeps evaluation centralized for comparable benchmark reporting. It does
not mean private client records are uploaded. Client-local datasets must use
the CIFAR-10 class IDs and image format documented in the
[deployment guide](USAGE.md#multi-machine-flower-deployment). The deployment
engine provides client/server orchestration, not privacy guarantees: enable
TLS and configure node authentication for the network, and use an appropriate
secure aggregation or differential privacy method if the experiment requires
those properties.

The repository's deployment profile is a starting configuration. It has not
been exercised against machines outside this development host, so a
multi-host run is not yet reported as completed evidence.

Use `scripts/run_network_experiment.py` on the coordinator after starting the
SuperLink and the selected number of SuperNodes. It exposes the same
`fedavg`, `clipped_fedavg`, `no_dp`, `vanilla_dp`, `smpc_dp`, and `lightdp`
settings as the application, and keeps each run on the Flower deployment
runtime. The network runner sets `simulate-stragglers=false`; connected nodes
therefore participate in real-time synchronous rounds while genuine disconnects
remain visible as client failures. See the deployment commands in
[USAGE.md](USAGE.md#run-every-maintained-method-through-the-same-connected-clients).
