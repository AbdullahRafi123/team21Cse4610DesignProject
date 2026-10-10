# Flower FedAvg and privacy research extensions

This repository contains Flower applications for CIFAR-10 federated learning.
The maintained implementation is in `src/lightdp_fl/`; archived notebooks and
results are historical research records.

The training baseline now follows the standard federated-averaging workflow:
Flower coordinates client rounds, each client trains locally for configured
epochs, and the server aggregates client model weights weighted by local
example counts. The project also retains per-example clipped local training
and client-update privacy methods as separately named research extensions.
This is a research-aligned baseline and experiment framework, not a claim that
the model or privacy mechanisms are state of the art.

## Clone and run

On Linux or macOS:

```bash
git clone https://github.com/AbdullahRafi123/team21Cse4610DesignProject.git
cd team21Cse4610DesignProject
bash setup/initialize.sh
source "$HOME/.venvs/lightdp-flower-project/bin/activate"
flwr run . --stream
```

On Windows, clone the repository and run `launchers\run_windows.bat` from the
clone. The launcher installs the environment if needed and starts the smoke
run.

The default run is a small, two-round, five-client Flower simulation. The
initializer installs the project in a virtual environment outside the clone;
the first run may download CIFAR-10 and pretrained ResNet-18 weights. On Linux
run `./launchers/run_linux.sh`; on macOS run `./launchers/run_macos.command` to
initialize and launch directly.
For all setup options and machine deployment, see the [run guide](docs/USAGE.md).

Completed runs receive a unique UTC, machine-labelled tag and are written to
`results/experiments/<machine-id>/<tag>/`. The canonical run records are staged
with `git add` automatically after completion. This does not create a commit;
review the staged files and commit them when ready.

## Start here

- [Installation and run guide](docs/USAGE.md): clone, setup, smoke runs, GPU
  runs, experiment scripts, result locations, and resuming.
- [Current experiment analysis](docs/EXPERIMENT_ANALYSIS_2026-10-10.md): the
  completed October 2026 comparison, its limitations, and the recommended
  follow-up.
- [Research standards](docs/RESEARCH_STANDARDS.md): protocol, reproducibility,
  reporting, privacy, and release requirements.
- [Experiment protocol template](docs/EXPERIMENT_PROTOCOL_TEMPLATE.md): fill
  this out before confirmatory runs.
- [Federated method definitions](docs/FEDERATED_METHODS.md): standard FedAvg,
  retained clipped local-SGD and privacy extensions, partitions, evaluation,
  and runtime tuning.
- [Multi-machine deployment guide](docs/USAGE.md#multi-machine-flower-deployment):
  SuperLink/SuperNode setup and client-local training data.
- [Ten-client same-host Flower lab](docs/TEN_CLIENT_LAB.md): non-interactive
  ten-client networked runs, all maintained method variants, continuation, and
  diagnostic records.
- [Results index](results/experiments/README.md): active run artifacts and
  provenance expectations.

## Research scope

The active application uses CIFAR-10 and torchvision ResNet-18 features. The
`lightdp` method simulates pairwise masks using matching seeds; it is not
production cryptography or a key-exchange implementation. `smpc_dp` is an
idealized secure-aggregation baseline, not a cryptographic transport. Privacy,
security, and reconstruction claims must stay within the implementation and
documented threat model. The networked Flower runtime transports client
updates, but Flower deployment alone does not provide secure aggregation or
differential privacy.

Archived notebook outputs are not independent replications of the maintained
Flower application. A smoke run verifies that the software starts and writes
records; it does not establish research findings. The October 2026 comparison
is exploratory and has unresolved evaluation and provenance limitations; see
its analysis before interpreting the numbers.

## Citation and licensing

See [CITATION.cff](CITATION.cff) for software metadata. Cite CIFAR-10, the exact
torchvision ResNet-18 weight variant, Flower, ResNet, and other borrowed methods
or code when reporting work. Confirm dataset and pretrained-weight terms before
redistributing artifacts. The repository source is MIT licensed; third-party
assets retain their own terms.
