# LightDP-FL CIFAR-10 experiments

This repository contains a Flower simulation of the LightDP federated learning comparison, its experiment notebooks, and archived results. The active implementation is the `src/lightdp_fl/` package.

## Set up after cloning

Requires Python 3.10–3.13 and internet access for the first CIFAR-10 and pretrained ResNet downloads. The initializers prefer Python 3.12 when available and accept a `PYTHON` override to select another supported interpreter.

```bash
git clone https://github.com/AbdullahRafi123/team21Cse4610DesignProject.git
cd team21Cse4610DesignProject
```

Run the platform initializer from the repository root. On macOS and Linux:

```bash
bash setup/initialize.sh
```

On Windows, install Python 3.10–3.13 and Git, then run:

```powershell
.\setup\initialize.bat
```

Each initializer creates a virtual environment if one does not already exist, installs the project and dependencies, and prints the activation command. They do not start training. Windows uses `%USERPROFILE%\.venvs\lightdp-flower-project` by default; macOS/Linux use `~/.venvs/lightdp-flower-project`. Keeping environments outside the checkout avoids paths that can interfere with Ray workers. Activate the environment, then run the quick smoke experiment from the repository root:

```text
flwr run . --stream
```

The first run may download CIFAR-10 and pretrained weights.

To choose a particular supported Python interpreter, set `PYTHON` before running the initializer, for example `PYTHON=python3.11 bash setup/initialize.sh` on macOS/Linux or `set "PYTHON=py -3.13"` before running `setup\initialize.bat` on Windows. Supported versions are 3.10–3.13; an existing environment with a different version is left untouched and the initializer explains how to select a new `VENV_DIR`.

### Click to run

The launchers initialize the virtual environment if needed, then start the default Flower smoke run and keep the terminal open to show its output:

- Windows: double-click [`launchers/run_windows.bat`](launchers/run_windows.bat).
- macOS: double-click [`launchers/run_macos.command`](launchers/run_macos.command). If macOS blocks it, open Terminal in the repository and run `bash launchers/run_unix.sh`.
- Linux: double-click [`launchers/run_linux.sh`](launchers/run_linux.sh) when the file manager is set to run executable scripts, or run `bash launchers/run_unix.sh` in a terminal.

## Experiments

The default Flower configuration is a small smoke run. The supplied configurations and scripts run the larger comparisons:

```bash
flwr run . local-simulation --stream --run-config configs/full_lightdp.toml \
  --federation-config "options.num-supernodes=50"
python scripts/run_main_suite.py
python scripts/run_sweeps.py
python scripts/collect_results.py
```

### Resume an interrupted run

Every completed communication round saves the global parameters and optimizer momentum. Continue
an interrupted run by tag; the helper reads its recorded configuration and resumes from the latest
checkpoint:

```bash
python scripts/resume_run.py --tag my_run
```

Keep the experiment settings and total `num-server-rounds` unchanged. A checkpoint already at the
target round cannot continue training. Runs are machine-local: the resume helper and server reject a
checkpoint whose machine ID differs from the current machine. Start a fresh run on each machine;
do not copy checkpoints between machines. Data and pretrained weights can be downloaded again or
copied from the local cache.

### Benchmark optional training modes

The default training path remains eager FP32. To compare it with available autocast and
`torch.compile` paths on the current machine, run:

```bash
python scripts/benchmark_training_modes.py
```

This uses synthetic inputs and writes timing and relative-error results under
`results/benchmarks/`. It is a microbenchmark, not evidence that an option improves full-run
utility or privacy behavior. Record and compare end-to-end experiment outcomes before using a
different mode in a research run.

Install the optional test dependency and run the focused clipping reference check with:

```bash
pip install -e '.[test]'
python -m pytest
```

Suite and sweep scripts add a UTC timestamp to run names. Outputs are organized under `results/experiments/<machine-id>/<tag>/`:

- `run_config.json` records parameters, a UTC run ID, machine ID, machine-local run number, hostname, OS/architecture, and available CUDA devices.
- `history.csv` records loss, accuracy, and privacy accounting per round.
- `train_rounds.csv` records active clients, simulated stragglers, learning rate, and client timing.
- `training.log` records run start and round summaries.
- `checkpoints/server_checkpoint.pt` stores that machine's latest resumable global parameters and momentum.
- `checkpoints/final_model.pt` stores its final model state; `.pt` files are excluded from Git.

Each new run gets an ID such as `20261009T120000Z_host-linux-x86_64-a1b2c3d4_run0007_9f8e7d6c`. Outputs are partitioned under `results/experiments/<machine-id>/<tag>/`, so equal tags on different machines stay separate when results are collected together. A stable machine ID and its sequence are stored in the user's cache directory (`~/.cache/lightdp_flower/` on Linux/macOS or `%LOCALAPPDATA%\lightdp_flower\` on Windows), so they persist across repository pulls without adding machine-specific state to Git. Resume events record the machine and sequence used to continue a run.

At startup, the server and each client process check `torch.cuda.is_available()`, list visible GPUs, and log the selected device. With `device="auto"` the process uses CUDA when available and otherwise logs that it selected CPU. An explicitly requested CUDA device stops early with a clear error if CUDA is unavailable or the requested GPU index is not visible.

Training logs and small tabular summaries are intentionally Git-trackable for reproducibility. Review and commit the relevant `run_config.json`, `history.csv`, `train_rounds.csv`, and `training.log` files after an experiment. `results/summary/all_runs.csv` includes run ID, machine ID, hostname, and run number so collected results can be compared across machines. Use a distinct `tag` for separate experiments on one machine; machine IDs partition outputs across machines. Local datasets, caches, machine IDs, and run-number counters are excluded from Git.

To pre-download data and model weights without training:

```bash
python -m lightdp_fl.prepare_data
```

## Project layout

- `src/lightdp_fl/`: client/server applications, model, data, privacy, training, and aggregation modules.
- `setup/`: platform initializers and the Windows redundancy-check wrapper.
- `launchers/`: platform-specific clickable smoke-run entry points plus their shared Unix launcher.
- `configs/`: full experiment configurations.
- `scripts/`: experiment runners and result collection.
- `notebooks/`: methodology and source notebooks; older notebooks are in `notebooks/archive/`.
- `results/archive/`: retained historical tables, logs, and figures grouped by experiment.
- `results/experiments/<machine-id>/<tag>/`: machine-partitioned Flower runs and their trackable logs/checkpoint directories.

## Check for repository redundancies

Run `setup\check_redundancies.bat` on Windows or `python scripts/check_redundancies.py` on macOS/Linux. The dependency-free checker reports byte-identical files, repeated top-level Python declarations, and broken local Markdown links. It never removes or edits files; review each finding before changing the repository. Add `--strict` to return a failure exit code when findings exist, or `--include-outputs` to include generated experiment and summary directories.

The `lightdp` method simulates pairwise masks using matching seeds; it is not production key exchange. `smpc_dp` is an ideal secure-aggregation baseline, not a real cryptographic transport implementation. The notebooks document earlier experiments and may not use the active Flower code.

## Research use and citation

Archived measurements are historical notebook outputs. The current smoke configuration checks execution and logging only; it does not reproduce the archived paper-style results. Follow the [research standards](docs/RESEARCH_STANDARDS.md) and record a protocol from the [experiment template](docs/EXPERIMENT_PROTOCOL_TEMPLATE.md) before presenting new confirmatory findings. Repository agents should follow [AGENTS.md](AGENTS.md).

The project uses CIFAR-10 and torchvision's pretrained ResNet-18 weights. Cite the [CIFAR-10 dataset and technical report](https://www.cs.toronto.edu/~kriz/cifar.html) and record the exact [ResNet-18 weight variant](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) used. Software citation metadata is in [CITATION.cff](CITATION.cff); the full author list and contribution order must be confirmed by the team before a paper release.

## License

The source code is released under the MIT License in [LICENSE](LICENSE). Dataset and pretrained-weight terms remain those of their respective providers.
