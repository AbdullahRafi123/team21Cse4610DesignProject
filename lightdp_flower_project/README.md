# LightDP-FL in Flower — CIFAR-10

This project ports the supplied notebook **(Final)Another_LightDP_FL_CIFAR10_PaperComparison_Cached.ipynb** into a Flower simulation project.

## What is preserved from the notebook

- CIFAR-10 (50,000 train / 10,000 test)
- IID partition by default; optional 2-shard label-skew non-IID partition
- Frozen public pretrained ResNet-18 features at 224x224
- Trainable CNN image branch + classifier (28,362 trainable parameters)
- Full-client averaged **per-example clipped gradients**
- Four methods: `no_dp`, `vanilla_dp`, `smpc_dp` (ideal), `lightdp`
- zCDP calibration, epsilon/delta accounting, Cmax/Smax threat states
- Server momentum 0.9 and the notebook cosine LR schedule
- The notebook's deterministic straggler schedule
- LightDP explicit pairwise + / - Gaussian masks, generated from matching pair seeds
- Main epsilon 3/6/9 comparisons and the ten controlled sweep configurations
- Centralized CIFAR-10 test evaluation every round
- Exact linear no-DP pixel probe from the notebook

## Important scientific boundary

`lightdp` is a **protocol simulation**: matching pair seeds simulate pairwise secret-derived masks. It is not a production cryptographic key-exchange implementation.

`smpc_dp` is also the notebook's **ideal secure-aggregation baseline**. It reproduces the noise behavior, not real Paillier/SMPC transport.

The supplied notebook describes a later hybrid-model reconstruction attack, but its executable attack code is not present in the notebook cells. This package therefore does **not invent** that missing algorithm. It includes the exact linear pixel probe that is present in the source.

## 1. Pull and initialize

Use Python 3.10, 3.11, or 3.12. After cloning or pulling the repository, go to this project directory and run the initializer once. It creates a virtual environment and installs the project in editable mode; it does not start an experiment.

Linux/macOS:
```bash
cd lightdp_flower_project
bash INIT_LINUX_MAC.sh
```

Windows Command Prompt:
```bat
cd lightdp_flower_project
INIT_WINDOWS.bat
```

Activate the environment in each new terminal before working:

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```bat
.venv\Scripts\activate
```

On Linux/macOS, if the repository path contains spaces, the initializer places the environment at `~/.venvs/lightdp-flower-project` because Ray needs a space-free Python executable path. In that case activate it with `source "$HOME/.venvs/lightdp-flower-project/bin/activate"`.

The project also includes `RUN_LINUX_MAC.sh` and `RUN_WINDOWS.bat` for initializing and immediately starting the default smoke run.

You need internet access the first time so torchvision can download **CIFAR-10** and the **ImageNet pretrained ResNet-18 weights**. After they are cached, later runs can reuse them.

### Linux: project path contains spaces

Ray starts workers through a shell and can split the Python executable path at a space. A symlink to the existing `.venv` is not enough because its generated launch scripts still contain the original path. Create a separate environment under a path with no spaces, install the project there, then activate that environment before running Flower:

```bash
cd "$HOME/Documents/Rafi der Thesis/lightdp_flower_project"
mkdir -p "$HOME/.venvs"
python3 -m venv "$HOME/.venvs/lightdp-flower-clean"
source "$HOME/.venvs/lightdp-flower-clean/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$HOME/Documents/Rafi der Thesis/lightdp_flower_project"
python -m flwr.cli.app run . --stream
```

The separate environment makes both Flower's and Ray's worker launch paths space-free; it does not move the project.

If the run reports `Temporary failure in name resolution`, the machine cannot reach the download host. Restore internet/DNS access and retry. An interrupted download may leave an incomplete archive in `.cache/lightdp_flower/data`; torchvision will retry the download when connectivity is available.

## 2. Run a quick Flower smoke test

```bash
flwr run . --stream
```

The default smoke test uses 5 Flower SuperNodes, 2 rounds, and only 64 records/client. It checks that Flower, partitioning, clipping, LightDP masks, aggregation and evaluation all work.

## 3. Run the notebook-equivalent LightDP experiment

The uploaded result bundle shows the executed source notebook actually completed **8 main rounds** (not 50) and **3 sweep rounds** under its timing pilot.

```bash
flwr run . local-simulation --stream \
  --run-config configs/full_lightdp.toml \
  --federation-config "options.num-supernodes=50"
```

On Windows PowerShell, put this on one line.

## 4. Run all 10 main experiments

```bash
python scripts/run_main_suite.py
```

This runs:
- No DP
- Vanilla local DP at epsilon 3, 6, 9
- SMPC+DP ideal at epsilon 3, 6, 9
- LightDP at epsilon 3, 6, 9

Each uses 50 clients and 8 rounds, matching the executed source result plan.

## 5. Run the 10 LightDP sweeps

```bash
python scripts/run_sweeps.py
```

Sweeps: reference, epsilon 3, epsilon 9, N=25, N=100, Cmax=20, Smax=20, clip=0.5, LR=0.05, non-IID.

## 6. Collect plots/tables

```bash
python scripts/collect_results.py
```

Outputs are written to `results/summary/`.

## 7. Run the exact pixel probe

```bash
python -m lightdp_fl.pixel_probe
```

## GPU configuration

The default Flower federation reserves no GPU, so the project can start on CPU-only machines. To use a CUDA GPU, override the client resource reservation, for example:

```bash
flwr run . local-simulation --stream \
  --federation-config "options.backend.client-resources.num-gpus=0.2"
```

If you have a small GPU and get OOM errors, reserve more GPU per ClientApp:

```toml
options.backend.client-resources.num-gpus = 0.5
```

This limits concurrency to about two GPU clients at once. For a single client at a time, set it to `1.0`.

If you have no CUDA GPU, the code can run on CPU, but the full experiment will be very slow.

## Files you do NOT need to add

You do not need the old `.pt` checkpoints to train the Flower version. CIFAR-10 and ResNet weights are downloaded automatically.

## Files you WOULD need for an exact hybrid reconstruction port

The uploaded notebook contains the textual description and the result image/CSV for the hybrid reconstruction, but not the executable attack cell. If you have an older notebook/script containing that attack implementation, add it and it can be ported into `lightdp_fl/attacks/` without guessing.
