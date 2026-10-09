from __future__ import annotations
import csv
import json
import math
import subprocess
import sys
import time
from importlib import metadata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .config import RunConfig
from .data import load_test_data, resolve_device
from .model import HybridClassifier, get_parameters, set_parameters
from .privacy import calibrate, eps_from_rho
from .progress import start_progress_monitor
from .run_tracking import machine_identity, new_run_identity
from .strategy import GradientMomentumStrategy
from .training import evaluate_global


def _records_per_client(cfg: RunConfig) -> int:
    full = 50000 // cfg.num_clients
    return min(full, cfg.max_records_per_client) if cfg.max_records_per_client > 0 else full


def _source_provenance() -> dict[str, str | bool | None]:
    """Return Git revision and dirty-state when the run comes from a checkout."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_revision": revision, "git_worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_worktree_dirty": None}


def _dependency_versions() -> dict[str, str]:
    distributions = (
        "flwr",
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-image",
    )
    return {
        package: metadata.version(package)
        for package in distributions
    }


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def server_fn(context: Context) -> ServerAppComponents:
    cfg = RunConfig.from_mapping(context.run_config)
    cfg.validate()
    device = resolve_device(cfg.device)
    cfg.run_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.checkpoint_dir / "server_checkpoint.pt"
    legacy_checkpoint_path = cfg.run_output_dir / "server_checkpoint.pt"
    if cfg.resume and not (checkpoint_path.is_file() or legacy_checkpoint_path.is_file()):
        raise FileNotFoundError(
            f"Resume was requested but no checkpoint exists in {cfg.checkpoint_dir}"
        )
    run_config_path = cfg.run_output_dir / "run_config.json"
    started_at = datetime.now(timezone.utc).isoformat()
    if cfg.resume and run_config_path.is_file():
        run_metadata = json.loads(run_config_path.read_text(encoding="utf-8"))
        current_machine = machine_identity()
        saved_machine_id = run_metadata.get("machine_id")
        if saved_machine_id != current_machine["machine_id"]:
            raise RuntimeError(
                "This run can only be resumed on its originating machine. "
                f"Checkpoint machine_id={saved_machine_id!r}; current "
                f"machine_id={current_machine['machine_id']!r}. Start a new run on this machine."
            )
        resume_events = run_metadata.setdefault("resume_events", [])
        resume_events.append(
            {
                "resumed_at_utc": started_at,
                "python": sys.version.split()[0],
                "dependencies": _dependency_versions(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": (
                    [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                    if torch.cuda.is_available()
                    else []
                ),
                "selected_device": str(device),
                **new_run_identity(),
                **current_machine,
                **_source_provenance(),
            }
        )
    else:
        run_identity = new_run_identity()
        run_metadata = {
            **asdict(cfg),
            **run_identity,
            "started_at_utc": started_at,
            "python": sys.version.split()[0],
            **machine_identity(),
            "dependencies": _dependency_versions(),
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "selected_device": str(device),
            "cuda_devices": (
                [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else []
            ),
            "dataset": {
                "name": "CIFAR-10",
                "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
                "train_examples": 50000,
                "test_examples": 10000,
            },
            "pretrained_backbone": (
                "torchvision ResNet18_Weights.IMAGENET1K_V1" if cfg.use_pretrained else None
            ),
            **_source_provenance(),
        }
    run_config_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    training_log = cfg.run_output_dir / "training.log"
    with training_log.open("a", encoding="utf-8") as log:
        action = "Resuming" if cfg.resume else "Started"
        run_id = run_metadata.get("run_id", cfg.tag)
        log.write(f"{action} run {cfg.tag} (run_id={run_id}) at {started_at}\n")
        log.write(
            f"CUDA check: available={torch.cuda.is_available()}; "
            f"devices={run_metadata.get('cuda_devices', [])}; selected_device={device}\n"
        )
        if not cfg.resume:
            log.write(json.dumps(run_metadata, sort_keys=True) + "\n")

    progress_stop = start_progress_monitor(
        cfg.run_output_dir / "client_progress",
        training_log,
        cfg.num_clients,
        cfg.num_server_rounds,
    )

    torch.manual_seed(cfg.seed)
    model = HybridClassifier().to(device)
    initial_arrays = get_parameters(model)

    records = _records_per_client(cfg)
    calibration = None
    if cfg.method != "no_dp":
        sensitivity = 2.0 * cfg.clip / records
        calibration = calibrate(
            cfg.epsilon, cfg.delta, cfg.num_server_rounds, sensitivity,
            cfg.num_clients, cfg.max_colluders, cfg.max_stragglers,
        )

    test_pixels, test_labels, test_features = load_test_data(cfg, device)
    eval_rows: dict[int, dict[str, float | int]] = {}
    resume_round = 0
    history_path = cfg.run_output_dir / "history.csv"
    if cfg.resume and history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as history_file:
            for saved_row in csv.DictReader(history_file):
                round_id = int(saved_row["round"])
                eval_rows[round_id] = {
                    "round": round_id,
                    "loss": float(saved_row["loss"]),
                    "accuracy": float(saved_row["accuracy"]),
                    "epsilon": float(saved_row["epsilon"]),
                }

    def evaluate_fn(server_round: int, parameters, config):
        global_round = resume_round + server_round
        set_parameters(model, parameters)
        loss, acc = evaluate_global(model, test_pixels, test_labels, test_features, device)
        eps = float("inf") if calibration is None else eps_from_rho(global_round * calibration.rho_round, cfg.delta)
        row = {"round": global_round, "loss": float(loss), "accuracy": float(acc), "epsilon": float(eps)}
        eval_rows[global_round] = row
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(eval_rows[key] for key in sorted(eval_rows))
        if global_round == cfg.num_server_rounds:
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.checkpoint_dir / "final_model.pt")
        completed_rounds = max(0, global_round - strategy.resume_round)
        if completed_rounds:
            elapsed = time.monotonic() - training_started
            average_round_seconds = elapsed / completed_rounds
            rounds_left = max(0, cfg.num_server_rounds - global_round)
            eta = _format_duration(average_round_seconds * rounds_left)
            progress = (
                f"progress={completed_rounds}/{session_rounds} "
                f"avg_round={average_round_seconds:.1f}s ETA~{eta}"
            )
        else:
            progress = f"progress=0/{session_rounds} ETA=estimating"
        message = (
            f"[{cfg.tag}] round={global_round} loss={loss:.6f} "
            f"accuracy={acc:.2f}% epsilon={eps:.6g} {progress}"
        )
        print(message, flush=True)
        with training_log.open("a", encoding="utf-8") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
        if global_round >= cfg.num_server_rounds:
            progress_stop.set()
        return float(loss), {"accuracy": float(acc), "epsilon": float(eps)}

    def fit_config(server_round: int) -> dict[str, int | float]:
        global_round = resume_round + server_round
        fit_settings: dict[str, int | float] = {"server_round": global_round}
        if calibration is not None:
            fit_settings.update(
                {f"calibration_{key}": value for key, value in calibration.to_dict().items()}
            )
        return fit_settings

    strategy = GradientMomentumStrategy(
        cfg=cfg,
        initial_arrays=initial_arrays,
        calibration=calibration,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_fn,
        accept_failures=False,
    )
    resume_round = strategy.resume_round
    for saved_round in list(eval_rows):
        if saved_round > strategy.resume_round:
            del eval_rows[saved_round]
    remaining_rounds = cfg.num_server_rounds - strategy.resume_round
    if remaining_rounds <= 0:
        raise ValueError("There are no training rounds left to run")
    session_rounds = remaining_rounds
    training_started = time.monotonic()

    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=remaining_rounds))


app = ServerApp(server_fn=server_fn)
