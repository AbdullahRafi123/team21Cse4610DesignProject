from __future__ import annotations
import csv
import json
from importlib import metadata
import platform
import subprocess
import sys
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


def server_fn(context: Context) -> ServerAppComponents:
    cfg = RunConfig.from_mapping(context.run_config)
    cfg.validate()
    cfg.run_output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.resume and not (cfg.run_output_dir / "server_checkpoint.pt").is_file():
        raise FileNotFoundError(
            f"Resume was requested but no checkpoint exists in {cfg.run_output_dir}"
        )
    run_config_path = cfg.run_output_dir / "run_config.json"
    started_at = datetime.now(timezone.utc).isoformat()
    if cfg.resume and run_config_path.is_file():
        run_metadata = json.loads(run_config_path.read_text(encoding="utf-8"))
        resume_events = run_metadata.setdefault("resume_events", [])
        resume_events.append(
            {
                "resumed_at_utc": started_at,
                "python": sys.version.split()[0],
                "dependencies": _dependency_versions(),
                **_source_provenance(),
            }
        )
    else:
        run_metadata = {
            **asdict(cfg),
            "started_at_utc": started_at,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "dependencies": _dependency_versions(),
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
        log.write(f"{action} run {cfg.tag} at {started_at}\n")
        if not cfg.resume:
            log.write(json.dumps(run_metadata, sort_keys=True) + "\n")

    device = resolve_device(cfg.device)
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
            torch.save(model.state_dict(), cfg.run_output_dir / "model.pt")
        message = (
            f"[{cfg.tag}] round={global_round} loss={loss:.6f} "
            f"accuracy={acc:.2f}% epsilon={eps:.6g}"
        )
        print(message, flush=True)
        with training_log.open("a", encoding="utf-8") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
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

    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=remaining_rounds))


app = ServerApp(server_fn=server_fn)
