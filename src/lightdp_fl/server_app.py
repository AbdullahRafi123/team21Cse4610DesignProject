from __future__ import annotations
import csv
import json
import math
import os
import subprocess
import sys
import time
from importlib import metadata
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from flwr.common import Context, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .config import RunConfig
from .data import load_test_data, load_validation_data, partition_record_counts, resolve_device
from .model import HybridClassifier, get_parameters, set_parameters
from .privacy import calibrate, eps_from_rho
from .progress import feature_progress_callback, start_progress_monitor
from .run_tracking import machine_identity, new_run_identity
from .strategy import PrivateFedAvgStrategy, StandardFedAvgStrategy
from .training import evaluate_global


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


def _stage_completed_run(cfg: RunConfig, metadata: dict[str, object]) -> None:
    """Stage this completed run's provenance in its containing Git checkout."""
    run_config_path = cfg.run_output_dir / "run_config.json"
    if not cfg.git_track_results:
        metadata["git_tracking"] = {"status": "disabled"}
        run_config_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return

    artifacts = (
        "run_config.json",
        "history.csv",
        "train_rounds.csv",
        "training.log",
        "final_metrics.json",
    )
    run_files = [cfg.run_output_dir / name for name in artifacts]
    run_files = [path for path in run_files if path.is_file()]
    try:
        repo_root = Path(
            subprocess.run(
                ["git", "-C", str(cfg.run_output_dir), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        relative_files = [path.resolve().relative_to(repo_root) for path in run_files]
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        metadata["git_tracking"] = {
            "status": "unavailable",
            "reason": str(exc),
        }
        run_config_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[git] run artifacts were not staged: {exc}",
            flush=True,
        )
        return

    metadata["git_tracking"] = {
        "status": "staged",
        "action": "git add; no commit created",
        "files": [path.as_posix() for path in relative_files],
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_config_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "add",
                "--",
                *(path.as_posix() for path in relative_files),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        metadata["git_tracking"] = {
            "status": "failed",
            "reason": str(exc),
        }
        run_config_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[git] could not stage run artifacts: {exc}", flush=True)
        return
    print(
        f"[git] staged {len(relative_files)} completed run records; no commit created",
        flush=True,
    )


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
    run_identity = None
    if not cfg.resume and cfg.tag == "auto":
        run_identity = new_run_identity()
        cfg = replace(cfg, tag=str(run_identity["run_id"]))
    device = resolve_device(cfg.device)
    cfg.run_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [
        cfg.checkpoint_dir / "fedavg_server_checkpoint.pt",
    ]
    if cfg.resume and not any(path.is_file() for path in checkpoint_paths):
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
        run_identity = run_identity or new_run_identity()
        run_metadata = {
            **asdict(cfg),
            "result_schema_version": 2,
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
                "torchvision ResNet18_Weights.IMAGENET1K_V1"
                if cfg.use_pretrained
                else "torchvision ResNet-18 with random initialization"
            ),
            "model_definition": {
                "name": "HybridClassifier",
                "trainable_parameters": "CNN branch and classifier head",
                "backbone_frozen": True,
            },
            "data_protocol": {
                "source_train_examples": (
                    50000 if cfg.client_data_mode == "partitioned_cifar10" else None
                ),
                "client_train_fraction": (
                    1.0
                    if cfg.client_data_mode == "local_npz"
                    else 1.0 - cfg.validation_fraction
                ),
                "validation_fraction": cfg.validation_fraction,
                "validation_selection": "deterministic class-stratified holdout",
                "validation_evaluated_each_round": True,
                "test_evaluated_each_round": False,
                "partition_method": (
                    cfg.partition_method
                    if cfg.client_data_mode == "partitioned_cifar10"
                    else None
                ),
                "client_data_mode": cfg.client_data_mode,
                "dirichlet_alpha": (
                    cfg.dirichlet_alpha
                    if cfg.client_data_mode == "partitioned_cifar10"
                    and cfg.partition_method == "dirichlet"
                    else None
                ),
            },
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

    if cfg.client_data_mode == "partitioned_cifar10":
        stop_progress_monitor = start_progress_monitor(
            cfg.run_output_dir / "client_progress",
            training_log,
            cfg.num_clients,
            cfg.num_server_rounds,
        )
    else:
        # Client progress files live on remote hosts and are not visible here.
        stop_progress_monitor = None

    torch.manual_seed(cfg.seed)
    model = HybridClassifier().to(device)
    initial_arrays = get_parameters(model)

    calibration = None
    if cfg.method != "no_dp":
        sensitivity = 2.0 * cfg.client_update_clip
        calibration = calibrate(
            cfg.epsilon, cfg.delta, cfg.num_server_rounds, sensitivity,
            cfg.num_clients, cfg.max_colluders, cfg.max_stragglers,
        )
        run_metadata["privacy_calibration"] = {
                "mechanism": cfg.method,
                "unit": "client_model_update",
                "client_update_l2_clip": cfg.client_update_clip,
                "calibration": calibration.to_dict(),
                "status": (
                    "research calibration adapted from the clipped-gradient pipeline; "
                    "independent formal review required before claiming a guarantee"
                ),
        }

    print("[setup] training-derived CIFAR-10 validation data/features: loading", flush=True)
    validation_pixels, validation_labels, validation_features = load_validation_data(
        cfg,
        device,
        progress_callback=feature_progress_callback("centralized validation features"),
    )
    print("[setup] validation data/features: ready", flush=True)
    # Load and cache public test features before training, but do not evaluate
    # the test labels until the final configured round.
    test_pixels, test_labels, test_features = load_test_data(
        cfg,
        device,
        progress_callback=feature_progress_callback("final test features"),
    )
    data_protocol = run_metadata.setdefault("data_protocol", {})
    data_protocol.update({
        "validation_examples": len(validation_labels),
        "test_examples": len(test_labels),
    })
    if cfg.client_data_mode == "partitioned_cifar10":
        record_counts = partition_record_counts(cfg)
        data_protocol.update({
            "client_train_records": record_counts,
            "train_examples_used": sum(record_counts),
        })
    else:
        data_protocol.update({
            "client_train_records": "client-local; aggregate count recorded per round",
            "train_examples_used": "reported by clients during training",
            "partition_method": None,
        })
    run_config_path.write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    eval_rows: dict[int, dict[str, float | int]] = {}
    resume_round = 0
    history_path = cfg.run_output_dir / "history.csv"
    if cfg.resume and history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as history_file:
            for saved_row in csv.DictReader(history_file):
                round_id = int(saved_row["round"])
                if "val_loss" not in saved_row or "val_accuracy" not in saved_row:
                    raise RuntimeError(
                        "This run predates validation-only histories; start a fresh run "
                        "instead of mixing test and validation metrics."
                    )
                eval_rows[round_id] = {
                    "round": round_id,
                    "val_loss": float(saved_row["val_loss"]),
                    "val_accuracy": float(saved_row["val_accuracy"]),
                    "epsilon": float(saved_row["epsilon"]),
                }

    best_validation_path = cfg.checkpoint_dir / "best_validation_checkpoint.pt"
    best_validation_accuracy = float("-inf")
    if cfg.resume and best_validation_path.is_file():
        best_checkpoint = torch.load(
            best_validation_path, map_location="cpu", weights_only=True
        )
        best_validation_accuracy = float(best_checkpoint["val_accuracy"])
    elif cfg.resume and eval_rows:
        raise RuntimeError(
            "Validation-best checkpoint is missing; start a fresh run rather than "
            "selecting from validation metrics without the corresponding model."
        )

    def save_best_validation_checkpoint(global_round: int, accuracy: float) -> None:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "round": global_round,
            "val_accuracy": accuracy,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        }
        with NamedTemporaryFile(
            dir=cfg.checkpoint_dir, suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, best_validation_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def evaluate_fn(server_round: int, parameters, config):
        nonlocal best_validation_accuracy
        global_round = resume_round + server_round
        set_parameters(model, parameters)
        loss, acc = evaluate_global(
            model, validation_pixels, validation_labels, validation_features, device
        )
        if acc > best_validation_accuracy:
            best_validation_accuracy = float(acc)
            save_best_validation_checkpoint(global_round, float(acc))
        eps = float("inf") if calibration is None else eps_from_rho(global_round * calibration.rho_round, cfg.delta)
        row = {
            "round": global_round,
            "val_loss": float(loss),
            "val_accuracy": float(acc),
            "epsilon": float(eps),
        }
        eval_rows[global_round] = row
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(eval_rows[key] for key in sorted(eval_rows))
        if global_round == cfg.num_server_rounds:
            best_checkpoint = torch.load(
                best_validation_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(best_checkpoint["model_state"])
            test_loss, test_accuracy = evaluate_global(
                model, test_pixels, test_labels, test_features, device
            )
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.checkpoint_dir / "final_model.pt")
            (cfg.run_output_dir / "final_metrics.json").write_text(
                json.dumps(
                    {
                        "round": global_round,
                        "test_loss": float(test_loss),
                        "test_accuracy": float(test_accuracy),
                        "test_evaluations": 1,
                        "selected_validation_round": int(best_checkpoint["round"]),
                        "selected_validation_accuracy": float(best_checkpoint["val_accuracy"]),
                        "validation_examples": len(validation_labels),
                        "test_examples": len(test_labels),
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
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
            f"[{cfg.tag}] round={global_round} val_loss={loss:.6f} "
            f"val_accuracy={acc:.2f}% epsilon={eps:.6g} {progress}"
        )
        print(message, flush=True)
        with training_log.open("a", encoding="utf-8") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
        if global_round >= cfg.num_server_rounds:
            if stop_progress_monitor is not None:
                stop_progress_monitor()
            _stage_completed_run(cfg, run_metadata)
        return float(loss), {"val_accuracy": float(acc), "epsilon": float(eps)}

    def fit_config(server_round: int) -> dict[str, Scalar]:
        global_round = resume_round + server_round
        fit_settings: dict[str, Scalar] = {
            "server_round": global_round,
            "run_tag": cfg.tag,
        }
        if calibration is not None:
            fit_settings.update(
                {f"calibration_{key}": value for key, value in calibration.to_dict().items()}
            )
        fit_settings.update(
            {
                "local_epochs": cfg.local_epochs,
                "batch_size": cfg.batch_size,
                "local_learning_rate": cfg.local_learning_rate,
                "local_momentum": cfg.local_momentum,
            }
        )
        return fit_settings

    if cfg.training_algorithm in {"fedavg", "clipped_fedavg"}:
        if cfg.method == "no_dp":
            min_fit = max(1, math.ceil(cfg.num_clients * cfg.fraction_fit))
            strategy = StandardFedAvgStrategy(
                cfg=cfg,
                initial_arrays=initial_arrays,
                fraction_fit=cfg.fraction_fit,
                fraction_evaluate=0.0,
                min_fit_clients=min_fit,
                min_available_clients=cfg.num_clients,
                on_fit_config_fn=fit_config,
                evaluate_fn=evaluate_fn,
                accept_failures=True,
            )
        else:
            min_fit = cfg.num_clients - cfg.max_stragglers
            strategy = PrivateFedAvgStrategy(
                cfg=cfg,
                initial_arrays=initial_arrays,
                fraction_fit=1.0,
                fraction_evaluate=0.0,
                min_fit_clients=min_fit,
                min_available_clients=min_fit,
                on_fit_config_fn=fit_config,
                evaluate_fn=evaluate_fn,
                accept_failures=True,
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

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(
            num_rounds=remaining_rounds,
            round_timeout=cfg.round_timeout,
        ),
    )


app = ServerApp(server_fn=server_fn)
