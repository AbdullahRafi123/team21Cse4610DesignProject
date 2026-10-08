from __future__ import annotations
import csv
import json
import platform
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


def server_fn(context: Context) -> ServerAppComponents:
    cfg = RunConfig.from_mapping(context.run_config)
    cfg.validate()
    cfg.run_output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = cfg.run_output_dir / "run_config.json"
    run_metadata = {
        **asdict(cfg),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    run_config_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    training_log = cfg.run_output_dir / "training.log"
    with training_log.open("a", encoding="utf-8") as log:
        log.write(f"Started run {cfg.tag} at {run_metadata['started_at_utc']}\n")
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
    eval_rows: list[dict] = []

    def evaluate_fn(server_round, parameters, config):
        set_parameters(model, parameters)
        loss, acc = evaluate_global(model, test_pixels, test_labels, test_features, device)
        eps = float("inf") if calibration is None else eps_from_rho(server_round * calibration.rho_round, cfg.delta)
        row = {"round": int(server_round), "loss": float(loss), "accuracy": float(acc), "epsilon": float(eps)}
        eval_rows.append(row)
        path = cfg.run_output_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader(); writer.writerows(eval_rows)
        torch.save(model.state_dict(), cfg.run_output_dir / "model.pt")
        message = (
            f"[{cfg.tag}] round={server_round} loss={loss:.6f} "
            f"accuracy={acc:.2f}% epsilon={eps:.6g}"
        )
        print(message, flush=True)
        with training_log.open("a", encoding="utf-8") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
        return float(loss), {"accuracy": float(acc), "epsilon": float(eps)}

    def fit_config(server_round: int):
        return {"server_round": int(server_round)}

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
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=cfg.num_server_rounds))


app = ServerApp(server_fn=server_fn)
