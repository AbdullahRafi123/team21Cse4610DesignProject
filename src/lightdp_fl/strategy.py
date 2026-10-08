from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from dataclasses import asdict

import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .config import RunConfig
from .model import HybridClassifier, arrays_to_flat, flat_to_arrays
from .privacy import Calibration, eps_from_rho
from .run_tracking import machine_id


class GradientMomentumStrategy(FedAvg):
    """Aggregate protected client gradients, then apply notebook server momentum."""
    def __init__(self, cfg: RunConfig, initial_arrays: list[np.ndarray], calibration: Calibration | None, **kwargs: Any):
        self.device = torch.device("cpu")
        self.model_template = HybridClassifier()
        self.out = cfg.run_output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "server_checkpoint.pt"
        self.legacy_checkpoint_path = self.out / "server_checkpoint.pt"
        self.resume_round = 0

        if cfg.resume:
            checkpoint = self._load_checkpoint(cfg)
            self.resume_round = int(checkpoint["round"])
            self.current = checkpoint["parameters"].to(device=self.device, dtype=torch.float32)
            self.velocity = checkpoint["velocity"].to(device=self.device, dtype=torch.float32)
            expected_shapes = [list(parameter.shape) for parameter in self.model_template.parameters()]
            parameter_count = sum(parameter.numel() for parameter in self.model_template.parameters())
            if (
                checkpoint.get("parameter_shapes") != expected_shapes
                or self.current.numel() != parameter_count
                or self.velocity.shape != self.current.shape
            ):
                raise ValueError("Resume checkpoint parameter or momentum shapes do not match this model")
            if self.resume_round >= cfg.num_server_rounds:
                raise ValueError(
                    f"Checkpoint is already at round {self.resume_round}; target is {cfg.num_server_rounds}"
                )
            initial_arrays = flat_to_arrays(self.current, self.model_template)
        else:
            self.current = arrays_to_flat(initial_arrays, self.device)
            self.velocity = torch.zeros_like(self.current)

        super().__init__(initial_parameters=ndarrays_to_parameters(initial_arrays), **kwargs)
        self.cfg = cfg
        self.calibration = calibration
        self.train_history = self._load_train_history(self.resume_round) if cfg.resume else []
        self._write_metadata()

    def _load_checkpoint(self, cfg: RunConfig) -> dict[str, Any]:
        checkpoint_path = (
            self.checkpoint_path
            if self.checkpoint_path.is_file()
            else self.legacy_checkpoint_path
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume was requested but no checkpoint exists at {self.checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported or incomplete server checkpoint schema")
        checkpoint_machine_id = checkpoint.get("machine_id")
        if checkpoint_machine_id != machine_id():
            raise RuntimeError(
                "This checkpoint can only be resumed on its originating machine. "
                f"Checkpoint machine_id={checkpoint_machine_id!r}; "
                f"current machine_id={machine_id()!r}. Start a new run on this machine."
            )
        expected_config = asdict(cfg)
        expected_config.pop("resume", None)
        if checkpoint.get("config") != expected_config:
            raise ValueError(
                "Resume configuration differs from the checkpoint. Keep the same experiment "
                "settings, output tag, and total round count when resuming."
            )
        return checkpoint

    def _load_train_history(self, through_round: int) -> list[dict[str, Any]]:
        path = self.out / "train_rounds.csv"
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as history_file:
            return [
                row for row in csv.DictReader(history_file)
                if int(row["round"]) <= through_round
            ]

    def _save_checkpoint(self, server_round: int) -> None:
        checkpoint = {
            "schema_version": 1,
            "machine_id": machine_id(),
            "round": server_round,
            "parameters": self.current.detach().cpu(),
            "velocity": self.velocity.detach().cpu(),
            "parameter_shapes": [
                list(parameter.shape) for parameter in self.model_template.parameters()
            ],
            "config": {key: value for key, value in asdict(self.cfg).items() if key != "resume"},
        }
        with NamedTemporaryFile(dir=self.out, suffix=".tmp", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, self.checkpoint_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _write_metadata(self) -> None:
        path = self.out / "run_config.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else dict(self.cfg.__dict__)
        data["calibration"] = None if self.calibration is None else self.calibration.to_dict()
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        global_round = self.resume_round + server_round
        active_vectors: list[torch.Tensor] = []
        local_times: list[float] = []
        noise_times: list[float] = []
        stragglers = None
        for _, fit_res in results:
            metrics = fit_res.metrics
            if int(metrics.get("active", 1)) != 1:
                stragglers = int(metrics.get("stragglers", 0))
                continue
            arrays = parameters_to_ndarrays(fit_res.parameters)
            active_vectors.append(arrays_to_flat(arrays, self.device))
            local_times.append(float(metrics.get("local_seconds", 0.0)))
            noise_times.append(float(metrics.get("noise_seconds", 0.0)))
            stragglers = int(metrics.get("stragglers", 0))

        if not active_vectors:
            raise RuntimeError("All clients were simulated as stragglers; notebook schedule should prevent this")

        gradient = torch.stack(active_vectors).mean(0)
        self.velocity.mul_(self.cfg.momentum).add_(gradient)
        t = global_round - 1
        lr = self.cfg.learning_rate * (
            0.2 + 0.8 * (1.0 + math.cos(math.pi * t / self.cfg.num_server_rounds)) / 2.0
        )
        self.current = self.current - lr * self.velocity
        arrays = flat_to_arrays(self.current, self.model_template)
        params = ndarrays_to_parameters(arrays)

        row = {
            "round": global_round,
            "active_clients": len(active_vectors),
            "stragglers": int(stragglers or 0),
            "learning_rate": lr,
            "mean_client_local_seconds": float(np.mean(local_times)) if local_times else 0.0,
            "mean_client_noise_seconds": float(np.mean(noise_times)) if noise_times else 0.0,
            "failures": len(failures),
        }
        self.train_history.append(row)
        self._write_csv(self.out / "train_rounds.csv", self.train_history)
        self._save_checkpoint(global_round)
        return params, {"active_clients": len(active_vectors), "stragglers": int(stragglers or 0), "lr": float(lr)}

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
