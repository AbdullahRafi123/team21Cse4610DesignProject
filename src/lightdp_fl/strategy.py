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
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .config import RunConfig
from .model import HybridClassifier, arrays_to_flat, flat_to_arrays
from .privacy import Calibration, eps_from_rho
from .run_tracking import machine_id


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a nonempty sequence of uniform records as a CSV file."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class StandardFedAvgStrategy(FedAvg):
    """Flower FedAvg with per-round provenance and resumable global weights."""

    def __init__(
        self,
        cfg: RunConfig,
        initial_arrays: list[np.ndarray],
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.out = cfg.run_output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "fedavg_server_checkpoint.pt"
        self.resume_round = 0
        if cfg.resume:
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
            if checkpoint.get("schema_version") != 1:
                raise ValueError("Unsupported FedAvg checkpoint schema")
            if checkpoint.get("machine_id") != machine_id():
                raise RuntimeError(
                    "FedAvg checkpoints can only be resumed on their originating machine"
                )
            expected = asdict(cfg)
            expected.pop("resume", None)
            saved_config = checkpoint.get("config")
            if isinstance(saved_config, dict):
                saved_config = dict(saved_config)
                for removed_key in ("learning_rate", "momentum", "microbatch"):
                    saved_config.pop(removed_key, None)
            if isinstance(saved_config, dict) and "git_track_results" not in saved_config:
                expected.pop("git_track_results", None)
            if isinstance(saved_config, dict) and "round_timeout" not in saved_config:
                expected.pop("round_timeout", None)
            if saved_config != expected:
                raise ValueError("FedAvg resume configuration differs from the saved run")
            self.resume_round = int(checkpoint["round"])
            initial_arrays = [
                tensor.cpu().numpy().copy() for tensor in checkpoint["parameters"]
            ]
            if self.resume_round >= cfg.num_server_rounds:
                raise ValueError("FedAvg checkpoint has already reached the configured round count")
        self.train_history = self._load_train_history(self.resume_round) if cfg.resume else []
        super().__init__(
            initial_parameters=ndarrays_to_parameters(initial_arrays),
            **kwargs,
        )

    def _load_train_history(self, through_round: int) -> list[dict[str, Any]]:
        path = self.out / "train_rounds.csv"
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as history_file:
            return [
                row for row in csv.DictReader(history_file)
                if int(row["round"]) <= through_round
            ]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if failures and not self.accept_failures:
            raise RuntimeError(
                f"FedAvg round {self.resume_round + server_round} had client failures"
            )
        if not results:
            raise RuntimeError(
                f"FedAvg round {self.resume_round + server_round} returned no client results"
            )
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            return None, metrics

        global_round = self.resume_round + server_round
        successful = [fit_res for _, fit_res in results]
        examples = sum(result.num_examples for result in successful)
        train_times = [float(result.metrics.get("local_seconds", 0.0)) for result in successful]
        weighted_loss = sum(
            float(result.metrics.get("train_loss", 0.0)) * result.num_examples
            for result in successful
        ) / max(examples, 1)
        row: dict[str, Any] = {
            "round": global_round,
            "active_clients": len(successful),
            "examples": examples,
            "training_algorithm": self.cfg.training_algorithm,
            "aggregation": "example_weighted_fedavg",
            "local_epochs": self.cfg.local_epochs,
            "learning_rate": self.cfg.local_learning_rate,
            "weighted_train_loss": weighted_loss,
            "mean_client_local_seconds": float(np.mean(train_times)) if train_times else 0.0,
            "failures": len(failures),
        }
        self.train_history.append(row)
        _write_csv(self.out / "train_rounds.csv", self.train_history)
        self._save_checkpoint(global_round, parameters)
        return parameters, {**metrics, "weighted_train_loss": weighted_loss}

    def _save_checkpoint(self, server_round: int, parameters: Parameters) -> None:
        tensors = [
            torch.from_numpy(array.copy())
            for array in parameters_to_ndarrays(parameters)
        ]
        checkpoint = {
            "schema_version": 1,
            "machine_id": machine_id(),
            "round": server_round,
            "parameters": tensors,
            "config": {key: value for key, value in asdict(self.cfg).items() if key != "resume"},
        }
        with NamedTemporaryFile(dir=self.checkpoint_dir, suffix=".tmp", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, self.checkpoint_path)
        finally:
            temp_path.unlink(missing_ok=True)


class PrivateFedAvgStrategy(StandardFedAvgStrategy):
    """Apply privacy mechanisms to clipped local-SGD model deltas.

    Private updates are uniformly averaged across participating clients. This
    is intentionally distinct from example-weighted FedAvg because the current
    research mechanisms and calibration assume equal client contribution.
    """

    def __init__(self, cfg: RunConfig, initial_arrays: list[np.ndarray], **kwargs: Any):
        self.model_template = HybridClassifier()
        self.round_start_arrays: list[np.ndarray] | None = None
        super().__init__(cfg, initial_arrays, **kwargs)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        self.round_start_arrays = [
            array.copy() for array in parameters_to_ndarrays(parameters)
        ]
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        global_round = self.resume_round + server_round
        active_results = [
            result for _, result in results if int(result.metrics.get("active", 1)) == 1
        ]
        missing = len(failures) + len(results) - len(active_results)
        if missing > self.cfg.max_stragglers:
            raise RuntimeError(
                f"Private FedAvg round {global_round} observed {missing} missing clients; "
                f"configured privacy bound is {self.cfg.max_stragglers}"
            )
        if not active_results:
            raise RuntimeError(f"Private FedAvg round {global_round} has no client updates")
        if self.round_start_arrays is None:
            raise RuntimeError("Private FedAvg has no saved round-start model parameters")

        device = torch.device("cpu")
        client_updates = [
            arrays_to_flat(parameters_to_ndarrays(result.parameters), device)
            for result in active_results
        ]
        mean_update = torch.stack(client_updates).mean(dim=0)
        base = arrays_to_flat(self.round_start_arrays, device)
        parameters = ndarrays_to_parameters(
            flat_to_arrays(base + mean_update, self.model_template)
        )

        examples = sum(result.num_examples for result in active_results)
        successful = len(active_results)
        losses = [
            float(result.metrics.get("train_loss", 0.0)) for result in active_results
        ]
        update_norms = [
            float(result.metrics.get("update_norm", 0.0))
            for result in active_results
        ]
        train_times = [
            float(result.metrics.get("local_seconds", 0.0))
            for result in active_results
        ]
        row: dict[str, Any] = {
            "round": global_round,
            "active_clients": successful,
            "missing_clients": missing,
            "examples": examples,
            "training_algorithm": self.cfg.training_algorithm,
            "aggregation": "uniform_client_mean",
            "privacy_method": self.cfg.method,
            "client_update_clip": self.cfg.client_update_clip,
            "mean_preclip_update_norm": (
                float(np.mean(update_norms)) if update_norms else 0.0
            ),
            "clients_clipped": sum(
                int(result.metrics.get("update_clipped", 0))
                for result in active_results
            ),
            "local_epochs": self.cfg.local_epochs,
            "learning_rate": self.cfg.local_learning_rate,
            "mean_train_loss": float(np.mean(losses)) if losses else 0.0,
            "mean_client_local_seconds": (
                float(np.mean(train_times)) if train_times else 0.0
            ),
            "failures": len(failures),
        }
        self.train_history.append(row)
        _write_csv(self.out / "train_rounds.csv", self.train_history)
        self._save_checkpoint(global_round, parameters)
        return parameters, {
            "active_clients": successful,
            "missing_clients": missing,
            "examples": examples,
        }
