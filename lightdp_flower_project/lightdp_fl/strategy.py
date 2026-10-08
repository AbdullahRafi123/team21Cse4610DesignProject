from __future__ import annotations
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .config import RunConfig
from .model import HybridClassifier, arrays_to_flat, flat_to_arrays
from .privacy import Calibration, eps_from_rho


class GradientMomentumStrategy(FedAvg):
    """Aggregate protected client gradients, then apply notebook server momentum."""
    def __init__(self, cfg: RunConfig, initial_arrays: list[np.ndarray], calibration: Calibration | None, **kwargs: Any):
        super().__init__(initial_parameters=ndarrays_to_parameters(initial_arrays), **kwargs)
        self.cfg = cfg
        self.calibration = calibration
        self.device = torch.device("cpu")
        model = HybridClassifier()
        self.model_template = model
        self.current = arrays_to_flat(initial_arrays, self.device)
        self.velocity = torch.zeros_like(self.current)
        self.out = cfg.run_output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.train_history: list[dict[str, Any]] = []
        self._write_metadata()

    def _write_metadata(self) -> None:
        data = dict(self.cfg.__dict__)
        data["calibration"] = None if self.calibration is None else self.calibration.to_dict()
        (self.out / "run_config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        active_vectors: list[torch.Tensor] = []
        local_times: list[float] = []
        noise_times: list[float] = []
        stragglers = None
        active_ids: list[int] = []
        for _, fit_res in results:
            metrics = fit_res.metrics
            if int(metrics.get("active", 1)) != 1:
                stragglers = int(metrics.get("stragglers", 0))
                continue
            arrays = parameters_to_ndarrays(fit_res.parameters)
            active_vectors.append(arrays_to_flat(arrays, self.device))
            local_times.append(float(metrics.get("local_seconds", 0.0)))
            noise_times.append(float(metrics.get("noise_seconds", 0.0)))
            active_ids.append(int(metrics.get("client_id", -1)))
            stragglers = int(metrics.get("stragglers", 0))

        if not active_vectors:
            raise RuntimeError("All clients were simulated as stragglers; notebook schedule should prevent this")

        gradient = torch.stack(active_vectors).mean(0)
        self.velocity.mul_(self.cfg.momentum).add_(gradient)
        t = server_round - 1
        lr = self.cfg.learning_rate * (
            0.2 + 0.8 * (1.0 + math.cos(math.pi * t / self.cfg.num_server_rounds)) / 2.0
        )
        self.current = self.current - lr * self.velocity
        arrays = flat_to_arrays(self.current, self.model_template)
        params = ndarrays_to_parameters(arrays)

        row = {
            "round": server_round,
            "active_clients": len(active_vectors),
            "stragglers": int(stragglers or 0),
            "learning_rate": lr,
            "mean_client_local_seconds": float(np.mean(local_times)) if local_times else 0.0,
            "mean_client_noise_seconds": float(np.mean(noise_times)) if noise_times else 0.0,
            "failures": len(failures),
        }
        self.train_history.append(row)
        self._write_csv(self.out / "train_rounds.csv", self.train_history)
        torch.save({"parameters": arrays, "velocity": self.velocity.numpy(), "round": server_round}, self.out / "latest_server_state.pt")
        return params, {"active_clients": len(active_vectors), "stragglers": int(stragglers or 0), "lr": float(lr)}

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
