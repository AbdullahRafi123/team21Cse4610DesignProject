from __future__ import annotations
import numpy as np
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, Scalar

from .config import RunConfig
from .data import load_client_data, resolve_device
from .model import HybridClassifier, get_parameters, set_parameters
from .privacy import Calibration
from .training import compute_client_upload


def _calibration_from_config(config: dict[str, Scalar]) -> Calibration | None:
    if "calibration_normal_var" not in config:
        return None
    return Calibration(
        normal_var=float(config["calibration_normal_var"]),
        u=float(config["calibration_u"]),
        k=float(config["calibration_k"]),
        rho_round=float(config["calibration_rho_round"]),
        eps=float(config["calibration_eps"]),
        steps=int(config["calibration_steps"]),
        sensitivity=float(config["calibration_sensitivity"]),
    )


class LightDPClient(NumPyClient):
    def __init__(self, partition_id: int, cfg: RunConfig) -> None:
        self.partition_id = partition_id
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.model = HybridClassifier().to(self.device)
        self.pixels, self.labels, self.features = load_client_data(cfg, partition_id, self.device)

    def get_parameters(self, config: dict[str, Scalar]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Scalar]
    ) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        server_round = int(config.get("server_round", 1))
        arrays, metrics = compute_client_upload(
            self.model, self.pixels, self.labels, self.features,
            self.cfg, self.partition_id, server_round - 1,
            _calibration_from_config(config), self.device,
        )
        return arrays, len(self.labels), metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        # Centralized server evaluation is used to match the notebook.
        return 0.0, len(self.labels), {"client_eval_disabled": 1}


def client_fn(context: Context) -> Client:
    cfg = RunConfig.from_mapping(context.run_config)
    cfg.validate()
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config.get("num-partitions", cfg.num_clients))
    if num_partitions != cfg.num_clients:
        raise ValueError(
            f"Flower has {num_partitions} SuperNodes but run config num-clients={cfg.num_clients}. "
            "Override both together."
        )
    return LightDPClient(partition_id, cfg).to_client()


app = ClientApp(client_fn=client_fn)
