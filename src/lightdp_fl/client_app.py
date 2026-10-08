from __future__ import annotations
import numpy as np
import torch
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .config import RunConfig
from .data import load_client_data, resolve_device
from .model import HybridClassifier, get_parameters, set_parameters
from .privacy import Calibration, calibrate
from .training import compute_client_upload


def _calibration(cfg: RunConfig, records_per_client: int) -> Calibration | None:
    if cfg.method == "no_dp":
        return None
    sensitivity = 2.0 * cfg.clip / records_per_client
    return calibrate(
        cfg.epsilon, cfg.delta, cfg.num_server_rounds, sensitivity,
        cfg.num_clients, cfg.max_colluders, cfg.max_stragglers,
    )


class LightDPClient(NumPyClient):
    def __init__(self, partition_id: int, cfg: RunConfig) -> None:
        self.partition_id = partition_id
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.model = HybridClassifier().to(self.device)
        self.pixels, self.labels, self.features = load_client_data(cfg, partition_id, self.device)
        self.calibration = _calibration(cfg, len(self.labels))

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        server_round = int(config.get("server_round", 1))
        arrays, metrics = compute_client_upload(
            self.model, self.pixels, self.labels, self.features,
            self.cfg, self.partition_id, server_round - 1,
            self.calibration, self.device,
        )
        return arrays, len(self.labels), metrics

    def evaluate(self, parameters, config):
        # Centralized server evaluation is used to match the notebook.
        return 0.0, len(self.labels), {"client_eval_disabled": 1}


def client_fn(context: Context):
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
