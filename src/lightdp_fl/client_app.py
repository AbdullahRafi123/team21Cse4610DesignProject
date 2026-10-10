from __future__ import annotations
from dataclasses import replace
import math
import os
from pathlib import Path
import numpy as np
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, Scalar

from .config import RunConfig
from .data import load_client_data, resolve_device
from .model import HybridClassifier, get_parameters, set_parameters
from .privacy import Calibration
from .progress import feature_progress_callback, write_client_progress
from .training import (
    train_clipped_fedavg_client,
    train_fedavg_client,
    train_private_fedavg_client,
)


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


class FederatedCifarClient(NumPyClient):
    def __init__(self, partition_id: int, cfg: RunConfig) -> None:
        self.partition_id = partition_id
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.model = HybridClassifier().to(self.device)
        local_data_value = os.environ.get("FEDABBA_CLIENT_DATA")
        if cfg.client_data_mode == "local_npz" and not local_data_value:
            raise RuntimeError(
                "client-data-mode=local_npz requires FEDABBA_CLIENT_DATA on every client host"
            )
        if cfg.client_data_mode == "partitioned_cifar10" and local_data_value:
            raise RuntimeError(
                "FEDABBA_CLIENT_DATA is set, but client-data-mode is partitioned_cifar10; "
                "set client-data-mode=local_npz in the Flower run config"
            )
        local_data_path = Path(local_data_value) if local_data_value else None
        scope = f"client {partition_id + 1}/{cfg.num_clients} data/features"
        print(f"[setup] {scope}: loading", flush=True)
        self.pixels, self.labels, self.features = load_client_data(
            cfg,
            partition_id,
            self.device,
            progress_callback=feature_progress_callback(scope),
            local_data_path=local_data_path,
        )
        source = (
            "machine-local dataset"
            if local_data_path is not None
            else "partitioned CIFAR-10"
        )
        print(
            f"[setup] {scope}: ready records={len(self.labels)} source={source}",
            flush=True,
        )

    def get_parameters(self, config: dict[str, Scalar]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Scalar]
    ) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        initial_parameters = [array.copy() for array in parameters]
        set_parameters(self.model, parameters)
        server_round = int(config.get("server_round", 1))
        run_tag = config.get("run_tag")
        if isinstance(run_tag, str) and run_tag != self.cfg.tag:
            self.cfg = replace(self.cfg, tag=run_tag)
        progress_dir = self.cfg.run_output_dir / "client_progress"
        total_microbatches = self.cfg.local_epochs * (
            (len(self.labels) + self.cfg.batch_size - 1) // self.cfg.batch_size
        )
        write_client_progress(
            progress_dir, server_round, self.partition_id, 0, total_microbatches
        )
        progress_interval = max(1, math.ceil(total_microbatches / 10))

        def report_progress(done: int, total: int) -> None:
            if done == total or done % progress_interval == 0:
                write_client_progress(
                    progress_dir, server_round, self.partition_id, done, total
                )

        progress_callback = report_progress
        if self.cfg.method == "no_dp":
            train_function = (
                train_clipped_fedavg_client
                if self.cfg.training_algorithm == "clipped_fedavg"
                else train_fedavg_client
            )
            metrics = train_function(
                self.model,
                self.pixels,
                self.labels,
                self.features,
                self.cfg,
                self.partition_id,
                server_round,
                self.device,
                progress_callback=progress_callback,
            )
            arrays = get_parameters(self.model)
        else:
            calibration = _calibration_from_config(config)
            if calibration is None:
                raise RuntimeError("FedAvg privacy variant received no calibration")
            arrays, metrics = train_private_fedavg_client(
                self.model,
                initial_parameters,
                self.pixels,
                self.labels,
                self.features,
                self.cfg,
                self.partition_id,
                server_round,
                calibration,
                self.device,
                progress_callback=progress_callback,
            )
        write_client_progress(
            progress_dir, server_round, self.partition_id,
            total_microbatches, total_microbatches, status="complete",
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
    return FederatedCifarClient(partition_id, cfg).to_client()


app = ClientApp(client_fn=client_fn)
