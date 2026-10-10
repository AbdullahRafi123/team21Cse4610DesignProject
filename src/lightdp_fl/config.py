from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, cast

from .run_tracking import machine_id

Method = Literal["no_dp", "vanilla_dp", "smpc_dp", "lightdp"]
TrainingAlgorithm = Literal["fedavg", "clipped_fedavg"]
PartitionMethod = Literal["iid", "label_shards", "dirichlet"]
ClientDataMode = Literal["partitioned_cifar10", "local_npz"]

@dataclass(frozen=True)
class RunConfig:
    seed: int = 2026
    num_clients: int = 50
    num_server_rounds: int = 8
    max_colluders: int = 10
    max_stragglers: int = 10
    epsilon: float = 6.0
    delta: float = 1e-5
    clip: float = 1.0
    client_update_clip: float = 1.0
    image_size: int = 224
    feature_batch: int = 128
    method: Method = "no_dp"
    training_algorithm: TrainingAlgorithm = "fedavg"
    partition_method: PartitionMethod = "iid"
    client_data_mode: ClientDataMode = "partitioned_cifar10"
    simulate_stragglers: bool = True
    dirichlet_alpha: float = 0.5
    validation_fraction: float = 0.1
    local_epochs: int = 1
    batch_size: int = 32
    local_learning_rate: float = 0.01
    local_momentum: float = 0.9
    fraction_fit: float = 1.0
    max_records_per_client: int = 0
    use_pretrained: bool = True
    cache_dir: str = ".cache/lightdp_flower"
    output_dir: str = "results/experiments"
    tag: str = "auto"
    git_track_results: bool = True
    device: str = "auto"
    resume: bool = False

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "RunConfig":
        return cls(
            seed=int(cfg.get("seed", 2026)),
            num_clients=int(cfg.get("num-clients", 50)),
            num_server_rounds=int(cfg.get("num-server-rounds", 8)),
            max_colluders=int(cfg.get("max-colluders", 10)),
            max_stragglers=int(cfg.get("max-stragglers", 10)),
            epsilon=float(cfg.get("epsilon", 6.0)),
            delta=float(cfg.get("delta", 1e-5)),
            clip=float(cfg.get("clip", 1.0)),
            client_update_clip=float(cfg.get("client-update-clip", 1.0)),
            image_size=int(cfg.get("image-size", 224)),
            feature_batch=int(cfg.get("feature-batch", 128)),
            method=cast(Method, str(cfg.get("method", "no_dp"))),
            training_algorithm=cast(
                TrainingAlgorithm,
                str(cfg.get("training-algorithm", "fedavg")),
            ),
            partition_method=cast(
                PartitionMethod,
                str(
                    cfg.get(
                        "partition-method",
                        "label_shards" if cfg.get("non-iid", False) else "iid",
                    )
                ),
            ),
            client_data_mode=cast(
                ClientDataMode, str(cfg.get("client-data-mode", "partitioned_cifar10"))
            ),
            simulate_stragglers=bool(cfg.get("simulate-stragglers", True)),
            dirichlet_alpha=float(cfg.get("dirichlet-alpha", 0.5)),
            validation_fraction=float(cfg.get("validation-fraction", 0.1)),
            local_epochs=int(cfg.get("local-epochs", 1)),
            batch_size=int(cfg.get("batch-size", 32)),
            local_learning_rate=float(cfg.get("local-learning-rate", 0.01)),
            local_momentum=float(cfg.get("local-momentum", 0.9)),
            fraction_fit=float(cfg.get("fraction-fit", 1.0)),
            max_records_per_client=int(cfg.get("max-records-per-client", 0)),
            use_pretrained=bool(cfg.get("use-pretrained", True)),
            cache_dir=str(cfg.get("cache-dir", ".cache/lightdp_flower")),
            output_dir=str(cfg.get("output-dir", "results/experiments")),
            tag=str(cfg.get("tag", "auto")),
            git_track_results=bool(cfg.get("git-track-results", True)),
            device=str(cfg.get("device", "auto")),
            resume=bool(cfg.get("resume", False)),
        )

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()

    @property
    def run_output_dir(self) -> Path:
        output_root = Path(self.output_dir).expanduser().resolve()
        machine_run_dir = output_root / machine_id() / self.tag
        legacy_run_dir = output_root / self.tag
        # Let existing pre-partition runs resume in place on their originating machine.
        if (
            self.resume
            and not (machine_run_dir / "run_config.json").is_file()
            and (legacy_run_dir / "run_config.json").is_file()
        ):
            return legacy_run_dir
        return machine_run_dir

    @property
    def checkpoint_dir(self) -> Path:
        """Keep binary artifacts isolated inside their experiment's output folder."""
        return self.run_output_dir / "checkpoints"

    def validate(self) -> None:
        methods = {"no_dp", "vanilla_dp", "smpc_dp", "lightdp"}
        algorithms = {"fedavg", "clipped_fedavg"}
        partitions = {"iid", "label_shards", "dirichlet"}
        data_modes = {"partitioned_cifar10", "local_npz"}
        if self.method not in methods:
            raise ValueError(f"method must be one of {sorted(methods)}, got {self.method!r}")
        if self.training_algorithm not in algorithms:
            raise ValueError(
                f"training-algorithm must be one of {sorted(algorithms)}, "
                f"got {self.training_algorithm!r}"
            )
        if self.partition_method not in partitions:
            raise ValueError(
                f"partition-method must be one of {sorted(partitions)}, "
                f"got {self.partition_method!r}"
            )
        if self.client_data_mode not in data_modes:
            raise ValueError(
                "client-data-mode must be one of "
                f"{sorted(data_modes)}, got {self.client_data_mode!r}"
            )
        if self.num_clients <= 0:
            raise ValueError("num-clients must be > 0")
        if self.num_server_rounds <= 0:
            raise ValueError("num-server-rounds must be > 0")
        if self.max_colluders < 0 or self.max_stragglers < 0:
            raise ValueError("max-colluders/max-stragglers must be >= 0")
        if (
            self.method != "no_dp"
            and self.num_clients <= self.max_colluders + self.max_stragglers
        ):
            raise ValueError("Need num-clients > max-colluders + max-stragglers")
        if self.client_data_mode == "local_npz" and self.training_algorithm not in {
            "fedavg", "clipped_fedavg"
        }:
            raise ValueError(
                "client-data-mode=local_npz currently supports local-SGD "
                "FedAvg algorithms only"
            )
        if (
            self.method != "no_dp"
            and self.training_algorithm in {"fedavg", "clipped_fedavg"}
            and self.fraction_fit != 1.0
        ):
            raise ValueError(
                "Privacy calibration currently requires fraction-fit=1.0 for FedAvg client updates"
            )
        if self.max_records_per_client < 0:
            raise ValueError("max-records-per-client must be >= 0")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet-alpha must be finite and > 0")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation-fraction must be in [0, 1)")
        if self.validation_fraction == 0:
            raise ValueError("validation-fraction must be > 0 for validation-based selection")
        if self.num_clients > int(50000 * (1.0 - self.validation_fraction)):
            raise ValueError(
                "num-clients cannot exceed the training records remaining "
                "after validation holdout"
            )
        if self.local_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("local-epochs and batch-size must be > 0")
        if not math.isfinite(self.local_learning_rate) or self.local_learning_rate <= 0:
            raise ValueError("local-learning-rate must be finite and > 0")
        if not 0.0 <= self.local_momentum < 1.0:
            raise ValueError("local-momentum must be in [0, 1)")
        if not 0.0 < self.fraction_fit <= 1.0:
            raise ValueError("fraction-fit must be in (0, 1]")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be between 0 and 1")
        if not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite")
        if self.method != "no_dp" and self.epsilon <= 0:
            raise ValueError("epsilon must be > 0 for private methods")
        if not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("clip must be finite and > 0")
        if not math.isfinite(self.client_update_clip) or self.client_update_clip <= 0:
            raise ValueError("client-update-clip must be finite and > 0")
        if self.image_size <= 0 or self.feature_batch <= 0:
            raise ValueError("image-size and feature-batch must both be > 0")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.tag):
            raise ValueError(
                "tag must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
            )
