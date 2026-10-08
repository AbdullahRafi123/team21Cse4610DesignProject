from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, cast

Method = Literal["no_dp", "vanilla_dp", "smpc_dp", "lightdp"]

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
    learning_rate: float = 1.0
    momentum: float = 0.9
    image_size: int = 224
    feature_batch: int = 128
    microbatch: int = 32
    method: Method = "lightdp"
    non_iid: bool = False
    max_records_per_client: int = 0
    use_pretrained: bool = True
    cache_dir: str = ".cache/lightdp_flower"
    output_dir: str = "results/experiments"
    tag: str = "lightdp_eps6"
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
            learning_rate=float(cfg.get("learning-rate", 1.0)),
            momentum=float(cfg.get("momentum", 0.9)),
            image_size=int(cfg.get("image-size", 224)),
            feature_batch=int(cfg.get("feature-batch", 128)),
            microbatch=int(cfg.get("microbatch", 32)),
            method=cast(Method, str(cfg.get("method", "lightdp"))),
            non_iid=bool(cfg.get("non-iid", False)),
            max_records_per_client=int(cfg.get("max-records-per-client", 0)),
            use_pretrained=bool(cfg.get("use-pretrained", True)),
            cache_dir=str(cfg.get("cache-dir", ".cache/lightdp_flower")),
            output_dir=str(cfg.get("output-dir", "results/experiments")),
            tag=str(cfg.get("tag", "lightdp_eps6")),
            device=str(cfg.get("device", "auto")),
            resume=bool(cfg.get("resume", False)),
        )

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()

    @property
    def run_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve() / self.tag

    def validate(self) -> None:
        methods = {"no_dp", "vanilla_dp", "smpc_dp", "lightdp"}
        if self.method not in methods:
            raise ValueError(f"method must be one of {sorted(methods)}, got {self.method!r}")
        if self.num_clients <= 0:
            raise ValueError("num-clients must be > 0")
        if self.num_server_rounds <= 0:
            raise ValueError("num-server-rounds must be > 0")
        if self.max_colluders < 0 or self.max_stragglers < 0:
            raise ValueError("max-colluders/max-stragglers must be >= 0")
        if self.num_clients <= self.max_colluders + self.max_stragglers:
            raise ValueError("Need num-clients > max-colluders + max-stragglers")
        if 50000 % self.num_clients != 0:
            raise ValueError("For notebook-equivalent partitions, num-clients must divide 50,000")
        if self.max_records_per_client < 0:
            raise ValueError("max-records-per-client must be >= 0")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be between 0 and 1")
        if not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite")
        if self.method != "no_dp" and self.epsilon <= 0:
            raise ValueError("epsilon must be > 0 for private methods")
        if not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("clip must be finite and > 0")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning-rate must be finite and >= 0")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if self.image_size <= 0 or self.feature_batch <= 0 or self.microbatch <= 0:
            raise ValueError("image-size, feature-batch, and microbatch must all be > 0")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.tag):
            raise ValueError(
                "tag must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
            )
