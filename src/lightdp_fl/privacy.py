from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass, asdict
from typing import Iterator

import numpy as np
import torch
from scipy.optimize import minimize_scalar


def rho_from_eps(eps: float, delta: float) -> float:
    L = math.log(1.0 / delta)
    return (math.sqrt(L + eps) - math.sqrt(L)) ** 2


def eps_from_rho(rho: float, delta: float) -> float:
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def configurations(N: int, Cmax: int, Smax: int) -> Iterator[tuple[int, int, int, int, int]]:
    for c in range(Cmax + 1):
        for s in range(Smax + 1):
            for overlap in range(max(0, c + s - N), min(c, s) + 1):
                h = N - c - s + overlap
                d = s - overlap
                if h > 0:
                    yield c, s, overlap, h, d


def precision_diag(u: float, k: float, h: int, d: int) -> float:
    a = u + (h + d) * k
    return 1.0 / a + k / (a * (u + d * k))


@dataclass(frozen=True)
class Calibration:
    normal_var: float
    u: float
    k: float
    rho_round: float
    eps: float
    steps: int
    sensitivity: float

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(
    eps: float,
    delta: float,
    steps: int,
    sensitivity: float,
    N: int,
    Cmax: int,
    Smax: int,
) -> Calibration:
    """Notebook-equivalent zCDP calibration for all threat states."""
    states = list(configurations(N, Cmax, Smax))
    rho = rho_from_eps(eps, delta) / steps
    base = sensitivity**2 / (2.0 * rho)

    def design(r: float) -> tuple[float, float, float]:
        worst = max(precision_diag(1.0, r, h, d) for _, _, _, h, d in states)
        u = base * worst
        k = r * u
        objective = np.mean([(u + s * k) / (N - s) for s in range(Smax + 1)])
        return float(objective), float(u), float(k)

    opt = minimize_scalar(lambda z: design(math.exp(z))[0], bounds=(-16, 12), method="bounded")
    _, u, k = min(design(0.0), design(math.exp(float(opt.x))), key=lambda z: z[0])
    achieved = max(
        sensitivity**2 * precision_diag(u, k, h, d) / 2.0
        for _, _, _, h, d in states
    )
    if achieved > rho * (1 + 1e-8):
        raise RuntimeError("Privacy calibration failed its internal precision check")
    return Calibration(base, u, k, rho, eps, steps, sensitivity)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def gaussian_vector(length: int, std: float, seed: int, device: torch.device) -> torch.Tensor:
    # Generate on CPU so matching pair seeds produce identical vectors regardless of GPU model.
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(length, generator=g, dtype=torch.float32).mul_(std).to(device)


def add_private_noise(
    gradient: torch.Tensor,
    method: str,
    client_id: int,
    round_zero_based: int,
    N: int,
    Cmax: int,
    Smax: int,
    calibration: Calibration | None,
    base_seed: int,
) -> torch.Tensor:
    """Apply the same four mechanisms as the notebook.

    LightDP uses explicit deterministic pair masks: client i adds +r_ij and client j
    adds -r_ij. Missing uploads therefore leave uncancelled masks, matching the
    covariance mechanism used in the notebook.
    """
    if method == "no_dp":
        return gradient
    if calibration is None:
        raise ValueError("Private mechanism requires calibration")

    out = gradient.clone()
    P = out.numel()
    device = out.device

    if method == "vanilla_dp":
        seed = stable_seed(base_seed, "vanilla", round_zero_based, client_id)
        out.add_(gaussian_vector(P, math.sqrt(calibration.normal_var), seed, device))
        return out

    if method == "smpc_dp":
        # Ideal secure-aggregation baseline from the source notebook. Cryptographic
        # transport is NOT implemented; only its post-aggregation noise behavior.
        hmin = N - Cmax - Smax
        seed = stable_seed(base_seed, "smpc", round_zero_based, client_id)
        out.add_(gaussian_vector(P, math.sqrt(calibration.normal_var / hmin), seed, device))
        return out

    if method == "lightdp":
        individual_seed = stable_seed(base_seed, "lightdp-individual", round_zero_based, client_id)
        out.add_(gaussian_vector(P, math.sqrt(calibration.u), individual_seed, device))
        std = math.sqrt(calibration.k)
        if std > 0:
            for other in range(N):
                if other == client_id:
                    continue
                a, b = sorted((client_id, other))
                seed = stable_seed(base_seed, "lightdp-pair", round_zero_based, a, b)
                mask = gaussian_vector(P, std, seed, device)
                out.add_(mask if client_id == a else -mask)
        return out

    raise ValueError(f"Unknown method: {method}")


def active_client_ids(seed: int, round_zero_based: int, N: int, Smax: int) -> tuple[np.ndarray, int]:
    """Exact straggler schedule used by the source notebook."""
    rng = np.random.default_rng(seed + 1000 + round_zero_based)
    s = int(rng.integers(Smax + 1))
    active = rng.choice(N, N - s, replace=False)
    return np.asarray(active, dtype=np.int64), s
