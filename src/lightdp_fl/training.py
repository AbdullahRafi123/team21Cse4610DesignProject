from __future__ import annotations
import math
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .config import RunConfig
from .model import HybridClassifier, get_parameters, set_parameters, flat_to_arrays
from .privacy import Calibration, add_private_noise, active_client_ids

GradientHelpers = tuple[
    list[str],
    dict[str, torch.Tensor],
    Callable[..., torch.Tensor],
    Callable[..., dict[str, torch.Tensor]],
]


def _functional_helpers(
    model: HybridClassifier, compile_mode: str | None = None
) -> GradientHelpers:
    names = [k for k, _ in model.named_parameters()]
    buffers = dict(model.named_buffers())

    def one_loss(params, buffers_, x, feat, y):
        logits = functional_call(model, (params, buffers_), (x[None], feat[None]))
        return F.cross_entropy(logits, y[None])

    per_example = vmap(grad(one_loss), in_dims=(None, None, 0, 0, 0), randomness="error")
    if compile_mode is not None:
        per_example = torch.compile(per_example, mode=compile_mode)
    return names, buffers, one_loss, per_example


def clipped_client_gradient(
    model: HybridClassifier,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    round_zero_based: int,
    clip: float,
    microbatch: int,
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None = None,
    gradient_helpers: GradientHelpers | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    names, buffers, _, per_example = gradient_helpers or _functional_helpers(model)
    params = {k: v for k, v in model.named_parameters()}
    totals = [torch.zeros_like(params[name], device=device) for name in names]
    view = round_zero_based % 2

    total_microbatches = math.ceil(len(labels) / microbatch)
    for batch_index, start in enumerate(range(0, len(labels), microbatch), start=1):
        x = pixels[start:start+microbatch].to(device)
        if view:
            x = x.flip(-1)
        feat = features[view, start:start+microbatch].to(device)
        y = labels[start:start+microbatch].to(device)
        if autocast_dtype is None:
            grads = per_example(params, buffers, x, feat, y)
        else:
            with torch.autocast(device.type, dtype=autocast_dtype):
                grads = per_example(params, buffers, x, feat, y)
        # Compute each example's global norm without materializing a [batch, P]
        # concatenation of all parameter gradients.
        norm_sq = torch.zeros(len(y), device=device)
        flat_grads = [grads[name].flatten(1) for name in names]
        for grad_part in flat_grads:
            norm_sq.add_(grad_part.square().sum(dim=1))
        norm = norm_sq.sqrt().clamp_min(1e-12)
        factors = (clip / norm).clamp(max=1.0)
        for total, grad_part in zip(totals, flat_grads, strict=True):
            total.add_((grad_part * factors[:, None]).sum(dim=0).reshape_as(total).detach())
        if progress_callback is not None:
            progress_callback(batch_index, total_microbatches)
    return torch.cat([total.reshape(-1) for total in totals]) / len(labels)


def compute_client_upload(
    model: HybridClassifier,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    cfg: RunConfig,
    partition_id: int,
    round_zero_based: int,
    calibration: Calibration | None,
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    started = time.monotonic()
    clean = clipped_client_gradient(
        model, pixels, labels, features, round_zero_based,
        cfg.clip, cfg.microbatch, device,
        progress_callback=progress_callback,
    )
    local_seconds = time.monotonic() - started

    noise_started = time.monotonic()
    protected = add_private_noise(
        clean, cfg.method, partition_id, round_zero_based,
        cfg.num_clients, cfg.max_colluders, cfg.max_stragglers,
        calibration, cfg.seed + 33,
    )
    noise_seconds = time.monotonic() - noise_started

    active, s = active_client_ids(cfg.seed, round_zero_based, cfg.num_clients, cfg.max_stragglers)
    is_active = int(bool(np.any(active == partition_id)))
    # We return the computed vector even for simulated stragglers. The server ignores it.
    # This preserves the notebook's "compute first, miss upload deadline second" timing semantics.
    arrays = flat_to_arrays(protected, model)
    metrics: dict[str, float | int] = {
        "active": is_active,
        "stragglers": s,
        "local_seconds": float(local_seconds),
        "noise_seconds": float(noise_seconds),
    }
    return arrays, metrics


@torch.inference_mode()
def evaluate_global(
    model: HybridClassifier,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total_loss = 0.0
    n = len(labels)
    for start in range(0, n, batch_size):
        x = pixels[start:start+batch_size].to(device)
        f = features[start:start+batch_size].to(device)
        y = labels[start:start+batch_size].to(device)
        logits = model(x, f)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += logits.argmax(1).eq(y).sum().item()
    return total_loss / n, 100.0 * correct / n
