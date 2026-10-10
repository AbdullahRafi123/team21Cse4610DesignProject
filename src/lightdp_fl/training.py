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


def train_fedavg_client(
    model: HybridClassifier,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    cfg: RunConfig,
    partition_id: int,
    server_round: int,
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    """Train locally with SGD as in FedAvg and return client-side measurements."""
    started = time.monotonic()
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.local_learning_rate,
        momentum=cfg.local_momentum,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed + partition_id * 1_000_003 + server_round * 9_176)
    loss_sum = torch.zeros((), device=device)
    example_count = 0
    batches_per_epoch = math.ceil(len(labels) / cfg.batch_size)
    total_batches = cfg.local_epochs * batches_per_epoch
    completed_batches = 0

    for _ in range(cfg.local_epochs):
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(order), cfg.batch_size):
            ids = order[start:start + cfg.batch_size]
            x = pixels[ids].to(device)
            y = labels[ids].to(device)
            flips = torch.rand(len(ids), generator=generator) < 0.5
            x = torch.where(flips.to(device)[:, None, None, None], x.flip(-1), x)
            view_ids = flips.long()
            feat = features[view_ids, ids].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, feat)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum.add_(loss.detach() * len(ids))
            example_count += len(ids)
            completed_batches += 1
            if progress_callback is not None:
                progress_callback(completed_batches, total_batches)

    return {
        "train_loss": float((loss_sum / max(example_count, 1)).item()),
        "local_seconds": time.monotonic() - started,
    }


def train_clipped_fedavg_client(
    model: HybridClassifier,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    cfg: RunConfig,
    partition_id: int,
    server_round: int,
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    """Run local SGD with per-example gradient clipping at every minibatch."""
    started = time.monotonic()
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.local_learning_rate, momentum=cfg.local_momentum
    )
    names, buffers, _, per_example = _functional_helpers(model)
    parameters = dict(model.named_parameters())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed + partition_id * 1_000_003 + server_round * 9_176)
    batches_per_epoch = math.ceil(len(labels) / cfg.batch_size)
    total_batches = cfg.local_epochs * batches_per_epoch
    loss_sum = torch.zeros((), device=device)
    example_count = 0
    completed_batches = 0

    for _ in range(cfg.local_epochs):
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(order), cfg.batch_size):
            ids = order[start:start + cfg.batch_size]
            x = pixels[ids].to(device)
            y = labels[ids].to(device)
            flips = torch.rand(len(ids), generator=generator) < 0.5
            x = torch.where(flips.to(device)[:, None, None, None], x.flip(-1), x)
            feat = features[flips.long(), ids].to(device)

            grads = per_example(parameters, buffers, x, feat, y)
            norm_sq = torch.zeros(len(y), device=device)
            flat_grads = {name: grads[name].flatten(1) for name in names}
            for grad_part in flat_grads.values():
                norm_sq.add_(grad_part.square().sum(dim=1))
            factors = (cfg.clip / norm_sq.sqrt().clamp_min(1e-12)).clamp(max=1.0)

            optimizer.zero_grad(set_to_none=True)
            for name in names:
                grad_part = grads[name]
                view_shape = (len(y),) + (1,) * (grad_part.ndim - 1)
                parameters[name].grad = (
                    grad_part * factors.reshape(view_shape)
                ).mean(dim=0).detach()
            with torch.no_grad():
                loss = F.cross_entropy(model(x, feat), y)
            optimizer.step()

            loss_sum.add_(loss.detach() * len(ids))
            example_count += len(ids)
            completed_batches += 1
            if progress_callback is not None:
                progress_callback(completed_batches, total_batches)

    return {
        "train_loss": float((loss_sum / max(example_count, 1)).item()),
        "local_seconds": time.monotonic() - started,
    }


def train_private_fedavg_client(
    model: HybridClassifier,
    initial_parameters: list[np.ndarray],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    cfg: RunConfig,
    partition_id: int,
    server_round: int,
    calibration: Calibration,
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """Train locally, clip the client model delta, and apply the selected mechanism.

    The return value is a protected delta, not a full model. The privacy-aware
    server strategy uniformly averages these client deltas and adds the result
    to the round's starting global model.
    """
    train_function = (
        train_clipped_fedavg_client
        if cfg.training_algorithm == "clipped_fedavg"
        else train_fedavg_client
    )
    metrics = train_function(
        model,
        pixels,
        labels,
        features,
        cfg,
        partition_id,
        server_round,
        device,
        progress_callback=progress_callback,
    )
    updated_parameters = get_parameters(model)
    initial = torch.cat([
        torch.as_tensor(array, device=device, dtype=torch.float32).reshape(-1)
        for array in initial_parameters
    ])
    updated = torch.cat([
        torch.as_tensor(array, device=device, dtype=torch.float32).reshape(-1)
        for array in updated_parameters
    ])
    delta = updated - initial
    update_norm = float(torch.linalg.vector_norm(delta).item())
    scale = min(1.0, cfg.client_update_clip / max(update_norm, 1e-12))
    delta.mul_(scale)

    noise_started = time.monotonic()
    protected = add_private_noise(
        delta,
        cfg.method,
        partition_id,
        server_round - 1,
        cfg.num_clients,
        cfg.max_colluders,
        cfg.max_stragglers,
        calibration,
        cfg.seed + 33,
    )
    noise_seconds = time.monotonic() - noise_started
    active, stragglers = active_client_ids(
        cfg.seed, server_round - 1, cfg.num_clients, cfg.max_stragglers
    )
    is_active = int(
        not cfg.simulate_stragglers or np.any(active == partition_id)
    )
    arrays = flat_to_arrays(protected, model)
    metrics.update({
        "active": is_active,
        "stragglers": stragglers if cfg.simulate_stragglers else 0,
        "update_norm": update_norm,
        "update_clipped": int(scale < 1.0),
        "noise_seconds": float(noise_seconds),
    })
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
    correct = torch.zeros((), dtype=torch.long, device=device)
    total_loss = torch.zeros((), device=device)
    n = len(labels)
    for start in range(0, n, batch_size):
        x = pixels[start:start+batch_size].to(device)
        f = features[start:start+batch_size].to(device)
        y = labels[start:start+batch_size].to(device)
        logits = model(x, f)
        total_loss.add_(F.cross_entropy(logits, y, reduction="sum"))
        correct.add_(logits.argmax(1).eq(y).sum())
    return (
        float((total_loss / n).item()),
        100.0 * float((correct / n).item()),
    )
