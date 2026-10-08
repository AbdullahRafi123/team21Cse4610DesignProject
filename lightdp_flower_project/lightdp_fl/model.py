from __future__ import annotations
from collections import OrderedDict
from typing import Iterable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class HybridClassifier(nn.Module):
    """Source notebook model: trainable CNN branch + classifier over frozen ResNet features."""
    def __init__(self) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(), nn.Linear(128, 64), nn.SiLU(),
        )
        self.head = nn.Linear(512 + 64, 10)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, pixels: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
        learned = self.branch((pixels - 0.5) / 0.5)
        learned = F.normalize(learned, dim=-1)
        return self.head(torch.cat([public, learned], dim=-1))


def parameter_names_shapes_sizes(model: nn.Module):
    names = [name for name, _ in model.named_parameters()]
    shapes = [tuple(p.shape) for p in model.parameters()]
    sizes = [p.numel() for p in model.parameters()]
    return names, shapes, sizes


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_parameters(model: nn.Module, arrays: Iterable[np.ndarray]) -> None:
    with torch.no_grad():
        for p, arr in zip(model.parameters(), arrays, strict=True):
            p.copy_(torch.as_tensor(arr, device=p.device, dtype=p.dtype))


def arrays_to_flat(arrays: Iterable[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.cat([torch.as_tensor(a, device=device, dtype=torch.float32).reshape(-1) for a in arrays])


def flat_to_arrays(vector: torch.Tensor, model: nn.Module) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    offset = 0
    for p in model.parameters():
        n = p.numel()
        result.append(vector[offset:offset+n].reshape(p.shape).detach().cpu().numpy().copy())
        offset += n
    if offset != vector.numel():
        raise ValueError("Flat vector size does not match model parameters")
    return result
