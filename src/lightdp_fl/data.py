from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, models

from .config import RunConfig

MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def dataset_root(cfg: RunConfig) -> Path:
    root = cfg.cache_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _raw(train: bool, cfg: RunConfig):
    try:
        return datasets.CIFAR10(str(dataset_root(cfg)), train=train, download=True)
    except (OSError, RuntimeError) as exc:
        split = "training" if train else "test"
        raise RuntimeError(
            f"Could not load the CIFAR-10 {split} split. The dataset is not cached completely and "
            "the automatic download failed. Restore internet/DNS access and run again; if an "
            "interrupted download left `.cache/lightdp_flower/data/cifar-10-python.tar.gz`, "
            "remove that incomplete archive and retry."
        ) from exc


def all_pixels_labels(train: bool, cfg: RunConfig) -> tuple[torch.Tensor, torch.Tensor]:
    ds = _raw(train, cfg)
    pixels = torch.tensor(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
    labels = torch.tensor(ds.targets, dtype=torch.long)
    return pixels, labels


def partition_indices(cfg: RunConfig, partition_id: int) -> np.ndarray:
    _, labels = all_pixels_labels(True, cfg)
    N = cfg.num_clients
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(labels))
    if cfg.non_iid:
        order = order[np.argsort(labels.numpy()[order], kind="stable")]
        shards = np.array_split(order, 2 * N)
        rng.shuffle(shards)
        parts = [np.concatenate(shards[2*i:2*i+2]) for i in range(N)]
    else:
        parts = list(order.reshape(N, -1))
    ids = np.asarray(parts[partition_id], dtype=np.int64)
    if cfg.max_records_per_client > 0:
        ids = ids[: min(cfg.max_records_per_client, len(ids))]
    return ids


def _weights(cfg: RunConfig):
    return models.ResNet18_Weights.IMAGENET1K_V1 if cfg.use_pretrained else None


def make_backbone(cfg: RunConfig, device: torch.device) -> nn.Module:
    try:
        backbone = models.resnet18(weights=_weights(cfg))
    except Exception as exc:
        if cfg.use_pretrained:
            raise RuntimeError(
                "Could not load/download pretrained ResNet-18 weights. Connect to the internet once, "
                "or pre-populate the PyTorch cache. Setting use-pretrained=false is only a smoke-test "
                "fallback and does NOT reproduce the source experiment."
            ) from exc
        raise
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


@torch.no_grad()
def public_features(backbone: nn.Module, x: torch.Tensor, image_size: int, device: torch.device) -> torch.Tensor:
    mean = MEAN.to(device)
    std = STD.to(device)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
    return F.normalize(backbone((x - mean) / std), dim=1)


def _feature_key(cfg: RunConfig, split: str, partition_id: int | None) -> str:
    payload = dict(
        split=split, partition_id=partition_id, N=cfg.num_clients, seed=cfg.seed,
        non_iid=cfg.non_iid, max_records=cfg.max_records_per_client,
        image_size=cfg.image_size, pretrained=cfg.use_pretrained, views=2 if split == "train" else 1,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _cache_file(cfg: RunConfig, split: str, partition_id: int | None) -> Path:
    folder = cfg.cache_path / "features"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{split}_{_feature_key(cfg, split, partition_id)}.pt"


def load_client_data(cfg: RunConfig, partition_id: int, device: torch.device):
    path = _cache_file(cfg, "train", partition_id)
    pixels_all, labels_all = all_pixels_labels(True, cfg)
    ids = partition_indices(cfg, partition_id)
    pixels = pixels_all[torch.as_tensor(ids)]
    labels = labels_all[torch.as_tensor(ids)]
    if path.exists():
        cached = torch.load(path, map_location="cpu")
        return pixels, labels, cached["features"]

    backbone = make_backbone(cfg, device)
    views = []
    for flip in (False, True):
        chunks = []
        for start in range(0, len(pixels), cfg.feature_batch):
            x = pixels[start:start+cfg.feature_batch].to(device)
            if flip:
                x = x.flip(-1)
            chunks.append(public_features(backbone, x, cfg.image_size, device).cpu())
        views.append(torch.cat(chunks, dim=0))
    features = torch.stack(views)
    temp = path.with_suffix(".tmp")
    torch.save({"features": features}, temp)
    os.replace(temp, path)
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pixels, labels, features


def load_test_data(cfg: RunConfig, device: torch.device):
    path = _cache_file(cfg, "test", None)
    pixels, labels = all_pixels_labels(False, cfg)
    if path.exists():
        cached = torch.load(path, map_location="cpu")
        return pixels, labels, cached["features"]

    backbone = make_backbone(cfg, device)
    chunks = []
    for start in range(0, len(pixels), cfg.feature_batch):
        x = pixels[start:start+cfg.feature_batch].to(device)
        chunks.append(public_features(backbone, x, cfg.image_size, device).cpu())
    features = torch.cat(chunks, dim=0)
    temp = path.with_suffix(".tmp")
    torch.save({"features": features}, temp)
    os.replace(temp, path)
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pixels, labels, features
