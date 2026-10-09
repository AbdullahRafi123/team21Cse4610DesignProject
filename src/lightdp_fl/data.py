from __future__ import annotations
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from functools import lru_cache
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, models

from .config import RunConfig

MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


def resolve_device(name: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if name == "auto":
        selected = "cuda" if cuda_available else "cpu"
    else:
        selected = name
    if selected.startswith("cuda") and not cuda_available:
        raise RuntimeError(
            f"CUDA was requested with device={name!r}, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build and check the GPU driver, or use device=auto/cpu. "
            f"torch.version.cuda={torch.version.cuda!r}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}."
        )
    device = torch.device(selected)
    if device.type == "cuda" and torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA was requested, but this process sees no CUDA devices. "
            "Check the NVIDIA driver, CUDA-enabled PyTorch, and Flower/Ray GPU reservations."
        )
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {device.index} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )
    visible_gpus = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    print(
        f"CUDA check: available={cuda_available}; visible_devices={visible_gpus}; "
        f"selected_device={device}",
        flush=True,
    )
    return device


def dataset_root(cfg: RunConfig) -> Path:
    root = cfg.cache_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _raw(train: bool, cfg: RunConfig) -> datasets.CIFAR10:
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
    pixels = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
    labels = torch.as_tensor(ds.targets, dtype=torch.long)
    return pixels, labels


def _partition_indices_from_labels(
    cfg: RunConfig, partition_id: int, labels: torch.Tensor
) -> np.ndarray:
    if not 0 <= partition_id < cfg.num_clients:
        raise ValueError(f"partition-id must be in [0, {cfg.num_clients}), got {partition_id}")
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


def partition_indices(cfg: RunConfig, partition_id: int) -> np.ndarray:
    labels = torch.as_tensor(_raw(True, cfg).targets, dtype=torch.long)
    return _partition_indices_from_labels(cfg, partition_id, labels)


def _weights(cfg: RunConfig) -> models.ResNet18_Weights | None:
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


@lru_cache(maxsize=8)
def _normalization(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return MEAN.to(device), STD.to(device)


@torch.no_grad()
def public_features(backbone: nn.Module, x: torch.Tensor, image_size: int) -> torch.Tensor:
    mean, std = _normalization(x.device)
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


def _save_feature_cache(path: Path, features: torch.Tensor) -> None:
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        torch.save({"features": features}, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_client_data(
    cfg: RunConfig, partition_id: int, device: torch.device
    , progress_callback: Callable[[str, int, int], None] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = _cache_file(cfg, "train", partition_id)
    dataset = _raw(True, cfg)
    labels_all = torch.as_tensor(dataset.targets, dtype=torch.long)
    ids = _partition_indices_from_labels(cfg, partition_id, labels_all)
    index = torch.as_tensor(ids)
    pixels = torch.from_numpy(dataset.data[ids]).permute(0, 3, 1, 2).float().div_(255.0)
    labels = labels_all[index]
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        return pixels, labels, cached["features"]

    backbone = make_backbone(cfg, device)
    views = []
    total_batches = 2 * ((len(pixels) + cfg.feature_batch - 1) // cfg.feature_batch)
    completed_batches = 0
    for flip_index, flip in enumerate((False, True), start=1):
        chunks = []
        for start in range(0, len(pixels), cfg.feature_batch):
            x = pixels[start:start+cfg.feature_batch].to(device)
            if flip:
                x = x.flip(-1)
            chunks.append(public_features(backbone, x, cfg.image_size).cpu())
            completed_batches += 1
            if progress_callback is not None:
                progress_callback(
                    f"client {partition_id + 1}/{cfg.num_clients}, view {flip_index}/2",
                    completed_batches,
                    total_batches,
                )
        views.append(torch.cat(chunks, dim=0))
    features = torch.stack(views)
    _save_feature_cache(path, features)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pixels, labels, features


def load_test_data(
    cfg: RunConfig, device: torch.device
    , progress_callback: Callable[[str, int, int], None] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = _cache_file(cfg, "test", None)
    pixels, labels = all_pixels_labels(False, cfg)
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        return pixels, labels, cached["features"]

    backbone = make_backbone(cfg, device)
    chunks = []
    total_batches = (len(pixels) + cfg.feature_batch - 1) // cfg.feature_batch
    for batch_index, start in enumerate(range(0, len(pixels), cfg.feature_batch), start=1):
        x = pixels[start:start+cfg.feature_batch].to(device)
        chunks.append(public_features(backbone, x, cfg.image_size).cpu())
        if progress_callback is not None:
            progress_callback("centralized test features", batch_index, total_batches)
    features = torch.cat(chunks, dim=0)
    _save_feature_cache(path, features)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pixels, labels, features
