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
    if len(labels) < N:
        raise ValueError("The training split must contain at least one record per client")
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(labels))
    if cfg.partition_method == "label_shards":
        order = order[np.argsort(labels.numpy()[order], kind="stable")]
        shards = np.array_split(order, 2 * N)
        rng.shuffle(shards)
        parts = [np.concatenate(shards[2*i:2*i+2]) for i in range(N)]
    elif cfg.partition_method == "dirichlet":
        label_values = labels.numpy()
        parts = [[] for _ in range(N)]
        for label in np.unique(label_values):
            class_ids = np.flatnonzero(label_values == label)
            rng.shuffle(class_ids)
            proportions = rng.dirichlet(np.full(N, cfg.dirichlet_alpha))
            counts = rng.multinomial(len(class_ids), proportions)
            offset = 0
            for client_id, count in enumerate(counts):
                parts[client_id].extend(class_ids[offset:offset + count])
                offset += count
        # Dirichlet draws can leave a client empty. Move seeded samples from
        # the largest partitions so every selected client can train locally.
        for client_id, part in enumerate(parts):
            if not part:
                donor_id = max(range(N), key=lambda candidate: len(parts[candidate]))
                donor = parts[donor_id]
                if len(donor) <= 1:
                    raise ValueError("Dirichlet partition cannot give every client a training record")
                moved = int(rng.integers(len(donor)))
                part.append(donor.pop(moved))
        parts = [np.asarray(part, dtype=np.int64) for part in parts]
    else:
        parts = list(np.array_split(order, N))
    ids = np.asarray(parts[partition_id], dtype=np.int64)
    rng.shuffle(ids)
    if cfg.max_records_per_client > 0:
        ids = ids[: min(cfg.max_records_per_client, len(ids))]
    return ids


def _train_validation_indices(
    cfg: RunConfig, labels: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic, class-stratified holdout from CIFAR-10 training data."""
    rng = np.random.default_rng(cfg.seed + 1)
    label_values = labels.numpy()
    train_ids: list[np.ndarray] = []
    validation_ids: list[np.ndarray] = []
    for label in np.unique(label_values):
        class_ids = np.flatnonzero(label_values == label)
        rng.shuffle(class_ids)
        count = max(1, int(round(len(class_ids) * cfg.validation_fraction)))
        validation_ids.append(class_ids[:count])
        train_ids.append(class_ids[count:])
    training = np.concatenate(train_ids)
    validation = np.concatenate(validation_ids)
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


def partition_indices(cfg: RunConfig, partition_id: int) -> np.ndarray:
    labels = torch.as_tensor(_raw(True, cfg).targets, dtype=torch.long)
    training_ids, _ = _train_validation_indices(cfg, labels)
    local = _partition_indices_from_labels(cfg, partition_id, labels[training_ids])
    return training_ids[local]


def partition_record_counts(cfg: RunConfig) -> list[int]:
    """Return effective record counts after validation holdout and client caps."""
    labels = torch.as_tensor(_raw(True, cfg).targets, dtype=torch.long)
    training_ids, _ = _train_validation_indices(cfg, labels)
    local_labels = labels[training_ids]
    counts = [
        len(_partition_indices_from_labels(cfg, partition_id, local_labels))
        for partition_id in range(cfg.num_clients)
    ]
    if cfg.max_records_per_client > 0:
        counts = [min(count, cfg.max_records_per_client) for count in counts]
    return counts


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
        partition_method=cfg.partition_method, dirichlet_alpha=cfg.dirichlet_alpha,
        validation_fraction=cfg.validation_fraction, max_records=cfg.max_records_per_client,
        image_size=cfg.image_size, pretrained=cfg.use_pretrained, views=2 if split == "train" else 1,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _local_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load one client's private training set from a local NPZ file.

    Accepted image keys are ``x_train`` or ``images``; label keys are
    ``y_train`` or ``labels``. Images must be NHWC RGB arrays and labels must
    be integer class IDs in [0, 9].
    """
    if not path.is_file():
        raise FileNotFoundError(f"Client training data does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        image_key = "x_train" if "x_train" in data else "images"
        label_key = "y_train" if "y_train" in data else "labels"
        if image_key not in data or label_key not in data:
            raise ValueError(
                f"{path} must contain x_train/y_train or images/labels arrays"
            )
        images = np.asarray(data[image_key])
        labels = np.asarray(data[label_key])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Client images must have NHWC RGB shape, got {images.shape}")
    if labels.ndim != 1 or len(images) != len(labels) or len(labels) == 0:
        raise ValueError("Client labels must be a non-empty vector matching the images")
    if (
        not np.issubdtype(labels.dtype, np.integer)
        or labels.min() < 0
        or labels.max() > 9
    ):
        raise ValueError("Client labels must be integer CIFAR-10 class IDs in [0, 9]")
    pixels = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    if pixels.dtype == torch.uint8:
        pixels = pixels.float().div_(255.0)
    else:
        pixels = pixels.float()
        if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("Floating point client images must be finite and scaled to [0, 1]")
    labels_tensor = torch.as_tensor(labels.astype(np.int64, copy=False), dtype=torch.long)
    identity = hashlib.sha256(
        f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}".encode()
    ).hexdigest()[:20]
    return pixels, labels_tensor, identity


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
    cfg: RunConfig,
    partition_id: int,
    device: torch.device,
    progress_callback: Callable[[str, int, int], None] | None = None,
    local_data_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if local_data_path is None:
        path = _cache_file(cfg, "train", partition_id)
        dataset = _raw(True, cfg)
        labels_all = torch.as_tensor(dataset.targets, dtype=torch.long)
        training_ids, _ = _train_validation_indices(cfg, labels_all)
        local_ids = _partition_indices_from_labels(cfg, partition_id, labels_all[training_ids])
        ids = training_ids[local_ids]
        index = torch.as_tensor(ids)
        pixels = torch.from_numpy(dataset.data[ids]).permute(0, 3, 1, 2).float().div_(255.0)
        labels = labels_all[index]
    else:
        pixels, labels, identity = _local_dataset(local_data_path.expanduser())
        cache_name = (
            f"train_local_{identity}_{cfg.image_size}_{int(cfg.use_pretrained)}.pt"
        )
        path = cfg.cache_path / "features" / cache_name
        path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_evaluation_data(
    cfg: RunConfig,
    device: torch.device,
    split: str,
    progress_label: str,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = _cache_file(cfg, split, None)
    if split == "test":
        pixels, labels = all_pixels_labels(False, cfg)
    elif split == "validation":
        dataset = _raw(True, cfg)
        labels_all = torch.as_tensor(dataset.targets, dtype=torch.long)
        _, ids = _train_validation_indices(cfg, labels_all)
        index = torch.as_tensor(ids)
        pixels = torch.from_numpy(dataset.data[ids]).permute(0, 3, 1, 2).float().div_(255.0)
        labels = labels_all[index]
    else:
        raise ValueError(f"Unsupported evaluation split: {split}")

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
            progress_callback(progress_label, batch_index, total_batches)
    features = torch.cat(chunks, dim=0)
    _save_feature_cache(path, features)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pixels, labels, features


def load_validation_data(
    cfg: RunConfig,
    device: torch.device,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _load_evaluation_data(
        cfg, device, "validation", "centralized validation features", progress_callback
    )


def load_test_data(
    cfg: RunConfig,
    device: torch.device,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _load_evaluation_data(
        cfg, device, "test", "final test features", progress_callback
    )
