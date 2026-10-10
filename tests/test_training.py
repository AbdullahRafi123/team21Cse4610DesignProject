from __future__ import annotations

import torch

from lightdp_fl.config import RunConfig
from lightdp_fl.model import HybridClassifier, get_parameters
from lightdp_fl.training import train_clipped_fedavg_client


def test_clipped_fedavg_trains_locally_and_reports_progress() -> None:
    torch.manual_seed(23)
    model = HybridClassifier()
    initial = get_parameters(model)
    cfg = RunConfig(
        num_clients=1,
        max_colluders=0,
        max_stragglers=0,
        local_epochs=1,
        batch_size=2,
        local_learning_rate=0.01,
        clip=1e-6,
    )
    pixels = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([1, 4])
    features = torch.randn(2, 2, 512)
    progress: list[tuple[int, int]] = []

    metrics = train_clipped_fedavg_client(
        model,
        pixels,
        labels,
        features,
        cfg,
        partition_id=0,
        server_round=1,
        device=torch.device("cpu"),
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    updated = get_parameters(model)
    assert any(not torch.equal(torch.from_numpy(before), torch.from_numpy(after))
               for before, after in zip(initial, updated, strict=True))
    assert progress == [(1, 1)]
    assert metrics["train_loss"] > 0
