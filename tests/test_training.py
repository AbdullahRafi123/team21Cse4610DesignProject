import torch

from lightdp_fl.model import HybridClassifier
from lightdp_fl.training import _functional_helpers, clipped_client_gradient


def _concatenated_reference(
    model: HybridClassifier,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    clip: float,
    microbatch: int,
) -> torch.Tensor:
    """Reference implementation retaining the original full-gradient concatenation."""
    names, buffers, _, per_example = _functional_helpers(model)
    params = dict(model.named_parameters())
    total = torch.zeros(sum(param.numel() for param in params.values()))

    for start in range(0, len(labels), microbatch):
        x = pixels[start : start + microbatch]
        feat = features[0, start : start + microbatch]
        y = labels[start : start + microbatch]
        gradients = per_example(params, buffers, x, feat, y)
        flat = torch.cat([gradients[name].flatten(1) for name in names], dim=1)
        norms = flat.norm(dim=1).clamp_min(1e-12)
        factors = (clip / norms).clamp(max=1.0)
        total.add_((flat * factors[:, None]).sum(dim=0).detach())

    return total / len(labels)


def test_layerwise_clipping_matches_concatenated_reference() -> None:
    torch.manual_seed(23)
    model = HybridClassifier()
    pixels = torch.rand(5, 3, 32, 32)
    labels = torch.tensor([0, 2, 4, 6, 8])
    features = torch.randn(2, len(labels), 512)

    actual = clipped_client_gradient(
        model,
        pixels,
        labels,
        features,
        round_zero_based=0,
        clip=0.05,
        microbatch=2,
        device=torch.device("cpu"),
    )
    expected = _concatenated_reference(model, pixels, labels, features, 0.05, 2)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
