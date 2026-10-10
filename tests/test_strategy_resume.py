import torch

from lightdp_fl.config import RunConfig
from lightdp_fl.model import HybridClassifier, get_parameters
from flwr.common import ndarrays_to_parameters

from lightdp_fl.strategy import StandardFedAvgStrategy


def test_fedavg_checkpoint_restores_round_and_parameters(tmp_path) -> None:
    common = {
        "num_clients": 5,
        "max_colluders": 1,
        "max_stragglers": 1,
        "num_server_rounds": 4,
        "method": "no_dp",
        "output_dir": str(tmp_path),
        "tag": "resume_check",
    }
    initial_arrays = get_parameters(HybridClassifier())
    fresh = StandardFedAvgStrategy(RunConfig(**common), initial_arrays)
    expected = [array.copy() for array in initial_arrays]
    for array in expected:
        array.fill(0.25)
    fresh._save_checkpoint(2, ndarrays_to_parameters(expected))

    resumed_config = RunConfig(**common, resume=True)
    resumed = StandardFedAvgStrategy(resumed_config, initial_arrays)

    assert resumed.resume_round == 2
    checkpoint = torch.load(resumed.checkpoint_path, map_location="cpu", weights_only=True)
    restored = [tensor.numpy() for tensor in checkpoint["parameters"]]
    for actual, wanted in zip(restored, expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(wanted))
