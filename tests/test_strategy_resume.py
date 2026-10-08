import torch

from lightdp_fl.config import RunConfig
from lightdp_fl.model import HybridClassifier, get_parameters
from lightdp_fl.strategy import GradientMomentumStrategy


def test_server_checkpoint_restores_round_parameters_and_momentum(tmp_path) -> None:
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
    fresh = GradientMomentumStrategy(RunConfig(**common), initial_arrays, None)
    fresh.current.fill_(0.25)
    fresh.velocity.fill_(-0.5)
    fresh._save_checkpoint(server_round=2)

    resumed_config = RunConfig(**common, resume=True)
    resumed = GradientMomentumStrategy(resumed_config, initial_arrays, None)

    assert resumed.resume_round == 2
    torch.testing.assert_close(resumed.current, torch.full_like(resumed.current, 0.25))
    torch.testing.assert_close(resumed.velocity, torch.full_like(resumed.velocity, -0.5))
