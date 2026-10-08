"""Resume an interrupted Flower run from its recorded configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from lightdp_fl.run_tracking import machine_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = {
    "seed": "seed",
    "num_clients": "num-clients",
    "num_server_rounds": "num-server-rounds",
    "max_colluders": "max-colluders",
    "max_stragglers": "max-stragglers",
    "epsilon": "epsilon",
    "delta": "delta",
    "clip": "clip",
    "learning_rate": "learning-rate",
    "momentum": "momentum",
    "image_size": "image-size",
    "feature_batch": "feature-batch",
    "microbatch": "microbatch",
    "method": "method",
    "non_iid": "non-iid",
    "max_records_per_client": "max-records-per-client",
    "use_pretrained": "use-pretrained",
    "cache_dir": "cache-dir",
    "output_dir": "output-dir",
    "tag": "tag",
    "device": "device",
}


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    raise TypeError(f"Unsupported run-config value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue a saved run using its original settings and checkpoint."
    )
    parser.add_argument("--tag", required=True, help="Run tag/output folder to resume")
    args = parser.parse_args()

    experiments_dir = PROJECT_ROOT / "results" / "experiments"
    run_dir = (experiments_dir / machine_id() / args.tag).resolve()
    if not (run_dir / "run_config.json").is_file():
        # Preserve resume access to pre-machine-partition runs on their source machine.
        legacy_dir = (experiments_dir / args.tag).resolve()
        if (legacy_dir / "run_config.json").is_file():
            run_dir = legacy_dir
    metadata_path = run_dir / "run_config.json"
    checkpoints = (
        run_dir / "checkpoints" / "server_checkpoint.pt",
        run_dir / "server_checkpoint.pt",  # legacy layout
    )
    if not metadata_path.is_file():
        raise SystemExit(f"No run metadata found at {metadata_path}")
    checkpoint = next((path for path in checkpoints if path.is_file()), None)
    if checkpoint is None:
        raise SystemExit(
            f"No resumable server checkpoint found in {run_dir / 'checkpoints'}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    current_machine_id = machine_id()
    if metadata.get("machine_id") != current_machine_id:
        raise SystemExit(
            "Runs are machine-local and cannot continue on another machine. "
            f"Recorded machine_id={metadata.get('machine_id')!r}; "
            f"current machine_id={current_machine_id!r}. Start a new run here."
        )
    config = " ".join(
        f"{key}={_toml_value(metadata[field])}"
        for field, key in CONFIG_KEYS.items()
        if field in metadata
    )
    config += " resume=true"
    num_clients = int(metadata["num_clients"])
    command = [
        "flwr", "run", ".", "local-simulation", "--stream",
        "--run-config", config,
        "--federation-config", f"options.num-supernodes={num_clients}",
    ]
    print(f"Resuming tag={args.tag} from round checkpoint: {checkpoint}", flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
