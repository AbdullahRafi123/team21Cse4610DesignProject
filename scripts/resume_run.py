"""Resume an interrupted Flower run from its recorded configuration."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from lightdp_fl.config import RunConfig
from lightdp_fl.flower_cli import run_flower_app, serialize_run_config
from lightdp_fl.run_tracking import machine_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = {
    field.name: field.name.replace("_", "-")
    for field in fields(RunConfig)
    if field.name != "resume"
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue a saved run using its original settings and checkpoint."
    )
    parser.add_argument("--tag", required=True, help="Run tag/output folder to resume")
    parser.add_argument(
        "--gpu", action="store_true",
        help="Resume with the CUDA simulation profile (requires CUDA-enabled PyTorch).",
    )
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
        run_dir / "checkpoints" / "fedavg_server_checkpoint.pt",
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
    config = serialize_run_config(
        {
            key: metadata[field]
            for field, key in CONFIG_KEYS.items()
            if field in metadata
        }
    )
    config = f"{config} resume=true"
    num_clients = int(metadata["num_clients"])
    federation = "local-simulation-gpu" if args.gpu else "local-simulation"
    print(f"Resuming tag={args.tag} from round checkpoint: {checkpoint}", flush=True)
    run_flower_app(
        federation,
        run_config=config,
        federation_config=f"options.num-supernodes={num_clients}",
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
