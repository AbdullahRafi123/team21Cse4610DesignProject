"""Submit one reproducible experiment to a running Flower SuperLink.

Start the SuperLink and one SuperNode per client before using this command.
The SuperNodes stay connected between experiments; this runner only submits
the ServerApp and streams its progress from the coordinator.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from lightdp_fl.flower_cli import find_flower_cli, run_flower_app, serialize_run_config
from lightdp_fl.config import RunConfig
from lightdp_fl.run_tracking import machine_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse a single networked Flower experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Run one Flower deployment experiment through network-tls."
    )
    parser.add_argument(
        "--ca-cert",
        type=Path,
        required=True,
        help="CA certificate trusted by the coordinator's local Flower CLI.",
    )
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument(
        "--algorithm", choices=("fedavg", "clipped_fedavg"), default="fedavg"
    )
    parser.add_argument(
        "--method",
        choices=("no_dp", "vanilla_dp", "smpc_dp", "lightdp"),
        default="no_dp",
    )
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--partition-method",
        choices=("iid", "label_shards", "dirichlet"),
        default="iid",
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument(
        "--client-data-mode",
        choices=("partitioned_cifar10", "local_npz"),
        default="partitioned_cifar10",
    )
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max-records-per-client", type=int, default=0)
    parser.add_argument("--client-update-clip", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-colluders", type=int, default=1)
    parser.add_argument("--max-stragglers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--tag", default="auto")
    parser.add_argument(
        "--resume-tag",
        help="Resume this interrupted run tag using its recorded configuration.",
    )
    parser.add_argument(
        "--client-log-dir",
        type=Path,
        help="Same-host SuperNode log directory to copy into the completed run output.",
    )
    parser.add_argument("--no-git-track-results", action="store_true")
    return parser.parse_args()


def find_run_dir(tag: str) -> Path:
    """Return this machine's unique experiment directory for a recorded tag."""
    run_dir = PROJECT_ROOT / "results" / "experiments" / machine_id() / tag
    if not (run_dir / "run_config.json").is_file():
        raise SystemExit(f"No recorded run configuration found for tag={tag!r}: {run_dir}")
    return run_dir


def recorded_resume_config(tag: str) -> tuple[str, int]:
    """Recreate the exact saved configuration accepted by the FedAvg checkpoint."""
    metadata = json.loads((find_run_dir(tag) / "run_config.json").read_text(encoding="utf-8"))
    values = {
        field.name.replace("_", "-"): metadata[field.name]
        for field in fields(RunConfig)
        if field.name != "resume" and field.name in metadata
    }
    if len(values) != len(fields(RunConfig)) - 1:
        raise SystemExit("Saved run configuration is incomplete and cannot be resumed safely.")
    values["resume"] = True
    return serialize_run_config(values), int(metadata["num_clients"])


def git_context() -> dict[str, object]:
    """Record repository identity without storing a potentially sensitive diff."""
    def run_git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "revision": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "status": (run_git("status", "--short") or "").splitlines(),
    }


def export_network_artifacts(tag: str, client_log_dir: Path | None) -> None:
    """Preserve same-host client logs and Git provenance beside run records."""
    run_dir = find_run_dir(tag)
    copied_logs = False
    if client_log_dir is not None:
        source = client_log_dir.expanduser().resolve()
        if source.is_dir():
            destination = run_dir / "client_logs"
            shutil.copytree(source, destination, dirs_exist_ok=True)
            copied_logs = True
    export_path = run_dir / "network_run_export.json"
    export_path.write_text(
        json.dumps(
            {
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "tag": tag,
                "client_logs_copied": copied_logs,
                "client_log_source": str(client_log_dir) if client_log_dir else None,
                "git": git_context(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    staged = subprocess.run(
        ["git", "add", "--", str(export_path.relative_to(PROJECT_ROOT))],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if staged.returncode == 0:
        print(f"Exported network logs and Git context: {export_path}", flush=True)


def main() -> None:
    """Validate coordinator inputs and submit a streaming Flower run."""
    args = parse_args()
    ca_cert = args.ca_cert.expanduser().resolve()
    if not ca_cert.is_file():
        raise SystemExit(f"CA certificate does not exist: {ca_cert}")
    if args.clients <= 0:
        raise SystemExit("--clients must be > 0")
    if args.resume_tag:
        if args.tag != "auto":
            raise SystemExit("Use --resume-tag alone; do not also supply --tag.")
        run_config, expected_clients = recorded_resume_config(args.resume_tag)
        if args.clients != 4 and args.clients != expected_clients:
            raise SystemExit(
                f"--clients={args.clients} differs from resumed run clients={expected_clients}"
            )
        run_tag = args.resume_tag
    elif args.method != "no_dp" and args.epsilon <= 0:
        raise SystemExit("--epsilon must be > 0 for a privacy method")
    elif not args.resume_tag:
        run_tag = args.tag
        run_config = serialize_run_config(
            {
                "seed": args.seed,
                "num-clients": args.clients,
                "num-server-rounds": args.rounds,
                "max-colluders": args.max_colluders,
                "max-stragglers": args.max_stragglers,
                "epsilon": args.epsilon,
                "clip": args.clip,
                "client-update-clip": args.client_update_clip,
                "method": args.method,
                "training-algorithm": args.algorithm,
                "partition-method": args.partition_method,
                "dirichlet-alpha": args.dirichlet_alpha,
                "client-data-mode": args.client_data_mode,
                "simulate-stragglers": False,
                "local-epochs": args.local_epochs,
                "batch-size": args.batch_size,
                "local-learning-rate": args.learning_rate,
                "local-momentum": args.momentum,
                "max-records-per-client": args.max_records_per_client,
                "device": args.device,
                "tag": args.tag,
                "git-track-results": not args.no_git_track_results,
            }
        )
    print(
        "Submitting networked Flower run: "
        f"algorithm={args.algorithm}, method={args.method}, "
        f"partition={args.partition_method}, clients={args.clients}, rounds={args.rounds}",
        flush=True,
    )
    run_flower_app(
        "network-tls",
        run_config=run_config,
        federation_config=f"root-certificates={json.dumps(str(ca_cert))}",
        cwd=PROJECT_ROOT,
        cli=find_flower_cli(),
    )
    if run_tag != "auto":
        export_network_artifacts(run_tag, args.client_log_dir)


if __name__ == "__main__":
    main()
