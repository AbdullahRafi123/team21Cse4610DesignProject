from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "results" / "experiments"

RUNS=[
    ("no_dp", 6.0, "No_DP"),
    *[(method,eps,f"{tag}_eps{int(eps)}") for eps in (3.0,6.0,9.0) for method,tag in [
        ("vanilla_dp","Vanilla_local_noise_adding"),("smpc_dp","SMPC_DP"),("lightdp","LightDP")
    ]],
]


def verify_completed_run(tag: str, rounds: int) -> None:
    """Reject Flower's zero-exit startup errors and incomplete runs."""
    run_dirs = list(RUNS_DIR.glob(f"*/{tag}"))
    if len(run_dirs) != 1:
        raise RuntimeError(
            f"Expected one machine-partitioned output directory for {tag}, found {len(run_dirs)}"
        )

    run_dir = run_dirs[0]
    config_path = run_dir / "run_config.json"
    history_path = run_dir / "history.csv"
    train_rounds_path = run_dir / "train_rounds.csv"
    log_path = run_dir / "training.log"
    if not all(
        path.is_file()
        for path in (config_path, history_path, train_rounds_path, log_path)
    ):
        raise RuntimeError(
            f"Flower returned without creating all required run records under {run_dir}; "
            "missing train_rounds.csv means no client aggregation was recorded"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    with history_path.open(newline="", encoding="utf-8") as history_file:
        completed_rounds = [int(row["round"]) for row in csv.DictReader(history_file)]
    with train_rounds_path.open(newline="", encoding="utf-8") as train_file:
        train_rows = list(csv.DictReader(train_file))
    expected_rounds = list(range(rounds + 1))
    machine_id = config.get("machine_id")
    if (
        config.get("tag") != tag
        or config.get("num_server_rounds") != rounds
        or not machine_id
        or machine_id != run_dir.parent.name
    ):
        raise RuntimeError(f"Run configuration does not match the requested run in {run_dir}")
    if completed_rounds != expected_rounds:
        last_round = completed_rounds[-1] if completed_rounds else "none"
        raise RuntimeError(
            f"Run {tag} stopped before all rounds completed: last logged round={last_round}, "
            f"expected={rounds}"
        )
    if len(train_rows) != rounds:
        raise RuntimeError(
            f"Run {tag} recorded {len(train_rows)} successful aggregation rounds; expected {rounds}"
        )
    failures = sum(int(row["failures"]) for row in train_rows)
    active_clients = min(int(row["active_clients"]) for row in train_rows)
    if failures or active_clients <= 0:
        raise RuntimeError(
            f"Run {tag} contains failed client updates (failures={failures}, "
            f"minimum active clients per round={active_clients})"
        )

def main():
    p=argparse.ArgumentParser(description="Run the main Flower experiments matching the source notebook.")
    p.add_argument("--rounds",type=int,default=8,help="Executed source results used 8 rounds.")
    p.add_argument(
        "--epsilon", type=float, choices=(3.0, 6.0, 9.0),
        help="Run the no-DP baseline and the three private methods at this epsilon.",
    )
    p.add_argument("--run-id", help="Optional output prefix. Defaults to the current UTC timestamp.")
    p.add_argument("--gpu", action="store_true", help="Use the CUDA simulation profile (requires CUDA-enabled PyTorch).")
    p.add_argument("--stream",action="store_true",default=True)
    args=p.parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    runs = RUNS if args.epsilon is None else [
        RUNS[0], *(run for run in RUNS[1:] if run[1] == args.epsilon)
    ]
    federation = "local-simulation-gpu" if args.gpu else "local-simulation-suite"
    device = "cuda" if args.gpu else "auto"
    # Keep the venv path lexical: resolving its Python symlink can jump back to
    # the base Conda interpreter and select that environment's Flower tools.
    flwr = Path(sys.executable).absolute().with_name("flwr")
    if not flwr.is_file():
        found_flwr = shutil.which("flwr")
        if found_flwr is None:
            p.error(
                f"Could not find the Flower CLI for {sys.executable}. Activate the project "
                "environment or install the project before running this script."
            )
        flwr = Path(found_flwr)

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PATH"] = f"{flwr.parent}{os.pathsep}{child_env.get('PATH', '')}"
    for index, (method,eps,tag) in enumerate(runs, start=1):
        cfg=(f'method="{method}" epsilon={eps} tag="{run_id}_{tag}" num-clients=50 num-server-rounds={args.rounds} '
             f'max-colluders=10 max-stragglers=10 max-records-per-client=0 microbatch=32 non-iid=false device="{device}"')
        # Avoid CLI federation overrides: Flower currently replaces the full
        # nested `options` table, which drops backend GPU reservations.
        cmd=[str(flwr),"run",".",federation,"--run-config",cfg]
        if args.stream: cmd.append("--stream")
        started = time.monotonic()
        print(
            f"\nRUN {index}/{len(runs)} START: method={method}, epsilon={eps:g}, "
            f"rounds={args.rounds}, federation={federation}, tag={run_id}_{tag}",
            flush=True,
        )
        try:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=child_env)
            verify_completed_run(f"{run_id}_{tag}", args.rounds)
        except subprocess.CalledProcessError as exc:
            print(
                f"RUN {index}/{len(runs)} FAILED after {time.monotonic() - started:.1f}s "
                f"(exit={exc.returncode}); stopping suite.",
                flush=True,
            )
            raise
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            print(
                f"RUN {index}/{len(runs)} FAILED validation after "
                f"{time.monotonic() - started:.1f}s: {exc}",
                flush=True,
            )
            raise
        print(
            f"RUN {index}/{len(runs)} COMPLETE in {time.monotonic() - started:.1f}s: "
            f"{run_id}_{tag}",
            flush=True,
        )
    print("Collecting results for completed runs...", flush=True)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/collect_results.py")],
        check=True,
        cwd=PROJECT_ROOT,
        env=child_env,
    )

if __name__=="__main__": main()
