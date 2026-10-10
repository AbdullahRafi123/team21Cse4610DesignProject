from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


def feature_progress_callback(scope: str) -> Callable[[str, int, int], None]:
    """Print occasional feature-preparation updates without logging every batch."""
    last_reported = 0

    def report(stage: str, completed: int, total: int) -> None:
        nonlocal last_reported
        interval = max(1, total // 4)
        if completed == 1 or completed - last_reported >= interval or completed == total:
            print(
                f"[setup] {scope}: {stage} batches={completed}/{total}",
                flush=True,
            )
            last_reported = completed

    return report


def write_client_progress(
    progress_dir: Path,
    round_id: int,
    client_id: int,
    completed: int,
    total: int,
    status: str = "training",
) -> None:
    """Atomically publish the latest microbatch state for one client."""
    client_dir = progress_dir / f"round_{round_id:04d}"
    client_dir.mkdir(parents=True, exist_ok=True)
    path = client_dir / f"client_{client_id:04d}.json"
    payload = {
        "round": round_id,
        "client": client_id,
        "completed_microbatches": completed,
        "total_microbatches": total,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def start_progress_monitor(
    progress_dir: Path,
    training_log: Path,
    num_clients: int,
    num_rounds: int,
) -> Callable[[], None]:
    """Print a live aggregate microbatch bar and periodically retain it in the log."""
    stop = threading.Event()

    def monitor() -> None:
        last_logged = 0.0
        while not stop.wait(2.0):
            round_dirs = sorted(progress_dir.glob("round_*"))
            if not round_dirs:
                continue
            round_id = int(round_dirs[-1].name.removeprefix("round_"))
            if round_id > num_rounds:
                continue
            completed_clients = 0
            total_clients = 0
            completed_batches = 0
            total_batches = 0
            for path in round_dirs[-1].glob("client_*.json"):
                try:
                    state = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                total_clients += 1
                done = int(state.get("completed_microbatches", 0))
                batches = int(state.get("total_microbatches", 0))
                completed_batches += done
                total_batches += batches
                if state.get("status") == "complete":
                    completed_clients += 1
            if total_batches == 0:
                continue
            percent = completed_batches / total_batches
            width = 24
            filled = min(width, math.floor(percent * width))
            bar = "#" * filled + "-" * (width - filled)
            message = (
                f"[clients] round {round_id}/{num_rounds} [{bar}] {percent:6.1%} "
                f"microbatches={completed_batches}/{total_batches} "
                f"clients_done={completed_clients}/{num_clients} "
                f"clients_started={total_clients}/{num_clients}"
            )
            print(message, flush=True)
            now = time.monotonic()
            if now - last_logged >= 10.0:
                with training_log.open("a", encoding="utf-8") as log:
                    log.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
                last_logged = now

    thread = threading.Thread(
        target=monitor, name="client-progress-monitor", daemon=True
    )
    thread.start()

    def stop_monitor() -> None:
        stop.set()
        thread.join()

    return stop_monitor
