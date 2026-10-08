"""Machine-local run numbering and portable machine identity."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import re
import socket
import time
import uuid


def machine_id() -> str:
    host = re.sub(r"[^A-Za-z0-9]+", "-", socket.gethostname()).strip("-").lower()
    host = host or "unknown-host"
    system = re.sub(r"[^A-Za-z0-9]+", "-", platform.system()).strip("-").lower()
    identity_file = _counter_path().with_name("machine_id.txt")
    identity_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(identity_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        token = ""
        for _ in range(100):
            token = identity_file.read_text(encoding="utf-8").strip()
            if token:
                break
            time.sleep(0.01)
        if not token:
            raise RuntimeError(f"Machine identity file is empty: {identity_file}")
    else:
        token = uuid.uuid4().hex[:8]
        with os.fdopen(descriptor, "w", encoding="utf-8") as identity:
            identity.write(f"{token}\n")
    return f"{host}-{system}-{platform.machine().lower()}-{token}"


def _counter_path() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path.home() / ".cache"
    return base / "lightdp_flower" / "run_sequence.txt"


def _next_run_number(path: Path) -> int:
    """Increment a local sequence, using an exclusive lock for concurrent launches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    deadline = time.monotonic() + 30
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for run-number lock: {lock}")
            time.sleep(0.05)

    try:
        try:
            previous = int(path.read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            previous = 0
        number = previous + 1
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(f"{number}\n", encoding="utf-8")
        os.replace(temporary, path)
        return number
    finally:
        lock.unlink(missing_ok=True)


def new_run_identity() -> dict[str, str | int]:
    """Return a UTC ID and machine-local sequence number for a new experiment."""
    current_machine_id = machine_id()
    run_number = _next_run_number(_counter_path())
    started = datetime.now(timezone.utc)
    run_id = (
        f"{started:%Y%m%dT%H%M%SZ}_{current_machine_id}_run{run_number:04d}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    return {
        "run_id": run_id,
        "run_number": run_number,
        "machine_id": current_machine_id,
        "hostname": socket.gethostname(),
    }


def machine_identity() -> dict[str, str]:
    return {
        "machine_id": machine_id(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine_architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
