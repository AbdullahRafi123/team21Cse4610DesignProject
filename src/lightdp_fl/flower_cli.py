"""Shared helpers for invoking Flower from project command-line scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections.abc import Mapping


def find_flower_cli() -> Path:
    """Find the Flower executable belonging to the active Python environment."""
    executable = Path(sys.executable).absolute().with_name("flwr")
    if executable.is_file():
        return executable
    found = shutil.which("flwr")
    if found is not None:
        return Path(found)
    raise RuntimeError(
        f"Could not find the Flower CLI for {sys.executable}. Activate the project "
        "environment or install the project before running this script."
    )


def toml_value(value: object) -> str:
    """Serialize scalar Flower run-config values using TOML syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    raise TypeError(f"Unsupported run-config value: {value!r}")


def serialize_run_config(values: Mapping[str, object]) -> str:
    """Serialize a mapping into Flower's space-separated run-config syntax."""
    return " ".join(f"{key}={toml_value(value)}" for key, value in values.items())


def flower_environment(cli: Path | None = None) -> dict[str, str]:
    """Return an environment that lets Flower find its sibling executables."""
    executable = cli or find_flower_cli()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PATH"] = os.pathsep.join(
        part for part in (str(executable.parent), environment.get("PATH", "")) if part
    )
    return environment


def flower_run_command(
    federation: str,
    *,
    run_config: str | None = None,
    federation_config: str | None = None,
    stream: bool = True,
    cli: Path | None = None,
) -> list[str]:
    """Build a Flower run command with optional configuration overrides."""
    command = [str(cli or find_flower_cli()), "run", ".", federation]
    if run_config is not None:
        command.extend(("--run-config", run_config))
    if federation_config is not None:
        command.extend(("--federation-config", federation_config))
    if stream:
        command.append("--stream")
    return command


def run_flower_app(
    federation: str,
    *,
    run_config: str | None = None,
    federation_config: str | None = None,
    stream: bool = True,
    cwd: Path,
    check: bool = True,
    cli: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the configured Flower app with a consistent project environment."""
    executable = cli or find_flower_cli()
    return subprocess.run(
        flower_run_command(
            federation,
            run_config=run_config,
            federation_config=federation_config,
            stream=stream,
            cli=executable,
        ),
        check=check,
        cwd=cwd,
        env=flower_environment(executable),
        text=True,
    )


def collect_results(*, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the standard result collector from a project checkout."""
    return subprocess.run(
        [sys.executable, str(cwd / "scripts" / "collect_results.py")],
        check=True,
        cwd=cwd,
        text=True,
    )
