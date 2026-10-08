from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightdp_fl.model import HybridClassifier
from lightdp_fl.training import _functional_helpers, clipped_client_gradient


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_mode(
    mode: str,
    device: torch.device,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    microbatch: int,
    repeats: int,
    warmups: int,
    compile_mode: str,
) -> dict[str, object]:
    torch.manual_seed(23)
    model = HybridClassifier().to(device)
    autocast_dtype = None
    compile_request = None
    if mode == "autocast_bfloat16":
        autocast_dtype = torch.bfloat16
    elif mode == "autocast_float16":
        autocast_dtype = torch.float16
    elif mode == "compile":
        compile_request = compile_mode

    helpers = _functional_helpers(model, compile_request)
    def run() -> torch.Tensor:
        return clipped_client_gradient(
            model,
            pixels,
            labels,
            features,
            round_zero_based=0,
            clip=0.05,
            microbatch=min(microbatch, len(labels)),
            device=device,
            autocast_dtype=autocast_dtype,
            gradient_helpers=helpers,
        )

    _sync(device)
    warm_started = time.perf_counter()
    output = None
    for _ in range(warmups):
        output = run()
    _sync(device)
    warmup_seconds = time.perf_counter() - warm_started

    timings = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        output = run()
        _sync(device)
        timings.append(time.perf_counter() - started)

    assert output is not None
    return {
        "median_seconds": statistics.median(timings),
        "samples_seconds": timings,
        "warmup_seconds": warmup_seconds,
        "result": output.detach().cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eager FP32 with optional autocast and torch.compile gradient paths."
    )
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.examples <= 0 or args.microbatch <= 0 or args.repeats <= 0 or args.warmups <= 0:
        parser.error("examples, microbatch, repeats, and warmups must all be > 0")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but it is not available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    torch.manual_seed(23)
    pixels = torch.rand(args.examples, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (args.examples,), device=device)
    features = torch.randn(2, args.examples, 512, device=device)

    modes = ["eager_fp32"]
    if device.type == "cuda":
        modes.append("autocast_float16")
        if torch.cuda.is_bf16_supported():
            modes.append("autocast_bfloat16")
    elif hasattr(torch, "autocast"):
        modes.append("autocast_bfloat16")
    if hasattr(torch, "compile"):
        modes.append("compile")

    raw: dict[str, dict[str, object]] = {}
    baseline = None
    for mode in modes:
        try:
            result = _run_mode(
                mode,
                device,
                pixels,
                labels,
                features,
                args.microbatch,
                args.repeats,
                args.warmups,
                args.compile_mode,
            )
            vector = result.pop("result")
            assert isinstance(vector, torch.Tensor)
            if baseline is None:
                baseline = vector
            baseline_norm = baseline.norm().clamp_min(1e-12)
            result["relative_l2_error_vs_eager"] = float(
                ((vector - baseline).norm() / baseline_norm).item()
            )
            raw[mode] = result
        except Exception as exc:  # Record unsupported compiler/backend combinations.
            raw[mode] = {"error": f"{type(exc).__name__}: {exc}"}

    eager_seconds = float(raw["eager_fp32"]["median_seconds"])
    for result in raw.values():
        if "median_seconds" in result:
            result["speedup_vs_eager"] = eager_seconds / float(result["median_seconds"])

    output_path = args.output or (
        PROJECT_ROOT
        / "results"
        / "benchmarks"
        / f"training_modes_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_kind": "synthetic clipped-gradient microbenchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "examples": args.examples,
        "microbatch": min(args.microbatch, args.examples),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "compile_mode": args.compile_mode,
        "results": raw,
        "interpretation": (
            "Synthetic microbenchmark only; compare task metrics and privacy behavior before "
            "adopting a mode in a research run."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    for mode, result in raw.items():
        print(mode, result)


if __name__ == "__main__":
    main()
