from __future__ import annotations
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUNS=[
    ("no_dp", 6.0, "No_DP"),
    *[(method,eps,f"{tag}_eps{int(eps)}") for eps in (3.0,6.0,9.0) for method,tag in [
        ("vanilla_dp","Vanilla_local_noise_adding"),("smpc_dp","SMPC+DP"),("lightdp","LightDP")
    ]],
]

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
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs = RUNS if args.epsilon is None else [
        RUNS[0], *(run for run in RUNS[1:] if run[1] == args.epsilon)
    ]
    federation = "local-simulation-gpu" if args.gpu else "local-simulation"
    device = "cuda" if args.gpu else "auto"
    for method,eps,tag in runs:
        cfg=(f'method="{method}" epsilon={eps} tag="{run_id}_{tag}" num-clients=50 num-server-rounds={args.rounds} '
             f'max-colluders=10 max-stragglers=10 max-records-per-client=0 microbatch=32 non-iid=false device="{device}"')
        cmd=["flwr","run",".",federation,"--run-config",cfg,"--federation-config","options.num-supernodes=50"]
        if args.stream: cmd.append("--stream")
        print("\nRUNNING:"," ".join(cmd),flush=True)
        subprocess.run(cmd,check=True,cwd=PROJECT_ROOT)
    subprocess.run([sys.executable,str(PROJECT_ROOT / "scripts/collect_results.py")],check=True,cwd=PROJECT_ROOT)

if __name__=="__main__": main()
