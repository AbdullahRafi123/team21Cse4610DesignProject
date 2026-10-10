from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path

from lightdp_fl.flower_cli import (
    collect_results,
    find_flower_cli,
    run_flower_app,
    serialize_run_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEPS=[
    ("Reference",{}),("epsilon_3",{"epsilon":3.0}),("epsilon_9",{"epsilon":9.0}),
    ("clients_25",{"num-clients":25,"max-colluders":10,"max-stragglers":10}),
    ("clients_100",{"num-clients":100,"max-colluders":10,"max-stragglers":10}),
    ("colluders_20",{"max-colluders":20}),("stragglers_20",{"max-stragglers":20}),
    ("client_update_clip_0.5",{"client-update-clip":0.5}),
    ("local_lr_0.005",{"local-learning-rate":0.005}),
    ("nonIID_dirichlet",{"partition-method":"dirichlet"}),
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--run-id", help="Optional output prefix. Defaults to the current UTC timestamp.")
    p.add_argument("--gpu", action="store_true", help="Use the CUDA simulation profile (requires CUDA-enabled PyTorch).")
    args=p.parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    device = "cuda" if args.gpu else "auto"
    flower_cli = find_flower_cli()
    for name,changes in SWEEPS:
        base={"method":"lightdp","training-algorithm":"fedavg","epsilon":6.0,
              "tag":f"{run_id}_Sweep_{name}","num-clients":50,"num-server-rounds":args.rounds,
              "max-colluders":10,"max-stragglers":10,"max-records-per-client":0,
              "client-update-clip":1.0,"simulate-stragglers":True,
              "partition-method":"iid","dirichlet-alpha":0.5,
              "validation-fraction":0.1,"device":device}
        base.update(changes);N=int(base["num-clients"])
        cfg=serialize_run_config(base)
        profile_key = "gpu" if args.gpu else "suite"
        if N == 50:
            federation = "local-simulation-gpu" if args.gpu else "local-simulation-suite"
        else:
            federation = f"local-simulation-{profile_key}-{N}"
        print(f"\nRUNNING: federation={federation}, tag={base['tag']}",flush=True)
        run_flower_app(federation,run_config=cfg,cwd=PROJECT_ROOT,cli=flower_cli)
    collect_results(cwd=PROJECT_ROOT)

if __name__=="__main__":main()
