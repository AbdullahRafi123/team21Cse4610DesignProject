from __future__ import annotations
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEPS=[
    ("Reference",{}),("epsilon_3",{"epsilon":3.0}),("epsilon_9",{"epsilon":9.0}),
    ("clients_25",{"num-clients":25,"max-colluders":10,"max-stragglers":10}),
    ("clients_100",{"num-clients":100,"max-colluders":10,"max-stragglers":10}),
    ("colluders_20",{"max-colluders":20}),("stragglers_20",{"max-stragglers":20}),
    ("clip_0.5",{"clip":0.5}),("lr_0.05",{"learning-rate":0.05}),("nonIID",{"non-iid":True}),
]

def toml(v):
    if isinstance(v,bool): return "true" if v else "false"
    if isinstance(v,str): return f'"{v}"'
    return str(v)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--run-id", help="Optional output prefix. Defaults to the current UTC timestamp.")
    args=p.parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name,changes in SWEEPS:
        base={"method":"lightdp","epsilon":6.0,"tag":f"{run_id}_Sweep_{name}","num-clients":50,"num-server-rounds":args.rounds,
              "max-colluders":10,"max-stragglers":10,"max-records-per-client":0,"microbatch":32,"non-iid":False}
        base.update(changes);N=int(base["num-clients"])
        cfg=" ".join(f'{k}={toml(v)}' for k,v in base.items())
        cmd=["flwr","run",".","local-simulation","--run-config",cfg,"--federation-config",f"options.num-supernodes={N}","--stream"]
        print("\nRUNNING:"," ".join(cmd),flush=True);subprocess.run(cmd,check=True,cwd=PROJECT_ROOT)
    subprocess.run([sys.executable,str(PROJECT_ROOT / "scripts/collect_results.py")],check=True,cwd=PROJECT_ROOT)

if __name__=="__main__":main()
