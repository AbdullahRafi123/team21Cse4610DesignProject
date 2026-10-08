from __future__ import annotations
import argparse, subprocess, sys

RUNS=[
    ("no_dp", 6.0, "No_DP"),
    *[(method,eps,f"{tag}_eps{int(eps)}") for eps in (3.0,6.0,9.0) for method,tag in [
        ("vanilla_dp","Vanilla_local_noise_adding"),("smpc_dp","SMPC+DP"),("lightdp","LightDP")
    ]],
]

def main():
    p=argparse.ArgumentParser(description="Run the 10 main Flower experiments matching the source notebook.")
    p.add_argument("--rounds",type=int,default=8,help="Executed source results used 8 rounds.")
    p.add_argument("--stream",action="store_true",default=True)
    args=p.parse_args()
    for method,eps,tag in RUNS:
        cfg=(f'method="{method}" epsilon={eps} tag="{tag}" num-clients=50 num-server-rounds={args.rounds} '
             'max-colluders=10 max-stragglers=10 max-records-per-client=0 microbatch=32 non-iid=false')
        cmd=["flwr","run",".","local-simulation","--run-config",cfg,"--federation-config","options.num-supernodes=50"]
        if args.stream: cmd.append("--stream")
        print("\nRUNNING:"," ".join(cmd),flush=True)
        subprocess.run(cmd,check=True)
    subprocess.run([sys.executable,"scripts/collect_results.py"],check=True)

if __name__=="__main__": main()
