from __future__ import annotations
import argparse
import torch
from .config import RunConfig
from .data import all_pixels_labels, load_client_data, load_test_data, resolve_device


def main() -> None:
    p = argparse.ArgumentParser(description="Download CIFAR-10/ResNet weights and optionally build feature caches.")
    p.add_argument("--num-clients", type=int, default=50)
    p.add_argument("--cache-all", action="store_true", help="Build feature cache for every client (can take several minutes).")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = RunConfig(num_clients=args.num_clients, max_colluders=min(10, max(0,args.num_clients//5)), max_stragglers=min(10,max(0,args.num_clients//5)), device=args.device)
    cfg.validate()
    device = resolve_device(cfg.device)
    all_pixels_labels(True, cfg); all_pixels_labels(False, cfg)
    print("CIFAR-10 downloaded/cached.")
    load_test_data(cfg, device)
    print("Test ResNet features cached.")
    if args.cache_all:
        for pid in range(cfg.num_clients):
            load_client_data(cfg, pid, device)
            print(f"Cached client {pid+1}/{cfg.num_clients}")
    print("Preparation complete.")

if __name__ == "__main__":
    main()
