from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "results" / "experiments"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"


def main() -> None:
    rows: list[dict] = []
    histories: list[pd.DataFrame] = []
    train_rounds: list[pd.DataFrame] = []

    for config_path in sorted(RUNS_DIR.rglob("run_config.json")):
        folder = config_path.parent
        history_path = folder / "history.csv"
        if not history_path.is_file():
            continue

        history = pd.read_csv(history_path)
        if history.empty:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_key = folder.relative_to(RUNS_DIR).as_posix()
        history["run"] = run_key
        histories.append(history)

        train_path = folder / "train_rounds.csv"
        if train_path.is_file():
            train = pd.read_csv(train_path)
            train["run"] = run_key
            train_rounds.append(train)

        last = history.iloc[-1].to_dict()
        final_metrics_path = folder / "final_metrics.json"
        final_metrics = (
            json.loads(final_metrics_path.read_text(encoding="utf-8"))
            if final_metrics_path.is_file()
            else {}
        )
        last.update(
            {
                "run": run_key,
                "tag": config.get("tag", folder.name),
                "run_id": config.get("run_id"),
                "run_number": config.get("run_number"),
                "machine_id": config.get("machine_id"),
                "hostname": config.get("hostname"),
                "method": config["method"],
                "training_algorithm": config.get("training_algorithm", "legacy_unknown"),
                "partition_method": config.get("partition_method", "legacy_unknown"),
                "client_data_mode": config.get("client_data_mode", "legacy_unknown"),
                "dirichlet_alpha": config.get("dirichlet_alpha"),
                "validation_fraction": config.get("validation_fraction"),
                "local_epochs": config.get("local_epochs"),
                "batch_size": config.get("batch_size"),
                "local_learning_rate": config.get("local_learning_rate"),
                "epsilon_target": config.get("epsilon"),
                "num_clients": config["num_clients"],
                "max_colluders": config["max_colluders"],
                "max_stragglers": config["max_stragglers"],
                "clip": config["clip"],
                "client_update_clip": config.get("client_update_clip"),
                "privacy_calibration": config.get("privacy_calibration"),
                "learning_rate": config["learning_rate"],
                "started_at_utc": config.get("started_at_utc"),
                **final_metrics,
            }
        )
        rows.append(last)

    if not rows:
        print(f"No completed runs found under {RUNS_DIR}")
        return

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    final = pd.DataFrame(rows).sort_values("run")
    all_history = pd.concat(histories, ignore_index=True)
    final.to_csv(SUMMARY_DIR / "all_runs.csv", index=False)
    all_history.to_csv(SUMMARY_DIR / "training_history.csv", index=False)
    if train_rounds:
        pd.concat(train_rounds, ignore_index=True).to_csv(
            SUMMARY_DIR / "training_rounds.csv", index=False
        )

    main_runs = final[final["tag"].str.contains("(?:FedAvg|No_DP|Vanilla_local_noise_adding_eps|SMPC(?:_D|\\+D)P_eps|LightDP_eps)", regex=True)]
    if not main_runs.empty:
        figure, axis = plt.subplots(figsize=(11, 6))
        for _, run in main_runs.iterrows():
            values = all_history[all_history["run"] == run["run"]]
            metric = "val_accuracy" if "val_accuracy" in values else "accuracy"
            axis.plot(values["round"], values[metric], marker="o", label=run["tag"])
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Recorded evaluation accuracy (%)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(SUMMARY_DIR / "main_accuracy.png", dpi=180)
        plt.close(figure)

    print(f"Wrote summaries to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
