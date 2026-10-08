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

    for folder in sorted(RUNS_DIR.glob("*")):
        history_path = folder / "history.csv"
        config_path = folder / "run_config.json"
        if not history_path.is_file() or not config_path.is_file():
            continue

        history = pd.read_csv(history_path)
        if history.empty:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        history["run"] = folder.name
        histories.append(history)

        train_path = folder / "train_rounds.csv"
        if train_path.is_file():
            train = pd.read_csv(train_path)
            train["run"] = folder.name
            train_rounds.append(train)

        last = history.iloc[-1].to_dict()
        last.update(
            {
                "run": folder.name,
                "tag": config.get("tag", folder.name),
                "method": config["method"],
                "epsilon_target": config.get("epsilon"),
                "num_clients": config["num_clients"],
                "max_colluders": config["max_colluders"],
                "max_stragglers": config["max_stragglers"],
                "clip": config["clip"],
                "learning_rate": config["learning_rate"],
                "started_at_utc": config.get("started_at_utc"),
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

    main_runs = final[final["tag"].str.contains("(?:No_DP|Vanilla_local_noise_adding_eps|SMPC\\+DP_eps|LightDP_eps)", regex=True)]
    if not main_runs.empty:
        figure, axis = plt.subplots(figsize=(11, 6))
        for _, run in main_runs.iterrows():
            values = all_history[all_history["run"] == run["run"]]
            axis.plot(values["round"], values["accuracy"], marker="o", label=run["tag"])
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Test accuracy (%)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(SUMMARY_DIR / "main_accuracy.png", dpi=180)
        plt.close(figure)

    print(f"Wrote summaries to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
