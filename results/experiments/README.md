# New experiment runs

Each run writes its configuration, evaluation history, client-training round metrics, and text log to `results/experiments/<machine-id>/<tag>/`. These small records are Git-trackable. Model checkpoints are kept in each run's `checkpoints/` subdirectory and ignored by Git; Flower/Ray state files are also ignored.
