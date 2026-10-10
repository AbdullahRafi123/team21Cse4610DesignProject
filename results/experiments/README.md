# Experiment records

Active Flower runs are written to
`results/experiments/<machine-id>/<tag>/`. Runs using the default `tag = "auto"`
get a unique UTC and machine-labelled ID. On completion, the app stages the
canonical run records in Git without creating a commit; set
`git-track-results = false` to disable staging. Keep the effective configuration,
evaluation history, per-round training metrics, and full text log together.
These are generated records: do not hand-edit metrics or use a plot without
preserving its inputs and generation procedure.

## October 2026 comparison

The four completed 30-round runs tagged `20261009T184343833086Z_*` are reviewed
in [the exploratory analysis](../../docs/EXPERIMENT_ANALYSIS_2026-10-10.md).
That review reports their observed curves and documents limitations including
one seed, repeated test-set evaluation, dirty worktrees, and differing recorded
Git revisions. Treat them as exploratory outputs, not confirmatory evidence.

## Expected files

- `run_config.json`: effective configuration and available provenance.
- `history.csv`: per-round validation loss and accuracy plus the configured
  epsilon estimate when a privacy mechanism is selected. For FedAvg privacy
  variants, see `privacy_calibration` in `run_config.json` for its research
  status and client-update sensitivity.
- `final_metrics.json`: final test loss/accuracy after the configured training
  is complete; the test split is evaluated once.
- `train_rounds.csv`: active/missing clients, training algorithm, aggregation
  rule, update clip statistics for private FedAvg, client timing, example
  counts, and failures.
  Plain FedAvg records example-weighted aggregation; private FedAvg records a
  uniform mean across client updates.
- `training.log`: setup and round progress.
- `git_tracking` in `run_config.json`: records automatic staging status.
- `checkpoints/`: machine-local resume/final checkpoints; excluded from Git.

Client progress records and Flower/Ray state files may be transient. Do not
commit datasets, caches, private data, credentials, virtual environments, or
generated checkpoints. See [research standards](../../docs/RESEARCH_STANDARDS.md)
before planning or reporting experiments.
