# Exploratory training analysis: 10 October 2026

**Status:** Exploratory post-run review; not a confirmatory result.  
**Machine:** `ndag-m-lab-linux-x86_64-7fa78ccc` (NVIDIA GeForce RTX 3090).  
**Dataset/model:** CIFAR-10, 50,000 train and 10,000 test examples; torchvision
`ResNet18_Weights.IMAGENET1K_V1`.  
**Run settings:** seed 2026, 50 clients, 30 rounds, IID partition, max 10
colluders and 10 simulated stragglers, configured epsilon 6 for private methods.

**Implementation follow-up:** this record describes the earlier runs and their
test-set histories. The application has since gained a standard FedAvg path
and validation-only round evaluation, with a single final test evaluation.
Those code changes have not yet produced replacement training results; all
values below remain historical observations of the earlier implementation.

## Observed results

Accuracy and loss are the values recorded in each run's `history.csv`. The
evaluation function uses the centralized CIFAR-10 test data; it is evaluated
at initialization and every communication round. “Best” below means the
largest accuracy observed in this repeatedly evaluated test history. It is
descriptive only and must not be used as an unbiased test estimate or as a
checkpoint-selection rule.

| Method | Accuracy at round 30 | Highest logged test accuracy (round) | Accuracy change, round 0 to 30 | Elapsed start-to-final-log time* |
|---|---:|---:|---:|---:|
| No DP | 53.17% | 71.31% (12) | +43.08 percentage points | 15.3 min |
| SMPC-DP, ε=6 | 53.26% | 71.60% (13) | +43.17 percentage points | 16.5 min |
| LightDP, ε=6 | 51.62% | 70.04% (12) | +41.53 percentage points | 19.9 min |
| Vanilla local noise, ε=6 | 51.25% | 67.98% (12) | +41.16 percentage points | 14.9 min |

All four runs logged 30 aggregation rounds, zero failures, 153 simulated
stragglers total, and an average of 44.9 active clients per round. The private
runs logged epsilon increasing from 0 to approximately 6. The no-DP baseline
logged infinite epsilon.

The common learning-curve pattern is a rise to roughly 68–72% test accuracy by
rounds 12–13, followed by a decline to roughly 51–53% at round 30. Loss
decreased from initialization to round 30 in all four runs (approximately
2.30 to 1.83–1.84), even as accuracy fell from its earlier maximum. This
accuracy/loss divergence and late decline need investigation. These logs alone
do not establish whether the cause is the optimization schedule, evaluation
pipeline, data/feature handling, or another implementation detail.

The round-30 accuracy differences are small for No DP versus SMPC-DP (0.09
points), and are 1.55 points for No DP versus LightDP and 1.92 points for No DP
versus vanilla local noise. These are single-run observations, not evidence of
statistical significance, a general method ranking, or a causal privacy cost.

*Elapsed time is computed from `started_at_utc` in the config to the timestamp
of the final line in `training.log`. It does not include prior setup/downloads
and is not a standardized timing benchmark.

## Run records

All four output directories are under
`results/experiments/ndag-m-lab-linux-x86_64-7fa78ccc/`:

- `20261009T184343833086Z_No_DP`
- `20261009T184343833086Z_SMPC_DP_eps6`
- `20261009T184343833086Z_LightDP_eps6`
- `20261009T184343833086Z_Vanilla_local_noise_adding_eps6`

Each has `run_config.json`, `history.csv`, `train_rounds.csv`, and
`training.log`. The complete machine-readable records are the source for the
table above; archived notebook results were not used.

## Limitations and provenance

- Each method has one run with configured seed 2026. There is no estimate of
  seed-to-seed variability or uncertainty interval.
- The same test split was evaluated after every round. The peak test accuracies
  are therefore repeated observations, not a fresh final test. Selecting a
  round or method using these test values would leak test information into
  model selection.
- All configs mark the Git worktree dirty. No-DP, SMPC-DP, and vanilla local
  noise record revision `b0c681e0a24423154a460308e7fb52e90aab392a`; LightDP
  records `afdb5a9e15bdfa0d97b761932f021c56556f54ae`. The LightDP revision
  differs, and dirty worktrees mean none of the configs fully identify the
  executed source state. Method comparisons are therefore not strictly
  reproducible or controlled.
- The run configs identify a CUDA RTX 3090, CUDA 12.8, Python 3.10.19,
  Flower 1.25.0, PyTorch 2.8.0, and torchvision 0.23.0. These details do not
  resolve the dirty-worktree uncertainty.
- The LightDP implementation simulates pairwise masks with matching seeds;
  SMPC-DP is an idealized secure-aggregation baseline. The runs do not
  demonstrate production cryptographic security.
- No confirmatory protocol was recorded before inspecting these outcomes.

## Recommended next step

Investigate the shared late-round accuracy decline before spending compute on a
larger comparison. Do not select a round based on the existing test history.

1. Review the round evaluation path and saved model state. Check that the model
   evaluated at each round corresponds to the parameters just aggregated, and
   that evaluation inputs/features and labels remain aligned and deterministic.
2. Review server update and learning-rate behavior around rounds 12–15. The
   configured learning rate is cosine-decayed with a nonzero floor; do not
   assume the schedule caused the decline without a controlled diagnostic.
3. Create a prospective experiment protocol before new outcome inspection.
   Define a validation split drawn from training data for any stopping or
   checkpoint selection, and evaluate the test split once under the fixed
   selection rule. Record this split change because it affects comparability.
4. Pin a clean code revision and rerun all methods under the same revisions,
   data split, client partitions, and compute settings. Use multiple
   independently specified seeds and report every run, aggregate statistics,
   and uncertainty. Determine the seed count and analysis plan in the protocol
   before launching.

Until those checks and reruns are complete, treat these outputs as a software
and exploratory training observation, not a claim that one privacy method
matches or outperforms another.
