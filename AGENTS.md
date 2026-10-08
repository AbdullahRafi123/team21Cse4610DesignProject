# Repository guidance for agents

This file applies to the entire repository. Read it before changing code, experiment configuration, notebooks, or reported results. The project-specific research rules are in [`docs/RESEARCH_STANDARDS.md`](docs/RESEARCH_STANDARDS.md).

## Research integrity

- Do not invent measurements, citations, protocol details, or claims. Mark a statement as measured, estimated, inferred, simulated, or proposed where that distinction matters.
- Do not describe the LightDP or SMPC implementation as production cryptography. The current implementation simulates pairwise masks, and the SMPC baseline is idealized; preserve these boundaries in code and prose.
- Treat `results/archive/` and notebooks as source records. Never edit historical metrics or plots to match current code. If a correction is needed, preserve the original and add a dated, reproducible replacement with an explanation.
- Do not present smoke tests, one-off runs, or archived notebook outputs as confirmatory evidence. Follow the replication and reporting requirements in the research standards.
- Cite datasets, model weights, prior methods, and borrowed code. Confirm licenses and attribution requirements before redistributing assets.

## Engineering and experiment changes

- The maintained application is `src/lightdp_fl/`; configs are in `configs/`; experiment and collection entry points are in `scripts/`.
- Keep reusable behavior in the package and keep scripts thin. Follow PEP 8, use clear names and type hints for public functions, validate configuration, and use deterministic seeds where the algorithm permits.
- Preserve algorithmic invariants with focused tests when changing privacy accounting, client partitioning, aggregation, or result schemas. Keep tests small and independent of network downloads and full training runs.
- Do not hard-code machine-specific paths, credentials, or local cache locations. Never add downloaded datasets, secrets, virtual environments, generated checkpoints, or private data to Git.
- New run outputs belong under `results/experiments/<run-id>/`. Preserve `run_config.json`, `history.csv`, `train_rounds.csv`, and `training.log` as machine-readable provenance. Checkpoints remain ignored. Never fabricate or hand-edit generated metrics.
- When changing a method, metric, data split, preprocessing step, or default, update its configuration documentation and describe the impact on comparability with earlier results.
- Keep archived output separate from active code. Remove an artifact only when it is demonstrably redundant or reproducibly regenerable; explain removals that affect research provenance.
- Do not rewrite shared Git history or force-push unless the user explicitly asks for it.

## Review and reporting

- Read the relevant code, configuration, and existing result provenance before editing.
- For code changes, identify relevant checks and report what was run and what was not. Do not run tests unless the user asks for testing or verification.
- Keep documentation consistent with the actual CLI, configuration keys, output paths, and limitations.
- In the final summary, state the key changes, validation performed, and any unresolved limitation. Do not claim publication readiness unless the evidence and checklist support it.
